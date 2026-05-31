#!/usr/bin/env python3
"""
Reachy Mini - Morning Greeter
==============================
Watches the camera for a new face and spontaneously greets them with
"Good morning! How are you today?" along with expressive robot gestures.

This is designed as the opening act of a longer conversation session.

Usage
-----
    # Real robot (auto-connect, robot's camera)
    python morning_greeter.py

    # Simulation / dev on Windows: daemon running with --sim, webcam for vision
    python morning_greeter.py --sim

    # Test TTS + vision without any robot connection
    python morning_greeter.py --no-robot

    # Override greeting cooldown (seconds before re-greeting same visitor)
    python morning_greeter.py --cooldown 30

How it works
------------
1. Continuously grabs frames (robot camera or local webcam).
2. Runs OpenCV Haar-cascade face detection on each frame.
3. Tracks presence/absence of faces:
   - "No face for N seconds" -> visitor has left, reset greeted flag.
   - New face detected and not greeted recently -> trigger greeting.
4. Greeting sequence (runs concurrently):
   - Robot: looks up attentively, wiggles antennas, head wobble during speech.
   - TTS:   says one of several warm "good morning" phrases.
5. Returns to neutral idle pose when done.

Extending to a full conversation
---------------------------------
After the greeting, you can hand off to the Reachy Mini Conversation App
(https://github.com/pollen-robotics/reachy_mini_conversation_app) or wire up
your own LLM pipeline by listening to mini.media audio after the greeting.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import threading
import time
from collections.abc import Sequence

import cv2
import numpy as np

# ── Optional robot imports ────────────────────────────────────────────────────
try:
    from reachy_mini import ReachyMini
    from reachy_mini.utils import create_head_pose

    REACHY_AVAILABLE = True
except ImportError:
    REACHY_AVAILABLE = False
    print("[WARNING] reachy_mini package not found — running in --no-robot mode.")

# ── Optional pyttsx3 TTS ──────────────────────────────────────────────────────
try:
    import pyttsx3

    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("[WARNING] pyttsx3 not found — TTS will be disabled (install: pip install pyttsx3).")


# ═════════════════════════════════════════════════════════════════════════════
#  Configuration
# ═════════════════════════════════════════════════════════════════════════════

# Seconds with no detected face before we consider the visitor "gone" and
# allow a fresh greeting on their return.
FACE_ABSENT_RESET_S: float = 5.0

# Smallest face bounding-box width (pixels) to accept.
# Increase to ignore distant/small detections.
MIN_FACE_PX: int = 80

# Haar cascade tuning
HAAR_SCALE_FACTOR: float = 1.3
HAAR_MIN_NEIGHBORS: int = 5

# Greeting messages (cycled in order so repeat visits get variety)
GREETINGS: list[str] = [
    "Good morning! How are you today?",
    "Good morning! It's wonderful to see you! How are you doing?",
    "Good morning! Hope you're feeling great! How are you?",
    "Good morning! What a lovely day! How are you doing today?",
]


# ═════════════════════════════════════════════════════════════════════════════
#  TTS helpers
# ═════════════════════════════════════════════════════════════════════════════

def _build_tts_engine() -> "pyttsx3.Engine | None":
    """Initialise pyttsx3 and pick a friendly voice if possible."""
    if not TTS_AVAILABLE:
        return None

    engine = pyttsx3.init()
    engine.setProperty("rate", 150)   # slightly slower → warmer feel
    engine.setProperty("volume", 1.0)

    voices = engine.getProperty("voices")
    # Prefer a female voice for a friendlier greeting (Windows: Zira / Hazel)
    for voice in voices:
        name_lower = voice.name.lower()
        if any(kw in name_lower for kw in ("zira", "hazel", "female", "eva", "susan")):
            engine.setProperty("voice", voice.id)
            print(f"[TTS] Using voice: {voice.name}")
            break

    return engine


def speak(engine: "pyttsx3.Engine | None", text: str) -> None:
    """Say *text* aloud. Falls back to console print if TTS not available."""
    print(f"[SPEAK] {text}")
    if engine is None:
        return
    engine.say(text)
    engine.runAndWait()


def speak_to_file(engine: "pyttsx3.Engine | None", text: str) -> str | None:
    """
    Synthesize *text* and save to a temporary WAV file.
    Returns the file path, or None if TTS is unavailable.
    Used when you want to pipe audio through the robot's speaker.
    """
    if engine is None:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    engine.save_to_file(text, tmp.name)
    engine.runAndWait()
    return tmp.name


# ═════════════════════════════════════════════════════════════════════════════
#  Robot motion helpers
# ═════════════════════════════════════════════════════════════════════════════

def greeting_animation(mini: "ReachyMini") -> None:  # type: ignore[name-defined]
    """
    Physical greeting sequence:
      1. Look up attentively.
      2. Enthusiastic antenna wiggle.
      3. Slight head tilt (curious / friendly).
    Runs synchronously — call from a background thread so TTS can overlap.
    """
    # Look up slightly
    mini.goto_target(
        head=create_head_pose(z=8, mm=True),
        antennas=[0.0, 0.0],
        duration=0.5,
        method="minjerk",
    )
    time.sleep(0.4)

    # Antenna wiggle × 3
    for _ in range(3):
        mini.goto_target(antennas=[0.9, -0.9], duration=0.18)
        time.sleep(0.16)
        mini.goto_target(antennas=[-0.9, 0.9], duration=0.18)
        time.sleep(0.16)

    # Hold antennas up (happy)
    mini.goto_target(antennas=[0.6, 0.6], duration=0.3)
    time.sleep(0.25)

    # Friendly head tilt
    mini.goto_target(
        head=create_head_pose(z=5, roll=12, mm=True, degrees=True),
        duration=0.6,
        method="minjerk",
    )
    time.sleep(0.5)


def idle_pose(mini: "ReachyMini") -> None:  # type: ignore[name-defined]
    """Return robot to neutral resting pose."""
    mini.goto_target(
        head=create_head_pose(),
        antennas=[0.0, 0.0],
        duration=1.2,
        method="minjerk",
    )


def gentle_scan(mini: "ReachyMini") -> None:  # type: ignore[name-defined]
    """Slow head pan left-right to signal the robot is 'looking around'."""
    for yaw_deg in (15, -15, 0):
        mini.goto_target(
            head=create_head_pose(z=0, mm=True),
            body_yaw=float(np.deg2rad(yaw_deg)),
            duration=1.5,
            method="ease_in_out",
        )
        time.sleep(1.3)


# ═════════════════════════════════════════════════════════════════════════════
#  Face detection
# ═════════════════════════════════════════════════════════════════════════════

class FaceDetector:
    def __init__(self) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"Could not load Haar cascade from {cascade_path}")

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Return list of (x, y, w, h) bounding boxes for detected faces."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raw = self._cascade.detectMultiScale(
            gray,
            scaleFactor=HAAR_SCALE_FACTOR,
            minNeighbors=HAAR_MIN_NEIGHBORS,
            minSize=(MIN_FACE_PX, MIN_FACE_PX),
        )
        if len(raw) == 0:
            return []
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in raw]


# ═════════════════════════════════════════════════════════════════════════════
#  Visitor tracker  (presence / absence logic)
# ═════════════════════════════════════════════════════════════════════════════

class VisitorTracker:
    """
    Tracks whether we've already greeted the current visitor.

    State machine:
      - Faces absent for > FACE_ABSENT_RESET_S  →  visitor left; reset flag.
      - Face detected + not yet greeted + cooldown elapsed  →  greet!
    """

    def __init__(self, cooldown_s: float) -> None:
        self._cooldown_s = cooldown_s
        self._last_face_at: float = 0.0
        self._last_greeted_at: float = 0.0
        self._greeted_this_visit: bool = False
        self._lock = threading.Lock()

    def on_face_seen(self) -> None:
        with self._lock:
            self._last_face_at = time.time()

    def should_greet(self) -> bool:
        now = time.time()
        with self._lock:
            # Reset greeted flag if visitor has been gone long enough
            if now - self._last_face_at > FACE_ABSENT_RESET_S:
                self._greeted_this_visit = False

            if self._greeted_this_visit:
                return False
            if now - self._last_greeted_at < self._cooldown_s:
                return False
            return True

    def mark_greeted(self) -> None:
        with self._lock:
            self._last_greeted_at = time.time()
            self._greeted_this_visit = True


# ═════════════════════════════════════════════════════════════════════════════
#  Main greeter class
# ═════════════════════════════════════════════════════════════════════════════

class MorningGreeter:
    def __init__(
        self,
        cooldown_s: float = 20.0,
        show_video: bool = False,
        debug_detections: bool = False,
    ) -> None:
        self._detector = FaceDetector()
        self._tracker = VisitorTracker(cooldown_s)
        self._tts = _build_tts_engine()
        self._greeting_idx = 0
        self._running = False
        self._show_video = show_video
        self._debug_detections = debug_detections
        self._debug_last_log_at = 0.0
        self._debug_frames = 0

    def _draw_debug_overlay(self, frame: np.ndarray, faces: list[tuple[int, int, int, int]]) -> np.ndarray:
        """Draw face boxes and status text on a copy of the frame for display."""
        overlay = frame.copy()
        for x, y, w, h in faces:
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 80), 2)
            cv2.putText(
                overlay,
                f"face {w}x{h}",
                (x, max(18, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 80),
                1,
            )

        status = "FACE DETECTED" if faces else "Watching..."
        cv2.putText(
            overlay,
            f"Reachy Mini Morning Greeter  [{status}]",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 200, 255),
            2,
        )
        return overlay

    def _log_detection_debug(self, faces: list[tuple[int, int, int, int]]) -> None:
        """Print low-rate detection debug to avoid flooding the console."""
        if not self._debug_detections:
            return

        now = time.time()
        self._debug_frames += 1
        if now - self._debug_last_log_at < 1.0:
            return

        self._debug_last_log_at = now
        if faces:
            faces_str = ", ".join(f"(x={x}, y={y}, w={w}, h={h})" for x, y, w, h in faces)
            print(f"[DEBUG] frames={self._debug_frames} faces={len(faces)} {faces_str}")
        else:
            print(f"[DEBUG] frames={self._debug_frames} faces=0")

    # ── Greeting orchestration ────────────────────────────────────────────────

    def _next_greeting(self) -> str:
        msg = GREETINGS[self._greeting_idx % len(GREETINGS)]
        self._greeting_idx += 1
        return msg

    def _do_greeting(self, mini: "ReachyMini | None") -> None:  # type: ignore[name-defined]
        """
        Run the full greeting: animation (background thread) + TTS (current thread).
        """
        greeting_text = self._next_greeting()
        print(f"\n[GREET] Greeting visitor: \"{greeting_text}\"")

        if mini is not None:
            # Start physical animation in background
            anim_thread = threading.Thread(
                target=greeting_animation, args=(mini,), daemon=True
            )
            anim_thread.start()

            # Enable head wobble while speaking (syncs with TTS duration)
            try:
                mini.enable_wobbling()
            except AttributeError:
                pass  # older SDK versions may not have this

        # Slight pause so animation starts before voice
        time.sleep(0.35)

        # Speak (blocks until done on main thread — safe for pyttsx3/SAPI5)
        speak(self._tts, greeting_text)

        if mini is not None:
            try:
                mini.disable_wobbling()
            except AttributeError:
                pass
            # Wait for animation to settle then go idle
            anim_thread.join(timeout=6.0)  # type: ignore[possibly-undefined]
            time.sleep(0.3)
            idle_pose(mini)

    # ── Run modes ─────────────────────────────────────────────────────────────

    def run_with_robot_camera(self) -> None:
        """
        Production mode: use Reachy Mini's built-in camera for face detection.
        Works with real Reachy Mini (Wireless or Lite via USB).
        """
        if not REACHY_AVAILABLE:
            print("[ERROR] reachy_mini not installed. Use --no-robot instead.")
            sys.exit(1)

        print("[INFO] Connecting to Reachy Mini (media_backend=default)...")
        with ReachyMini(media_backend="default") as mini:
            print("[INFO] Connected. Morning Greeter is watching... (Ctrl+C to stop)")
            idle_pose(mini)
            self._running = True

            while self._running:
                frame = mini.media.get_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue

                faces = self._detector.detect(frame)
                self._log_detection_debug(faces)

                if self._show_video:
                    overlay = self._draw_debug_overlay(frame, faces)
                    cv2.imshow("Reachy Mini - Morning Greeter (Robot Camera)", overlay)

                if faces:
                    self._tracker.on_face_seen()
                    if self._tracker.should_greet():
                        self._tracker.mark_greeted()
                        self._do_greeting(mini)
                        print("[INFO] Back to watching...")

                if self._show_video and (cv2.waitKey(1) & 0xFF == ord("q")):
                    print("[INFO] Q pressed — stopping.")
                    break

                time.sleep(0.08)   # ~12 fps polling

        if self._show_video:
            cv2.destroyAllWindows()

    def run_with_webcam(self, mini: "ReachyMini | None" = None) -> None:
        """
        Dev / simulation mode: use local webcam for face detection.
        Pass a connected ReachyMini instance to also drive robot animations.
        Shows an OpenCV preview window with face bounding boxes.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Could not open webcam (device 0). Plug in a camera.")
            sys.exit(1)

        print("[INFO] Webcam open. Morning Greeter is watching... (press Q to quit)")
        if mini is not None:
            idle_pose(mini)

        self._running = True
        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                faces = self._detector.detect(frame)
                self._log_detection_debug(faces)

                if self._show_video:
                    overlay = self._draw_debug_overlay(frame, faces)
                    cv2.imshow("Reachy Mini - Morning Greeter", overlay)

                # ── Greeting logic ────────────────────────────────────────────
                if faces:
                    self._tracker.on_face_seen()
                    if self._tracker.should_greet():
                        self._tracker.mark_greeted()
                        self._do_greeting(mini)
                        print("[INFO] Back to watching...")

                if self._show_video and (cv2.waitKey(1) & 0xFF == ord("q")):
                    print("[INFO] Q pressed — stopping.")
                    break

                time.sleep(0.05)   # ~20 fps

        finally:
            cap.release()
            cv2.destroyAllWindows()
            if mini is not None:
                idle_pose(mini)

    def stop(self) -> None:
        self._running = False


# ═════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Reachy Mini Morning Greeter — greets visitors spontaneously.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--sim",
        action="store_true",
        help=(
            "Simulation / dev mode: connect to the local MuJoCo daemon for "
            "robot movement, but use a webcam for face detection. "
            "Start the daemon first with: reachy-mini-daemon --sim"
        ),
    )
    mode.add_argument(
        "--no-robot",
        action="store_true",
        help="Run face detection + TTS only, no robot connection required.",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=20.0,
        metavar="SECONDS",
        help=(
            "Minimum seconds between greetings of the same continuous visit "
            "(default: 20). Set to 0 to greet every time a face reappears."
        ),
    )
    parser.add_argument(
        "--show-video",
        action="store_true",
        help=(
            "Show live video with face boxes and status text. "
            "Works for both robot camera and webcam mode. Press Q to quit."
        ),
    )
    parser.add_argument(
        "--debug-detections",
        action="store_true",
        help="Print face detection debug logs (~1 line/second) to the console.",
    )
    args = parser.parse_args(argv)

    greeter = MorningGreeter(
        cooldown_s=args.cooldown,
        show_video=args.show_video,
        debug_detections=args.debug_detections,
    )

    try:
        if args.no_robot or not REACHY_AVAILABLE:
            # Vision + TTS only — no robot needed
            greeter.run_with_webcam(mini=None)

        elif args.sim:
            # Simulation: robot available via daemon, webcam for vision
            print("[INFO] Connecting to Reachy Mini simulation daemon...")
            with ReachyMini() as mini:
                print("[INFO] Connected to simulation.")
                greeter.run_with_webcam(mini=mini)

        else:
            # Real robot: use the robot's own camera
            greeter.run_with_robot_camera()

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted — shutting down.")
        greeter.stop()


if __name__ == "__main__":
    main()
