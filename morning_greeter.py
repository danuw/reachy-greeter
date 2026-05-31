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
from pathlib import Path

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

# Tracking and diagnostics tuning
TRACK_MAX_MISSING_S: float = 2.5
TRACK_MATCH_MAX_DIST_PX: float = 120.0
NO_FRAME_WARN_INTERVAL_S: float = 3.0

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

class PersonTracker:
    """
    Lightweight online face tracking by centroid distance.

    This is not biometric identification, but it keeps a stable ID while the
    same person stays in view and prevents repetitive greetings.
    """

    def __init__(self, global_cooldown_s: float, same_person_cooldown_s: float) -> None:
        self._global_cooldown_s = global_cooldown_s
        self._same_person_cooldown_s = same_person_cooldown_s
        self._tracks: dict[int, dict[str, float | tuple[int, int, int, int] | bool]] = {}
        self._next_id = 1
        self._last_any_greet_at = 0.0
        self._lock = threading.Lock()

    @staticmethod
    def _center(box: tuple[int, int, int, int]) -> tuple[float, float]:
        x, y, w, h = box
        return (x + w / 2.0, y + h / 2.0)

    @staticmethod
    def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return float((dx * dx + dy * dy) ** 0.5)

    def update(self, faces: list[tuple[int, int, int, int]]) -> list[tuple[int, tuple[int, int, int, int]]]:
        """Assign detections to tracks and return [(track_id, box), ...]."""
        now = time.time()
        with self._lock:
            # Drop stale tracks that disappeared.
            stale_ids = [
                track_id
                for track_id, track in self._tracks.items()
                if now - float(track["last_seen_at"]) > TRACK_MAX_MISSING_S
            ]
            for track_id in stale_ids:
                del self._tracks[track_id]

            unmatched_ids = set(self._tracks.keys())
            assignments: list[tuple[int, tuple[int, int, int, int]]] = []

            for face in faces:
                fc = self._center(face)
                best_id = None
                best_dist = float("inf")

                for track_id in unmatched_ids:
                    track_box = self._tracks[track_id]["box"]
                    if not isinstance(track_box, tuple):
                        continue
                    tc = self._center(track_box)
                    d = self._dist(fc, tc)
                    if d < best_dist:
                        best_dist = d
                        best_id = track_id

                if best_id is not None and best_dist <= TRACK_MATCH_MAX_DIST_PX:
                    self._tracks[best_id]["box"] = face
                    self._tracks[best_id]["last_seen_at"] = now
                    assignments.append((best_id, face))
                    unmatched_ids.remove(best_id)
                else:
                    new_id = self._next_id
                    self._next_id += 1
                    self._tracks[new_id] = {
                        "box": face,
                        "last_seen_at": now,
                        "last_greeted_at": 0.0,
                    }
                    assignments.append((new_id, face))

            return assignments

    def should_greet(self, track_id: int) -> bool:
        now = time.time()
        with self._lock:
            track = self._tracks.get(track_id)
            if track is None:
                return False

            if now - self._last_any_greet_at < self._global_cooldown_s:
                return False

            last_greeted_at = float(track["last_greeted_at"])
            if last_greeted_at > 0 and now - last_greeted_at < self._same_person_cooldown_s:
                return False

            return True

    def mark_greeted(self, track_id: int) -> None:
        now = time.time()
        with self._lock:
            track = self._tracks.get(track_id)
            if track is None:
                return
            track["last_greeted_at"] = now
            self._last_any_greet_at = now


# ═════════════════════════════════════════════════════════════════════════════
#  Main greeter class
# ═════════════════════════════════════════════════════════════════════════════

class MorningGreeter:
    def __init__(
        self,
        cooldown_s: float = 20.0,
        show_video: bool = False,
        debug_detections: bool = False,
        save_detection_frames: int = 10,
        save_dir: str = "debug_detections",
        same_person_cooldown_s: float = 300.0,
        robot_media_backend: str = "default",
        no_frame_timeout_s: float = 8.0,
        fallback_webcam_if_no_robot_video: bool = False,
        webcam_index: int = 0,
        webcam_backend: str = "auto",
        robot_name: str = "reachy_mini",
        robot_host: str = "reachy-mini.local",
        robot_port: int = 8000,
        connection_mode: str = "auto",
    ) -> None:
        self._detector = FaceDetector()
        self._tracker = PersonTracker(
            global_cooldown_s=cooldown_s,
            same_person_cooldown_s=same_person_cooldown_s,
        )
        self._tts = _build_tts_engine()
        self._greeting_idx = 0
        self._running = False
        self._show_video = show_video
        self._debug_detections = debug_detections
        self._debug_last_log_at = 0.0
        self._debug_frames = 0
        self._save_detection_frames_remaining = max(0, save_detection_frames)
        self._save_detection_frame_idx = 0
        self._save_dir = Path(save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._video_error_reported = False
        self._last_no_frame_warn_at = 0.0
        self._first_frame_logged = False
        self._robot_media_backend = robot_media_backend
        self._no_frame_timeout_s = max(1.0, no_frame_timeout_s)
        self._fallback_webcam_if_no_robot_video = fallback_webcam_if_no_robot_video
        self._webcam_index = webcam_index
        self._webcam_backend = webcam_backend
        self._robot_name = robot_name
        self._robot_host = robot_host
        self._robot_port = robot_port
        self._connection_mode = connection_mode

        if self._save_detection_frames_remaining > 0:
            print(
                "[DEBUG] Will save first "
                f"{self._save_detection_frames_remaining} frames with detections to "
                f"{self._save_dir.resolve()}"
            )

    def _draw_debug_overlay(
        self,
        frame: np.ndarray,
        tracked_faces: list[tuple[int, tuple[int, int, int, int]]],
        source_label: str,
    ) -> np.ndarray:
        """Draw tracked face boxes and status text on a copy of the frame."""
        overlay = frame.copy()
        for track_id, (x, y, w, h) in tracked_faces:
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 80), 2)
            cv2.putText(
                overlay,
                f"id:{track_id} {w}x{h}",
                (x, max(18, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 80),
                1,
            )

        status = "FACE DETECTED" if tracked_faces else "Watching..."
        cv2.putText(
            overlay,
            f"Reachy Mini Morning Greeter ({source_label})  [{status}]",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 200, 255),
            2,
        )
        return overlay

    def _log_detection_debug(self, tracked_faces: list[tuple[int, tuple[int, int, int, int]]]) -> None:
        """Print low-rate detection debug to avoid flooding the console."""
        if not self._debug_detections:
            return

        now = time.time()
        self._debug_frames += 1
        if now - self._debug_last_log_at < 1.0:
            return

        self._debug_last_log_at = now
        if tracked_faces:
            faces_str = ", ".join(
                f"id={track_id}(x={x}, y={y}, w={w}, h={h})"
                for track_id, (x, y, w, h) in tracked_faces
            )
            print(f"[DEBUG] frames={self._debug_frames} faces={len(tracked_faces)} {faces_str}")
        else:
            print(f"[DEBUG] frames={self._debug_frames} faces=0")

    def _report_no_frame(self, source: str) -> None:
        now = time.time()
        if now - self._last_no_frame_warn_at < NO_FRAME_WARN_INTERVAL_S:
            return
        self._last_no_frame_warn_at = now
        print(
            f"[WARNING] No frames received from {source} yet. "
            "If this persists, camera streaming may be unavailable."
        )

    def _report_first_frame(self, frame: np.ndarray, source: str) -> None:
        if self._first_frame_logged:
            return
        h, w = frame.shape[:2]
        print(f"[INFO] First video frame received from {source}: {w}x{h}")
        self._first_frame_logged = True

    @staticmethod
    def _pick_primary_track(
        tracked_faces: list[tuple[int, tuple[int, int, int, int]]]
    ) -> int | None:
        if not tracked_faces:
            return None
        # Pick biggest visible face as active speaker target.
        best_track_id, best_box = max(tracked_faces, key=lambda item: item[1][2] * item[1][3])
        _ = best_box
        return best_track_id

    def _save_detection_frame(self, frame: np.ndarray, faces: list[tuple[int, int, int, int]]) -> None:
        """Save first N frames that contain detections for offline debugging."""
        if self._save_detection_frames_remaining <= 0 or not faces:
            return

        self._save_detection_frame_idx += 1
        filename = (
            f"det_{self._save_detection_frame_idx:03d}_"
            f"{int(time.time() * 1000)}_faces{len(faces)}.jpg"
        )
        output_path = self._save_dir / filename
        ok = cv2.imwrite(str(output_path), frame)
        if ok:
            self._save_detection_frames_remaining -= 1
            print(
                f"[DEBUG] Saved detection frame: {output_path} "
                f"({self._save_detection_frames_remaining} remaining)"
            )
            if self._save_detection_frames_remaining == 0:
                print("[DEBUG] Detection frame capture limit reached.")

    def _show_debug_window(self, title: str, frame: np.ndarray) -> bool:
        """Show debug window and return True if user requested quit with Q."""
        if not self._show_video:
            return False

        try:
            cv2.imshow(title, frame)
            return (cv2.waitKey(1) & 0xFF) == ord("q")
        except cv2.error as exc:
            if not self._video_error_reported:
                print(
                    "[WARNING] Could not display debug window. "
                    "Continuing without video preview. "
                    f"Reason: {exc}"
                )
                self._video_error_reported = True
            self._show_video = False
            return False

    def _open_webcam(self) -> cv2.VideoCapture:
        """Open webcam with backend fallback (Windows-friendly)."""
        backend_candidates: list[tuple[str, int | None]]
        if self._webcam_backend == "auto":
            backend_candidates = [
                ("msmf", getattr(cv2, "CAP_MSMF", None)),
                ("dshow", getattr(cv2, "CAP_DSHOW", None)),
                ("default", None),
            ]
        elif self._webcam_backend == "msmf":
            backend_candidates = [("msmf", getattr(cv2, "CAP_MSMF", None))]
        elif self._webcam_backend == "dshow":
            backend_candidates = [("dshow", getattr(cv2, "CAP_DSHOW", None))]
        else:
            backend_candidates = [("default", None)]

        for backend_name, backend_flag in backend_candidates:
            if backend_flag is None and backend_name != "default":
                continue

            if backend_flag is None:
                cap = cv2.VideoCapture(self._webcam_index)
            else:
                cap = cv2.VideoCapture(self._webcam_index, backend_flag)

            if not cap.isOpened():
                cap.release()
                continue

            # Ensure this backend can actually deliver frames.
            got_frame = False
            for _ in range(20):
                ok, _frame = cap.read()
                if ok:
                    got_frame = True
                    break
                time.sleep(0.03)

            if got_frame:
                print(
                    f"[INFO] Webcam opened on index {self._webcam_index} "
                    f"using backend={backend_name}."
                )
                return cap

            cap.release()

        raise RuntimeError(
            "Could not open webcam with a working backend. "
            "Try closing other camera apps, switching --webcam-index, "
            "or forcing --webcam-backend dshow."
        )

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

        print(
            "[INFO] Connecting to Reachy Mini "
            f"(host={self._robot_host}:{self._robot_port}, "
            f"connection_mode={self._connection_mode}, "
            f"media_backend={self._robot_media_backend})..."
        )
        with ReachyMini(
            robot_name=self._robot_name,
            host=self._robot_host,
            port=self._robot_port,
            connection_mode=self._connection_mode,
            media_backend=self._robot_media_backend,
        ) as mini:
            print("[INFO] Connected. Morning Greeter is watching... (Ctrl+C to stop)")
            idle_pose(mini)
            self._running = True
            first_none_at: float | None = None

            while self._running:
                frame = mini.media.get_frame()
                if frame is None:
                    self._report_no_frame("Reachy Mini camera")

                    if first_none_at is None:
                        first_none_at = time.time()

                    if (
                        self._fallback_webcam_if_no_robot_video
                        and (time.time() - first_none_at) >= self._no_frame_timeout_s
                    ):
                        print(
                            "[WARNING] Robot camera stream unavailable. "
                            "Falling back to webcam for vision while keeping "
                            "robot control active."
                        )
                        self.run_with_webcam(mini=mini)
                        break

                    time.sleep(0.05)
                    continue

                first_none_at = None
                self._report_first_frame(frame, "Reachy Mini camera")

                faces = self._detector.detect(frame)
                tracked_faces = self._tracker.update(faces)
                self._log_detection_debug(tracked_faces)

                overlay = None
                if self._show_video or (self._save_detection_frames_remaining > 0 and tracked_faces):
                    overlay = self._draw_debug_overlay(
                        frame,
                        tracked_faces,
                        source_label="robot",
                    )

                if tracked_faces:
                    boxes_only = [box for _, box in tracked_faces]
                    self._save_detection_frame(overlay if overlay is not None else frame, boxes_only)

                if self._show_video:
                    if self._show_debug_window(
                        "Reachy Mini - Morning Greeter (Robot Camera)",
                        overlay if overlay is not None else frame,
                    ):
                        print("[INFO] Q pressed — stopping.")
                        break

                primary_track_id = self._pick_primary_track(tracked_faces)
                if primary_track_id is not None:
                    if self._tracker.should_greet(primary_track_id):
                        self._tracker.mark_greeted(primary_track_id)
                        self._do_greeting(mini)
                        print("[INFO] Back to watching...")

                time.sleep(0.08)   # ~12 fps polling

        if self._show_video:
            cv2.destroyAllWindows()

    def run_with_webcam(self, mini: "ReachyMini | None" = None) -> None:
        """
        Dev / simulation mode: use local webcam for face detection.
        Pass a connected ReachyMini instance to also drive robot animations.
        Shows an OpenCV preview window with face bounding boxes.
        """
        try:
            cap = self._open_webcam()
        except RuntimeError as exc:
            print(f"[ERROR] {exc}")
            sys.exit(1)

        print("[INFO] Webcam open. Morning Greeter is watching... (press Q to quit)")
        if mini is not None:
            idle_pose(mini)

        self._running = True
        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    self._report_no_frame("webcam")
                    time.sleep(0.05)
                    continue

                self._report_first_frame(frame, "webcam")

                faces = self._detector.detect(frame)
                tracked_faces = self._tracker.update(faces)
                self._log_detection_debug(tracked_faces)

                overlay = None
                if self._show_video or (self._save_detection_frames_remaining > 0 and tracked_faces):
                    overlay = self._draw_debug_overlay(
                        frame,
                        tracked_faces,
                        source_label="webcam",
                    )

                if tracked_faces:
                    boxes_only = [box for _, box in tracked_faces]
                    self._save_detection_frame(overlay if overlay is not None else frame, boxes_only)

                if self._show_video:
                    if self._show_debug_window(
                        "Reachy Mini - Morning Greeter",
                        overlay if overlay is not None else frame,
                    ):
                        print("[INFO] Q pressed — stopping.")
                        break

                # ── Greeting logic ────────────────────────────────────────────
                primary_track_id = self._pick_primary_track(tracked_faces)
                if primary_track_id is not None:
                    if self._tracker.should_greet(primary_track_id):
                        self._tracker.mark_greeted(primary_track_id)
                        self._do_greeting(mini)
                        print("[INFO] Back to watching...")

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
    parser.add_argument(
        "--save-detection-frames",
        type=int,
        default=10,
        metavar="N",
        help=(
            "Save the first N frames containing face detections to disk "
            "for debugging (default: 10, set 0 to disable)."
        ),
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="debug_detections",
        help="Directory where detection debug frames are saved.",
    )
    parser.add_argument(
        "--same-person-cooldown",
        type=float,
        default=300.0,
        metavar="SECONDS",
        help=(
            "Minimum seconds before greeting the same tracked person again "
            "(default: 300)."
        ),
    )
    parser.add_argument(
        "--robot-media-backend",
        type=str,
        choices=["default", "local", "webrtc"],
        default="default",
        help=(
            "Media backend for real-robot mode. "
            "Use webrtc for remote/network robots, local when daemon and app "
            "run on same machine."
        ),
    )
    parser.add_argument(
        "--no-frame-timeout",
        type=float,
        default=8.0,
        metavar="SECONDS",
        help="Seconds to wait for robot camera frames before fallback actions.",
    )
    parser.add_argument(
        "--fallback-webcam-if-no-robot-video",
        action="store_true",
        help=(
            "If robot camera frames never arrive, switch to webcam for vision "
            "while still controlling the robot."
        ),
    )
    parser.add_argument(
        "--webcam-index",
        type=int,
        default=0,
        metavar="N",
        help="Webcam device index for webcam-based vision modes (default: 0).",
    )
    parser.add_argument(
        "--webcam-backend",
        type=str,
        choices=["auto", "msmf", "dshow", "default"],
        default="auto",
        help=(
            "OpenCV backend for webcam capture. Use dshow on Windows if msmf "
            "fails to grab frames."
        ),
    )
    parser.add_argument(
        "--robot-name",
        type=str,
        default="reachy_mini",
        help="Robot name used by Reachy SDK discovery (default: reachy_mini).",
    )
    parser.add_argument(
        "--robot-host",
        type=str,
        default="reachy-mini.local",
        help="Reachy host or IP for real-robot mode (default: reachy-mini.local).",
    )
    parser.add_argument(
        "--robot-port",
        type=int,
        default=8000,
        help="Reachy daemon port for real-robot mode (default: 8000).",
    )
    parser.add_argument(
        "--connection-mode",
        type=str,
        choices=["auto", "localhost_only", "network"],
        default="auto",
        help=(
            "Reachy SDK connection mode. Use network for explicit remote robot host, "
            "localhost_only for local daemon, auto for SDK auto-detection."
        ),
    )
    args = parser.parse_args(argv)

    greeter = MorningGreeter(
        cooldown_s=args.cooldown,
        show_video=args.show_video,
        debug_detections=args.debug_detections,
        save_detection_frames=args.save_detection_frames,
        save_dir=args.save_dir,
        same_person_cooldown_s=args.same_person_cooldown,
        robot_media_backend=args.robot_media_backend,
        no_frame_timeout_s=args.no_frame_timeout,
        fallback_webcam_if_no_robot_video=args.fallback_webcam_if_no_robot_video,
        webcam_index=args.webcam_index,
        webcam_backend=args.webcam_backend,
        robot_name=args.robot_name,
        robot_host=args.robot_host,
        robot_port=args.robot_port,
        connection_mode=args.connection_mode,
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
