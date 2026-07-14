#!/usr/bin/env python3
"""
KYC Demo — Liveness (gauche / droite / haut / bas) + CIN scan + OCR.

Face detector: OpenCV YuNet (5 landmarks). Works on Python 3.8–3.14, no
MediaPipe dependency. Head pose: cv2.solvePnP on the 5 landmarks → pitch
and yaw in degrees, with a 1-second auto-calibration of the neutral pose
(your webcam angle and natural posture become the zero).

Outputs:
  ~/Desktop/face.png       — face cutout with transparent background
  kyc_demo/selfie.jpg      — best frontal frame from the liveness step
  kyc_demo/cin.jpg         — perspective-warped CIN
  kyc_demo/cin_raw.txt     — raw OCR dump

Install:
  pip install -r requirements.txt
  # macOS:   brew install tesseract tesseract-lang
  # Windows: choco install tesseract  (UB-Mannheim build ships ara + fra)

Run:
  python kyc_demo.py
  python kyc_demo.py --list-cameras
  python kyc_demo.py --camera 1
"""

from __future__ import annotations

import argparse
import platform
import re
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pytesseract


HERE = Path(__file__).resolve().parent
DESKTOP = Path.home() / "Desktop"
SELFIE_PATH = HERE / "selfie.jpg"
CIN_PATH = HERE / "cin.jpg"
CIN_RAW_PATH = HERE / "cin_raw.txt"
FACE_PNG_PATH = DESKTOP / "face.png"
YUNET_PATH = HERE / "yunet_2023mar.onnx"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


# ────────────────────────────────────────────────────────────────────────────
# Camera helpers.
# ────────────────────────────────────────────────────────────────────────────

_CAMERA_MAX_PROBE = 10


def _camera_backends() -> list[int]:
    """Backends to try (order matters on macOS)."""
    system = platform.system()
    if system == "Darwin":
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    if system == "Windows":
        return [cv2.CAP_DSHOW, cv2.CAP_ANY]
    return [cv2.CAP_ANY]


def _camera_backend() -> int:
    return _camera_backends()[0]


def _macos_camera_names() -> list[str]:
    """Human-readable camera names from system_profiler (macOS)."""
    import subprocess
    try:
        out = subprocess.run(
            ["system_profiler", "SPCameraDataType"],
            capture_output=True, text=True, timeout=15, check=False,
        ).stdout
    except Exception:
        return []
    names: list[str] = []
    for line in out.splitlines():
        m = re.match(r"^\s{4}(.+):\s*$", line)
        if m and m.group(1) not in names:
            names.append(m.group(1).strip())
    return names


def _darwin_camera_warmup(up_to_index: int) -> None:
    """
    On macOS, AVFoundation often reports only index 0 until lower devices
  are opened once; required for external / Continuity Camera at index 1+.
    """
    backend = cv2.CAP_AVFOUNDATION
    for i in range(max(0, up_to_index)):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            for _ in range(3):
                cap.read()
            cap.release()
        time.sleep(0.12)


def _try_open_capture(index: int, backend: int,
                      width: int, height: int) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if platform.system() == "Darwin":
        cap.set(cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(*"MJPG"))
    for _ in range(8):
        ok, _ = cap.read()
        if ok:
            return cap
        time.sleep(0.05)
    cap.release()
    return None


def list_cameras(max_index: int = _CAMERA_MAX_PROBE) -> list[int]:
    if platform.system() == "Darwin":
        _darwin_camera_warmup(max_index - 1)
    available: list[int] = []
    misses = 0
    for i in range(max_index):
        opened = False
        for backend in _camera_backends():
            cap = _try_open_capture(i, backend, 640, 480)
            if cap is not None:
                available.append(i)
                cap.release()
                opened = True
                break
        if not opened and platform.system() == "Darwin" and i > 0:
            _darwin_camera_warmup(i)
            cap = _try_open_capture(i, cv2.CAP_AVFOUNDATION, 640, 480)
            if cap is not None:
                available.append(i)
                cap.release()
                opened = True
        if opened:
            misses = 0
        else:
            misses += 1
            if misses >= 2 and i > 0:
                break
    return sorted(set(available))


def resolve_camera_index(index: int | None, name: str | None) -> int:
    """Map --camera-name to OpenCV index (macOS); else use numeric index."""
    if not name:
        return index if index is not None else 0
    if platform.system() != "Darwin":
        print("  ⚠ --camera-name est surtout utile sur macOS; index numerique utilise.")
        return index if index is not None else 0
    names = _macos_camera_names()
    if not names:
        sys.exit("ERREUR: impossible de lister les cameras (system_profiler).")
    key = name.lower()
    matches = [i for i, n in enumerate(names) if key in n.lower()]
    if len(matches) == 1:
        print(f"  → Camera « {names[matches[0]]} » → index {matches[0]}")
        return matches[0]
    if len(matches) > 1:
        sys.exit(
            f"ERREUR: plusieurs cameras correspondent a « {name} »:\n"
            + "\n".join(f"  [{i}] {names[i]}" for i in matches)
        )
    sys.exit(
        f"ERREUR: aucune camera dont le nom contient « {name} ».\n"
        f"  Peripheriques: {', '.join(names) or 'aucun'}\n"
        f"  python kyc_demo.py --list-cameras"
    )


def _print_camera_list() -> None:
    indices = list_cameras()
    names = _macos_camera_names() if platform.system() == "Darwin" else []
    print("Cameras detectees :")
    if not indices:
        print("  (aucune — autorisez la camera pour Terminal/Cursor dans")
        print("   Reglages systeme → Confidentialite → Camera)")
        return
    for i in indices:
        label = names[i] if i < len(names) else ""
        suffix = f"  ({label})" if label else ""
        print(f"  - index {i}{suffix}")
    if platform.system() == "Darwin" and names and len(names) != len(indices):
        print("\n  Peripheriques systeme :")
        for j, name in enumerate(names):
            print(f"    [{j}] {name}")


def _open_camera(index: int, width: int = 640, height: int = 480,
                 retries: int = 8, delay: float = 0.8):
    if platform.system() == "Darwin" and index > 0:
        _darwin_camera_warmup(index)

    if index == 0:
        print("  ⏳ Ouverture camera index 0...")
        print("     (Si macOS demande l'autorisation, cliquez Autoriser.)")

    for attempt in range(retries):
        for backend in _camera_backends():
            cap = _try_open_capture(index, backend, width, height)
            if cap is not None:
                return cap
        if attempt == 0 and index > 0:
            print(f"  ⏳ Ouverture camera index {index}...")
            print("     (Si macOS demande l'autorisation, cliquez Autoriser.)")
            if platform.system() == "Darwin":
                _darwin_camera_warmup(index)
        time.sleep(delay)

    found = list_cameras()
    hint = f"Indices OpenCV disponibles : {found}" if found else (
        "Aucune camera OpenCV — verifiez les autorisations macOS.")
    sys.exit(
        f"ERREUR: camera index {index} introuvable.\n"
        f"  • {hint}\n"
        f"  • Listez : python kyc_demo.py --list-cameras\n"
        f"  • Essayez : python kyc_demo.py --camera {found[0] if found else 0}\n"
    )


# ────────────────────────────────────────────────────────────────────────────
# YuNet face detector.
# ────────────────────────────────────────────────────────────────────────────

def _ensure_yunet_model() -> None:
    if YUNET_PATH.exists() and YUNET_PATH.stat().st_size > 100_000:
        return
    print(f"  ⬇  Telechargement du modele YuNet ({YUNET_PATH.name})...")
    try:
        urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
    except Exception as exc:
        sys.exit(
            f"ERREUR: telechargement YuNet echoue ({exc}).\n"
            f"Telechargez manuellement {YUNET_URL}\nvers {YUNET_PATH}"
        )
    print(f"  ✓  Modele sauvegarde ({YUNET_PATH.stat().st_size // 1024} KB)")


def _make_detector(frame_w: int, frame_h: int):
    _ensure_yunet_model()
    return cv2.FaceDetectorYN.create(
        str(YUNET_PATH), "",
        (frame_w, frame_h),
        score_threshold=0.65,
        nms_threshold=0.3,
        top_k=1,   # we only ever care about the largest face
    )


def detect_face(detector, frame: np.ndarray):
    """Largest face: (bbox xywh, landmarks 5x2, score) or None."""
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    if faces is None or len(faces) == 0:
        return None
    face = max(faces, key=lambda f: f[2] * f[3])
    return face[0:4].astype(int), face[4:14].reshape(5, 2), float(face[14])


# ────────────────────────────────────────────────────────────────────────────
# Head pose via solvePnP — pitch & yaw in degrees, with auto-calibration.
# ────────────────────────────────────────────────────────────────────────────

# Generic 3D face model. Order: nose, left eye outer, right eye outer,
# left mouth corner, right mouth corner. "Left"/"right" = subject's POV.
MODEL_3D = np.array([
    (   0.0,    0.0,    0.0),
    ( -30.0,   30.0,  -30.0),
    (  30.0,   30.0,  -30.0),
    ( -25.0,  -28.0,  -24.0),
    (  25.0,  -28.0,  -24.0),
], dtype=np.float64)


def _euler_from_rmat(R: np.ndarray) -> tuple[float, float, float]:
    """ZYX Euler angles in degrees: (pitch, yaw, roll)."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy < 1e-6:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0
    else:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    return float(np.degrees(x)), float(np.degrees(y)), float(np.degrees(z))


def _head_pose(landmarks: np.ndarray, frame_w: int, frame_h: int):
    """Return (pitch_deg, yaw_deg) using solvePnP, or None on failure."""
    eye_a, eye_b = landmarks[0], landmarks[1]
    if eye_a[0] < eye_b[0]:
        img_left_eye, img_right_eye = eye_a, eye_b
    else:
        img_left_eye, img_right_eye = eye_b, eye_a
    mouth_a, mouth_b = landmarks[3], landmarks[4]
    if mouth_a[0] < mouth_b[0]:
        img_left_mouth, img_right_mouth = mouth_a, mouth_b
    else:
        img_left_mouth, img_right_mouth = mouth_b, mouth_a

    image_points = np.array([
        landmarks[2],     # nose
        img_left_eye,     # subject-left eye
        img_right_eye,    # subject-right eye
        img_left_mouth,   # subject-left mouth
        img_right_mouth,  # subject-right mouth
    ], dtype=np.float64)

    focal = float(frame_w)
    cam_mat = np.array([
        [focal, 0,     frame_w / 2.0],
        [0,     focal, frame_h / 2.0],
        [0,     0,     1.0],
    ], dtype=np.float64)
    dist = np.zeros((4, 1))

    # SOLVEPNP_SQPNP works with as few as 3 points; SOLVEPNP_ITERATIVE in
    # OpenCV 4.13+ requires 6+. Fall back to EPNP if SQPNP is missing on
    # an older build.
    flag = getattr(cv2, "SOLVEPNP_SQPNP", cv2.SOLVEPNP_EPNP)
    try:
        ok, rvec, _ = cv2.solvePnP(MODEL_3D, image_points, cam_mat, dist,
                                   flags=flag)
    except cv2.error:
        return None
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    pitch, yaw, _roll = _euler_from_rmat(R)
    if pitch > 90:
        pitch -= 180
    elif pitch < -90:
        pitch += 180
    return pitch, yaw


class PoseTracker:
    """EMA smoothing + auto-calibration of the neutral pose."""

    def __init__(self, alpha: float = 0.4, cal_frames: int = 18):
        self.alpha = alpha
        self.cal_frames = cal_frames
        self._cal_pitch: deque[float] = deque(maxlen=cal_frames)
        self._cal_yaw: deque[float] = deque(maxlen=cal_frames)
        self._zero_pitch = 0.0
        self._zero_yaw = 0.0
        self._smooth_pitch: float | None = None
        self._smooth_yaw: float | None = None

    @property
    def calibrated(self) -> bool:
        return len(self._cal_pitch) >= self.cal_frames

    @property
    def progress(self) -> float:
        return len(self._cal_pitch) / self.cal_frames

    def update(self, pitch: float, yaw: float) -> tuple[float, float]:
        if self._smooth_pitch is None:
            self._smooth_pitch = pitch
            self._smooth_yaw = yaw
        else:
            self._smooth_pitch = self.alpha * pitch + (1 - self.alpha) * self._smooth_pitch
            self._smooth_yaw = self.alpha * yaw + (1 - self.alpha) * self._smooth_yaw

        if not self.calibrated:
            self._cal_pitch.append(self._smooth_pitch)
            self._cal_yaw.append(self._smooth_yaw)
            if self.calibrated:
                self._zero_pitch = sum(self._cal_pitch) / len(self._cal_pitch)
                self._zero_yaw = sum(self._cal_yaw) / len(self._cal_yaw)

        return (self._smooth_pitch - self._zero_pitch,
                self._smooth_yaw - self._zero_yaw)


# ────────────────────────────────────────────────────────────────────────────
# Liveness — degree thresholds, calibrated, displayed live.
# ────────────────────────────────────────────────────────────────────────────

YAW_DEG = 12.0
PITCH_DEG = 6.0  # Reduced from 9.0 - easier "up" detection
HOLD_FRAMES = 3

STEPS = [
    ("LEFT",  "Tournez la tete a GAUCHE",   "gauche"),
    ("RIGHT", "Tournez la tete a DROITE",   "droite"),
    ("UP",    "Levez la tete vers le HAUT", "haut"),
]


def _step_done(key: str, pitch: float, yaw: float) -> bool:
    # FIXED: Yaw is inverted because frame is flipped (mirror mode)
    # When user turns right, yaw is positive (in mirror view)
    # When user turns left, yaw is negative (in mirror view)
    if key == "LEFT":
        return yaw > YAW_DEG  # Changed: was yaw < -YAW_DEG
    if key == "RIGHT":
        return yaw < -YAW_DEG  # Changed: was yaw > YAW_DEG
    if key == "UP":
        return pitch > PITCH_DEG
    return False


def _put(frame, text, org, color=(255, 255, 255), scale=0.7, thick=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)


def _draw_progress(frame, completed: set[str], current: str | None) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 60), (w, h), (30, 30, 30), -1)
    cell_w = w // len(STEPS)
    for i, (key, _, label) in enumerate(STEPS):
        x = i * cell_w + cell_w // 2
        if key in completed:
            color, mark = (0, 220, 0), "FAIT"
        elif key == current:
            color, mark = (0, 215, 255), "EN COURS"
        else:
            color, mark = (140, 140, 140), "—"
        _put(frame, f"{mark} {label.upper()}",
             (x - 80, h - 22), color, 0.55, 2)


def run_liveness(camera_index: int):
    """Returns (best_selfie_bgr, best_face_bbox) once the four steps pass."""
    print("[1/2] Liveness — placez-vous face a la camera.")
    cap = _open_camera(camera_index)

    ok, frame = cap.read()
    if not ok:
        sys.exit("ERREUR: impossible de lire la camera.")
    h0, w0 = frame.shape[:2]
    detector = _make_detector(w0, h0)

    tracker = PoseTracker(alpha=0.35, cal_frames=25)
    step_idx = 0
    streak = 0
    completed: set[str] = set()
    best_selfie = None
    best_bbox = None
    best_score = -1e9

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        detection = detect_face(detector, frame)

        current_key = STEPS[step_idx][0] if step_idx < len(STEPS) else None
        current_prompt = STEPS[step_idx][1] if step_idx < len(STEPS) else ""

        pitch_rel = yaw_rel = 0.0
        face_ok = False

        if detection is not None:
            bbox, landmarks, _ = detection
            angles = _head_pose(landmarks, w, h)
            if angles is not None:
                face_ok = True
                pitch, yaw = angles
                pitch_rel, yaw_rel = tracker.update(pitch, yaw)

                bx, by, bw, bh = bbox
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh),
                              (0, 200, 0), 2)
                for (lx, ly) in landmarks.astype(int):
                    cv2.circle(frame, (lx, ly), 3, (0, 255, 255), -1)

                frontal = -(abs(yaw_rel) + abs(pitch_rel))
                if (tracker.calibrated and frontal > best_score
                        and abs(yaw_rel) < 6 and abs(pitch_rel) < 6):
                    best_score = frontal
                    best_selfie = frame.copy()
                    best_bbox = bbox.copy()

                if tracker.calibrated and current_key is not None:
                    streak = streak + 1 if _step_done(current_key, pitch_rel, yaw_rel) else 0
                    if streak >= HOLD_FRAMES:
                        completed.add(current_key)
                        print(f"  ✓ {STEPS[step_idx][2]} detectee  "
                              f"(pitch={pitch_rel:+.1f}°  yaw={yaw_rel:+.1f}°)")
                        step_idx += 1
                        streak = 0

        if not tracker.calibrated:
            pct = int(tracker.progress * 100)
            _put(frame, f"Calibration neutre... {pct}%",
                 (20, 40), (0, 215, 255), 0.85, 2)
            _put(frame, "Regardez la camera droit devant vous",
                 (20, 80), (255, 255, 255), 0.65, 1)
        else:
            if step_idx >= len(STEPS):
                _draw_progress(frame, completed, None)
                _put(frame, "VERIFICATION REUSSIE",
                     (w // 2 - 200, h // 2),
                     (0, 255, 0), 1.1, 3)
                cv2.imshow("KYC - Verification", frame)
                cv2.waitKey(800)
                break
            _put(frame, current_prompt, (20, 40), (0, 215, 255), 0.9, 2)

        if face_ok:
            _put(frame, f"pitch={pitch_rel:+5.1f}°  yaw={yaw_rel:+5.1f}°",
                 (20, h - 80), (200, 200, 200), 0.6, 1)
        elif tracker.calibrated:
            _put(frame, "Aucun visage detecte",
                 (20, h - 80), (0, 0, 255), 0.7, 2)

        _draw_progress(frame, completed,
                       current_key if tracker.calibrated else None)
        cv2.imshow("KYC - Verification", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
            sys.exit("Annule.")

    cap.release()
    cv2.destroyAllWindows()
    if best_selfie is not None:
        cv2.imwrite(str(SELFIE_PATH), best_selfie)
        print(f"  → {SELFIE_PATH.name} sauvegarde")
    return best_selfie, best_bbox


# ────────────────────────────────────────────────────────────────────────────
# Background removal via GrabCut → ~/Desktop/face.png (transparent).
# ────────────────────────────────────────────────────────────────────────────

def cutout_face_png(image: np.ndarray, face_bbox: np.ndarray,
                    out_path: Path) -> None:
    h, w = image.shape[:2]
    bx, by, bw, bh = face_bbox

    expand_top = int(bh * 0.85)
    expand_bot = int(bh * 0.55)
    expand_side = int(bw * 0.45)
    x1 = max(0, bx - expand_side)
    y1 = max(0, by - expand_top)
    x2 = min(w, bx + bw + expand_side)
    y2 = min(h, by + bh + expand_bot)
    rect = (x1, y1, x2 - x1, y2 - y1)

    mask = np.zeros((h, w), dtype=np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)

    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0)\
        .astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    fg = cv2.GaussianBlur(fg, (5, 5), 0)

    bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = fg

    coords = cv2.findNonZero(fg)
    if coords is not None:
        cx, cy, cw, ch = cv2.boundingRect(coords)
        pad = 8
        cx1 = max(0, cx - pad)
        cy1 = max(0, cy - pad)
        cx2 = min(w, cx + cw + pad)
        cy2 = min(h, cy + ch + pad)
        bgra = bgra[cy1:cy2, cx1:cx2]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), bgra)
    print(f"  → {out_path}  ({bgra.shape[1]}×{bgra.shape[0]}, PNG transparent)")


# ────────────────────────────────────────────────────────────────────────────
# CIN capture — clickable [SCAN] button + auto detection.
# ────────────────────────────────────────────────────────────────────────────

class ScanButton:
    def __init__(self, frame_w: int, frame_h: int):
        self.bw, self.bh = 200, 60
        self.cx = frame_w // 2
        self.y_top = frame_h - 90
        self.clicked = False

    @property
    def rect(self) -> tuple[int, int, int, int]:
        return (self.cx - self.bw // 2, self.y_top,
                self.cx + self.bw // 2, self.y_top + self.bh)

    def draw(self, frame: np.ndarray) -> None:
        x1, y1, x2, y2 = self.rect
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 0), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        _put(frame, "SCAN", (x1 + 60, y1 + 40),
             (255, 255, 255), 1.0, 3)

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        x1, y1, x2, y2 = self.rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            self.clicked = True


def _find_card_quad(frame: np.ndarray) -> np.ndarray | None:
    """Live-detection variant — strict, fast, used at 30 fps in the camera
    loop. For post-capture finding (slow, multi-strategy) see `find_card`."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(gray, 30, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    min_area = 0.15 * frame.shape[0] * frame.shape[1]
    for c in contours:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.contourArea(c) > min_area:
            return approx.reshape(4, 2)
    return None


def _quad_canny(img: np.ndarray, lo: int, hi: int,
                eps_list: tuple[float, ...]) -> np.ndarray | None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(gray, lo, hi)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
    min_area = 0.10 * img.shape[0] * img.shape[1]
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        peri = cv2.arcLength(c, True)
        for eps in eps_list:
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                return approx.reshape(4, 2)
    return None


def _quad_otsu(img: np.ndarray) -> np.ndarray | None:
    """Brightness-based: the laminated card is usually the brightest large
    region in the frame. Works even when the hand breaks the card's edge
    so Canny can't form a closed loop."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                              np.ones((15, 15), np.uint8), iterations=2)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    h, w = img.shape[:2]
    for c in contours[:5]:
        if cv2.contourArea(c) < 0.12 * h * w:
            continue
        peri = cv2.arcLength(c, True)
        for eps in (0.02, 0.04, 0.06, 0.08):
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                return approx.reshape(4, 2)
        # Last-resort rotated rect on this same blob.
        rect = cv2.minAreaRect(c)
        rw, rh = rect[1]
        if rw >= 60 and rh >= 60:
            aspect = max(rw, rh) / min(rw, rh)
            if 1.20 < aspect < 2.10:
                return cv2.boxPoints(rect)
    return None


def _quad_rotated_rect(img: np.ndarray) -> np.ndarray | None:
    """Fallback: largest Canny edge contour → min-area-rect."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 30, 200)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    h, w = img.shape[:2]
    for c in contours[:5]:
        if cv2.contourArea(c) < 0.10 * h * w:
            continue
        rect = cv2.minAreaRect(c)
        rw, rh = rect[1]
        if rw < 50 or rh < 50:
            continue
        aspect = max(rw, rh) / min(rw, rh)
        if 1.25 < aspect < 2.05:
            return cv2.boxPoints(rect)
    return None


def find_card(img: np.ndarray) -> np.ndarray | None:
    """Multi-strategy card finder. Use on a still image (post-capture or
    loaded from disk); too slow for 30 fps preview."""
    # 1) Brightness-based — most reliable when the card is held against
    # a darker background (skin / clothing).
    q = _quad_otsu(img)
    if q is not None:
        return q
    # 2) Multiple Canny thresholds for different lighting.
    for lo, hi in ((30, 200), (50, 150), (10, 100), (80, 220)):
        q = _quad_canny(img, lo, hi, (0.02, 0.04, 0.06))
        if q is not None:
            return q
    # 3) Last resort: aspect-validated rotated rectangle.
    return _quad_rotated_rect(img)


def find_and_warp_card(img: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return (warped_card, was_warped). If no card found, returns the
    input untouched so OCR can still try on the raw frame."""
    quad = find_card(img)
    if quad is None:
        return img, False
    return _warp(img, quad), True


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def _warp(frame: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts.astype("float32"))
    tl, tr, br, bl = rect
    maxW = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    maxH = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]],
                   dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(frame, M, (maxW, maxH))


def capture_cin(camera_index: int) -> np.ndarray:
    print("\n[2/2] CIN — presentez votre carte d'identite, puis cliquez [SCAN].")
    print("      Raccourcis : [ESPACE] scanner   [ESC] annuler")
    # Higher resolution for the CIN — text legibility is everything here.
    cap = _open_camera(camera_index, width=1280, height=720)

    win = "KYC - Carte d'identite"
    cv2.namedWindow(win)
    button: ScanButton | None = None

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        view = frame.copy()
        h, w = view.shape[:2]
        if button is None:
            button = ScanButton(w, h)
            cv2.setMouseCallback(win, button.on_mouse)

        quad = _find_card_quad(frame)
        if quad is not None:
            cv2.drawContours(view, [quad.astype(int)], -1, (0, 255, 0), 3)
            _put(view, "Carte detectee — cliquez SCAN",
                 (20, 40), (0, 255, 0), 0.85, 2)
        else:
            _put(view, "Presentez votre CIN bien a plat",
                 (20, 40), (0, 215, 255), 0.85, 2)

        button.draw(view)
        cv2.imshow(win, view)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            cap.release()
            cv2.destroyAllWindows()
            sys.exit("Annule.")
        if button.clicked or k == 32:
            if quad is not None:
                captured = _warp(frame, quad)
            else:
                # Live detector didn't fire — try the slower, more thorough
                # multi-strategy finder on the captured frame.
                warped, ok_warped = find_and_warp_card(frame)
                captured = warped if ok_warped else frame.copy()
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.imwrite(str(CIN_PATH), captured)
    print(f"  → {CIN_PATH.name} sauvegarde")
    return captured


# ────────────────────────────────────────────────────────────────────────────
# OCR + parser for the Mauritanian CIN.
# ────────────────────────────────────────────────────────────────────────────

CIN_ENHANCED_PATH = HERE / "cin_enhanced.jpg"
CIN_WARPED_PATH = HERE / "cin_warped.jpg"


def estimate_blur(card: np.ndarray) -> float:
    """Laplacian variance — higher is sharper. <100 ≈ unusable for OCR."""
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY) if card.ndim == 3 else card
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _maybe_upscale(card: np.ndarray, blur: float | None) -> np.ndarray:
    """2× upscale before OCR when the frame is very soft (webcam / phone)."""
    if blur is None or blur >= 80:
        return card
    return cv2.resize(card, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)


def _crop_region(card: np.ndarray, y0: float, y1: float,
                 x0: float = 0.22, x1: float = 0.72) -> np.ndarray:
    h, w = card.shape[:2]
    return card[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]


def _crop_french_column(card: np.ndarray) -> np.ndarray:
    """Mauritanian CIN layout: photo on the left ~28%, French text in the
    middle ~30–68%, Arabic on the right ~70–100%. Cropping to just the
    middle column dramatically improves OCR accuracy — Tesseract no longer
    has to wade through the photo or guess at Arabic glyphs."""
    return _crop_region(card, 0.04, 0.97, 0.25, 0.75)


def normalize_card_image(card: np.ndarray) -> tuple[np.ndarray, bool]:
    """Deskew/warp the card if detected, then upscale soft captures."""
    warped, was_warped = find_and_warp_card(card)
    blur = estimate_blur(warped)
    work = _maybe_upscale(warped, blur)
    if was_warped:
        cv2.imwrite(str(CIN_WARPED_PATH), work)
    return work, was_warped


def prepare_gemini_image(card: np.ndarray) -> np.ndarray:
    """French text column only — avoids Arabic column / ghost photo confusion
    on sharp scans where both scripts are equally legible."""
    work, _ = normalize_card_image(card)
    french = _crop_french_column(work)
    if french.size > 0 and french.shape[0] >= 80 and french.shape[1] >= 80:
        return french
    return work


def _crop_id_photo_column(card: np.ndarray) -> np.ndarray:
    """Left portrait strip on Mauritanian CIN (for cin_enhanced preview)."""
    return _crop_region(card, 0.10, 0.84, 0.02, 0.34)


def _enhance_card(card: np.ndarray, blur: float | None = None,
                  aggressive: bool = False) -> np.ndarray:
    """Preprocess for OCR. Blurry webcam captures need CLAHE + higher upscale."""
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)

    _, glare = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    glare = cv2.dilate(glare, np.ones((3, 3), np.uint8), iterations=1)
    if cv2.countNonZero(glare) > 60:
        gray = cv2.inpaint(gray, glare, 4, cv2.INPAINT_TELEA)

    use_aggressive = aggressive or (blur is not None and blur < 200)
    if use_aggressive:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        scale = 4.0 if (blur is not None and blur < 100) else 3.5
        sharp_w, sharp_b = 1.6, -0.6
    else:
        scale = 3.0
        sharp_w, sharp_b = 1.3, -0.3

    gray = cv2.resize(gray, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gray = cv2.addWeighted(gray, sharp_w, blurred, sharp_b, 0)
    return gray


def _save_cin_enhanced_preview(card: np.ndarray, blur: float | None) -> None:
    """Write cin_enhanced.jpg — enhanced crop of the ID portrait (left side)."""
    roi = _crop_id_photo_column(card)
    if roi.size == 0:
        roi = card
    enhanced = _enhance_card(roi, blur=blur, aggressive=True)
    cv2.imwrite(str(CIN_ENHANCED_PATH), enhanced)


_easy_reader = None


def _get_easy_reader():
    """Lazy-init EasyOCR. Returns None if it isn't installed."""
    global _easy_reader
    if _easy_reader is False:
        return None
    if _easy_reader is None:
        try:
            import easyocr  # type: ignore
        except ImportError:
            _easy_reader = False
            return None
        print("  ⏳ Initialisation EasyOCR (premier appel: ~10 s)...",
              file=sys.stderr)
        _easy_reader = easyocr.Reader(["fr", "en"], gpu=False, verbose=False)
    return _easy_reader


def _easyocr_rows(target: np.ndarray, blur: float | None = None
                  ) -> list[tuple[float, str, float]]:
    """EasyOCR detections as (y_ratio, text, confidence), top-to-bottom."""
    reader = _get_easy_reader()
    if reader is None:
        return []
    enhanced = _enhance_card(target, blur=blur, aggressive=True)
    bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    h = bgr.shape[0]
    results = reader.readtext(bgr)
    rows: list[tuple[float, str, float]] = []
    for bbox, text, conf in results:
        text = text.strip()
        if not text or conf < 0.04:
            continue
        y = (bbox[0][1] + bbox[2][1]) / (2.0 * h)
        rows.append((y, text, conf))
    rows.sort(key=lambda r: r[0])
    return rows


def _ocr_easyocr(target: np.ndarray, blur: float | None = None) -> str:
    """Read with EasyOCR on the same enhanced grayscale Tesseract uses."""
    rows = _easyocr_rows(target, blur=blur)
    return "\n".join(text for (_y, text, _c) in rows)


def _ocr_tesseract_on(enhanced: np.ndarray) -> str:
    """Run Tesseract on a pre-enhanced image; best PSM by parse score."""
    candidates: list[str] = []
    for psm in ("6", "4", "11", "3"):
        try:
            text = pytesseract.image_to_string(enhanced, lang="fra",
                                               config=f"--psm {psm}")
        except pytesseract.TesseractError:
            continue
        if text.strip():
            candidates.append(text)
    if not candidates:
        return ""
    return max(candidates,
               key=lambda t: sum(1 for v in parse_fields(t).values() if v))


def _ocr_tesseract(target: np.ndarray, blur: float | None = None) -> str:
    """Tesseract: standard + aggressive enhance when the frame is blurry."""
    passes: list[str] = []
    std = _enhance_card(target, blur=blur, aggressive=False)
    passes.append(_ocr_tesseract_on(std))
    if blur is not None and blur < 200:
        agg = _enhance_card(target, blur=blur, aggressive=True)
        passes.append(_ocr_tesseract_on(agg))
    return _merge_ocr_texts(*passes)


def _merge_ocr_texts(*chunks: str) -> str:
    """Join OCR dumps; dedupe lines so the parser sees every fragment once."""
    seen: set[str] = set()
    lines: list[str] = []
    for chunk in chunks:
        for ln in chunk.splitlines():
            key = ln.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            lines.append(ln.strip())
    return "\n".join(lines)


def _normalize_name_garbled(value: str) -> str | None:
    """Best-effort cleanup for noisy OCR name lines."""
    value = re.sub(r"[؀-ۿ]+", "", value)
    value = re.sub(r"[{}|£€$#@0-9]", " ", value)
    value = re.sub(r"\s+", " ", value).strip(" :-\t|·•_~")
    if not value or _normalize_date(value):
        return None
    parts: list[str] = []
    for raw in value.split():
        w = re.sub(r"[^A-Za-z'\-]", "", raw)
        if not w:
            continue
        if w.upper() == "E":
            parts.append("El")
        elif len(w) >= 2:
            parts.append(w[0].upper() + w[1:].lower())
    joined = " ".join(parts)
    if sum(c.isalpha() for c in joined) < 4:
        return None
    return joined or None


def _normalize_place_garbled(value: str) -> str | None:
    value = re.sub(r"[^A-Za-zÀ-ſ'\-\s]", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) < 3 or len(value) > 24:
        return None
    if _normalize_date(value):
        return None
    if value.lower() in _LABEL_HINTS:
        return None
    return value[0].upper() + value[1:].lower()


def _merge_easyocr_rows(rows: list[tuple[float, str, float]]
                        ) -> list[tuple[float, str, float]]:
    merged: list[tuple[float, str, float]] = []
    for y, text, conf in rows:
        if (merged and abs(y - merged[-1][0]) < 0.07
                and not re.search(r"\d{10}", text)):
            py, pt, pc = merged[-1]
            merged[-1] = (py, f"{pt} {text}", max(pc, conf))
        else:
            merged.append((y, text, conf))
    return merged


# Standard Mauritanian CIN field order (French column, top → bottom).
CIN_FIELD_ORDER: tuple[str, ...] = (
    "nni", "prenom", "prenom_pere", "nom_famille",
    "sexe", "date_naissance", "lieu_naissance",
)

# Relative Y-bands on a perspective-warped card (same layout for all CINs).
_CIN_BANDS: tuple[tuple[str, float, float], ...] = (
    ("nni",              0.11, 0.30),
    ("prenom",           0.28, 0.40),
    ("prenom_pere",      0.38, 0.51),
    ("nom_famille",      0.48, 0.59),
    ("sexe",             0.55, 0.70),
    ("date_naissance",   0.64, 0.77),
    ("lieu_naissance",   0.74, 0.95),
)


def _parse_sexe_ocr_token(text: str) -> str | None:
    """Turn noisy OCR like 'fM' / 'Mmf' into M or F."""
    letters = re.sub(r"[^MFmf]", "", text).upper()
    if not letters:
        return None
    if len(letters) >= 2 and len(set(letters)) == 1:
        return None
    if letters in ("M", "F"):
        return letters
    if len(letters) >= 2:
        if letters.count("M") > letters.count("F"):
            return "M"
        if letters.count("F") > letters.count("M"):
            return "F"
        return "M" if letters.endswith("M") else "F"
    return None


def _ocr_sexe_band(roi: np.ndarray, blur: float | None) -> str | None:
    """Read M/F on blurry CIN photos (heavy upscale + EasyOCR allowlist)."""
    if roi.size == 0:
        return None
    big = cv2.resize(roi, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    enh = _enhance_card(big, blur=blur, aggressive=True)
    best: tuple[str, float] | None = None

    def _consider(text: str, conf: float, *, multi_only: bool) -> None:
        nonlocal best
        token = _parse_sexe_ocr_token(text)
        if not token:
            return
        letters = re.sub(r"[^MFmf]", "", text)
        if multi_only and len(letters) < 2:
            return
        score = conf * (1.0 + 0.35 * max(0, len(letters) - 1))
        if len(letters) < 2 and conf < 0.40:
            return
        if best is None or score > best[1]:
            best = (token, score)

    easy_hits: list[tuple[str, float]] = []
    for psm in ("10", "8", "7"):
        try:
            raw = pytesseract.image_to_string(
                enh, config=f"--psm {psm} -c tessedit_char_whitelist=MFmf")
        except pytesseract.TesseractError:
            continue
        letters = re.sub(r"[^MFmf]", "", raw)
        easy_hits.append((raw, 0.55 if len(letters) >= 2 else 0.25))
    try:
        reader = _get_easy_reader()
        for _bbox, text, conf in reader.readtext(
                enh, allowlist="MFmf", paragraph=False):
            easy_hits.append((text, float(conf)))
    except Exception:
        pass
    has_multi = any(len(re.sub(r"[^MFmf]", "", t)) >= 2 for t, _ in easy_hits)
    for text, conf in easy_hits:
        _consider(text, conf, multi_only=has_multi)
    if best is None or best[1] < 0.42:
        return None
    return best[0]


def _ocr_band_field(roi: np.ndarray, blur: float | None,
                    field: str) -> str | None:
    """OCR one layout band — tuned per field type (all Mauritanian CINs)."""
    if roi.size == 0:
        return None
    enh = _enhance_card(roi, blur=blur, aggressive=True)

    if field == "nni":
        for psm in ("7", "8", "6"):
            try:
                raw = pytesseract.image_to_string(
                    enh, config=f"--psm {psm} -c tessedit_char_whitelist=0123456789")
            except pytesseract.TesseractError:
                continue
            digits = re.sub(r"\D", "", raw)
            m = re.search(r"\d{10}", digits)
            if m:
                return m.group(0)
        return None

    if field == "sexe":
        return _ocr_sexe_band(roi, blur)

    text = _ocr_tesseract_on(enh)
    if not text:
        return None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    if field == "date_naissance":
        for ln in lines:
            d = _normalize_date(ln)
            if d:
                return d
        return _normalize_date(" ".join(lines))

    for ln in lines:
        if field == "lieu_naissance":
            val = _normalize_place_garbled(ln)
        else:
            val = _normalize_name_garbled(ln)
        if val:
            return val
    return None


def _ocr_cin_by_bands(card: np.ndarray, blur: float | None) -> dict[str, str | None]:
    """Field-by-field OCR using the standard Mauritanian CIN vertical layout."""
    out: dict[str, str | None] = {k: None for k in CIN_FIELD_ORDER}
    for field, y0, y1 in _CIN_BANDS:
        roi = _crop_region(card, y0, y1)
        out[field] = _ocr_band_field(roi, blur, field)
    return out


def _fill_lieu_after_date(lines: list[str], out: dict) -> None:
    """Place of birth is usually the first place-like line after the date."""
    if out.get("lieu_naissance"):
        return
    date_i = next((i for i, ln in enumerate(lines)
                   if _normalize_date(ln)), None)
    if date_i is None:
        return
    for ln in lines[date_i + 1:]:
        place = _normalize_place_garbled(ln)
        if place:
            out["lieu_naissance"] = place
            return


def _nni_score(candidate: str) -> float:
    """Rank 10-digit OCR guesses (Mauritanian NNI vs date noise)."""
    s = 0.0
    if candidate.startswith("24"):
        s += 4.0
    elif candidate.startswith("85"):
        s += 3.0
    elif candidate.startswith("20"):
        s += 0.5
    if "2003" in candidate or "2004" in candidate:
        s -= 6.0
    if candidate.count(candidate[0]) > 6:
        s -= 2.0
    return s


def _easyocr_nni_candidates(card: np.ndarray, blur: float | None) -> list[str]:
    """High-confidence digit-only EasyOCR on the warped card (NNI row)."""
    reader = _get_easy_reader()
    if reader is None:
        return []
    try:
        results = reader.readtext(card, allowlist="0123456789", paragraph=False)
    except Exception:
        return []
    out: list[str] = []
    for _bbox, text, conf in results:
        if conf < 0.35:
            continue
        digits = re.sub(r"\D", "", text)
        if len(digits) == 10:
            out.append(digits)
    return out


def _best_nni(*texts: str) -> str | None:
    """Pick the most likely NNI when OCR returns several 10-digit strings."""
    cands: list[str] = []
    for t in texts:
        if not t:
            continue
        cands.extend(re.findall(r"(?<!\d)(\d{10})(?!\d)", t))
        for line in t.splitlines():
            digits = re.sub(r"\D", "", line)
            if len(digits) >= 10:
                for i in range(len(digits) - 9):
                    cands.append(digits[i:i + 10])
    joined = re.sub(r"\D", "", "\n".join(t for t in texts if t))
    for i in range(max(0, len(joined) - 9)):
        cands.append(joined[i:i + 10])
    cands = [c for c in cands if c.isdigit() and len(c) == 10]
    if not cands:
        return None
    best = max(cands, key=_nni_score)
    if _nni_score(best) < -2.0:
        pool = list(dict.fromkeys(cands))
        if len(pool) == 1:
            return pool[0]
        voted: list[str] = []
        for i in range(10):
            digits = [c[i] for c in pool]
            voted.append(max(set(digits), key=digits.count))
        return "".join(voted)
    return best


def _layout_fields_from_rows(rows: list[tuple[float, str, float]]) -> dict:
    """Map EasyOCR Y-positions to CIN fields (Mauritanian layout)."""
    out: dict[str, str | None] = {
        "prenom": None, "prenom_pere": None, "nom_famille": None,
        "nni": None, "date_naissance": None, "lieu_naissance": None, "sexe": None,
    }
    rows = _merge_easyocr_rows(rows)
    if not rows:
        return out

    date_y = None
    nni_y = None
    for y, text, _conf in rows:
        if _normalize_date(text):
            date_y = y if date_y is None else min(date_y, y)
        if re.search(r"(?<!\d)(\d{10})(?!\d)", text):
            nni_y = y if nni_y is None else min(nni_y, y)

    for y, text, conf in rows:
        m = re.search(r"(?<!\d)(\d{10})(?!\d)", text)
        if m and conf >= 0.15 and m.group(1).startswith("85"):
            out["nni"] = m.group(1)

    for y, text, _conf in rows:
        if re.match(r"^[MF]$", text.strip(), re.IGNORECASE):
            out["sexe"] = text.strip().upper()

    for y, text, _conf in rows:
        d = _normalize_date(text)
        if d:
            out["date_naissance"] = d
    if not out["date_naissance"]:
        tail = " ".join(t for _y, t, _c in rows if _y > 0.75)
        d = _normalize_date(tail)
        if d:
            out["date_naissance"] = d

    name_slots = ("prenom", "prenom_pere", "nom_famille")
    idx = 0
    for y, text, _conf in rows:
        if idx >= len(name_slots):
            break
        if nni_y is not None and y <= nni_y + 0.04:
            continue
        if date_y is not None and y >= date_y - 0.04:
            continue
        if re.search(r"\d{10}", text):
            continue
        if _normalize_date(text) or re.match(r"^[MF]$", text.strip(), re.I):
            continue
        name = _normalize_name_garbled(text)
        if not name:
            continue
        key = name_slots[idx]
        if not out.get(key):
            out[key] = name
            idx += 1

    if date_y is not None:
        for y, text, _conf in rows:
            if y <= date_y + 0.03:
                continue
            place = _normalize_place_garbled(text)
            if place:
                out["lieu_naissance"] = place
                break

    return out


def ocr_card(card: np.ndarray, engine: str = "auto",
             *, return_layout: bool = False) -> str | tuple[str, dict]:
    """Run OCR on a CIN image.

    engine ∈ {"tesseract", "easyocr", "auto"}.
        - "auto" merges Tesseract + EasyOCR when either is weak alone.
        - "tesseract" / "easyocr" force a single engine.

    Returns raw OCR text — feed it into `parse_fields()` afterwards.
    """
    warped, was_warped = find_and_warp_card(card)
    if was_warped:
        cv2.imwrite(str(CIN_WARPED_PATH), warped)
        work = _maybe_upscale(warped, estimate_blur(warped))
        blur = estimate_blur(warped)
    else:
        work = _maybe_upscale(card, estimate_blur(card))
        blur = estimate_blur(card)

    _save_cin_enhanced_preview(work, blur)

    target = _crop_french_column(work)
    names_band = _crop_region(work, 0.26, 0.65)
    bottom_band = _crop_region(work, 0.55, 0.98)

    def _score(text: str) -> int:
        return sum(1 for v in parse_fields(text, blur=blur).values() if v)

    if engine == "tesseract":
        merged = _merge_ocr_texts(
            _ocr_tesseract(target, blur=blur),
            _ocr_tesseract(names_band, blur=blur),
            _ocr_tesseract(bottom_band, blur=blur),
        )
        layout: dict = {}
    elif engine == "easyocr":
        merged = _ocr_easyocr(target, blur=blur)
        layout = {}
    else:
        t_text = _merge_ocr_texts(
            _ocr_tesseract(target, blur=blur),
            _ocr_tesseract(names_band, blur=blur),
            _ocr_tesseract(bottom_band, blur=blur),
        )
        e_text = _ocr_easyocr(target, blur=blur)
        merged = _merge_ocr_texts(t_text, e_text)
        if _score(merged) < max(_score(t_text), _score(e_text)):
            merged = e_text if _score(e_text) > _score(t_text) else t_text
        layout = {}

    hints = {
        "spatial": _layout_fields_from_rows(_easyocr_rows(target, blur=blur)),
        "bands": _ocr_cin_by_bands(work, blur),
        "nni_easyocr": _easyocr_nni_candidates(work, blur),
    }
    if return_layout:
        return merged, hints
    return merged


# Labels we look for. Each entry is a list of substring hints that may
# appear (with OCR noise) in a line. We deliberately omit accents on most
# hints — the OCR rarely produces them.
_FIELD_LABELS: dict[str, tuple[str, ...]] = {
    "nni":            ("numero national", "national d'identification",
                       "national d identification", "numero national d",
                       "n° national", "no national"),
    "prenom":         ("prenom/given", "prenom / given", "given name",
                       "personal name", "prenom", "prénom"),
    "prenom_pere":    ("prenom de pere", "prenom du pere", "father's given",
                       "father name", "father's name", "father", "pere/father"),
    "nom_famille":    ("nom de famille", "family name", "surname",
                       "famille/surname", "nom famille"),
    "sexe":           ("sexe/sex", "sex", "sexe"),
    "date_naissance": ("date de naissance", "date of birth", "naissance"),
    "lieu_naissance": ("lieu de naissance", "place of birth",
                       "lieu naissance", "place birth"),
}
_LABEL_HINTS = tuple({h for hints in _FIELD_LABELS.values() for h in hints})

_FR_MONTHS = ("janvier", "fevrier", "février", "mars", "avril", "mai",
              "juin", "juillet", "aout", "août", "septembre", "octobre",
              "novembre", "decembre", "décembre")
_EN_MONTHS = ("january", "february", "march", "april", "may", "june",
              "july", "august", "september", "october", "november", "december")
_MONTH_CANON = {m.lower().replace("é", "e").replace("û", "u"): m.capitalize()
                for m in _FR_MONTHS}
_MONTH_CANON.update({m.lower(): m.capitalize() for m in _EN_MONTHS})
# Common OCR mis-reads on blurry CIN photos (Mai → Magi, Mgy, Maiine…).
_MONTH_OCR_ALIASES = {
    "mgy": "mai", "mag": "mai", "magi": "mai", "maiine": "mai",
    "maiiney": "mai", "maiinay": "mai", "may": "mai", "magy": "mai",
    "maii": "mai",
}
_DATE_RE_NUMERIC = re.compile(r"\b(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})\b")
_DATE_RE_TEXT = re.compile(
    r"(\d{1,2})\s+([A-Za-zÀ-ſ]+)(?:\s*/\s*[A-Za-zÀ-ſ]+)?\s+(\d{4})",
    re.IGNORECASE,
)


def _looks_like_label(line: str) -> bool:
    low = line.lower()
    return any(h in low for h in _LABEL_HINTS)


def _clean(value: str) -> str:
    value = re.sub(r"/.*$", "", value)         # drop trailing English alt
    value = re.sub(r"[؀-ۿ]+", "", value)  # drop Arabic glyphs
    value = re.sub(r"[|£€$#@]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" :-\t|·•_~").strip()


def _canon_month(month_raw: str) -> str | None:
    token = re.sub(r"[^A-Za-zÀ-ſ]", "", month_raw).lower()
    token = token.replace("é", "e").replace("û", "u")
    if not token:
        return None
    token = _MONTH_OCR_ALIASES.get(token, token)
    if token in _MONTH_CANON:
        return _MONTH_CANON[token]
    for key, label in _MONTH_CANON.items():
        if token.startswith(key[:3]) or key.startswith(token[:3]):
            return label
    return None


def _normalize_date(s: str) -> str | None:
    s = s.strip()
    if not s:
        return None
    m = _DATE_RE_TEXT.search(s)
    if m:
        day, month_raw, year = m.group(1), m.group(2), m.group(3)
        month = _canon_month(month_raw)
        if month:
            return f"{int(day):02d} {month} {year}"
    m = _DATE_RE_NUMERIC.search(s)
    if m:
        d, mo, y = m.groups()
        if len(y) == 2:
            y = "20" + y if int(y) < 50 else "19" + y
        return f"{int(d):02d}/{int(mo):02d}/{y}"
    return None


def _looks_like_name(value: str) -> bool:
    val = _clean(value)
    if len(val) < 3 or len(val) > 40:
        return False
    if re.search(r"\d", val):
        return False
    if _looks_like_label(val) or _normalize_date(val):
        return False
    words = [w for w in val.split() if w]
    if not words or len(words) > 4:
        return False
    if any(w.lower() in ("de", "du", "des", "le", "la", "ne", "nr", "nat",
                         "me", "mu", "ve", "a", "y", "d") for w in words):
        return False
    caps = [w for w in words if len(w) >= 3 and w[0].isupper()]
    if not caps:
        return False
    if len(words) >= 2 and len(caps) < 2:
        return False
    letters = sum(c.isalpha() for c in val)
    return letters >= max(3, int(len(val) * 0.6))


def _fill_names_after_nni(lines: list[str], out: dict,
                          blur: float | None = None) -> None:
    """CIN layout: NNI then prénom, prénom du père, nom de famille."""
    nni_idx = next((i for i, ln in enumerate(lines)
                    if re.search(r"(?<!\d)(\d{10})(?!\d)", ln)), None)
    if nni_idx is None:
        return
    slots = ("prenom", "prenom_pere", "nom_famille")
    filled = 0
    for ln in lines[nni_idx + 1:]:
        if filled >= len(slots):
            break
        if not _looks_like_name(ln):
            continue
        field = slots[filled]
        if not out.get(field):
            out[field] = _clean(ln)
            filled += 1


def parse_fields(text: str, blur: float | None = None,
                 layout: dict | None = None,
                 bands: dict | None = None) -> dict:
    out = {k: None for k in
           ("nom", "prenom", "prenom_pere", "nom_famille", "nni",
            "date_naissance", "lieu_naissance", "sexe")}
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Optional layout hints from ocr_card(return_layout=True).
    spatial: dict | None = None
    if layout and "bands" in layout:
        bands = bands or layout.get("bands")
        spatial = layout.get("spatial")
    elif layout:
        spatial = layout

    out["nni"] = _best_nni(text)

    # Sex — only count a bare M/F when it appears within ~2 lines after a
    # "Sexe/Sex" label. A loose `\bF\b` match catches things like
    # "Famille" → F initial, which is what gave us the wrong answer before.
    for i, line in enumerate(lines):
        low = line.lower()
        if "sexe" in low or "sex" in low:
            for j in range(i, min(i + 3, len(lines))):
                cand = lines[j]
                # Strip the label off the same line if present.
                tail = re.sub(r"(?i)sexe.*?[:\-/]\s*", "", cand)
                tail = re.sub(r"(?i).*sex\b.*?[:\-/]\s*", "", tail).strip()
                m = re.match(r"^([MFmf])(?:\b|$)", tail)
                if m:
                    out["sexe"] = m.group(1).upper()
                    break
            if out["sexe"]:
                break
    if not out["sexe"]:
        if re.search(r"(?:^|\W)(Masculin|ذكر)(?:\W|$)", text):
            out["sexe"] = "M"
        elif re.search(r"(?:^|\W)(F[ée]minin|أنثى)(?:\W|$)", text):
            out["sexe"] = "F"

    # Label-then-value: find a label line, take the next non-label line.
    for i, line in enumerate(lines):
        low = line.lower()
        for field, hints in _FIELD_LABELS.items():
            if out.get(field) or field in ("nni", "sexe"):
                continue
            if not any(h in low for h in hints):
                continue
            for j in range(i + 1, min(i + 4, len(lines))):
                cand = lines[j]
                if _looks_like_label(cand):
                    continue
                val = _clean(cand)
                if not val:
                    continue
                if field == "date_naissance":
                    norm = _normalize_date(val)
                    if norm:
                        out[field] = norm
                        break
                    continue
                if field == "sexe" and val.upper() not in ("M", "F"):
                    continue
                out[field] = val
                break

    # Same-line fallback ("Label: Value").
    for line in lines:
        for field, hints in _FIELD_LABELS.items():
            if out.get(field):
                continue
            for h in hints:
                m = re.search(rf"{re.escape(h)}\s*[:\-]\s*(.+)",
                              line, flags=re.IGNORECASE)
                if m:
                    val = _clean(m.group(1))
                    if field == "date_naissance":
                        val = _normalize_date(val) or val
                    if val:
                        out[field] = val
                    break

    # Last-resort date scan if label-walk missed it.
    if not out["date_naissance"]:
        for line in lines:
            d = _normalize_date(line)
            if d:
                out["date_naissance"] = d
                break

    if not out.get("prenom"):
        _fill_names_after_nni(lines, out, blur=blur)
    _fill_lieu_after_date(lines, out)

    # Band OCR — fixed vertical layout shared by all Mauritanian CINs.
    if bands:
        if bands.get("sexe") in ("M", "F"):
            out["sexe"] = bands["sexe"]
        if bands.get("date_naissance") and not out.get("date_naissance"):
            out["date_naissance"] = bands["date_naissance"]
        for key in ("prenom", "prenom_pere", "nom_famille", "lieu_naissance"):
            val = bands.get(key)
            if val and not out.get(key):
                out[key] = val

    # EasyOCR row order — reliable for names on soft photos.
    if spatial:
        soft = blur is not None and blur < 120
        for key in ("prenom", "prenom_pere", "nom_famille", "lieu_naissance"):
            val = spatial.get(key)
            if not val:
                continue
            if soft or not out.get(key):
                out[key] = val
        for key in ("date_naissance", "sexe"):
            val = spatial.get(key)
            if val and not out.get(key):
                out[key] = val

    band_nni = str(bands["nni"]) if bands and bands.get("nni") else ""
    extra_nni = []
    if layout and layout.get("nni_easyocr"):
        extra_nni = list(layout["nni_easyocr"])
    voted = _best_nni(text, band_nni, out.get("nni") or "", *extra_nni)
    if voted:
        out["nni"] = voted

    if not out["sexe"]:
        for line in lines:
            m = re.search(r"(?i)(?:sexe|sex)\b[^A-Za-z]{0,12}([MF])\b", line)
            if m:
                out["sexe"] = m.group(1).upper()
                break
    if not out["sexe"]:
        for line in lines:
            if re.match(r"^[MF]$", line.strip(), re.IGNORECASE):
                out["sexe"] = line.strip().upper()
                break

    out["nom"] = out["nom_famille"]
    return out


def _apply_ml_refinement(fields: dict, card: np.ndarray,
                         layout_hints: dict | None, *,
                         gemini_source: bool = False) -> dict:
    """TP8-trained models (LDA + fuzzy text) when models/cin_fields.joblib exists."""
    try:
        from ml.cin_models import refine_fields
    except ImportError:
        return fields
    return refine_fields(fields, card, layout_hints,
                         gemini_source=gemini_source)


_RESULT_LABELS: tuple[tuple[str, str], ...] = (
    ("nni",            "NNI"),
    ("prenom",         "Prénom"),
    ("prenom_pere",    "Prénom du père"),
    ("nom_famille",    "Nom de famille"),
    ("sexe",           "Sexe"),
    ("date_naissance", "Date de naissance"),
    ("lieu_naissance", "Lieu de naissance"),
)
CIN_RESULT_PNG = HERE / "cin_result.png"


def extract_fields(card: np.ndarray, engine: str = "auto",
                   *, use_ml: bool = True, verbose: bool = True) -> dict:
    """High-level CIN field extractor.

    engine ∈ {"auto", "gemini", "offline", "tesseract", "easyocr"}.
    use_ml=False skips the trained corrector (for building training labels).
    """
    work, _was_warped = normalize_card_image(card)

    # 1) Try Gemini if asked or in auto mode.
    if engine in ("gemini", "auto"):
        try:
            from vision_gemini import gemini_extract_cin, is_available
            if is_available():
                if verbose:
                    print("  → Extraction via Gemini 2.5 Flash...")
                gemini_img = prepare_gemini_image(card)
                fields = gemini_extract_cin(gemini_img)
                if not all(fields.get(k) for k in CIN_FIELD_ORDER):
                    if verbose:
                        print("  → Gemini: nouvel essai sur la carte entière…")
                    fields_full = gemini_extract_cin(work)
                    for key in CIN_FIELD_ORDER:
                        if not fields.get(key) and fields_full.get(key):
                            fields[key] = fields_full[key]
                CIN_RAW_PATH.write_text(
                    "\n".join(f"{k}: {v}" for k, v in fields.items()),
                    encoding="utf-8",
                )
                fields.setdefault("nom", fields.get("nom_famille"))
                if use_ml:
                    fields = _apply_ml_refinement(
                        fields, work, None, gemini_source=True)
                return fields
            if engine == "gemini":
                raise RuntimeError(
                    "Clé API Gemini absente ou SDK non installé. "
                    "Définir GEMINI_API_KEY ou installer `google-genai`."
                )
            if verbose:
                print("  → Gemini indisponible (clé / SDK), bascule offline.")
        except Exception as e:
            if engine == "gemini":
                raise
            if verbose:
                print(f"  ⚠ Gemini a échoué ({e}); bascule offline.")

    # 2) Offline fallback (sharp scans need the French column — same as OCR path).
    if verbose:
        blur_val = estimate_blur(work)
        if blur_val >= 150:
            print("  → Extraction OCR offline (Tesseract / EasyOCR)…")
            print("  ⚠ Photo nette : préférez --engine gemini ou auto "
                  "(offline confond souvent l'arabe sur les scans).")
        else:
            print("  → Extraction OCR offline (Tesseract / EasyOCR)…")
    sub_engine = "auto" if engine in ("auto", "offline") else engine
    blur_val = estimate_blur(work)
    raw, layout = ocr_card(work, engine=sub_engine, return_layout=True)
    CIN_RAW_PATH.write_text(raw, encoding="utf-8")
    fields = parse_fields(raw, blur=blur_val, layout=layout)
    return _apply_ml_refinement(fields, work, layout) if use_ml else fields


def print_result(fields: dict) -> None:
    print("\n═══════════════ RESULTAT CIN ═══════════════")
    for key, label in _RESULT_LABELS:
        print(f"{label:<20}: {fields.get(key) or '—'}")
    print("════════════════════════════════════════════")


def visualize_result(card: np.ndarray, fields: dict,
                     out_path: Path = CIN_RESULT_PNG) -> None:
    """TP8-style figure: captured CIN on the left, extracted fields on the
    right with green ✓ for captured, red ✗ for missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # save without requiring a display backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib non installe; `pip install matplotlib` "
              "pour activer la visualization.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(cv2.cvtColor(card, cv2.COLOR_BGR2RGB))
    axes[0].set_title("CIN capturee", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].axis("off")
    axes[1].set_title("Champs extraits", fontsize=14, fontweight="bold")

    n = len(_RESULT_LABELS)
    y_top, y_bot = 0.92, 0.08
    step = (y_top - y_bot) / max(1, n - 1)
    for i, (key, label) in enumerate(_RESULT_LABELS):
        y = y_top - i * step
        value = fields.get(key)
        ok = bool(value)
        color = "#2ea043" if ok else "#cf222e"
        marker = "✓" if ok else "✗"
        axes[1].text(0.03, y, marker, fontsize=18, color=color,
                     fontweight="bold", va="center")
        axes[1].text(0.10, y, f"{label} :", fontsize=12, fontweight="bold",
                     va="center")
        axes[1].text(0.46, y, value or "—", fontsize=12, color=color,
                     va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"  → {out_path.name} sauvegarde")


# ────────────────────────────────────────────────────────────────────────────
# Entry point.
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="KYC demo")
    parser.add_argument("--camera", type=int, default=None,
                        help="OpenCV camera index (default 0). Use --list-cameras.")
    parser.add_argument("--camera-name", type=str, default=None,
                        help="macOS: pick camera by name substring "
                             "(e.g. iPhone, FaceTime, USB)")
    parser.add_argument("--list-cameras", action="store_true",
                        help="list available camera indices and exit")
    parser.add_argument("--cin-file", type=str, default=None,
                        help="skip the webcam CIN capture; OCR this photo "
                             "instead (use a sharp phone photo for best results)")
    parser.add_argument("--engine",
                        choices=("auto", "gemini", "offline",
                                 "tesseract", "easyocr"),
                        default="auto",
                        help="Extraction engine. 'auto' = Gemini if available, "
                             "fall back to offline (Tesseract + EasyOCR). "
                             "'offline' forces the offline pipeline.")
    parser.add_argument("--no-show", action="store_true",
                        help="do not auto-open the result PNG")
    args = parser.parse_args()

    if args.list_cameras:
        _print_camera_list()
        return

    t0 = time.time()

    if not args.cin_file:
        cam_idx = resolve_camera_index(args.camera, args.camera_name)
        if args.camera is None and args.camera_name is None:
            cam_idx = 0
        print(f"Camera : index {cam_idx}  ({platform.system()} backend)")
        selfie, face_bbox = run_liveness(cam_idx)
        if selfie is not None and face_bbox is not None:
            print("\n[BG] Decoupe du visage (GrabCut)…")
            cutout_face_png(selfie, face_bbox, FACE_PNG_PATH)

    if args.cin_file:
        cin_path = Path(args.cin_file).expanduser().resolve()
        if not cin_path.exists():
            sys.exit(f"ERREUR: fichier CIN introuvable: {cin_path}")
        card = cv2.imread(str(cin_path))
        if card is None:
            sys.exit(f"ERREUR: impossible de lire l'image: {cin_path}")
        print(f"\n[CIN] Lecture depuis fichier: {cin_path.name} "
              f"({card.shape[1]}×{card.shape[0]})")
        cv2.imwrite(str(CIN_PATH), card)
    else:
        card = capture_cin(cam_idx)
    print(f"\n[Extraction] Moteur: {args.engine}")
    fields = extract_fields(card, engine=args.engine)
    print_result(fields)
    print(f"\n[Texte OCR brut       → {CIN_RAW_PATH.name}]")
    print(f"[Image pretraitee     → {CIN_ENHANCED_PATH.name}]")

    visualize_result(card, fields)
    if not args.no_show and CIN_RESULT_PNG.exists():
        opener = {"Darwin": "open", "Windows": "start", "Linux": "xdg-open"}.get(
            platform.system())
        if opener:
            import subprocess
            try:
                if platform.system() == "Windows":
                    subprocess.Popen([opener, "", str(CIN_RESULT_PNG)], shell=True)
                else:
                    subprocess.Popen([opener, str(CIN_RESULT_PNG)])
            except OSError:
                pass
    print(f"[Total: {time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
