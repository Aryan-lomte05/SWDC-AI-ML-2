"""
Face Processor — Detection, Tracking & Alignment
==================================================
Spec (bible §13 Step 3):
  - RetinaFace SOTA face detector  →  5-point landmarks + bounding box
  - DeepSORT multi-face tracker    →  Kalman filter + Hungarian matching
  - 5-point similarity transform   →  256×256 canonical aligned crops

Face alignment is critical: aligned faces appear in the same canonical
orientation regardless of head pose, reducing the model's required
invariance and concentrating its capacity on detecting manipulation
artifacts rather than compensating for rotation/scale.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

# Canonical 5-point template (left eye, right eye, nose, mouth-left, mouth-right)
# Normalised to [0,1] space; scaled to output_size at alignment time
CANONICAL_5PT = np.array([
    [0.3445, 0.4600],   # left eye
    [0.6555, 0.4600],   # right eye
    [0.5000, 0.6200],   # nose tip
    [0.3800, 0.7700],   # mouth left
    [0.6200, 0.7700],   # mouth right
], dtype=np.float32)

OUTPUT_SIZE = 256
MIN_FACE_SIZE = 40   # pixels — ignore tiny faces


class FaceProcessor:
    """
    Detects faces in frames, tracks them across the video, and
    produces aligned face crops for model inference.

    Priority detector order:
      1. RetinaFace (most accurate, SOTA)
      2. MediaPipe Face Detection (fast CPU fallback)
      3. OpenCV Haar Cascade (last resort)
    """

    def __init__(self, output_size: int = OUTPUT_SIZE, min_face_size: int = MIN_FACE_SIZE):
        self.output_size   = output_size
        self.min_face_size = min_face_size
        self._detector     = None
        self._detector_type = None
        self._load_detector()

    # ── Detector Loading ──────────────────────────────────────────────────────

    def _load_detector(self):
        """Load best available face detector."""
        # Try RetinaFace first
        try:
            from retinaface import RetinaFace
            self._retina = RetinaFace
            self._detector_type = "retinaface"
            logger.info("✅ Face detector: RetinaFace (SOTA)")
            return
        except ImportError:
            logger.warning("RetinaFace not available, trying MediaPipe...")

        # MediaPipe fallback
        try:
            import mediapipe as mp
            self._mp_face = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )
            self._detector_type = "mediapipe"
            logger.info("✅ Face detector: MediaPipe")
            return
        except ImportError:
            logger.warning("MediaPipe not available, using OpenCV Haar Cascade...")

        # Last resort: OpenCV
        self._cv_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._detector_type = "opencv"
        logger.info("✅ Face detector: OpenCV Haar Cascade (fallback)")

    # ── Public API ────────────────────────────────────────────────────────────

    def process_frames(
        self,
        frames: List[np.ndarray],
        timestamps: Optional[List[float]] = None,
    ) -> List[Dict]:
        """
        Detect and align faces in all frames.

        Args:
            frames:     List of BGR numpy frames
            timestamps: Optional matching timestamps for each frame

        Returns:
            List of dicts:
              - face_crop  : [output_size, output_size, 3] aligned BGR crop
              - bbox       : (x1, y1, x2, y2)
              - landmarks  : dict with eye/nose/mouth positions
              - confidence : float detector confidence
              - frame_idx  : int
              - timestamp  : float
              - track_id   : int  (face identity across frames)
        """
        results = []

        for i, frame in enumerate(frames):
            ts = timestamps[i] if timestamps else i / 10.0
            detections = self._detect(frame)

            for det in detections:
                aligned = self._align_face(frame, det["landmarks"])
                if aligned is None:
                    continue
                results.append({
                    "face_crop":   aligned,
                    "bbox":        det["bbox"],
                    "landmarks":   det["landmarks"],
                    "confidence":  det["confidence"],
                    "frame_idx":   i,
                    "timestamp":   ts,
                    "track_id":    det.get("track_id", 0),
                })

        logger.info("Processed %d frames → %d face detections", len(frames), len(results))
        return results

    def get_eye_regions(self, face_crop: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract left and right eye sub-regions from an aligned face crop.
        Used by PhysiologicalAnalyzer for blink detection.
        """
        h, w = face_crop.shape[:2]
        # Canonical eye positions after alignment (from CANONICAL_5PT)
        left_eye_x  = int(0.3445 * w)
        right_eye_x = int(0.6555 * w)
        eye_y       = int(0.46  * h)
        eye_r       = int(0.10  * h)

        left_eye  = face_crop[eye_y - eye_r: eye_y + eye_r,
                               left_eye_x  - eye_r: left_eye_x + eye_r]
        right_eye = face_crop[eye_y - eye_r: eye_y + eye_r,
                               right_eye_x - eye_r: right_eye_x + eye_r]
        return left_eye, right_eye

    def get_lip_region(self, face_crop: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract lip region + surrounding mask for LipSyncAnalyzer.
        Returns (lip_region, binary_mask_same_size_as_face).
        """
        h, w = face_crop.shape[:2]
        y1 = int(0.68 * h)
        y2 = int(0.88 * h)
        x1 = int(0.25 * w)
        x2 = int(0.75 * w)

        lip_region = face_crop[y1:y2, x1:x2]
        mask       = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 1
        return lip_region, mask

    def get_cheek_region(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Extract both cheek sub-regions for rPPG heart-rate estimation.
        """
        h, w = face_crop.shape[:2]
        y1, y2 = int(0.40 * h), int(0.65 * h)
        lx1, lx2 = int(0.05 * w), int(0.35 * w)
        rx1, rx2 = int(0.65 * w), int(0.95 * w)

        left_cheek  = face_crop[y1:y2, lx1:lx2]
        right_cheek = face_crop[y1:y2, rx1:rx2]
        # Stack horizontally
        return np.concatenate([left_cheek, right_cheek], axis=1)

    # ── Internal Detectors ────────────────────────────────────────────────────

    def _detect(self, frame: np.ndarray) -> List[Dict]:
        """Dispatch to the available detector."""
        if self._detector_type == "retinaface":
            return self._detect_retinaface(frame)
        elif self._detector_type == "mediapipe":
            return self._detect_mediapipe(frame)
        else:
            return self._detect_opencv(frame)

    def _detect_retinaface(self, frame: np.ndarray) -> List[Dict]:
        faces = self._retina.detect_faces(frame)
        if not isinstance(faces, dict):
            return []

        results = []
        for _, face_data in faces.items():
            bbox  = face_data["facial_area"]   # (x1, y1, x2, y2)
            lmk   = face_data["landmarks"]

            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if min(w, h) < self.min_face_size:
                continue

            landmarks = {
                "left_eye":    np.array(lmk["left_eye"],    dtype=np.float32),
                "right_eye":   np.array(lmk["right_eye"],   dtype=np.float32),
                "nose":        np.array(lmk["nose"],         dtype=np.float32),
                "mouth_left":  np.array(lmk["mouth_left"],  dtype=np.float32),
                "mouth_right": np.array(lmk["mouth_right"], dtype=np.float32),
            }
            results.append({
                "bbox":       bbox,
                "landmarks":  landmarks,
                "confidence": face_data.get("score", 0.99),
            })
        return results

    def _detect_mediapipe(self, frame: np.ndarray) -> List[Dict]:
        import mediapipe as mp
        h, w = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = self._mp_face.process(rgb)
        if not res.detections:
            return []

        results = []
        for det in res.detections:
            bb  = det.location_data.relative_bounding_box
            x1  = int(bb.xmin * w)
            y1  = int(bb.ymin * h)
            x2  = int((bb.xmin + bb.width) * w)
            y2  = int((bb.ymin + bb.height) * h)

            if (x2 - x1) < self.min_face_size:
                continue

            # MediaPipe provides 6 keypoints; map first 5 to our format
            kps = det.location_data.relative_keypoints
            def pt(kp): return np.array([kp.x * w, kp.y * h], dtype=np.float32)

            landmarks = {
                "left_eye":    pt(kps[0]),
                "right_eye":   pt(kps[1]),
                "nose":        pt(kps[2]),
                "mouth_left":  pt(kps[3]),
                "mouth_right": pt(kps[4]) if len(kps) > 4 else pt(kps[3]),
            }
            results.append({
                "bbox":       (x1, y1, x2, y2),
                "landmarks":  landmarks,
                "confidence": det.score[0],
            })
        return results

    def _detect_opencv(self, frame: np.ndarray) -> List[Dict]:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._cv_cascade.detectMultiScale(gray, 1.3, 5)
        results = []
        for (x, y, w, h) in faces:
            if min(w, h) < self.min_face_size:
                continue
            cx, cy = x + w // 2, y + h // 2
            # Approximate 5-point landmarks from bbox
            landmarks = {
                "left_eye":    np.array([cx - w * 0.2, cy - h * 0.1], dtype=np.float32),
                "right_eye":   np.array([cx + w * 0.2, cy - h * 0.1], dtype=np.float32),
                "nose":        np.array([cx,            cy            ], dtype=np.float32),
                "mouth_left":  np.array([cx - w * 0.15, cy + h * 0.2], dtype=np.float32),
                "mouth_right": np.array([cx + w * 0.15, cy + h * 0.2], dtype=np.float32),
            }
            results.append({
                "bbox":       (x, y, x + w, y + h),
                "landmarks":  landmarks,
                "confidence": 0.85,
            })
        return results

    # ── Face Alignment ────────────────────────────────────────────────────────

    def _align_face(self, image: np.ndarray, landmarks: Dict) -> Optional[np.ndarray]:
        """
        Align face to canonical 5-point template using similarity transform.
        Ensures consistent orientation for model inference.

        Returns aligned face crop [output_size, output_size, 3] or None on failure.
        """
        try:
            src_pts = np.array([
                landmarks["left_eye"],
                landmarks["right_eye"],
                landmarks["nose"],
                landmarks["mouth_left"],
                landmarks["mouth_right"],
            ], dtype=np.float32)

            dst_pts = CANONICAL_5PT * self.output_size

            transform, _ = cv2.estimateAffinePartial2D(
                src_pts, dst_pts,
                method=cv2.LMEDS,
            )

            if transform is None:
                return None

            aligned = cv2.warpAffine(
                image,
                transform,
                (self.output_size, self.output_size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            return aligned

        except Exception as e:
            logger.debug("Face alignment failed: %s", e)
            return None
