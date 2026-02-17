# HandTracking module using the new MediaPipe Tasks API
# Compatible with mediapipe >= 0.10.18 (where mp.solutions was removed)

import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Model download URL and local path
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")


def ensure_model_downloaded():
    """Download the hand_landmarker model if it doesn't exist locally."""
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading hand_landmarker model to {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete!")


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.lmlist = []
        self.results = None

        # Ensure model is available
        ensure_model_downloaded()

        # Create HandLandmarker with VIDEO mode (synchronous, frame-by-frame)
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.maxHands,
            min_hand_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.frame_timestamp_ms = 0

    def findHands(self, img, draw=True):
        """Detect hands and optionally draw landmarks on the image."""
        # Convert BGR (OpenCV) to RGB for MediaPipe
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)

        # Detect for video mode (needs increasing timestamps)
        self.frame_timestamp_ms += 33  # ~30 fps
        self.results = self.landmarker.detect_for_video(mp_image, self.frame_timestamp_ms)

        if draw and self.results.hand_landmarks:
            h, w, c = img.shape
            for hand_landmarks in self.results.hand_landmarks:
                # Draw connections between landmarks
                landmark_points = []
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmark_points.append((cx, cy))

                # Draw landmark connections (hand skeleton)
                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
                    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
                    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
                    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
                    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
                    (5, 9), (9, 13), (13, 17),             # Palm
                ]
                for start, end in connections:
                    if start < len(landmark_points) and end < len(landmark_points):
                        cv2.line(img, landmark_points[start], landmark_points[end],
                                 (255, 0, 255), 2)

                # Draw circles on each landmark
                for cx, cy in landmark_points:
                    cv2.circle(img, (cx, cy), 3, (0, 0, 255), cv2.FILLED)

        return img

    def findPosition(self, img, draw=True):
        """Get list of landmark positions: [id, x, y] for the first detected hand."""
        self.lmlist = []

        if self.results and self.results.hand_landmarks:
            hand_landmarks = self.results.hand_landmarks[0]  # First hand only
            h, w, c = img.shape
            for id, lm in enumerate(hand_landmarks):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return self.lmlist

    def findWorldPosition(self):
        """Get 3D world landmarks in METERS for the first detected hand.
        Returns list of [id, x, y, z] — real-world coordinates independent
        of camera distance. Returns empty list if no hand detected."""
        self.world_lmlist = []

        if self.results and self.results.hand_world_landmarks:
            world_landmarks = self.results.hand_world_landmarks[0]
            for id, lm in enumerate(world_landmarks):
                # x, y, z are in meters
                self.world_lmlist.append([id, lm.x, lm.y, lm.z])

        return self.world_lmlist

    def FingerUp(self):
        """Determine which fingers are up. Returns list of 5 values (0 or 1)."""
        tipIds = [4, 8, 12, 16, 20]
        fingers = []
        if len(self.lmlist) < 21:
            return fingers
        # Thumb
        if self.lmlist[tipIds[0]][1] < self.lmlist[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 Fingers
        for id in range(1, 5):
            if self.lmlist[tipIds[id]][2] < self.lmlist[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers


def main():
    pTime = 0
    cTime = 0

    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Ignoring empty frame")
            continue
        img = detector.findHands(img)
        lmlist = detector.findPosition(img)
        if len(lmlist) != 0:
            print(lmlist[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
