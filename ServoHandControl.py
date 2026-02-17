"""
Hand Gesture Servo Motor Control (On-Screen Visualization)
===========================================================
Uses thumb and index finger distance to control a virtual servo motor (0-180°).
The servo position is visualized on screen as a gauge with a rotating needle.
Shows real-world distance estimates: camera-to-hand and finger-to-finger.
No real hardware required — purely visual simulation.
"""

import math
import time

import cv2
import numpy as np
import HandTrackingModule as htm

################################################################
webcam_height = 640
webcam_width = 480
# Servo range
SERVO_MIN_ANGLE = 0
SERVO_MAX_ANGLE = 180
# Normalized gap range (finger_gap / palm_size)
# Depth-independent: same gesture = same value at ANY distance
NORM_GAP_MIN = 0.15   # pinched fingers → 0°
NORM_GAP_MAX = 0.85   # fully spread    → 180°
################################################################


def draw_servo_gauge(img, angle, center, radius):
    """Draw a semicircular servo gauge with a rotating needle."""
    cx, cy = center

    # --- Background arc (gray) ---
    cv2.ellipse(img, (cx, cy), (radius, radius), 180, 0, 180, (60, 60, 60), 3)

    # --- Colored arc up to current angle ---
    # Map angle 0-180 to arc sweep: 0° (left) → 180° (right)
    if angle > 0:
        # Green-to-Red gradient effect via color interpolation
        ratio = angle / 180.0
        color = (
            int(0 + ratio * 0),       # B
            int(255 * (1 - ratio)),    # G
            int(255 * ratio),          # R
        )
        cv2.ellipse(img, (cx, cy), (radius, radius), 180, 0, int(angle), color, 6)

    # --- Tick marks at 0°, 45°, 90°, 135°, 180° ---
    for tick_angle in range(0, 181, 45):
        rad = math.radians(180 + tick_angle)
        x_outer = int(cx + (radius + 10) * math.cos(rad))
        y_outer = int(cy + (radius + 10) * math.sin(rad))
        x_inner = int(cx + (radius - 10) * math.cos(rad))
        y_inner = int(cy + (radius - 10) * math.sin(rad))
        cv2.line(img, (x_inner, y_inner), (x_outer, y_outer), (200, 200, 200), 2)
        # Label
        x_label = int(cx + (radius + 25) * math.cos(rad)) - 10
        y_label = int(cy + (radius + 25) * math.sin(rad)) + 5
        cv2.putText(img, f"{tick_angle}", (x_label, y_label),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # --- Needle ---
    needle_rad = math.radians(180 + angle)
    needle_x = int(cx + (radius - 20) * math.cos(needle_rad))
    needle_y = int(cy + (radius - 20) * math.sin(needle_rad))
    cv2.line(img, (cx, cy), (needle_x, needle_y), (0, 255, 255), 3)
    cv2.circle(img, (cx, cy), 8, (0, 255, 255), cv2.FILLED)

    # --- Angle text below gauge ---
    cv2.putText(img, f"{int(angle)} deg", (cx - 35, cy + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def draw_angle_bar(img, angle):
    """Draw a vertical bar indicator for the servo angle (like a progress bar)."""
    bar_x = 50
    bar_top = 150
    bar_bot = 400
    bar_w = 35

    # Map angle to bar fill level
    fill_y = int(np.interp(angle, [0, 180], [bar_bot, bar_top]))

    # Background bar (dark)
    cv2.rectangle(img, (bar_x, bar_top), (bar_x + bar_w, bar_bot), (60, 60, 60), cv2.FILLED)
    # Filled portion (green to red gradient)
    ratio = angle / 180.0
    color = (0, int(255 * (1 - ratio)), int(255 * ratio))
    cv2.rectangle(img, (bar_x, fill_y), (bar_x + bar_w, bar_bot), color, cv2.FILLED)
    # Border
    cv2.rectangle(img, (bar_x, bar_top), (bar_x + bar_w, bar_bot), (255, 255, 255), 2)
    # Percentage text
    cv2.putText(img, f"{int(angle)} deg", (bar_x - 5, bar_bot + 30),
                cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 2)


def draw_distance_info(img, finger_cm, servo_angle):
    """Draw an info panel showing gesture measurements."""
    h, w = img.shape[:2]
    panel_x = w - 280
    panel_y = h - 120
    panel_w = 270
    panel_h = 110

    # Semi-transparent dark background
    overlay = img.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h),
                  (30, 30, 30), cv2.FILLED)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    # Border
    cv2.rectangle(img, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h),
                  (0, 255, 255), 1)

    # Title
    cv2.putText(img, "SERVO INFO", (panel_x + 10, panel_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    # Finger gap in cm
    cv2.putText(img, f"Finger Gap: {finger_cm:.1f} cm",
                (panel_x + 10, panel_y + 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

    # Servo angle
    cv2.putText(img, f"Servo     : {servo_angle} deg",
                (panel_x + 10, panel_y + 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 1)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, webcam_width)
    cap.set(4, webcam_height)

    detector = htm.handDetector(detectionCon=0.7)
    pTime = 0
    servo_angle = 0
    normalized_gap = 0.0
    finger_cm = 0.0

    # Gauge position (top-right area of frame)
    gauge_center = (550, 120)
    gauge_radius = 80

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty frame")
            continue

        image = detector.findHands(image)
        lmList = detector.findPosition(image, draw=False)

        if len(lmList) != 0:
            # ---- Thumb tip (4) and Index finger tip (8) ----
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]

            # ---- Wrist (0) and Middle finger MCP (9) as palm reference ----
            x0, y0 = lmList[0][1], lmList[0][2]
            x9, y9 = lmList[9][1], lmList[9][2]

            finger_gap = math.hypot(x2 - x1, y2 - y1)
            palm_size = math.hypot(x9 - x0, y9 - y0)

            # ---- Normalized gap for SERVO (depth independent) ----
            if palm_size > 0:
                normalized_gap = finger_gap / palm_size
            normalized_gap = float(np.clip(normalized_gap, NORM_GAP_MIN, NORM_GAP_MAX))

            # ---- World landmarks for CM display (depth independent) ----
            worldList = detector.findWorldPosition()
            finger_cm = 0.0
            if len(worldList) >= 21:
                tx, ty, tz = worldList[4][1], worldList[4][2], worldList[4][3]
                ix, iy, iz = worldList[8][1], worldList[8][2], worldList[8][3]
                finger_cm = math.sqrt(
                    (ix - tx)**2 + (iy - ty)**2 + (iz - tz)**2
                ) * 100.0  # meters → cm

            # Midpoint between the two fingers
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Draw finger points and connecting line
            cv2.circle(image, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(image, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(image, (cx, cy), 10, (0, 0, 255), cv2.FILLED)

            # Map normalized gap → servo angle (DEPTH INDEPENDENT)
            servo_angle = int(np.interp(normalized_gap,
                                        [NORM_GAP_MIN, NORM_GAP_MAX],
                                        [SERVO_MIN_ANGLE, SERVO_MAX_ANGLE]))

            # Show finger distance in cm on the connecting line
            cv2.putText(image, f"{finger_cm:.1f} cm",
                        (cx + 15, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

            # Visual feedback when fingers are pinched together
            if normalized_gap <= NORM_GAP_MIN:
                cv2.circle(image, (cx, cy), 10, (255, 255, 255), cv2.FILLED)

        # --- Draw the servo gauge on the frame ---
        draw_servo_gauge(image, servo_angle, gauge_center, gauge_radius)

        # --- Draw the vertical angle bar ---
        draw_angle_bar(image, servo_angle)

        # --- Draw distance info panel ---
        draw_distance_info(image, finger_cm, servo_angle)

        # --- Title ---
        cv2.putText(image, "SERVO CONTROL", (400, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # --- FPS ---
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        cv2.putText(image, f"FPS: {int(fps)}", (20, 70),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        cv2.imshow("Servo Control by Hand Gesture", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
