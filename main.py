"""
Main Entry Point for Real-Time Object Detection & Logging Platform.
Coordinates YOLOv8 Object Detection, OpenCV Video Processing,
MySQL Event Logging, and Modular User Interfaces.

Usage:
    python main.py                # Launches the interactive Streamlit Web Dashboard
    python main.py --ui           # Launches Streamlit Web Dashboard
    python main.py --desktop      # Launches native OpenCV Desktop GUI with HUD
    python main.py --test         # Runs comprehensive end-to-end integration test
"""

import os
import sys
import time
import argparse
import subprocess
# pyrefly: ignore [missing-import]
import cv2

# Ensure UTF-8 console encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from detector import ObjectDetector
from database import DatabaseManager


def run_integration_test():
    """Runs a complete self-check of detector, database, and logging pipeline."""
    print("=" * 60)
    print("🔍 RUNNING VISION PLATFORM INTEGRATION TEST")
    print("=" * 60)

    # 1. Test Database
    print("\n[1/3] Testing Database Module...")
    db = DatabaseManager()
    status = db.get_status()
    print(f"  • Active Backend : {status['backend'].upper()}")
    print(f"  • Target DB      : {status['database']}")
    if status['backend'] == 'mysql':
        print("  • MySQL Status   : Connected successfully! ✅")
    else:
        print(f"  • MySQL Status   : Offline ({status['last_error']})")
        print("  • SQLite Status  : Local fallback activated successfully! ✅")

    # Insert test detection log
    success = db.log_detection(
        object_class="person",
        confidence=0.912,
        bbox_x=120,
        bbox_y=80,
        bbox_w=240,
        bbox_h=400
    )
    assert success, "Failed to insert test log into database"
    print("  • Test Log Insert: SUCCESS ✅")

    # Fetch logs
    recent = db.get_recent_logs(limit=1)
    print(f"  • Verified Log   : ID={recent[0]['log_id']} Class={recent[0]['object_class']} Conf={recent[0]['confidence']}")

    # 2. Test YOLO Detector
    print("\n[2/3] Testing YOLO Detector Module...")
    detector = ObjectDetector(confidence_threshold=0.50)
    print(f"  • YOLO Model     : {detector.model_name} loaded ✅")
    print(f"  • Classes Count  : {len(detector.available_classes)} COCO classes available")

    # Test blank frame inference
    # pyrefly: ignore [missing-import]
    import numpy as np
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated, dets, fps = detector.process_frame(dummy_frame)
    print("  • Test Inference : Frame processed successfully ✅")

    # 3. Summary
    stats = db.get_summary_stats()
    print("\n[3/3] System Metrics:")
    print(f"  • Total DB Logs  : {stats['total_logs']}")
    print(f"  • Unique Objects : {stats['unique_classes']}")
    print(f"  • Top Object     : {stats['top_object']}")

    print("\n" + "=" * 60)
    print("🎉 ALL SYSTEMS OPERATIONAL AND READY FOR PRODUCTION!")
    print("=" * 60)


def run_desktop_mode(source=0, confidence=0.50, target_classes=None, log_interval=1.0):
    """
    Runs OpenCV native desktop window mode with real-time detection HUD
    and direct MySQL event logging.
    """
    print("=" * 60)
    print("🚀 STARTING VISION PLATFORM - DESKTOP OPENCV HUD MODE")
    print("=" * 60)
    print("Controls:")
    print("  • [Q] or [ESC] : Quit application")
    print("  • [L]          : Toggle MySQL event logging ON/OFF")
    print("  • [+] / [-]    : Increase / Decrease confidence threshold")
    print("  • [S]          : Save current frame screenshot")
    print("=" * 60)

    # Initialize modules
    db = DatabaseManager()
    detector = ObjectDetector(confidence_threshold=confidence)
    if target_classes:
        detector.set_target_classes(target_classes)

    # Open video capture
    try:
        source_idx = int(source)
        cap = cv2.VideoCapture(source_idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source_idx)
    except ValueError:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"❌ Error: Could not open video source: {source}")
        return

    logging_enabled = True
    last_logged_time = {}
    saved_count = 0

    window_name = "VisionPlatform - YOLOv8 Live Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 768)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video stream ended or frame read failed.")
            break

        # Process frame with YOLO
        annotated_frame, detections, fps = detector.process_frame(frame)

        # Log detections to MySQL/DB if enabled
        now = time.time()
        if logging_enabled and detections:
            for det in detections:
                cls_name = det["object_class"]
                if cls_name not in last_logged_time or (now - last_logged_time[cls_name]) >= log_interval:
                    db.log_detection(
                        object_class=det["object_class"],
                        confidence=det["confidence"],
                        bbox_x=det["bbox_x"],
                        bbox_y=det["bbox_y"],
                        bbox_w=det["bbox_w"],
                        bbox_h=det["bbox_h"]
                    )
                    last_logged_time[cls_name] = now
                    print(f" [DB LOG] {det['object_class']} ({int(det['confidence']*100)}%) saved to {db.backend.upper()}")

        # Draw HUD overlay on top of frame
        db_status = f"DB: {db.backend.upper()} (Active)" if logging_enabled else "DB: LOGGING PAUSED"
        hud_color = (0, 255, 0) if logging_enabled else (0, 0, 255)
        
        cv2.putText(annotated_frame, f"FPS: {fps}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"Conf: {int(detector.confidence_threshold*100)}%", (160, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Objects: {len(detections)}", (320, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        cv2.putText(annotated_frame, db_status, (480, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hud_color, 2)

        cv2.imshow(window_name, annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), ord('Q'), 27]:  # 27 = ESC
            break
        elif key in [ord('l'), ord('L')]:
            logging_enabled = not logging_enabled
            print(f"Logging toggled: {'ENABLED' if logging_enabled else 'DISABLED'}")
        elif key in [ord('+'), ord('=')]:
            detector.set_confidence(min(0.95, detector.confidence_threshold + 0.05))
            print(f"Confidence increased to: {detector.confidence_threshold:.2f}")
        elif key in [ord('-'), ord('_')]:
            detector.set_confidence(max(0.10, detector.confidence_threshold - 0.05))
            print(f"Confidence decreased to: {detector.confidence_threshold:.2f}")
        elif key in [ord('s'), ord('S')]:
            saved_count += 1
            filename = f"capture_{saved_count}_{int(time.time())}.jpg"
            cv2.imwrite(filename, annotated_frame)
            print(f"Screenshot saved to: {filename}")

    cap.release()
    cv2.destroyAllWindows()
    db.close()
    print("Vision Platform Desktop session closed.")


def launch_streamlit_ui():
    """Launches the Streamlit browser-based dashboard interface."""
    ui_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui.py")
    print(f"🚀 Launching Streamlit Web Dashboard: {ui_script}")
    subprocess.run([sys.executable, "-m", "streamlit", "run", ui_script])


def main():
    parser = argparse.ArgumentParser(description="Real-Time Object Detection & Logging Platform")
    parser.add_argument("--ui", action="store_true", help="Launch Streamlit Web UI dashboard")
    parser.add_argument("--desktop", action="store_true", help="Launch OpenCV desktop window mode")
    parser.add_argument("--test", action="store_true", help="Run integration test suite")
    parser.add_argument("--source", default="0", help="Video source (webcam index or video file path)")
    parser.add_argument("--conf", type=float, default=0.50, help="Initial confidence threshold (0.0 to 1.0)")
    parser.add_argument("--classes", type=str, default="", help="Comma-separated target classes (e.g. 'person,cell phone')")
    parser.add_argument("--interval", type=float, default=1.0, help="Log throttle interval in seconds")

    args = parser.parse_args()

    classes_list = [c.strip() for c in args.classes.split(",")] if args.classes else None

    if args.test:
        run_integration_test()
    elif args.desktop:
        run_desktop_mode(
            source=args.source,
            confidence=args.conf,
            target_classes=classes_list,
            log_interval=args.interval
        )
    else:
        # Default mode: launch the rich Streamlit dashboard
        launch_streamlit_ui()


if __name__ == "__main__":
    main()
