# run_worker.py
from app.detector import LivePhoneDetector

if __name__ == "__main__":
    print("🚀 Starting phone detection worker...")
    detector = LivePhoneDetector()
    detector.run_live_detection(camera_index=0)  # Or use a video file
