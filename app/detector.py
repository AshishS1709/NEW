import cv2
import numpy as np
from ultralytics import YOLO
import time
import signal
import sys
from datetime import datetime
 
class LivePhoneDetector:
    def __init__(self):
        print("🔄 Loading models...")
        self.model1 = YOLO('best.pt')      # Your custom model
        self.model2 = YOLO('yolov8n.pt')   # YOLOv8 model
        self.confidence_threshold = 0.35
        self.running = True
       
        # Real-time statistics
        self.frame_count = 0
        self.detection_count = 0
        self.violation_count = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
       
        print("✅ Models loaded successfully!")
       
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
 
    def signal_handler(self, sig, frame):
        print("\n🛑 Shutting down...")
        self.running = False
 
    def is_in_driving_area(self, box, frame_shape):
        """Check if detected phone is in the driving area (center region)"""
        frame_height, frame_width = frame_shape[:2]
        x1, y1, x2, y2 = map(int, box)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
       
        # Define driving area as center 60% of the frame
        return (frame_width * 0.2 < cx < frame_width * 0.8) and (frame_height * 0.15 < cy < frame_height * 0.85)
 
    def check_phone_size(self, box, frame_shape):
        """Validate phone detection based on reasonable size"""
        frame_height, frame_width = frame_shape[:2]
        x1, y1, x2, y2 = map(int, box)
        w, h = x2 - x1, y2 - y1
        box_area = w * h
        frame_area = frame_width * frame_height
       
        # Phone should be between 0.1% and 25% of frame area
        return 0.001 < (box_area / frame_area) < 0.25
 
    def detect_and_annotate(self, frame):
        """Detect phones and annotate frame in real-time"""
        # Run detection on both models
        results1 = self.model1(frame, conf=self.confidence_threshold, verbose=False)
        results2 = self.model2(frame, conf=self.confidence_threshold, verbose=False)
       
        phone_detected = False
        violation_detected = False
        annotated_frame = frame.copy()
       
        # Process results from both models
        all_results = [results1, results2]
        model_names = ["Custom Model", "YOLOv8n"]
       
        for results, model_name in zip(all_results, model_names):
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        class_id = int(box.cls.item())
                        class_name = r.names.get(class_id, "unknown")
                        conf = float(box.conf.item())
                       
                        # Look for phone/cell phone detections
                        if 'phone' in class_name.lower() or 'cell' in class_name.lower():
                            box_coords = box.xyxy[0].cpu().numpy()
                           
                            # Validate detection
                            if not self.check_phone_size(box_coords, frame.shape):
                                continue
                           
                            phone_detected = True
                            x1, y1, x2, y2 = map(int, box_coords)
                           
                            # Check if it's a violation (in driving area)
                            is_violation = self.is_in_driving_area(box_coords, frame.shape)
                           
                            if is_violation:
                                violation_detected = True
                                color = (0, 0, 255)  # Red for violation
                                thickness = 3
                                label = f"🚨 VIOLATION! {class_name} ({conf:.2f})"
                                text_color = (255, 255, 255)
                                bg_color = (0, 0, 255)
                            else:
                                color = (0, 255, 0)  # Green for normal detection
                                thickness = 2
                                label = f"📱 {class_name} ({conf:.2f})"
                                text_color = (0, 0, 0)
                                bg_color = (0, 255, 0)
                           
                            # Draw bounding box
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                           
                            # Add label with background
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            cv2.rectangle(annotated_frame,
                                        (x1, y1 - label_size[1] - 10),
                                        (x1 + label_size[0], y1),
                                        bg_color, -1)
                            cv2.putText(annotated_frame, label, (x1, y1 - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                           
                            # Add model info
                            model_text = f"Model: {model_name}"
                            cv2.putText(annotated_frame, model_text, (x1, y2 + 20),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
       
        return annotated_frame, phone_detected, violation_detected
 
    def add_live_stats_overlay(self, frame):
        """Add real-time statistics overlay"""
        height, width = frame.shape[:2]
       
        # Create semi-transparent background for stats
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
       
        # Current time
        current_time = datetime.now().strftime("%H:%M:%S")
       
        # Statistics text
        stats = [
            f"🕐 Time: {current_time}",
            f"🎬 Frames: {self.frame_count}",
            f"📱 Detections: {self.detection_count}",
            f"🚨 Violations: {self.violation_count}",
            f"🚀 FPS: {self.current_fps:.1f}"
        ]
       
        # Draw statistics
        for i, stat in enumerate(stats):
            y_pos = 30 + (i * 22)
            cv2.putText(frame, stat, (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
       
        # Add driving area indicator
        frame_height, frame_width = frame.shape[:2]
        x1 = int(frame_width * 0.2)
        y1 = int(frame_height * 0.15)
        x2 = int(frame_width * 0.8)
        y2 = int(frame_height * 0.85)
       
        # Draw driving area boundary (dotted line effect)
        for i in range(x1, x2, 20):
            cv2.line(frame, (i, y1), (i + 10, y1), (255, 255, 0), 1)
            cv2.line(frame, (i, y2), (i + 10, y2), (255, 255, 0), 1)
        for i in range(y1, y2, 20):
            cv2.line(frame, (x1, i), (x1, i + 10), (255, 255, 0), 1)
            cv2.line(frame, (x2, i), (x2, i + 10), (255, 255, 0), 1)
       
        # Add zone label
        cv2.putText(frame, "DRIVING ZONE", (x1 + 10, y1 + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
       
        return frame
 
    def calculate_fps(self):
        """Calculate real-time FPS"""
        self.fps_counter += 1
        if self.fps_counter >= 30:  # Update FPS every 30 frames
            current_time = time.time()
            elapsed = current_time - self.fps_start_time
            self.current_fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_start_time = current_time
 
    def test_camera(self, camera_index):
        """Test camera accessibility"""
        print(f"🔍 Testing camera {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
       
        if not cap.isOpened():
            cap.release()
            return False
       
        ret, frame = cap.read()
        cap.release()
       
        if not ret:
            return False
       
        print(f"✅ Camera {camera_index} is working!")
        return True
 
    def run_live_detection(self, camera_index=0):
        """Run live phone detection with real-time display"""
        print("="*60)
        print("🚀 LIVE PHONE DETECTION SYSTEM")
        print("="*60)
        print("📋 Controls:")
        print("   - Press 'q' to quit")
        print("   - Press 's' to save current frame")
        print("   - Press SPACE to pause/resume")
        print("   - Press Ctrl+C for emergency stop")
        print("="*60)
       
        # Find working camera
        working_camera = None
        for cam_idx in [camera_index, 0, 1, 2]:
            if self.test_camera(cam_idx):
                working_camera = cam_idx
                break
       
        if working_camera is None:
            print("❌ No working camera found!")
            return
       
        # Initialize camera
        cap = cv2.VideoCapture(working_camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
       
        print(f"📹 Using camera {working_camera}")
        print("🎬 Starting live detection...")
       
        paused = False
       
        try:
            while self.running:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("⚠️ Failed to read frame, retrying...")
                        continue
                   
                    self.frame_count += 1
                   
                    # Detect and annotate
                    annotated_frame, phone_detected, violation_detected = self.detect_and_annotate(frame)
                   
                    # Update counters
                    if phone_detected:
                        self.detection_count += 1
                    if violation_detected:
                        self.violation_count += 1
                        print(f"🚨 VIOLATION DETECTED! Frame #{self.frame_count}")
                   
                    # Calculate FPS
                    self.calculate_fps()
                   
                    # Add live statistics overlay
                    final_frame = self.add_live_stats_overlay(annotated_frame)
                   
                    # Display frame
                    cv2.imshow('Live Phone Detection - Press Q to quit', final_frame)
               
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("👋 Quitting...")
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"detection_frame_{timestamp}.jpg"
                    cv2.imwrite(filename, final_frame if not paused else annotated_frame) # type: ignore
                    print(f"💾 Frame saved as {filename}")
                elif key == ord(' '):  # Spacebar
                    paused = not paused
                    status = "⏸️ PAUSED" if paused else "▶️ RESUMED"
                    print(f"{status}")
       
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            print("\n📊 Final Statistics:")
            print(f"   Total Frames: {self.frame_count}")
            print(f"   Phone Detections: {self.detection_count}")
            print(f"   Violations: {self.violation_count}")
            if self.frame_count > 0:
                print(f"   Detection Rate: {(self.detection_count/self.frame_count)*100:.1f}%")
                print(f"   Violation Rate: {(self.violation_count/self.frame_count)*100:.1f}%")
            print("👋 Detection stopped!")
 
def main():
    print("🚀 Initializing Live Phone Detection System...")
   
    try:
        detector = LivePhoneDetector()
        detector.run_live_detection(camera_index=0)  # Try camera 0 first
    except Exception as e:
        print(f"❌ Failed to start detection: {e}")
        print("💡 Make sure your models (best.pt, yolov8n.pt) are in the current directory")
 
if __name__ == "__main__":
    main()