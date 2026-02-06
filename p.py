import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time
from datetime import datetime

class EyeTrackingStudyAssistant:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Eye landmark indices for MediaPipe Face Mesh
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        
        # Attention tracking
        self.blink_counter = 0
        self.blink_threshold = 0.21
        self.attention_scores = deque(maxlen=300)  # 10 seconds at 30fps
        self.gaze_positions = deque(maxlen=100)
        self.fixation_points = []
        
        # Cognitive state variables
        self.start_time = time.time()
        self.total_blinks = 0
        self.last_blink_time = 0
        self.engagement_level = "Unknown"
        
        # Screen dimensions for gaze mapping
        self.screen_w, self.screen_h = 1920, 1080
        
    def calculate_ear(self, eye_landmarks):
        """Calculate Eye Aspect Ratio for blink detection"""
        # Vertical distances
        v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        # Horizontal distance
        h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def get_gaze_direction(self, iris_center, eye_center):
        """Calculate gaze direction from iris position"""
        gaze_x = iris_center[0] - eye_center[0]
        gaze_y = iris_center[1] - eye_center[1]
        return gaze_x, gaze_y
    
    def estimate_attention_level(self, ear_left, ear_right, gaze_variance):
        """Estimate attention level based on eye metrics"""
        # Average EAR
        avg_ear = (ear_left + ear_right) / 2
        
        # Calculate attention score (0-100)
        # Higher EAR = more open eyes = better attention
        # Lower gaze variance = more focused = better attention
        ear_score = min(avg_ear * 250, 100)  # Normalize to 0-100
        focus_score = max(0, 100 - gaze_variance * 50)
        
        attention_score = (ear_score * 0.6 + focus_score * 0.4)
        
        return max(0, min(100, attention_score))
    
    def classify_cognitive_state(self, attention_score, blink_rate):
        """Classify cognitive state based on metrics"""
        # Blink rate analysis (normal: 15-20 per minute)
        if blink_rate < 10:
            blink_state = "High Focus"
        elif blink_rate < 25:
            blink_state = "Normal"
        else:
            blink_state = "Fatigue/Stress"
        
        # Attention level
        if attention_score > 75:
            return "Highly Engaged", blink_state
        elif attention_score > 50:
            return "Moderately Engaged", blink_state
        elif attention_score > 30:
            return "Low Engagement", blink_state
        else:
            return "Distracted", blink_state
    
    def process_frame(self, frame):
        """Process video frame for eye tracking"""
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])
            
            # Extract eye landmarks
            left_eye = landmarks[self.LEFT_EYE]
            right_eye = landmarks[self.RIGHT_EYE]
            left_iris = landmarks[self.LEFT_IRIS]
            right_iris = landmarks[self.RIGHT_IRIS]
            
            # Calculate EAR for blink detection
            ear_left = self.calculate_ear(left_eye)
            ear_right = self.calculate_ear(right_eye)
            avg_ear = (ear_left + ear_right) / 2
            
            # Blink detection
            if avg_ear < self.blink_threshold:
                self.blink_counter += 1
            else:
                if self.blink_counter >= 2:
                    self.total_blinks += 1
                    self.last_blink_time = time.time()
                self.blink_counter = 0
            
            # Calculate gaze
            left_iris_center = np.mean(left_iris, axis=0)
            right_iris_center = np.mean(right_iris, axis=0)
            left_eye_center = np.mean(left_eye, axis=0)
            right_eye_center = np.mean(right_eye, axis=0)
            
            gaze_left = self.get_gaze_direction(left_iris_center, left_eye_center)
            gaze_right = self.get_gaze_direction(right_iris_center, right_eye_center)
            avg_gaze = ((gaze_left[0] + gaze_right[0]) / 2, 
                       (gaze_left[1] + gaze_right[1]) / 2)
            
            self.gaze_positions.append(avg_gaze)
            
            # Calculate gaze variance (measure of focus)
            if len(self.gaze_positions) > 10:
                gaze_array = np.array(list(self.gaze_positions))
                gaze_variance = np.mean(np.std(gaze_array, axis=0))
            else:
                gaze_variance = 0
            
            # Calculate attention score
            attention_score = self.estimate_attention_level(ear_left, ear_right, gaze_variance)
            self.attention_scores.append(attention_score)
            
            # Calculate blink rate (blinks per minute)
            elapsed_time = time.time() - self.start_time
            blink_rate = (self.total_blinks / elapsed_time) * 60 if elapsed_time > 0 else 0
            
            # Classify cognitive state
            engagement, blink_state = self.classify_cognitive_state(attention_score, blink_rate)
            
            # Draw visualizations
            self.draw_visualizations(frame, landmarks, left_eye, right_eye, 
                                   left_iris_center, right_iris_center,
                                   attention_score, blink_rate, engagement, blink_state)
            
            return frame, attention_score, engagement
        
        return frame, 0, "No Face Detected"
    
    def draw_visualizations(self, frame, landmarks, left_eye, right_eye, 
                          left_iris_center, right_iris_center,
                          attention_score, blink_rate, engagement, blink_state):
        """Draw all visualizations on frame"""
        h, w, _ = frame.shape
        
        # Draw eyes
        for point in left_eye:
            cv2.circle(frame, tuple(point.astype(int)), 2, (0, 255, 0), -1)
        for point in right_eye:
            cv2.circle(frame, tuple(point.astype(int)), 2, (0, 255, 0), -1)
        
        # Draw iris centers
        cv2.circle(frame, tuple(left_iris_center.astype(int)), 3, (255, 0, 0), -1)
        cv2.circle(frame, tuple(right_iris_center.astype(int)), 3, (255, 0, 0), -1)
        
        # Draw attention bar
        bar_width = 300
        bar_height = 30
        bar_x, bar_y = 20, 20
        
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (50, 50, 50), -1)
        
        # Attention level bar
        fill_width = int((attention_score / 100) * bar_width)
        color = (0, 255, 0) if attention_score > 70 else \
                (0, 255, 255) if attention_score > 40 else (0, 0, 255)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                     color, -1)
        
        # Text overlay - Info panel
        cv2.rectangle(frame, (10, 60), (400, 280), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 60), (400, 280), (255, 255, 255), 2)
        
        info_texts = [
            f"Attention Score: {attention_score:.1f}%",
            f"Engagement: {engagement}",
            f"Blink State: {blink_state}",
            f"Total Blinks: {self.total_blinks}",
            f"Blink Rate: {blink_rate:.1f}/min",
            f"Session Time: {int(time.time() - self.start_time)}s"
        ]
        
        y_offset = 90
        for text in info_texts:
            cv2.putText(frame, text, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30
        
        # Draw attention graph
        if len(self.attention_scores) > 1:
            graph_h = 100
            graph_w = 300
            graph_x, graph_y = w - graph_w - 20, 20
            
            cv2.rectangle(frame, (graph_x, graph_y), 
                         (graph_x + graph_w, graph_y + graph_h), (0, 0, 0), -1)
            cv2.rectangle(frame, (graph_x, graph_y), 
                         (graph_x + graph_w, graph_y + graph_h), (255, 255, 255), 2)
            
            scores = list(self.attention_scores)
            points = []
            for i, score in enumerate(scores):
                x = graph_x + int((i / len(scores)) * graph_w)
                y = graph_y + graph_h - int((score / 100) * graph_h)
                points.append((x, y))
            
            for i in range(len(points) - 1):
                cv2.line(frame, points[i], points[i + 1], (0, 255, 0), 2)
    
    def run(self):
        """Main run loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("Eye Tracking Study Assistant Started")
        print("Press 'q' to quit, 's' to save report")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            processed_frame, attention, engagement = self.process_frame(frame)
            
            cv2.imshow('Eye Tracking Study Assistant', processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_report()
        
        cap.release()
        cv2.destroyAllWindows()
    
    def save_report(self):
        """Save attention report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"attention_report_{timestamp}.txt"
        
        avg_attention = np.mean(list(self.attention_scores)) if self.attention_scores else 0
        elapsed_time = time.time() - self.start_time
        blink_rate = (self.total_blinks / elapsed_time) * 60 if elapsed_time > 0 else 0
        
        with open(filename, 'w') as f:
            f.write("=" * 50 + "\n")
            f.write("EYE TRACKING STUDY ASSISTANT - SESSION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Session Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {int(elapsed_time)} seconds\n\n")
            f.write(f"Average Attention Score: {avg_attention:.2f}%\n")
            f.write(f"Total Blinks: {self.total_blinks}\n")
            f.write(f"Blink Rate: {blink_rate:.2f} per minute\n")
            f.write(f"Peak Attention: {max(self.attention_scores) if self.attention_scores else 0:.2f}%\n")
            f.write(f"Lowest Attention: {min(self.attention_scores) if self.attention_scores else 0:.2f}%\n")
        
        print(f"Report saved: {filename}")

if __name__ == "__main__":
    assistant = EyeTrackingStudyAssistant()
    assistant.run()