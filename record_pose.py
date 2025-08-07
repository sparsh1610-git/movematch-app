import cv2
import mediapipe as mp
import csv
import time
import os

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize video capture and MediaPipe Pose
cap = cv2.VideoCapture(0)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Create a filename with timestamp
timestamp = time.strftime("%Y%m%d-%H%M%S")
filename = f"pose_data_{timestamp}.csv"

# Make sure output directory exists
os.makedirs("poses", exist_ok=True)
filepath = os.path.join("poses", filename)

with open(filepath, mode='w', newline='') as f:
    csv_writer = csv.writer(f)

    # Write header: frame, 33 landmarks * (x, y, z, visibility)
    headers = [f"{i}_{axis}" for i in range(33) for axis in ["x", "y", "z", "visibility"]]
    csv_writer.writerow(["frame"] + headers)

    frame_num = 0
    print("🎥 Recording pose data. Press 'q' to stop.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB and process with MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            # Collect pose landmark data
            row = []
            for lm in results.pose_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z, lm.visibility])
            csv_writer.writerow([frame_num] + row)

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow("Pose Recording", frame)
        frame_num += 1

        # Exit recording on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"✅ Pose data saved to: {filepath}")
