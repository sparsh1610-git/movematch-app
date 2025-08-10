import csv
import numpy as np
import os

# ====== Use this if you're using MediaPipe (33 keypoints) ======
KEYPOINT_WEIGHTS = np.ones(33)
KEYPOINT_WEIGHTS = np.repeat(KEYPOINT_WEIGHTS, 2)

# ====== OR, use this for 17-keypoint models like OpenPose ======
# keypoint_weights = np.array([
#     1.0,  # nose
#     0.8, 0.8,  # eyes
#     0.8, 0.8,  # ears
#     1.2, 1.2,  # shoulders
#     1.3, 1.3,  # elbows
#     1.5, 1.5,  # wrists
#     1.0, 1.0,  # hips
#     1.1, 1.1,  # knees
#     1.2, 1.2   # ankles
# ])
# KEYPOINT_WEIGHTS = np.repeat(keypoint_weights, 2)

def load_pose_csv(filename):
    if not os.path.exists(filename):
        print(f"❌ File not found: {filename}")
        return None
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        data = [row[1:] for row in reader]  # skip frame number
    return np.array(data, dtype=np.float32)

def calculate_pose_distance(pose1, pose2):
    if pose1.shape != pose2.shape:
        min_frames = min(pose1.shape[0], pose2.shape[0])
        pose1 = pose1[:min_frames]
        pose2 = pose2[:min_frames]

    frame_scores = []

    for f1, f2 in zip(pose1, pose2):
        dists = np.abs(f1 - f2)
        dists = dists.reshape(-1, 2)
        euclidean_dists = np.linalg.norm(dists, axis=1)
        
        # Weights are per keypoint (not per x/y)
        per_frame_weights = KEYPOINT_WEIGHTS[:len(euclidean_dists)]
        weighted = euclidean_dists * per_frame_weights

        # Exponential decay for scoring (more intuitive: lower distance → higher score)
        score = np.mean(np.exp(-weighted))
        frame_scores.append(score)

    return np.mean(frame_scores)  # overall average match score across frames

def main():
    ref_file = "poses/ref.csv"
    test_file = "poses/test.csv"

    ref_pose = load_pose_csv(ref_file)
    test_pose = load_pose_csv(test_file)

    if ref_pose is None or test_pose is None:
        return

    score = 100 * calculate_pose_distance(ref_pose, test_pose)
    print(f"✅ Dance Match Score: {score:.4f} (0 to 100 scale)")

if __name__ == "__main__":
    main()
