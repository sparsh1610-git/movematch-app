import csv
import numpy as np
import os

def load_pose_csv(filename):
    if not os.path.exists(filename):
        print(f"❌ File not found: {filename}")
        return None

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)  # skip header
        data = [row[1:] for row in reader]  # skip frame number
    return np.array(data, dtype=np.float32)

def compare_poses(file1, file2):
    pose1 = load_pose_csv(file1)
    pose2 = load_pose_csv(file2)

    if pose1 is None or pose2 is None:
        return

    min_frames = min(len(pose1), len(pose2))
    pose1 = pose1[:min_frames]
    pose2 = pose2[:min_frames]

    # Compute L2 distance per frame
    distances = np.linalg.norm(pose1 - pose2, axis=1)
    avg_distance = np.mean(distances)

    # Normalize score: smaller distance = better match
    score = max(0, 100 - avg_distance * 1000)
    print("\n📊 Pose Comparison Results")
    print(f"🔍 Average Frame Distance: {avg_distance:.4f}")
    print(f"✅ Similarity Score: {score:.2f} / 100")

# === USAGE EXAMPLE ===
# Make sure both reference and user pose files are inside the 'poses/' directory
reference_file = "poses/pose_data_reference.csv"
user_file = "poses/pose_data_user.csv"

compare_poses(reference_file, user_file)
