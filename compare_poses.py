#!/usr/bin/env python3
"""
compare_poses.py

Automatic choreography detection + DTW alignment + weighted exponential scoring
for pose CSVs.

Usage:
 - Put your reference CSV at "poses/ref.csv"
 - Put your test CSV at "poses/test.csv"
 - Run inside your virtualenv:
     python compare_poses.py

CSV format expected (per row = frame):
 - first column may be frame index (it will be skipped)
 - remaining columns are flattened keypoint coordinates per frame:
     either x,y pairs repeated (x0,y0,x1,y1,...)
     or x,y,z triplets (x0,y0,z0,x1,y1,z1,...). Only x,y are used.
"""

import csv
import numpy as np
import os

# ------------- USER-TUNABLE PARAMETERS -------------
ALPHA = 5.0            # sensitivity for exponential scoring (higher = stricter)
MIN_SEGMENT_FRAMES = 5 # minimum detected choreo length in frames (fallback to whole if not)
SMOOTH_WINDOW = 5      # smoothing window for motion energy
ENERGY_THRESH_FACTOR = 0.20  # fraction of max energy to consider active (0.2 = 20%)
# --------------------------------------------------


def load_pose_csv(filename):
    """
    Loads CSV and returns numpy array of shape (num_frames, num_values).
    Skips the first column (frame index) if present.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    rows = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        # Peek first row to see number of columns and header style
        header = next(reader)
        # If header contains non-numeric strings, skip it as header and continue
        try:
            # try to parse the first header row as numbers (if possible)
            [float(x) for x in header[1:]]  # attempt skip first column
            rows.append(header)
        except Exception:
            # header was non-numeric -> skip it, read rest normally
            pass
        for r in reader:
            # skip empty rows
            if not r:
                continue
            # if there is a frame index column, drop it
            if len(r) > 0 and not is_float(r[0]):
                # first entry non-numeric => maybe header remained, skip
                continue
            # if first column looks like frame index (integer), drop it
            if len(r) > 1 and is_int_like(r[0]):
                vals = r[1:]
            else:
                vals = r[:]
            # convert strings to floats and append
            rows.append([float(x) for x in vals])
    arr = np.array(rows, dtype=np.float32)
    return arr


def is_int_like(s):
    try:
        int(float(s))
        return True
    except Exception:
        return False


def is_float(s):
    try:
        float(s)
        return True
    except Exception:
        return False


def frame_to_keypoints(frame_row):
    """
    Convert a flattened 1D array (frame_row) to Nx2 array of keypoints (x,y).
    If input has triplets (x,y,z) we ignore z.
    Returns (K,2)
    """
    L = len(frame_row)
    # if divisible by 3, assume (x,y,z) triplets
    if L % 3 == 0:
        K = L // 3
        arr = np.array(frame_row, dtype=np.float32).reshape(K, 3)
        return arr[:, :2].copy()
    elif L % 2 == 0:
        K = L // 2
        arr = np.array(frame_row, dtype=np.float32).reshape(K, 2)
        return arr.copy()
    else:
        raise ValueError("Frame length not divisible by 2 or 3; unexpected CSV format.")


def normalize_frame_kps(kps):
    """
    Normalize a single frame keypoints (K,2):
     - center: subtract mean of keypoints (centering)
     - scale: divide by max distance between any two keypoints (scale-invariant)
    Returns flattened vector (K*2,) normalized.
    """
    if kps.size == 0:
        return kps.flatten()
    center = np.mean(kps, axis=0)
    kps_c = kps - center
    # max pairwise distance (approx) - to avoid O(K^2) do max norm from center
    max_dist = np.max(np.linalg.norm(kps_c, axis=1))
    if max_dist <= 1e-8:
        max_dist = 1.0
    kps_c /= max_dist
    return kps_c.flatten()


def normalize_sequence(frames):
    """
    frames: numpy array shape (F, D) where D is flattened coordinates
    returns normalized_frames shape (F, D2) using normalize_frame_kps
    """
    normed = []
    for i in range(frames.shape[0]):
        kps = frame_to_keypoints(frames[i])
        normed.append(normalize_frame_kps(kps))
    return np.vstack(normed)  # shape (F, D2)


def motion_energy_from_normed(normed_frames):
    """
    Compute motion energy per frame:
      energy[t] = sum over keypoints of ||kp_t - kp_{t-1}||
    normed_frames shape: (F, D) where D = K*2
    returns energy array shape (F,) (first frame energy = 0)
    """
    F = normed_frames.shape[0]
    energy = np.zeros(F, dtype=np.float32)
    if F < 2:
        return energy
    diffs = np.abs(normed_frames[1:] - normed_frames[:-1])  # (F-1, D)
    # compute Euclidean per keypoint: reshape to (F-1,K,2)
    D = diffs.shape[1]
    if D % 2 != 0:
        raise ValueError("Normalized frame length not divisible by 2")
    K = D // 2
    diffs_kp = diffs.reshape(-1, K, 2)
    per_kp = np.linalg.norm(diffs_kp, axis=2)  # (F-1, K)
    energy[1:] = np.sum(per_kp, axis=1)
    return energy


def smooth(x, w):
    if w <= 1:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode='same')


def detect_active_segment(energy, window=SMOOTH_WINDOW, factor=ENERGY_THRESH_FACTOR, min_len=MIN_SEGMENT_FRAMES):
    """
    Detect the longest contiguous segment where energy > threshold.
    threshold = factor * max_energy (after smoothing).
    Returns (start_idx, end_idx) inclusive (end exclusive like Python slicing).
    If no segment found of sufficient length, returns (0, len(energy)).
    """
    if len(energy) == 0:
        return 0, 0
    s = smooth(energy, window)
    mx = np.max(s)
    if mx <= 0:
        return 0, len(energy)
    thresh = max(np.mean(s) * 0.2, factor * mx)  # combine mean and max heuristics
    mask = s > thresh
    # find contiguous True segments
    best_start, best_end, best_len = 0, len(energy), 0
    cur_start = None
    for i, v in enumerate(mask):
        if v and cur_start is None:
            cur_start = i
        if not v and cur_start is not None:
            cur_len = i - cur_start
            if cur_len > best_len:
                best_len = cur_len
                best_start, best_end = cur_start, i
            cur_start = None
    # close if ended in a segment
    if cur_start is not None:
        cur_len = len(mask) - cur_start
        if cur_len > best_len:
            best_len = cur_len
            best_start, best_end = cur_start, len(mask)
    if best_len >= min_len:
        return best_start, best_end
    # fallback: expand around max energy index to min_len
    idx_max = int(np.argmax(s))
    half = min_len // 2
    start = max(0, idx_max - half)
    end = min(len(energy), start + min_len)
    return start, end


def dtw_path(cost):
    """
    Compute DTW cumulative cost and return optimal path as list of (i,j) pairs.
    cost: 2D array (N_ref, N_test) of pairwise distances.
    Returns path (list of (i,j)) from start->end (aligned indices).
    """
    N, M = cost.shape
    D = np.full((N + 1, M + 1), np.inf, dtype=np.float64)
    D[0, 0] = 0.0
    # dynamic programming
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            choices = (D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
            D[i, j] = cost[i - 1, j - 1] + min(choices)
    # backtrack to get path
    i, j = N, M
    path = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        # choose predecessor
        prevs = [(i - 1, j), (i, j - 1), (i - 1, j - 1)]
        vals = [D[p] for p in prevs]
        argmin = int(np.argmin(vals))
        if argmin == 0:
            i -= 1
        elif argmin == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
    path.reverse()
    return path


def align_and_score(ref_norm, test_norm, keypoint_weights=None, alpha=ALPHA):
    """
    ref_norm: (F_ref, D) normalized frames
    test_norm: (F_test, D) normalized frames
    keypoint_weights: array length K (per keypoint) or None -> equal weights
    Returns (score_0_1, details_dict)
    """
    # Prepare pairwise cost between frames (use flattened L2 distance)
    # cost[i,j] = Euclidean distance between ref_norm[i] and test_norm[j]
    N = ref_norm.shape[0]
    M = test_norm.shape[0]
    # quick edge cases
    if N == 0 or M == 0:
        return 0.0, {"reason": "empty sequence"}

    # compute cost matrix efficiently
    # squared distances: (a-b)^2 = a^2 + b^2 - 2ab
    A2 = np.sum(ref_norm ** 2, axis=1).reshape(N, 1)
    B2 = np.sum(test_norm ** 2, axis=1).reshape(1, M)
    AB = ref_norm.dot(test_norm.T)
    cost = np.sqrt(np.maximum(0.0, A2 + B2 - 2.0 * AB))  # shape (N,M)

    # compute DTW path
    path = dtw_path(cost)

    # per-pair scoring using keypoint distances
    # determine number of keypoints K from D
    D = ref_norm.shape[1]
    if D % 2 != 0:
        raise ValueError("Frame dimensionality not divisible by 2")
    K = D // 2

    # default per-keypoint weights
    if keypoint_weights is None:
        kp_w = np.ones(K, dtype=np.float32)
    else:
        if len(keypoint_weights) != K:
            # if user passed flattened weights, try to reduce
            kp_w = np.array(keypoint_weights, dtype=np.float32)
            if kp_w.size == 2 * K:
                kp_w = kp_w.reshape(K, 2).mean(axis=1)
            else:
                # fallback to ones
                kp_w = np.ones(K, dtype=np.float32)
        else:
            kp_w = np.array(keypoint_weights, dtype=np.float32)

    frame_scores = []
    for (i, j) in path:
        f_ref = ref_norm[i].reshape(K, 2)
        f_test = test_norm[j].reshape(K, 2)
        # Euclidean per keypoint
        dists = np.linalg.norm(f_ref - f_test, axis=1)  # (K,)
        # weighted distances
        wd = dists * kp_w
        # exponential similarity per keypoint
        sim_k = np.exp(-alpha * wd)
        frame_score = np.mean(sim_k)
        frame_scores.append(frame_score)

    overall = float(np.mean(frame_scores))
    details = {
        "path_length": len(path),
        "frames_ref": N,
        "frames_test": M,
        "frame_scores": frame_scores
    }
    return overall, details


def auto_compare(ref_csv, test_csv, verbose=True):
    # load
    ref = load_pose_csv(ref_csv)
    test = load_pose_csv(test_csv)

    if ref is None or test is None:
        raise FileNotFoundError("One of the files could not be loaded.")

    # normalize per-frame
    ref_norm = normalize_sequence(ref)
    test_norm = normalize_sequence(test)

    # detect activity segments
    ref_energy = motion_energy_from_normed(ref_norm)
    test_energy = motion_energy_from_normed(test_norm)

    r_start, r_end = detect_active_segment(ref_energy)
    t_start, t_end = detect_active_segment(test_energy)

    # clip to segments
    ref_seg = ref_norm[r_start:r_end]
    test_seg = test_norm[t_start:t_end]

    if verbose:
        print(f"Reference frames: {ref.shape[0]}, detected active: {r_start}..{r_end} (len {len(ref_seg)})")
        print(f"Test frames:      {test.shape[0]}, detected active: {t_start}..{t_end} (len {len(test_seg)})")

    # if detected segments are too short, fallback to whole normalized sequences
    if len(ref_seg) < MIN_SEGMENT_FRAMES or len(test_seg) < MIN_SEGMENT_FRAMES:
        if verbose:
            print("Detected segments too short — falling back to whole sequences.")
        ref_seg = ref_norm
        test_seg = test_norm

    # align using DTW + score
    score01, details = align_and_score(ref_seg, test_seg, keypoint_weights=None, alpha=ALPHA)
    score100 = score01 * 100.0

    if verbose:
        print(f"Aligned path length: {details.get('path_length')}, score (0-100): {score100:.2f}")
    return score100, details, (r_start, r_end), (t_start, t_end)


def main():
    ref_csv = os.path.join("poses", "ref.csv")
    test_csv = os.path.join("poses", "test.csv")

    if not os.path.exists(ref_csv) or not os.path.exists(test_csv):
        print("Place your ref/test CSVs at poses/ref.csv and poses/test.csv")
        return

    score, details, ref_seg, test_seg = auto_compare(ref_csv, test_csv, verbose=True)
    print(f"\nFinal MoveMatch score: {score:.2f} / 100")


if __name__ == "__main__":
    main()
