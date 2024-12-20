import os
import cv2
import numpy as np
import json
import csv
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.interpolate import interp1d


dir_path = os.path.dirname(os.path.realpath(__file__))
os.environ['PATH'] += os.pathsep + os.path.join(dir_path, 'bin')

import pyopenpose as op

#########################################
#   1) Extract keypoints (with interpolation)
#########################################
def extract_keypoints_with_full_interpolation(video_path, model_folder="./models/", number_people_max=1):
    """
    Extract keypoints using OpenPose and handle interpolation for missing frames.
    Returns a numpy array of shape=(frames, num_joints, 2).
    """
 
    params = {
        "model_folder": model_folder,
        "number_people_max": number_people_max
    }
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    cap = cv2.VideoCapture(video_path)
    keypoints_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        datum = op.Datum()
        datum.cvInputData = frame
        datums_vec = op.VectorDatum()
        datums_vec.append(datum)
        success = opWrapper.emplaceAndPop(datums_vec)
        if not success:
            keypoints_list.append(None)
            continue

        pose_keypoints = datums_vec[0].poseKeypoints
        if pose_keypoints is not None and pose_keypoints.shape[0] > 0:
            keypoints_list.append(pose_keypoints[0][:, :2]) 
        else:
            keypoints_list.append(None)

    cap.release()
    opWrapper.stop()

    if len(keypoints_list) == 0:
        return np.zeros((0, 0, 2), dtype=np.float32)

    max_joints = max(kp.shape[0] for kp in keypoints_list if kp is not None)
    all_kp = []
    for kp in keypoints_list:
        if kp is None:
            all_kp.append(np.full((max_joints, 2), np.nan, dtype=np.float32))
        else:
            padded_kp = np.vstack([kp, np.full((max_joints - kp.shape[0], 2), np.nan)]) if kp.shape[0] < max_joints else kp
            all_kp.append(padded_kp)

    all_kp = np.array(all_kp, dtype=np.float32)

    for joint_idx in range(max_joints):
        coords = all_kp[:, joint_idx, :]
        valid_mask = ~np.isnan(coords[:, 0])
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > 0:
            coords[:valid_indices[0]] = coords[valid_indices[0]]
            coords[valid_indices[-1] + 1:] = coords[valid_indices[-1]]
            for i in range(len(valid_indices) - 1):
                start, end = valid_indices[i], valid_indices[i + 1]
                coords[start + 1:end] = np.linspace(coords[start], coords[end], end - start, endpoint=False)[1:]
        all_kp[:, joint_idx, :] = coords

    return np.nan_to_num(all_kp, nan=0.0)

#########################################
#   2) Calculate skeleton angles
#########################################
def calculate_angles(keypoints):
    """
    Calculate skeleton angle features based on keypoints.
    Input:
        keypoints: numpy array of shape=(num_joints, 2).
    Output:
        angles: angles between each pair of keypoints, in radians.
    """
    connections = [
        (1, 2), (2, 3),  # Head to torso
        (2, 5), (5, 6), (6, 7),  # Left arm
        (2, 8), (8, 9), (9, 10),  # Right arm
        (2, 12), (12, 13), (13, 14),  # Left leg
        (2, 15), (15, 16), (16, 17),  # Right leg
    ]
    angles = []
    for start, end in connections:
        if not np.any(keypoints[start]) or not np.any(keypoints[end]):
            angles.append(np.nan)
            continue
        vector = keypoints[end] - keypoints[start]
        angle = np.arctan2(vector[1], vector[0])  # Calculate angle
        angles.append(angle)

    angles = np.nan_to_num(angles, nan=0.0, posinf=0.0, neginf=0.0)
    return np.array(angles)

#########################################
#   3) Compute angle differences & DTW
#########################################
def compute_angle_offset(student_kp, teacher_kp, use_dtw=False, align_method='truncate'):
    """
    Compute the angle differences between student and teacher, with DTW support.
    """
    if student_kp.shape[0] != teacher_kp.shape[0]:
        print(f"Aligning frame counts: student_kp={student_kp.shape[0]}, teacher_kp={teacher_kp.shape[0]}")
        if align_method == 'truncate':
            # Truncate to the shortest frame count
            min_frames = min(student_kp.shape[0], teacher_kp.shape[0])
            student_kp = student_kp[:min_frames]
            teacher_kp = teacher_kp[:min_frames]
        elif align_method == 'interpolate':
            # Align to the same frame count using interpolation
            target_frames = max(student_kp.shape[0], teacher_kp.shape[0])
            student_kp = interpolate_keypoints(student_kp, target_frames)
            teacher_kp = interpolate_keypoints(teacher_kp, target_frames)
        else:
            raise ValueError(f"Unsupported align_method: {align_method}")

    frames_count = student_kp.shape[0]

    # Calculate angle features
    student_angles = [calculate_angles(student_kp[f]) for f in range(frames_count)]
    teacher_angles = [calculate_angles(teacher_kp[f]) for f in range(frames_count)]

    # Check for NaN or Inf data
    for i, (s_angles, t_angles) in enumerate(zip(student_angles, teacher_angles)):
        if np.any(np.isnan(s_angles)) or np.any(np.isnan(t_angles)):
            print(f"Frame {i}: Contains NaN in angle data.")
        if np.any(np.isinf(s_angles)) or np.any(np.isinf(t_angles)):
            print(f"Frame {i}: Contains Inf in angle data.")

     # Calculate per-frame or use DTW
    if not use_dtw:
        offset_table = []
        for s_angles, t_angles in zip(student_angles, teacher_angles):
            valid_mask = ~np.isnan(s_angles) & ~np.isnan(t_angles)
            if np.any(valid_mask):
                angle_diff = np.abs(s_angles[valid_mask] - t_angles[valid_mask])
                angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)  # Account for angle periodicity
                offset_table.append(angle_diff.tolist())
            else:
                offset_table.append([])
        return offset_table

    # Dynamic Time Warping (DTW)
    student_seq = [angles.flatten() for angles in student_angles]
    teacher_seq = [angles.flatten() for angles in teacher_angles]

    # Filter out invalid values
    student_seq = [np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0) for seq in student_seq]
    teacher_seq = [np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0) for seq in teacher_seq]

    _, path = fastdtw(student_seq, teacher_seq, dist=euclidean)
    offset_table = []
    for s_idx, t_idx in path:
        valid_mask = ~np.isnan(student_angles[s_idx]) & ~np.isnan(teacher_angles[t_idx])
        if np.any(valid_mask):
            angle_diff = np.abs(student_angles[s_idx][valid_mask] - teacher_angles[t_idx][valid_mask])
            angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)  
            offset_table.append(angle_diff.tolist())
        else:
            offset_table.append([])
    return offset_table

def interpolate_keypoints(keypoints, target_frames):
    """
    Interpolate keypoint data to the specified number of frames.
    :param keypoints: Original keypoint data (frames, num_joints, 2)
    :param target_frames: Target number of frames
    :return: Interpolated keypoint data
    """
    num_frames, num_joints, _ = keypoints.shape
    x = np.arange(num_frames)
    x_new = np.linspace(0, num_frames - 1, target_frames)
    interp_func = interp1d(x, keypoints, axis=0, kind='linear', fill_value="extrapolate")
    return interp_func(x_new)

#########################################
#   4) Save angle offset table
#########################################
def save_offset_table(offset_table, output_path, save_format='csv'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if save_format == 'csv':
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["frame_index"] + [f"angle_{i}" for i in range(len(offset_table[0]))])
            for i, row in enumerate(offset_table):
                writer.writerow([i] + row)
    elif save_format == 'json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([{"frame_index": i, "angles": row} for i, row in enumerate(offset_table)], f, indent=2)


def save_keypoints_to_csv(keypoints, output_path):
    """
    Save keypoints to a CSV file.
    :param keypoints: Keypoint array (frames, num_joints, 2)
    :param output_path: Output CSV file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for frame in keypoints:
            if frame is None:
                writer.writerow([])   # Write an empty line if the frame is None
            else:
                flat_frame = frame.flatten()  # Flatten to a 1D list
                writer.writerow(flat_frame)


#######################################
#   5) Main function
#########################################
if __name__ == "__main__":
    output_dir = "table"
    os.makedirs(output_dir, exist_ok=True)

    student_kp = extract_keypoints_with_full_interpolation("videos/lip_me_resampled_800.mp4")
    teacher_kp = extract_keypoints_with_full_interpolation("videos/lip_teacher_resampled_800.mp4")

    print(f"Student keypoints shape: {student_kp.shape}")
    print(f"Teacher keypoints shape: {teacher_kp.shape}")

     # 保存关键点数据
    save_keypoints_to_csv(student_kp, os.path.join(output_dir, "lip_me_keypoints.csv"))
    save_keypoints_to_csv(teacher_kp, os.path.join(output_dir, "lip_teacher_keypoints.csv"))


    offset_table = compute_angle_offset(student_kp, teacher_kp, use_dtw=True)
    save_offset_table(offset_table, os.path.join(output_dir, "lip_angle_offset_table.csv"), save_format='csv')

    flattened = [angle for frame in offset_table for angle in frame if not np.isnan(angle)]
    if flattened:
        mean_angle_diff = np.mean(flattened)
        similarity_score = 1.0 / (1.0 + mean_angle_diff)  # Closer to 1 indicates higher similarity
        print(f"Mean angle difference: {mean_angle_diff:.4f}, similarity_score={similarity_score:.4f}")
    else:
        print("No valid offsets, cannot compute similarity.")
