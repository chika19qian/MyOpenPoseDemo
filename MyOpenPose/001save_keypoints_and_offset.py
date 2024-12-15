import os
import cv2
import numpy as np
import json
import csv
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

dir_path = os.path.dirname(os.path.realpath(__file__))
os.environ['PATH'] += os.pathsep + os.path.join(dir_path, 'bin')

import pyopenpose as op

#########################################
#   1) 提取关键点（含插值）
#########################################
def extract_keypoints_with_full_interpolation(video_path, model_folder="./models/", number_people_max=1):
    """
    使用 OpenPose 提取关键点并对空帧做插值处理。
    返回 shape=(frames, num_joints, 2) 的numpy数组。
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
            keypoints_list.append(pose_keypoints[0][:, :2])  # 只取第一个人的 2D 关键点
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
#   2) 加权偏移 & DTW
#########################################
def compute_weighted_offset(student_kp, teacher_kp, weights=None, use_dtw=False):
    """
    计算学生/老师关键点偏移。
    """
    if student_kp.shape != teacher_kp.shape:
        raise ValueError(f"Shape mismatch: student_kp={student_kp.shape}, teacher_kp={teacher_kp.shape}")

    frames_count, num_joints, _ = student_kp.shape
    if weights is None:
        weights = np.ones(num_joints, dtype=np.float32)

    if not use_dtw:
        offset_table = []
        for f in range(frames_count):
            dist = np.linalg.norm(student_kp[f] - teacher_kp[f], axis=1) * weights
            offset_table.append(dist.tolist())
        return offset_table

    student_seq = [(kp * weights[:, None]).flatten() for kp in student_kp]
    teacher_seq = [(kp * weights[:, None]).flatten() for kp in teacher_kp]

    _, path = fastdtw(student_seq, teacher_seq, dist=euclidean)
    offset_table = []
    for s_idx, t_idx in path:
        dist = np.linalg.norm(student_kp[s_idx] - teacher_kp[t_idx], axis=1) * weights
        offset_table.append(dist.tolist())
    return offset_table

#########################################
#   3) 保存偏移表
#########################################
def save_offset_table(offset_table, output_path, save_format='csv'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if save_format == 'csv':
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["frame_index"] + [f"joint_{i}" for i in range(len(offset_table[0]))])
            for i, row in enumerate(offset_table):
                writer.writerow([i] + row)
    elif save_format == 'json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([{"frame_index": i, "distances": row} for i, row in enumerate(offset_table)], f, indent=2)

#########################################
#   4) 主函数
#########################################
if __name__ == "__main__":
    output_dir = "table"
    os.makedirs(output_dir, exist_ok=True)

    student_kp = extract_keypoints_with_full_interpolation("video/student_resampled_800x1200.mp4") #student_resampled
    teacher_kp = extract_keypoints_with_full_interpolation("video/teacher_resampled_800x1200.mp4") #teacher_resampled

    num_joints = student_kp.shape[1]
    weights = np.ones(num_joints, dtype=np.float32)
    important_joints = [4, 7, 10, 13]
    for j in important_joints:
        weights[j] = 2.0

    offset_table = compute_weighted_offset(student_kp, teacher_kp, weights=weights, use_dtw=True)
    save_offset_table(offset_table, os.path.join(output_dir, "offset_table.csv"), save_format='csv')

    flattened = [dist for frame in offset_table for dist in frame]
    if flattened:
        mean_dist = np.mean(flattened)
        similarity_score = 1.0 / (1.0 + mean_dist)
        print(f"Mean distance: {mean_dist:.4f}, similarity_score={similarity_score:.4f}")
    else:
        print("No valid offsets, cannot compute similarity.")
