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
#   2) 计算骨骼角度
#########################################
def calculate_angles(keypoints):
    """
    根据关节点计算骨骼角度特征。
    输入：
        keypoints: shape=(num_joints, 2) 的 numpy 数组。
    输出：
        angles: 每一对关节点之间的角度，单位为弧度。
    """
    connections = [
        (1, 2), (2, 3),  # 头部到躯干
        (2, 5), (5, 6), (6, 7),  # 左臂
        (2, 8), (8, 9), (9, 10),  # 右臂
        (2, 12), (12, 13), (13, 14),  # 左腿
        (2, 15), (15, 16), (16, 17),  # 右腿
    ]
    angles = []
    for start, end in connections:
        if not np.any(keypoints[start]) or not np.any(keypoints[end]):
            angles.append(np.nan)
            continue
        vector = keypoints[end] - keypoints[start]
        angle = np.arctan2(vector[1], vector[0])  # 计算角度
        angles.append(angle)

    angles = np.nan_to_num(angles, nan=0.0, posinf=0.0, neginf=0.0)
    return np.array(angles)

#########################################
#   3) 计算角度差异 & DTW
#########################################
def compute_angle_offset(student_kp, teacher_kp, use_dtw=False, align_method='truncate'):
    """
    计算学生与老师的角度差异，支持 DTW。
    """
    if student_kp.shape[0] != teacher_kp.shape[0]:
        print(f"Aligning frame counts: student_kp={student_kp.shape[0]}, teacher_kp={teacher_kp.shape[0]}")
        if align_method == 'truncate':
            # 截短到最短帧数
            min_frames = min(student_kp.shape[0], teacher_kp.shape[0])
            student_kp = student_kp[:min_frames]
            teacher_kp = teacher_kp[:min_frames]
        elif align_method == 'interpolate':
            # 插值对齐到相同帧数
            target_frames = max(student_kp.shape[0], teacher_kp.shape[0])
            student_kp = interpolate_keypoints(student_kp, target_frames)
            teacher_kp = interpolate_keypoints(teacher_kp, target_frames)
        else:
            raise ValueError(f"Unsupported align_method: {align_method}")

    frames_count = student_kp.shape[0]

    # 计算角度特征
    student_angles = [calculate_angles(student_kp[f]) for f in range(frames_count)]
    teacher_angles = [calculate_angles(teacher_kp[f]) for f in range(frames_count)]

    # 检查是否存在 NaN 或 Inf 数据
    for i, (s_angles, t_angles) in enumerate(zip(student_angles, teacher_angles)):
        if np.any(np.isnan(s_angles)) or np.any(np.isnan(t_angles)):
            print(f"Frame {i}: Contains NaN in angle data.")
        if np.any(np.isinf(s_angles)) or np.any(np.isinf(t_angles)):
            print(f"Frame {i}: Contains Inf in angle data.")

    # 使用 DTW 或逐帧计算
    if not use_dtw:
        offset_table = []
        for s_angles, t_angles in zip(student_angles, teacher_angles):
            valid_mask = ~np.isnan(s_angles) & ~np.isnan(t_angles)
            if np.any(valid_mask):
                angle_diff = np.abs(s_angles[valid_mask] - t_angles[valid_mask])
                angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)  # 考虑角度周期性
                offset_table.append(angle_diff.tolist())
            else:
                offset_table.append([])
        return offset_table

    # 动态时间规整
    student_seq = [angles.flatten() for angles in student_angles]
    teacher_seq = [angles.flatten() for angles in teacher_angles]

    # 过滤掉无效值
    student_seq = [np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0) for seq in student_seq]
    teacher_seq = [np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0) for seq in teacher_seq]

    _, path = fastdtw(student_seq, teacher_seq, dist=euclidean)
    offset_table = []
    for s_idx, t_idx in path:
        valid_mask = ~np.isnan(student_angles[s_idx]) & ~np.isnan(teacher_angles[t_idx])
        if np.any(valid_mask):
            angle_diff = np.abs(student_angles[s_idx][valid_mask] - teacher_angles[t_idx][valid_mask])
            angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)  # 考虑角度周期性
            offset_table.append(angle_diff.tolist())
        else:
            offset_table.append([])
    return offset_table

def interpolate_keypoints(keypoints, target_frames):
    """
    插值对齐关键点数据到指定帧数。
    :param keypoints: 原始关键点数据 (frames, num_joints, 2)
    :param target_frames: 目标帧数
    :return: 插值后的关键点数据
    """
    num_frames, num_joints, _ = keypoints.shape
    x = np.arange(num_frames)
    x_new = np.linspace(0, num_frames - 1, target_frames)
    interp_func = interp1d(x, keypoints, axis=0, kind='linear', fill_value="extrapolate")
    return interp_func(x_new)

#########################################
#   4) 保存角度偏移表
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
    将关键点保存到 CSV 文件
    :param keypoints: 关键点数组 (frames, num_joints, 2)
    :param output_path: 输出 CSV 文件路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for frame in keypoints:
            if frame is None:
                writer.writerow([])  # 如果帧为空，写入空行
            else:
                flat_frame = frame.flatten()  # 展平为 1D 列表
                writer.writerow(flat_frame)


#######################################
#   5) 主函数
#########################################
if __name__ == "__main__":
    output_dir = "table"
    os.makedirs(output_dir, exist_ok=True)

    student_kp = extract_keypoints_with_full_interpolation("videos/girls_student_resampled_800.mp4")
    teacher_kp = extract_keypoints_with_full_interpolation("videos/girls_teacher_resampled_800.mp4")

    print(f"Student keypoints shape: {student_kp.shape}")
    print(f"Teacher keypoints shape: {teacher_kp.shape}")

     # 保存关键点数据
    save_keypoints_to_csv(student_kp, os.path.join(output_dir, "girls_student_keypoints.csv"))
    save_keypoints_to_csv(teacher_kp, os.path.join(output_dir, "girls_teacher_keypoints.csv"))


    offset_table = compute_angle_offset(student_kp, teacher_kp, use_dtw=True)
    save_offset_table(offset_table, os.path.join(output_dir, "girls_angle_offset_table.csv"), save_format='csv')

    flattened = [angle for frame in offset_table for angle in frame if not np.isnan(angle)]
    if flattened:
        mean_angle_diff = np.mean(flattened)
        similarity_score = 1.0 / (1.0 + mean_angle_diff)  # 越接近 1 表示越相似
        print(f"Mean angle difference: {mean_angle_diff:.4f}, similarity_score={similarity_score:.4f}")
    else:
        print("No valid offsets, cannot compute similarity.")
