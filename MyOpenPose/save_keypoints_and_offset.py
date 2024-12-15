import os
import cv2
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/bin;'
import pyopenpose as op

import json
import csv

def extract_keypoints(video_path, output_path=None, save_format='csv'):
    """
    使用 pyopenpose 提取视频中的关键点序列。
    如果指定 output_path，则会将关键点保存成 CSV 或 JSON（取决于 save_format）。
    
    :param video_path: 视频路径
    :param output_path: 输出文件路径，如 "table/student_keypoints.csv"
    :param save_format: 'csv' 或 'json'
    :return: 一个列表 all_keypoints，每一帧 keypoints[i] 是 shape=(num_joints,2) 或 None
    """

    params = {
        "model_folder": "./models/",
        "number_people_max": 1
    }
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    cap = cv2.VideoCapture(video_path)
    all_keypoints = []

    frame_idx = 0
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
            print(f"OpenPose failed to process frame {frame_idx}")
            all_keypoints.append(None)
            frame_idx += 1
            continue

        pose_keypoints = datums_vec[0].poseKeypoints
        if pose_keypoints is not None and pose_keypoints.shape[0] > 0:
            # 只拿第一个人
            keypts_2d = pose_keypoints[0, :, :2]  # (num_joints, 2)
            all_keypoints.append(keypts_2d.tolist())
        else:
            all_keypoints.append(None)

        frame_idx += 1

    cap.release()
    opWrapper.stop()

    if output_path is not None:
        if save_format == 'csv':
            save_keypoints_csv(all_keypoints, output_path)
        elif save_format == 'json':
            save_keypoints_json(all_keypoints, output_path)
        else:
            print(f"Unknown save_format={save_format}, skip saving file.")

    return all_keypoints


def save_keypoints_csv(all_keypoints, csv_path):
    """
    将 all_keypoints 列表保存成 CSV。每帧一行，每个关节点 (x,y) 成对出现。
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)  # 确保目录存在
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for frame_data in all_keypoints:
            if frame_data is None:
                writer.writerow([])
            else:
                row = []
                for (x, y) in frame_data:
                    row.extend([x, y])
                writer.writerow(row)
    print(f"Keypoints CSV saved to {csv_path}")


def save_keypoints_json(all_keypoints, json_path):
    """
    将 all_keypoints 列表保存成 JSON。 每帧就是 [ [x1,y1], [x2,y2], ... ]
    """
    os.makedirs(os.path.dirname(json_path), exist_ok=True)  # 确保目录存在
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_keypoints, f, indent=2)
    print(f"Keypoints JSON saved to {json_path}")


def compute_offset_table(student_keypoints, teacher_keypoints, output_path=None, save_format='csv'):
    """
    逐帧逐关节点计算偏移 (欧几里得距离)，生成一个 offset_table。
    offset_table[i] = [dist_joint0, dist_joint1, ..., dist_jointN]
    如果指定 output_path，则会将偏移表保存到CSV或JSON。
    """
    num_frames = min(len(student_keypoints), len(teacher_keypoints))
    offset_table = []  # offset_table[i] -> list of distances for frame i

    for i in range(num_frames):
        sk = student_keypoints[i]
        tk = teacher_keypoints[i]

        if sk is None or tk is None:
            # 该帧没有检测到人
            offset_table.append(None)
            continue

        sk_arr = np.array(sk)  # shape=(num_joints,2)
        tk_arr = np.array(tk)  # shape=(num_joints,2)
        if sk_arr.shape != tk_arr.shape:
            offset_table.append(None)
            continue

        dist = np.linalg.norm(sk_arr - tk_arr, axis=1)  # (num_joints,)
        offset_table.append(dist.tolist())

    if output_path is not None:
        if save_format == 'csv':
            save_offset_csv(offset_table, output_path)
        elif save_format == 'json':
            save_offset_json(offset_table, output_path)
        else:
            print(f"Unknown save_format={save_format}, skip saving file.")

    return offset_table


def save_offset_csv(offset_table, csv_path):
    """
    偏移表保存到 CSV：每帧一行，每列为某个关节点距离
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 不一定知道关节点数，这里假设用 offset_table 里第一个非 None 的条目确定关节点数
        num_joints = 0
        for row in offset_table:
            if row is not None:
                num_joints = len(row)
                break
        # 写header
        header = [f"joint_{j}" for j in range(num_joints)]
        writer.writerow(["frame_index"] + header)

        for i, row in enumerate(offset_table):
            if row is None:
                writer.writerow([i] + [""] * num_joints)
            else:
                writer.writerow([i] + row)
    print(f"Offset CSV saved to {csv_path}")


def save_offset_json(offset_table, json_path):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    output = []
    for i, row in enumerate(offset_table):
        if row is None:
            output.append({"frame_index": i, "distances": None})
        else:
            output.append({"frame_index": i, "distances": row})
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    print(f"Offset JSON saved to {json_path}")


# =============== Main script example ===============
if __name__ == "__main__":
    # 学生 & 老师视频路径（这里假设已重采样过）
    student_video = "student_resampled.mp4"
    teacher_video = "teacher_resampled.mp4"

    # 选 "csv" 或 "json"，也可以只写一个保存
    save_format = "csv"  # or "json"

    # 输出目录
    table_dir = "table"  # 统一保存到 table/ 文件夹
    os.makedirs(table_dir, exist_ok=True)

    # 提取并保存
    student_keypoints = extract_keypoints(
        student_video,
        output_path=os.path.join(table_dir, f"student_keypoints.{save_format}"),
        save_format=save_format
    )
    teacher_keypoints = extract_keypoints(
        teacher_video,
        output_path=os.path.join(table_dir, f"teacher_keypoints.{save_format}"),
        save_format=save_format
    )

    # 计算 offset
    offset_table = compute_offset_table(
        student_keypoints,
        teacher_keypoints,
        output_path=os.path.join(table_dir, f"offset_table.{save_format}"),
        save_format=save_format
    )

    print("Done. Keypoints & offset table saved in 'table' folder.")
