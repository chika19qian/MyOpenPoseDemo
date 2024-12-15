import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# 设置环境变量 PATH（确保能找到 pyopenpose.so / .dll 等）
dir_path = os.path.dirname(os.path.realpath(__file__))
os.environ['PATH'] += os.pathsep + os.path.join(dir_path, 'bin')

print("成功引入 pyopenpose")
import pyopenpose as op


def extract_keypoints(video_path):
    """
    使用 pyopenpose 提取视频中的关键点序列
    """
    params = {
        "model_folder": "./models/",
        "number_people_max": 1
    }
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    cap = cv2.VideoCapture(video_path)
    keypoints = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        datum = op.Datum()
        datum.cvInputData = frame
        datums_vec = op.VectorDatum()
        datums_vec.append(datum)

        success = opWrapper.emplaceAndPop(datums_vec)
        if not success:
            print("OpenPose failed to process the frame.")
            keypoints.append(None)
            continue

        # 提取关键点 (如果有人体检测结果)
        if datums_vec[0].poseKeypoints is not None:
            keypoints.append(datums_vec[0].poseKeypoints[0])  # 只拿第一个人的
        else:
            keypoints.append(None)  # 没检测到人时填 None

    cap.release()
    opWrapper.stop()
    return keypoints


def normalize_keypoints(keypoints):
    """
    对关键点进行基于鼻子的平移、再按最大边界进行缩放归一化
    """
    normalized = []
    for frame in keypoints:
        if frame is None:
            normalized.append(None)
            continue

        # (x, y) 坐标
        x = frame[:, 0].copy()
        y = frame[:, 1].copy()

        # 以鼻子为基准点（索引0）
        nose = frame[0]
        if nose[0] > 0 and nose[1] > 0:
            x -= nose[0]
            y -= nose[1]

        # 再按最大边界进行缩放
        scale = max(x.max() - x.min(), y.max() - y.min())
        if scale > 1e-6:  # 防止除 0
            x /= scale
            y /= scale

        normalized.append(np.column_stack((x, y)))
    return normalized


def align_and_interpolate(seq1, seq2, target_length):
    """
    使用 DTW 对齐关键点序列，并将得到的匹配路径索引映射到同一长度
    返回两个对齐后帧索引列表（aligned_seq1, aligned_seq2）
    """
    valid_seq1 = [np.array(kp).flatten() for kp in seq1 if kp is not None]
    valid_seq2 = [np.array(kp).flatten() for kp in seq2 if kp is not None]

    # 动态时间规整 (DTW)
    _, path = fastdtw(valid_seq1, valid_seq2, dist=euclidean)

    # 提取路径中的索引
    seq1_indices, seq2_indices = zip(*path)

    # 将对齐索引映射到 target_length 的帧数上
    aligned_seq1 = [
        seq1_indices[min(len(seq1_indices) - 1, int(i * len(seq1_indices) / target_length))]
        for i in range(target_length)
    ]
    aligned_seq2 = [
        seq2_indices[min(len(seq2_indices) - 1, int(i * len(seq2_indices) / target_length))]
        for i in range(target_length)
    ]

    return list(map(int, aligned_seq1)), list(map(int, aligned_seq2))


def compute_similarity(student_keypoints, teacher_keypoints, 
                       aligned_student_indices, aligned_teacher_indices):
    """
    根据对齐后的索引，计算两段舞蹈关键点序列的平均距离。
    """
    distances = []
    for s_idx, t_idx in zip(aligned_student_indices, aligned_teacher_indices):
        s_frame = student_keypoints[s_idx]  # shape: (num_joints, 2)
        t_frame = teacher_keypoints[t_idx]

        if s_frame is None or t_frame is None:
            # 如果该帧关键点为空(检测失败)，可以跳过或设一个缺省值
            continue

        # 计算每个关节点的欧几里得距离
        diff = s_frame - t_frame   # shape: (num_joints, 2)
        frame_dist = np.sqrt((diff**2).sum(axis=1))  # 每个关节点距离
        avg_dist_this_frame = frame_dist.mean()      # 当前帧所有关节点距离的平均
        distances.append(avg_dist_this_frame)

    if len(distances) == 0:
        return None

    mean_distance = np.mean(distances)  # 所有对齐帧平均距离
    return mean_distance

def generate_combined_skeleton_video(student_video, teacher_video, 
                                     student_indices, teacher_indices, 
                                     output_path, fps=30):
    """
    根据DTW索引映射，将两段视频的帧对齐后，并排输出带骨骼渲染的视频。
    student_indices[i] 对应 teacher_indices[i]
    """
    # 1. 首先获取两个视频的分辨率
    cap_s = cv2.VideoCapture(student_video)
    cap_t = cv2.VideoCapture(teacher_video)

    width_s = int(cap_s.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_s = int(cap_s.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width_t = int(cap_t.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_t = int(cap_t.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 2. 取最小高度，做等比例缩放
    target_height = min(height_s, height_t)

    # 分别计算等比例缩放后宽度
    scale_s = target_height / float(height_s) if height_s > 0 else 1.0
    scale_t = target_height / float(height_t) if height_t > 0 else 1.0

    new_width_s = int(width_s * scale_s)
    new_width_t = int(width_t * scale_t)

    # 3. 创建VideoWriter。最终输出宽度 = 两个缩放后宽度之和，高度=target_height
    out_width = new_width_s + new_width_t
    out_height = target_height

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

    # 4. 初始化 OpenPose
    params = {
        "model_folder": "./models/",
        "number_people_max": 1,
        "display": 0,
        "render_pose": 1,
    }
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    written_frames = 0
    for s_idx, t_idx in zip(student_indices, teacher_indices):
        cap_s.set(cv2.CAP_PROP_POS_FRAMES, s_idx)
        cap_t.set(cv2.CAP_PROP_POS_FRAMES, t_idx)

        ret_s, frame_s = cap_s.read()
        ret_t, frame_t = cap_t.read()
        if not (ret_s and ret_t):
            print(f"Failed to read frame: s_idx={s_idx}, t_idx={t_idx}")
            continue

        # 分别渲染骨骼
        datum_s = op.Datum()
        datum_s.cvInputData = frame_s
        datums_vec_s = op.VectorDatum()
        datums_vec_s.append(datum_s)
        opWrapper.emplaceAndPop(datums_vec_s)
        skeleton_s = datum_s.cvOutputData

        datum_t = op.Datum()
        datum_t.cvInputData = frame_t
        datums_vec_t = op.VectorDatum()
        datums_vec_t.append(datum_t)
        opWrapper.emplaceAndPop(datums_vec_t)
        skeleton_t = datum_t.cvOutputData

        # 5. 等比缩放到相同 target_height
        skeleton_s_resized = cv2.resize(skeleton_s, (new_width_s, out_height))
        skeleton_t_resized = cv2.resize(skeleton_t, (new_width_t, out_height))

        # 并排拼接
        combined_frame = np.hstack((skeleton_s_resized, skeleton_t_resized))
        out.write(combined_frame)
        written_frames += 1

    cap_s.release()
    cap_t.release()
    out.release()
    opWrapper.stop()

    print(f"DTW-aligned skeleton comparison video saved to: {output_path}")
    print(f"Total written frames: {written_frames}")
    if written_frames == 0:
        print("警告：没有成功写入任何帧，输出视频可能为空文件。")

if __name__ == "__main__":
    student_video = "student_resampled.mp4"
    teacher_video = "teacher_resampled.mp4"

    # 1) 提取关键点
    student_keypoints = extract_keypoints(student_video)
    teacher_keypoints = extract_keypoints(teacher_video)

    # 2) 可选: 关键点归一化
    student_keypoints = normalize_keypoints(student_keypoints)
    teacher_keypoints = normalize_keypoints(teacher_keypoints)

    # 3) 使用 DTW 对齐并得到帧匹配索引
    target_length = min(len(student_keypoints), len(teacher_keypoints))
    aligned_student_indices, aligned_teacher_indices = align_and_interpolate(
        student_keypoints, teacher_keypoints, target_length
    )

    # 4) 计算相似度
    mean_distance = compute_similarity(student_keypoints, teacher_keypoints,
                                       aligned_student_indices, aligned_teacher_indices)
    if mean_distance is not None:
        print(f"两个舞蹈的平均关键点距离: {mean_distance:.4f}")
        similarity_score = 1 / (1 + mean_distance)
        print(f"相似度得分: {similarity_score:.4f} (越接近1越相似)")
    else:
        print("无法计算相似度，关键点序列可能为空或提取失败。")

    # 5) 生成并排骨骼可视化视频
    generate_combined_skeleton_video(
        student_video, teacher_video,
        aligned_student_indices, aligned_teacher_indices,
        output_path="comparison_side_by_side_aligned.mp4",
        fps=30
    )
