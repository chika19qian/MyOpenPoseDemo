import cv2
import numpy as np
import csv

def load_keypoints_from_csv(csv_path):
    """
    从 CSV 文件加载关键点数据
    :param csv_path: CSV 文件路径
    :return: 每一帧的关键点列表，格式 [[(x1, y1), (x2, y2), ...], ...]
    """
    keypoints_list = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:  # 如果该行为空，跳过
                keypoints_list.append(None)
                continue
            # 将数据转换为 (x, y) 坐标对
            keypoints = []
            for i in range(0, len(row), 2):
                if row[i] and row[i+1]:
                    keypoints.append((int(float(row[i])), int(float(row[i+1]))))
                else:
                    keypoints.append(None)
            keypoints_list.append(keypoints)
    return keypoints_list


def draw_skeleton(frame, keypoints, color=(255, 255, 255), thickness=2):
    """
    在帧上绘制骨架
    """
    POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], 
                  [1, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], 
                  [0, 14], [14, 16], [0, 15], [15, 17]]
    for pair in POSE_PAIRS:
        p1, p2 = pair
        if keypoints[p1] and keypoints[p2]:
            cv2.line(frame, keypoints[p1], keypoints[p2], color, thickness)
    for kp in keypoints:
        if kp:
            cv2.circle(frame, kp, 3, color, -1)


def create_skeleton_video(student_csv, teacher_csv, output_video_path, frame_size=(720, 1280), fps=30):
    """
    生成带骨架的视频
    :param student_csv: 学生的关键点 CSV 文件路径
    :param teacher_csv: 老师的关键点 CSV 文件路径
    :param output_video_path: 输出视频路径
    :param frame_size: 视频帧的尺寸 (height, width)
    :param fps: 视频帧率
    """
    # 加载关键点
    student_keypoints = load_keypoints_from_csv(student_csv)
    teacher_keypoints = load_keypoints_from_csv(teacher_csv)

    # 初始化视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 格式
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_size[1], frame_size[0]))

    num_frames = min(len(student_keypoints), len(teacher_keypoints))
    for i in range(num_frames):
        sk = student_keypoints[i]
        tk = teacher_keypoints[i]

        # 创建空白背景图像
        frame = np.zeros((frame_size[0], frame_size[1], 3), dtype=np.uint8)

        # 绘制骨架和差异
        if sk and tk:
            for j, (s_kp, t_kp) in enumerate(zip(sk, tk)):
                if s_kp and t_kp:
                    distance = np.linalg.norm(np.array(s_kp) - np.array(t_kp))
                    # 差异映射到颜色（蓝 -> 红）
                    color = (0, int(255 * (1 - min(distance / 100, 1))), int(255 * min(distance / 100, 1)))
                    cv2.circle(frame, s_kp, 5, color, -1)

            # 绘制骨架
            draw_skeleton(frame, sk, color=(0, 255, 0))  # 学生：绿色
            draw_skeleton(frame, tk, color=(255, 0, 0))  # 老师：红色

        # 写入视频
        out.write(frame)
        print(f"Processed frame {i+1}/{num_frames}")

    out.release()
    print(f"视频已保存到: {output_video_path}")


if __name__ == "__main__":
    # 输入 CSV 文件路径
    student_csv = "table/student_keypoints.csv"
    teacher_csv = "table/teacher_keypoints.csv"

    # 输出视频路径
    output_video_path = "output/skeleton_diff_video.mp4"
    import os
    os.makedirs("output", exist_ok=True)

    # 生成骨架差异视频
    create_skeleton_video(student_csv, teacher_csv, output_video_path)
