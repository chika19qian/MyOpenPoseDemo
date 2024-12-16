import cv2
import numpy as np
import csv
import os

def load_keypoints_from_csv(csv_path):
    """
    Load keypoints from a CSV file.
    :param csv_path: Path to keypoints CSV file.
    :return: List of keypoints for each frame.
    """
    keypoints_list = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                keypoints_list.append(None)
                continue
            keypoints = []
            for i in range(0, len(row), 2):
                if row[i] and row[i + 1]:
                    keypoints.append((int(float(row[i])), int(float(row[i + 1]))))
                else:
                    keypoints.append(None)
            keypoints_list.append(keypoints)
    return keypoints_list

def load_angles_from_csv(csv_path):
    """
    从 CSV 文件加载角度数据
    :param csv_path: CSV 文件路径
    :return: 每一帧的角度差异列表，格式 [[angle1, angle2, ...], ...]
    """
    angles_list = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过标题行
        for row in reader:
            if not row:  # 如果该行为空，跳过
                angles_list.append(None)
                continue
            angles = [float(a) if a else None for a in row[1:]]  # 忽略第一列 (frame_index)
            angles_list.append(angles)
    return angles_list

def load_high_diff_joints(csv_path):
    """
    Load high angle difference joints from CSV file.
    :param csv_path: Path to high angle diff CSV file.
    :return: List of high-difference joint markers for each frame.
    """
    high_diff_list = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header
        for row in reader:
            if not row:
                high_diff_list.append(None)
                continue
            high_diff_list.append([float(val) if val else 0 for val in row[1:]])
    return high_diff_list

def is_valid_point(point, frame_size):
    """Check if a point is within the frame size."""
    x, y = point
    return 0 <= x < frame_size[1] and 0 <= y < frame_size[0]

def draw_skeleton_with_highlight(frame, keypoints, high_diff_joints, frame_size, threshold=0.6):
    """
    Draw skeleton on a frame and highlight high difference joints with their angle values.
    :param frame: The video frame.
    :param keypoints: List of keypoints.
    :param high_diff_joints: List of angle differences per joint.
    :param frame_size: Size of the frame.
    :param threshold: Threshold for highlighting significant angle differences.
    """
    POSE_PAIRS = [[1, 2], [2, 3], [2, 5], [5, 6], [6, 7], [2, 8], [8, 9], [9, 10],
                  [2, 12], [12, 13], [13, 14], [2, 15], [15, 16], [16, 17]]

    # Draw skeleton lines
    for pair in POSE_PAIRS:
        p1, p2 = pair
        if keypoints[p1] and keypoints[p2]:
            if is_valid_point(keypoints[p1], frame_size) and is_valid_point(keypoints[p2], frame_size):
                cv2.line(frame, keypoints[p1], keypoints[p2], (200, 200, 200), 2)

    # Draw joints and highlight significant angle differences
    for idx, kp in enumerate(keypoints):
        if kp and is_valid_point(kp, frame_size):
            angle_diff = high_diff_joints[idx] if idx < len(high_diff_joints) else 0
            if angle_diff > threshold:
                color = (0, 0, 255)  # Red for high difference
                cv2.putText(frame, f"{angle_diff:.2f}", (kp[0] + 5, kp[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            else:
                color = (0, 255, 0)  # Green for normal joints
            cv2.circle(frame, kp, 5, color, -1)

def create_highlighted_video(student_csv, teacher_csv, angles_csv, high_diff_csv, teacher_video_path, student_video_path, output_video_path, frame_size=(1200, 1600), fps=30, threshold=0.6):
    """
    Create a video with skeletons and highlight joints with high angle differences.
    :param student_csv: Path to student keypoints CSV.
    :param teacher_csv: Path to teacher keypoints CSV.
    :param high_diff_csv: Path to high difference joints CSV.
    :param teacher_video_path: Path to teacher video.
    :param student_video_path: Path to student video.
    :param output_video_path: Path to output video.
    :param frame_size: Size of the combined frame.
    :param fps: Frames per second for the output video.
    :param threshold: Threshold for highlighting significant angle differences.
    """
    student_keypoints = load_keypoints_from_csv(student_csv)
    teacher_keypoints = load_keypoints_from_csv(teacher_csv)
    angle_differences = load_angles_from_csv(angles_csv)
    high_diff_joints = load_high_diff_joints(high_diff_csv)

    teacher_cap = cv2.VideoCapture(teacher_video_path)
    student_cap = cv2.VideoCapture(student_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_size[1], frame_size[0]))

    num_frames = min(len(student_keypoints), len(teacher_keypoints), len(high_diff_joints))
    for frame_idx in range(num_frames):
        ret_teacher, teacher_frame = teacher_cap.read()
        ret_student, student_frame = student_cap.read()
        if not ret_teacher or not ret_student:
            break

        teacher_frame = cv2.resize(teacher_frame, (800, 1200))
        student_frame = cv2.resize(student_frame, (800, 1200))
        combined_frame = np.zeros((1200, 1600, 3), dtype=np.uint8)
        combined_frame[:, :800, :] = teacher_frame
        combined_frame[:, 800:, :] = student_frame

        tk = teacher_keypoints[frame_idx]
        sk = student_keypoints[frame_idx]
        angles = angle_differences[frame_idx]

        high_diff = high_diff_joints[frame_idx]

        if tk and high_diff:
            draw_skeleton_with_highlight(combined_frame[:, :800, :], tk, high_diff, (1200, 800), threshold)
        if sk and high_diff:
            draw_skeleton_with_highlight(combined_frame[:, 800:, :], sk, high_diff, (1200, 800), threshold)

        avg_angle_diff = np.mean([abs(a) for a in angles if a is not None])
        cv2.putText(combined_frame, f"Frame {frame_idx + 1}/{num_frames}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined_frame, f"Avg Angle Diff: {avg_angle_diff:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(combined_frame)
        print(f"Processed frame {frame_idx + 1}/{num_frames}")

    teacher_cap.release()
    student_cap.release()
    out.release()
    print(f"Highlighted video saved to: {output_video_path}")

if __name__ == "__main__":
    student_csv = "table/girls_student_keypoints.csv"
    teacher_csv = "table/girls_teacher_keypoints.csv"
    angles_csv = "table/girls_angle_offset_table.csv"
    high_diff_csv = "table/girls_high_angle_diff_joints.csv"
    teacher_video_path = "videos/girls_teacher_resampled_800.mp4"
    student_video_path = "videos/girls_student_resampled_800.mp4"
    output_video_path = "output/girls_highlighted_skeleton_video.mp4"

    os.makedirs("output", exist_ok=True)

    create_highlighted_video(student_csv, teacher_csv, angles_csv, high_diff_csv, teacher_video_path, student_video_path, output_video_path, threshold=0.6)
