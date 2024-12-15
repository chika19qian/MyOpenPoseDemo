import cv2
import os

def resample_video(input_path, output_path, target_frames, target_fps=30):
    """
    将 input_path 视频重采样到固定的 target_frames 帧数，并以 target_fps 输出。
    最终输出视频时长 = target_frames / target_fps (秒).
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: cannot open video {input_path}")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Resample from {input_path}")
    print(f"  Original frames: {original_frames}, FPS: {original_fps}, size=({width}x{height})")
    print(f"  => Target frames: {target_frames}, target_fps: {target_fps}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))

    # 逐帧写入目标视频
    written_count = 0
    for i in range(target_frames):
        # 根据比例定位到原视频帧
        src_idx = i * (original_frames / target_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, src_idx)
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        written_count += 1

    cap.release()
    out.release()
    print(f"Resampled video saved to {output_path}, total written frames: {written_count}")
    if written_count < target_frames:
        print("Warning: did not write all target frames (video might be shorter).")


def main():
    student_video = "examples/chuxue_me.mp4"
    teacher_video = "examples/chuxue_teacher.mp4"

    # 打开老师视频, 获取它的帧数和FPS, 以它为基准
    cap_t = cv2.VideoCapture(teacher_video)
    teacher_frames = int(cap_t.get(cv2.CAP_PROP_FRAME_COUNT))
    teacher_fps = cap_t.get(cv2.CAP_PROP_FPS)
    cap_t.release()

    print(f"Teacher video has {teacher_frames} frames at {teacher_fps} fps.")

    # 假设我们想把「学生视频」也重采样到和老师相同的帧数+FPS
    student_output = "student_resampled.mp4"
    teacher_output = "teacher_resampled.mp4"

    # 重采样学生视频
    resample_video(
        input_path=student_video,
        output_path=student_output,
        target_frames=teacher_frames,
        target_fps=teacher_fps
    )

    # 如果也想让老师视频做一下“标准化导出”（有时你想把两段都输出到新的地方）
    # 也可以调用 resample_video:
    resample_video(
        input_path=teacher_video,
        output_path=teacher_output,
        target_frames=teacher_frames,
        target_fps=teacher_fps
    )

    print("Done.")

if __name__ == "__main__":
    main()
