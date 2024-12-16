import os
import cv2

def resample_video(input_path, output_path, target_frames, target_fps=30, resize_dim=None):
    """
    Resample the video at input_path to have target_frames and target_fps.
    If resize_dim=(w, h), scale frames to (w, h).
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: cannot open video {input_path}")
        return

    original_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\nResample from {input_path}")
    print(f"  Original frames: {original_frames}, fps: {original_fps}, size=({width}x{height})")
    print(f"  => target frames: {target_frames}, target_fps: {target_fps}, resize={resize_dim}")

    if resize_dim is None:
        target_w, target_h = width, height
    else:
        target_w, target_h = resize_dim

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (target_w, target_h))

    written_count = 0
    for i in range(target_frames):
        # Locate the source frame in the original video based on the proportion
        src_idx = i * (original_frames / target_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, src_idx)
        ret, frame = cap.read()
        if not ret:
            break

        if resize_dim is not None:
            frame = cv2.resize(frame, (target_w, target_h))

        out.write(frame)
        written_count += 1

    cap.release()
    out.release()
    print(f"Resampled video saved to {output_path}, total written frames: {written_count}")


def main():
    student_video = "examples/lip_me.mp4"
    teacher_video = "examples/lip_teacher.mp4"

    # Open the teacher's video to get its frame count and FPS for unifying duration
    cap_t = cv2.VideoCapture(teacher_video)
    teacher_frames = int(cap_t.get(cv2.CAP_PROP_FRAME_COUNT))
    teacher_fps = cap_t.get(cv2.CAP_PROP_FPS)
    cap_t.release()

    # Fixed resolution 800×1200 ; 1280x700
    fixed_w, fixed_h = 800, 1200

   # Resample the student video to match (teacher_frames, teacher_fps) and force resolution to 800×1200
    resample_video(
        input_path=student_video,
        output_path="videos/lip_me_resampled_800.mp4",
        target_frames=teacher_frames,
        target_fps=teacher_fps,
        resize_dim=(fixed_w, fixed_h)  # 固定宽高
    )

    # Save a new version of the teacher video as well, making both 800×1200 in size
    resample_video(
        input_path=teacher_video,
        output_path="videos/lip_teacher_resampled_800.mp4",
        target_frames=teacher_frames,
        target_fps=teacher_fps,
        resize_dim=(fixed_w, fixed_h)
    )

    print("Done. Both videos are now 800x1200 and have the same FPS/frames.")


if __name__ == "__main__":
    main()
