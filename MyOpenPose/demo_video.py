import os
import sys
import cv2
from sys import platform
import argparse

# 设置路径
dir_path = os.path.dirname(os.path.realpath(__file__))
os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/bin;'

# 引入 OpenPose
import pyopenpose as op
print(op)
print("成功引入pyopenpose")

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--video_path", default="examples/chuxue_me.mp4",
                    help="Path to video. Read all standard formats (mp4, avi, etc.).")
parser.add_argument("--output_path", default="output/chuxue_me_out.mp4",
                    help="Path to save the output video.")
args = parser.parse_args()

# OpenPose 参数设置
params = dict()
params["model_folder"] = "models/"
params["net_resolution"] = "368x256"  # 修改网络分辨率，降低显存占用

# 初始化 OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# 打开输入视频
video_path = args.video_path
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print(f"无法打开视频文件: {video_path}")
    sys.exit(1)

# 获取视频参数
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 初始化视频输出
output_path = args.output_path
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 使用 XVID 编码
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

print("开始处理视频...")

# 按帧处理视频
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # 视频结束

    # 将当前帧传递给 OpenPose
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    # Print keypoints
    keypoints = datum.poseKeypoints
    if keypoints is not None:
        print(f"Detected {keypoints.shape[0]} people")
        for person_id, person_keypoints in enumerate(keypoints):
            print(f"Person {person_id + 1}:")
            for keypoint_id, keypoint in enumerate(person_keypoints):
                x, y, confidence = keypoint
                print(f"  Keypoint {keypoint_id}: x={x}, y={y}, confidence={confidence}")
    else:
        print("No keypoints detected.")

    # 显示和保存输出帧
    cv2.imshow("OpenPose Video", datum.cvOutputData)
    out.write(datum.cvOutputData)

    # 按 `q` 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"视频处理完成！输出保存到: {output_path}")
