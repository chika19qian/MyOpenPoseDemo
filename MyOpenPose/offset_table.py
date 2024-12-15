import numpy as np
import csv

def compute_offset_table(student_keypoints, teacher_keypoints, output_csv="offset.csv"):
    """
    逐帧、逐关节点计算偏移 (欧几里得距离)。并保存成 CSV。
    student_keypoints[i] 和 teacher_keypoints[i] 均是 (num_joints, 2) 大小的数组或 None。
    """
    num_frames = min(len(student_keypoints), len(teacher_keypoints))
    offset_rows = []  # 存储 [frame_idx, dist_joint0, dist_joint1, ... ]

    for i in range(num_frames):
        sk = student_keypoints[i]
        tk = teacher_keypoints[i]
        if sk is None or tk is None:
            # 该帧没有检测到人
            offset_rows.append([i] + [""]*25)  # 根据关节点数量写空
            continue

        sk = np.array(sk)  # shape=(num_joints,2)
        tk = np.array(tk)  # shape=(num_joints,2)
        if sk.shape != tk.shape:
            offset_rows.append([i] + [""]*25)
            continue
        
        num_joints = sk.shape[0]
        distances = np.linalg.norm(sk - tk, axis=1)  # shape=(num_joints,)
        row = [i] + distances.tolist()
        offset_rows.append(row)

    # 写CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 第一行写个header: frame_idx + joint0...jointN
        header = ["frame_idx"] + [f"joint_{j}" for j in range(sk.shape[0])]
        writer.writerow(header)

        for row in offset_rows:
            writer.writerow(row)

    print(f"Offset table saved to {output_csv}")


def main():
    import json
    # 读取之前生成的 JSON 或 CSV
    with open("student_keypoints.json", 'r') as f:
        student_keypoints = json.load(f)
    with open("teacher_keypoints.json", 'r') as f:
        teacher_keypoints = json.load(f)

    # 计算偏移表格
    compute_offset_table(student_keypoints, teacher_keypoints, output_csv="offset_table.csv")

if __name__ == "__main__":
    main()
