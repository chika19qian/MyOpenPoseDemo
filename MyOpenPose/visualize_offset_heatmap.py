import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import json

def load_offset_table_csv(csv_path):
    """
    从 CSV 文件读取 offset_table（帧×关节点的误差矩阵）。
    返回: numpy array shape = (num_frames, num_joints), 以及frames列表
    """
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)

    # rows[0] 是 header: ["frame_index", "joint_0", "joint_1", ...]
    header = rows[0]
    data_rows = rows[1:]  # 从第二行开始是实际数据

    # 找出有多少 joints
    # header结构: [frame_index, joint_0, joint_1, ..., joint_n]
    num_joints = len(header) - 1

    frames = []
    offset_matrix = []
    for row in data_rows:
        if not row:
            continue
        frame_idx = int(row[0])
        frames.append(frame_idx)

        # 剩下是 offset
        offset_values = []
        for val in row[1:]:  # joint_0...joint_n
            if val.strip() == '':
                offset_values.append(np.nan)
            else:
                offset_values.append(float(val))
        offset_matrix.append(offset_values)

    offset_matrix = np.array(offset_matrix)  # shape=(num_frames, num_joints)
    return frames, offset_matrix


def load_offset_table_json(json_path):
    """
    从 JSON 文件读取 offset_table。JSON格式大致是：
    [
      {"frame_index": i, "distances": [d0, d1, d2, ...]}, ...
    ]
    返回 numpy array shape=(num_frames, num_joints)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    frames = []
    offsets = []
    for entry in data:
        frames.append(entry["frame_index"])
        if entry["distances"] is None:
            offsets.append([])
        else:
            offsets.append(entry["distances"])
    offset_matrix = np.array(offsets)  # shape=(num_frames, num_joints)
    return frames, offset_matrix


def plot_offset_heatmap(offset_matrix, frames, title="Offset Heatmap", save_path=None):
    """
    绘制偏移的热力图。offset_matrix shape=(num_frames, num_joints)
    横轴=帧index, 纵轴=关节点index
    """
    # 如果数据里有 NaN，可以选择用 np.nan_to_num() 或者 masked array
    offset_matrix = np.nan_to_num(offset_matrix, nan=0.0)

    num_frames, num_joints = offset_matrix.shape

    plt.figure(figsize=(10, 6))
    # imshow expects (num_joints, num_frames) 如果想让横轴为 frame，就要转置
    # Option A: 让 x-axis= frames, y-axis= joints => offset_matrix.T
    # Option B: x= joints, y= frames => offset_matrix
    # 这里我们选择 x=frames, y=joints => 需要转置
    data_for_plot = offset_matrix.T  # shape=(num_joints, num_frames)

    # 绘制热力图, aspect='auto' 可以自动适应宽高比
    plt.imshow(data_for_plot, aspect='auto', cmap='turbo', origin='lower') 
    # 也可以用其他 colormap: 'hot', 'jet', 'viridis', 'plasma', 'coolwarm'...

    plt.colorbar(label="Offset Distance")  # 颜色条

    plt.xlabel("Frame Index")
    plt.ylabel("Joint Index")
    plt.title(title)

    # 设置 x 坐标刻度
    # 如果帧数很多，可以不全显示
    plt.xticks(np.linspace(0, num_frames-1, min(num_frames, 10)).astype(int))  
    plt.yticks(np.arange(num_joints))  # 每个关节点一个刻度

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Heatmap saved to {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    # 你可以根据实际情况选择读取CSV还是JSON
    csv_path = os.path.join("table", "offset_table.csv")
    json_path = os.path.join("table", "offset_table.json")

    if os.path.exists(csv_path):
        frames, offset_matrix = load_offset_table_csv(csv_path)
        print(f"Loaded offset_table from {csv_path}, shape={offset_matrix.shape}")
        plot_offset_heatmap(
            offset_matrix, 
            frames,
            title="Offset Heatmap (CSV)",
            save_path="offset_heatmap_csv.png"
        )
    elif os.path.exists(json_path):
        frames, offset_matrix = load_offset_table_json(json_path)
        print(f"Loaded offset_table from {json_path}, shape={offset_matrix.shape}")
        plot_offset_heatmap(
            offset_matrix,
            frames,
            title="Offset Heatmap (JSON)",
            save_path="offset_heatmap_json.png"
        )
    else:
        print("No offset_table CSV or JSON found in 'table' folder. Please check your path.")


if __name__ == "__main__":
    main()
