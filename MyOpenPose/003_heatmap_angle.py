import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import json

def load_angle_offset_table_csv(csv_path):
    """
    Load angle offset table (angle difference matrix of frames × joints) from a CSV file.
    Returns: numpy array shape = (num_frames, num_joints), and a list of frames.
    """
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)

    # rows[0] is the header: ["frame_index", "joint_0", "joint_1", ...]
    header = rows[0]
    data_rows = rows[1:]  # Actual data starts from the second row

    num_joints = len(header) - 1  # Header structure: [frame_index, joint_0, ..., joint_n]

    frames = []
    angle_matrix = []
    for row in data_rows:
        if not row:
            continue
        frame_idx = int(row[0])
        frames.append(frame_idx)

        # Parse angle data
        angle_values = []
        for val in row[1:]:  # joint_0...joint_n
            if val.strip() == '':
                angle_values.append(np.nan)
            else:
                angle_values.append(float(val))
        angle_matrix.append(angle_values)

    angle_matrix = np.array(angle_matrix)  # shape=(num_frames, num_joints)
    return frames, angle_matrix

def load_angle_offset_table_json(json_path):
    """
    Load angle offset table from a JSON file. The JSON format is roughly:
    [
      {"frame_index": i, "angles": [a0, a1, a2, ...]}, ...
    ]
    Returns numpy array shape=(num_frames, num_joints).
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    frames = []
    angles = []
    for entry in data:
        frames.append(entry["frame_index"])
        if entry["angles"] is None:
            angles.append([])
        else:
            angles.append(entry["angles"])
    angle_matrix = np.array(angles)  # shape=(num_frames, num_joints)
    return frames, angle_matrix

def plot_angle_heatmap(angle_matrix, frames, title="Angle Difference Heatmap", save_path=None):
    """
    Plot a heatmap of angle offsets. angle_matrix shape=(num_frames, num_joints)
    X-axis = frame index, Y-axis = joint index.
    """
    angle_matrix = np.nan_to_num(angle_matrix, nan=0.0)  # Replace NaN with 0

    num_frames, num_joints = angle_matrix.shape

    plt.figure(figsize=(10, 6))
    # Transpose matrix to ensure x=frames, y=joints
    data_for_plot = angle_matrix.T  # shape=(num_joints, num_frames)

    # Use a color map such as 'coolwarm' or 'viridis' for better visual distinction
    plt.imshow(data_for_plot, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label="Angle Difference (radians)")

    plt.xlabel("Frame Index")
    plt.ylabel("Joint Index")
    plt.title(title)

    # Set x-axis ticks
    plt.xticks(np.linspace(0, num_frames - 1, min(num_frames, 10)).astype(int))
    plt.yticks(np.arange(num_joints))  # One tick for each joint

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Heatmap saved to {save_path}")
    else:
        plt.show()
    plt.close()

def main():
    # Update paths to the angle offset table
    csv_path = os.path.join("table", "lip_angle_offset_table.csv")
    json_path = os.path.join("table", "angle_offset_table.json")

    if os.path.exists(csv_path):
        frames, angle_matrix = load_angle_offset_table_csv(csv_path)
        print(f"Loaded angle_offset_table from {csv_path}, shape={angle_matrix.shape}")
        plot_angle_heatmap(
            angle_matrix, 
            frames,
            title="Angle Difference Heatmap (CSV)",
            save_path="output/lip_angle_heatmap_csv.png"
        )
    elif os.path.exists(json_path):
        frames, angle_matrix = load_angle_offset_table_json(json_path)
        print(f"Loaded angle_offset_table from {json_path}, shape={angle_matrix.shape}")
        plot_angle_heatmap(
            angle_matrix,
            frames,
            title="Angle Difference Heatmap (JSON)",
            save_path="angle_heatmap_json.png"
        )
    else:
        print("No angle_offset_table CSV or JSON found in 'table' folder. Please check your path.")

if __name__ == "__main__":
    main()
