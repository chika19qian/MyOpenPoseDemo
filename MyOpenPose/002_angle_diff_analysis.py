import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

def load_angle_offset_table(csv_path):
    """
    Load angle difference table from CSV file.
    :param csv_path: Path to angle offset CSV file.
    :return: numpy array of shape (num_frames, num_joints)
    """
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
    header = rows[0]
    data_rows = rows[1:]

    angle_matrix = []
    for row in data_rows:
        if not row:
            continue
        angle_values = [float(val) if val else np.nan for val in row[1:]]
        angle_matrix.append(angle_values)

    return np.array(angle_matrix)

def plot_frame_average_difference(angle_matrix, save_path=None):
    """
    Plot frame-wise average angle difference.
    """
    avg_angle_per_frame = np.nanmean(angle_matrix, axis=1)
    plt.figure(figsize=(10, 6))
    plt.plot(avg_angle_per_frame, label="Average Angle Difference per Frame")
    plt.xlabel("Frame Index")
    plt.ylabel("Average Angle Difference (radians)")
    plt.title("Frame-wise Average Angle Difference")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved frame-wise average difference plot to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_joint_average_difference(angle_matrix, save_path=None):
    """
    Plot joint-wise average angle difference.
    """
    avg_angle_per_joint = np.nanmean(angle_matrix, axis=0)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(avg_angle_per_joint)), avg_angle_per_joint)
    plt.xlabel("Joint Index")
    plt.ylabel("Average Angle Difference (radians)")
    plt.title("Joint-wise Average Angle Difference")
    if save_path:
        plt.savefig(save_path)
        print(f"Saved joint-wise average difference plot to {save_path}")
    else:
        plt.show()
    plt.close()

def detect_high_difference_frames(angle_matrix, threshold=1.5):
    """
    Detect frames where average angle difference exceeds a threshold.
    :param angle_matrix: Numpy array of shape (num_frames, num_joints)
    :param threshold: Threshold for angle difference.
    :return: List of high-difference frame indices.
    """
    avg_angle_per_frame = np.nanmean(angle_matrix, axis=1)
    high_diff_frames = np.where(avg_angle_per_frame > threshold)[0]
    print(f"Frames with high angle differences (>{threshold} radians): {high_diff_frames}")
    return high_diff_frames

def cluster_frames_by_angles(angle_matrix, n_clusters=3, save_path=None):
    """
    Perform clustering on angle data to identify motion patterns.
    :param angle_matrix: Numpy array of shape (num_frames, num_joints)
    :param n_clusters: Number of clusters.
    :return: Cluster labels for each frame.
    """
    angle_matrix_filled = np.nan_to_num(angle_matrix, nan=0.0)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    labels = kmeans.fit_predict(angle_matrix_filled)

    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(labels)), labels, c=labels, cmap='viridis', label='Cluster Label')
    plt.xlabel("Frame Index")
    plt.ylabel("Cluster Label")
    plt.title("Frame Clustering Based on Angle Differences")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved clustering plot to {save_path}")
    else:
        plt.show()
    plt.close()
    return labels

def dynamic_time_warping(angle_matrix):
    """
    Perform DTW analysis to evaluate time synchronization of joint angles.
    :param angle_matrix: Numpy array of shape (num_frames, num_joints)
    """
    avg_angles = np.nanmean(angle_matrix, axis=1)
    avg_angles = avg_angles.reshape(-1, 1)  # Ensure 1-D shape for DTW
    reference = avg_angles  # Using self-reference here, adjust if you have teacher data.
    distance, path = fastdtw(avg_angles, reference, dist=euclidean)
    print(f"DTW Distance: {distance}")
    return distance

def save_high_angle_diff_joints(angle_matrix, threshold, output_path):
    """
    保存高角度差异关节点信息。
    :param angle_matrix: 角度差异矩阵 (num_frames, num_joints)
    :param threshold: 高角度差异的阈值
    :param output_path: 输出 CSV 文件路径
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['frame_index'] + [f'joint_{i}' for i in range(angle_matrix.shape[1])]
        writer.writerow(header)

        for frame_idx, angles in enumerate(angle_matrix):
            row = [frame_idx] + [1 if angle > threshold else 0 for angle in angles]
            writer.writerow(row)

    print(f"High angle difference joints saved to: {output_path}")


def main():
    input_csv = "table/girls_angle_offset_table.csv"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    angle_matrix = load_angle_offset_table(input_csv)
    print(f"Loaded angle matrix of shape: {angle_matrix.shape}")

    # Save high angle difference joints
    high_angle_diff_path =  "table/girls_high_angle_diff_joints.csv"
    save_high_angle_diff_joints(angle_matrix, threshold=0.6, output_path=high_angle_diff_path)

    # Other analysis and plotting (already implemented)
    plot_frame_average_difference(angle_matrix, save_path=os.path.join(output_dir, "1frame_average_difference.png"))
    plot_joint_average_difference(angle_matrix, save_path=os.path.join(output_dir, "1joint_average_difference.png"))
    cluster_frames_by_angles(angle_matrix, n_clusters=3, save_path=os.path.join(output_dir, "1frame_clustering.png"))
    dynamic_time_warping(angle_matrix)

    print("Analysis complete. Outputs saved in 'output' folder.")


if __name__ == "__main__":
    main()
