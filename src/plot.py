#!/usr/bin/env python3
import csv
import matplotlib.pyplot as plt

def read_csv(filename):
    timestamps = []
    # 12関節分のリストを用意
    joint_data = [[] for _ in range(12)]
    
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # ヘッダー行をスキップ
        for row in reader:
            try:
                # 1列目がタイムスタンプ（秒）
                timestamps.append(float(row[0]))
                # 以降12列が各関節の値
                for i in range(12):
                    joint_data[i].append(float(row[i+1]))
            except Exception as e:
                print(f"Error processing row {row}: {e}")
    return timestamps, joint_data

def plot_data(timestamps, joint_data):
    plt.figure(figsize=(12, 8))
    # 各関節の時系列データをプロット
    for i in range(12):
        plt.plot(timestamps, joint_data[i], label=f"Joint {i}")
    
    plt.xlabel("Time (s)")
    plt.ylabel("Target DOF Position")
    plt.title("Time Series of Target DOF Positions")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    filename = "target_dof_pos_log.csv"
    timestamps, joint_data = read_csv(filename)
    
    if not timestamps:
        print("CSVファイルにデータが見つかりませんでした。")
        return
    
    plot_data(timestamps, joint_data)

if __name__ == "__main__":
    main()
