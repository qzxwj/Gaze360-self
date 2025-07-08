import numpy as np
import cv2
import os
from torch.utils.data import Dataset, DataLoader
import torch

def gazeto2d(gaze):
    yaw = np.arctan2(-gaze[0], -gaze[2])
    pitch = np.arcsin(-gaze[1])
    return np.array([yaw, pitch])


class loader(torch.utils.data.Dataset):
    def __init__(self, label_dir, image_root, header=True):
        self.image_root = image_root
        self.samples = []

        for fname in os.listdir(label_dir):
            if fname.endswith(".label"):
                label_path = os.path.join(label_dir, fname)
                with open(label_path) as f:
                    if header:
                        next(f)
                    for line in f:
                        self.samples.append(line.strip())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        line = self.samples[idx].strip().split(" ")

        name = line[3]             # 原始图像名
        gaze2d = line[7]           # 注视方向（2D）
        head2d = line[8]           # 头部姿态（2D）
        lefteye = line[1]          # 左眼图像相对路径
        righteye = line[2]         # 右眼图像相对路径
        face = line[0]             # 脸部图像路径（相对路径）

        # 处理 label 和 headpose
        label = np.array(gaze2d.split(",")).astype("float32")
        label = torch.from_numpy(label)

        headpose = np.array(head2d.split(",")).astype("float32")
        headpose = torch.from_numpy(headpose)

        # --- 读取 face 图像 ---
        fimg_path = os.path.join(self.image_root, face.replace("\\", "/"))
        fimg = cv2.imread(fimg_path)
        fimg = cv2.resize(fimg, (448, 448)) / 255.0
        fimg = fimg.transpose(2, 0, 1)  # (HWC) -> (CHW)
        fimg = torch.from_numpy(fimg).type(torch.FloatTensor)

        # --- 读取 left eye 图像 ---
        lefteye_path = os.path.join(self.image_root, lefteye.replace("\\", "/"))
        limg = cv2.imread(lefteye_path)
        limg = cv2.resize(limg, (60, 36)) / 255.0
        limg = limg.transpose(2, 0, 1)
        limg = torch.from_numpy(limg).type(torch.FloatTensor)

        # --- 读取 right eye 图像 ---
        righteye_path = os.path.join(self.image_root, righteye.replace("\\", "/"))
        rimg = cv2.imread(righteye_path)
        rimg = cv2.resize(rimg, (60, 36)) / 255.0
        rimg = rimg.transpose(2, 0, 1)
        rimg = torch.from_numpy(rimg).type(torch.FloatTensor)


        # 返回图像数据与标签
        data = {
            "face": fimg,
            "left": limg,
            "right": rimg,
            "head_pose": headpose,
            "name": name
        }

        return data, label


def txtload(labelpath, imagepath, batch_size, shuffle=True, num_workers=0, header=True):
    dataset = loader(labelpath, imagepath, header)
    print(f"[Read Data]: Total samples: {len(dataset)}")
    print(f"[Read Data]: Label path: {labelpath}")
    load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return load


# 可测试读取一条数据样本
if __name__ == "__main__":
    label_folder = "E:/Datasets/MPIIFaceGaze_normalized/Label"
    image_folder = "E:/Datasets/MPIIFaceGaze_normalized/Image"
    d = loader(label_folder, image_folder)
    print(f"总样本数量: {len(d)}")
    data, label = d[0]
    print("样本 keys:", data.keys())
    print("face shape:", data["face"].shape)
    print("left shape:", data["left"].shape)
    print("right shape:", data["right"].shape)
    print("label:", label)
