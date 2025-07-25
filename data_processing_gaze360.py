import numpy as np
import scipy.io as sio
import cv2 
import os
import sys
sys.path.append("../core/")
# import data_processing_core as dpc

root = r"E:\Datasets\gaze360_dataset_htrht37t43t9723kdfnJKhf_v2"
out_root = r"E:\Datasets\Gaze360"

def ImageProcessing_Gaze360():
    msg = sio.loadmat(os.path.join(root, "metadata.mat"))

    recordings = msg["recordings"]
    gazes = msg["gaze_dir"]
    head_bbox = msg["person_head_bbox"]
    face_bbox = msg["person_face_bbox"]
    lefteye_bbox = msg["person_eye_left_bbox"]
    righteye_bbox = msg["person_eye_right_bbox"]
    splits = msg["splits"]

    split_index = msg["split"]
    recording_index = msg["recording"]
    person_index = msg["person_identity"]
    frame_index = msg["frame"]

    total_num = recording_index.shape[1]
    outfiles = []

    # 构建保存图像和标签的文件夹
    if not os.path.exists(os.path.join(out_root, "Label")):
        os.makedirs(os.path.join(out_root, "Label"))

    for i in range(4):
        if not os.path.exists(os.path.join(out_root, "Image", splits[0, i][0])):
            os.makedirs(os.path.join(out_root, "Image", splits[0, i][0], "Left"))
            os.makedirs(os.path.join(out_root, "Image", splits[0, i][0], "Right"))
            os.makedirs(os.path.join(out_root, "Image", splits[0, i][0], "Face"))

        outfiles.append(open(os.path.join(out_root, "Label", f"{splits[0, i][0]}.label"), 'w'))
        outfiles[i].write("Face Left Right Origin 3DGaze 2DGaze\n")

    # 处理每张图像
    for i in range(total_num):
        # 构建图像完整路径
        im_path = os.path.join(root, "imgs",
                               recordings[0, recording_index[0, i]][0],
                               "head", '%06d' % person_index[0, i],
                               '%06d.jpg' % frame_index[0, i]
                               )

        # 更新进度条
        progress_bar = "".join(["\033[41m%s\033[0m" % '   '] * int(i / total_num * 20))
        progress_bar = f"\r{progress_bar} {i}|{total_num}"
        print(progress_bar, end="", flush=True)

        # 跳过无效人脸边界框
        if (face_bbox[i] == np.array([-1, -1, -1, -1])).all():
            continue

        category = splits[0, split_index[0, i]][0]
        gaze = gazes[i]

        # 读取图像并检查是否加载成功
        img = cv2.imread(im_path)
        if img is None:
            print(f"\n警告: 无法加载图像 {im_path}，已跳过")
            continue

        try:
            # 裁剪人脸和眼睛区域
            face = CropFaceImg(img, head_bbox[i], face_bbox[i])
            lefteye = CropEyeImg(img, head_bbox[i], lefteye_bbox[i])
            righteye = CropEyeImg(img, head_bbox[i], righteye_bbox[i])

            # 保存裁剪后的图像
            cv2.imwrite(os.path.join(out_root, "Image", category, "Face", f"{i + 1}.jpg"), face)
            cv2.imwrite(os.path.join(out_root, "Image", category, "Left", f"{i + 1}.jpg"), lefteye)
            cv2.imwrite(os.path.join(out_root, "Image", category, "Right", f"{i + 1}.jpg"), righteye)

            # 计算2D注视点
            gaze2d = GazeTo2d(gaze)

            # 构建保存路径和标签
            save_name_face = os.path.join(category, "Face", f"{i + 1}.jpg")
            save_name_left = os.path.join(category, "Left", f"{i + 1}.jpg")
            save_name_right = os.path.join(category, "Right", f"{i + 1}.jpg")

            save_origin = os.path.join(recordings[0, recording_index[0, i]][0],
                                       "head", "%06d" % person_index[0, i], "%06d.jpg" % frame_index[0, i])

            save_gaze = ",".join(gaze.astype("str"))
            save_gaze2d = ",".join(gaze2d.astype("str"))

            save_str = " ".join([save_name_face, save_name_left, save_name_right, save_origin, save_gaze, save_gaze2d])
            outfiles[split_index[0, i]].write(save_str + "\n")

        except Exception as e:
            print(f"\n处理图像 {im_path} 时出错: {str(e)}，已跳过")
            continue

    # 关闭所有标签文件
    for file in outfiles:
        file.close()
    print("\n处理完成")

def GazeTo2d(gaze):
  yaw = np.arctan2(gaze[0], -gaze[2])
  pitch = np.arcsin(gaze[1])
  return np.array([yaw, pitch])

def CropFaceImg(img, head_bbox, cropped_bbox):
    bbox =np.array([ (cropped_bbox[0] - head_bbox[0])/head_bbox[2],
              (cropped_bbox[1] - head_bbox[1])/head_bbox[3],
              cropped_bbox[2] / head_bbox[2],
              cropped_bbox[3] / head_bbox[3]])

    size = np.array([img.shape[1], img.shape[0]])

    bbox_pixel = np.concatenate([bbox[:2] * size, bbox[2:] * size]).astype("int")

    # Find the image center and crop head images with length = max(weight, height)
    center = np.array([bbox_pixel[0]+bbox_pixel[2]//2, bbox_pixel[1]+bbox_pixel[3]//2])

    length = int(max(bbox_pixel[2], bbox_pixel[3])/2) 

    center[0] = max(center[0], length)
    center[1] = max(center[1], length)

    result = img[(center[1] - length) : (center[1] + length),
                (center[0] - length) : (center[0] + length)] 

    result = cv2.resize(result, (224, 224))
    return result

def CropEyeImg(img, head_bbox, cropped_bbox):
    bbox =np.array([ (cropped_bbox[0] - head_bbox[0])/head_bbox[2],
              (cropped_bbox[1] - head_bbox[1])/head_bbox[3],
              cropped_bbox[2] / head_bbox[2],
              cropped_bbox[3] / head_bbox[3]])

    size = np.array([img.shape[1], img.shape[0]])

    bbox_pixel = np.concatenate([bbox[:2] * size, bbox[2:] * size]).astype("int")

    center = np.array([bbox_pixel[0]+bbox_pixel[2]//2, bbox_pixel[1]+bbox_pixel[3]//2])
    height = bbox_pixel[3]/36
    weight = bbox_pixel[2]/60
    ratio = max(height, weight) 

    size = np.array([ratio*30, ratio*18]).astype("int")

    center[0] = max(center[0], size[0])
    center[1] = max(center[1], size[1])


    result = img[(center[1] - size[1]): (center[1] + size[1]),
                (center[0] - size[0]): (center[0] + size[0])]

    result = cv2.resize(result, (60, 36)) 
    return result

if __name__ == "__main__":
    ImageProcessing_Gaze360()
