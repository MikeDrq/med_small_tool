import os
import time
import cv2
import numpy as np
import torch
from torchvision.transforms import transforms

from src import u2net_full

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


# 20250119 用于生成左右腿对应的mask
def pred_bone(source_path):

    weights_path = "/data2/home_backup/lc/code/BYSY/bone_for_ligament/segmentation/Seg/save_weights/model_femur_202500625.pth"

    #assert os.path.exists(img_path), f"image file {img_path} dose not exists."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(320),
    ])

    # 加载模型
    model = u2net_full(2)
    model.to(device)
    model = torch.nn.DataParallel(model)
    weights = torch.load(weights_path, map_location=device)
    if "model" in weights:
        model.load_state_dict(weights["model"])
    else:
        model.load_state_dict(weights)

    model.eval()

    # 逐个文件夹遍历
    srcpath = source_path

    # 读取原始带mask标签的训练集图片
    for diease_id in os.listdir(srcpath):
        patient_src_path = os.path.join(srcpath, diease_id)
        patient_save_path = os.path.join(srcpath.replace('extracted_data', 'segmask_data'), diease_id)
        if not os.path.exists(patient_save_path):
            os.makedirs(patient_save_path)
        for img_or_mask in os.listdir(patient_src_path):
            # img
            if 'png' in img_or_mask:
                img_path = os.path.join(patient_src_path, img_or_mask)
                origin_img = cv2.cvtColor(cv2.imread(img_path, flags=cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                h, w = origin_img.shape[:2]
                img = data_transform(origin_img)
                img = torch.unsqueeze(img, 0).to(device)  # [C, H, W] -> [1, C, H, W]
                with torch.no_grad():
                    # init model
                    img_height, img_width = img.shape[-2:]
                    init_img = torch.zeros((1, 3, img_height, img_width), device=device)
                    model(init_img)

                    pred = model(img)

                    # for multi-class
                    pred = torch.argmax(pred, dim=1)
                    pred = torch.squeeze(pred).to("cpu").numpy().astype(float)  # [1, 1, H, W] -> [H, W]


                    pred = cv2.resize(pred, dsize=(w, h), interpolation=cv2.INTER_LINEAR)

                    # 大于0的值设为255，小于等于1的值设为0
                    pred = np.where(pred > 0.5, 255, 0)

                    cv2.imwrite(os.path.join(patient_save_path, img_or_mask.replace('image', 'femur')), pred)

                    print(img_path, 'done------------')


if __name__ == '__main__':
    # 存放待分割图像的目录
    pred_bone('/data2/home_backup/lc/code/BYSY/bone_for_ligament/segmentation/Seg/src/test/extracted_data')
