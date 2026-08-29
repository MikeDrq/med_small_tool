import os
import time
import pickle
import cv2
import numpy as np
import torch
from torchvision.transforms import transforms
import shutil

from src import u2net_full


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def pred_femur():

    weights_path = "/data2/home_backup/lc/code/BYSY/bone_for_ligament/segmentation/Seg/save_weights/model_patella_202500625.pth"

    threshold = 0.5

    # assert os.path.exists(img_path), f"image file {img_path} dose not exists."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(320),
        # transforms.Normalize(mean=(0.485, 0.456, 0.406),
        #                      std=(0.229, 0.224, 0.225))
    ])

    # 加载模型
    model_sag = u2net_full(2)
    model_sag.to(device)
    model_sag = torch.nn.DataParallel(model_sag)
    weights = torch.load(weights_path, map_location='cpu')
    if "model" in weights:
        model_sag.load_state_dict(weights["model"])
    else:
        model_sag.load_state_dict(weights)

    model_sag.eval()

    # 逐个文件夹遍历
    srcpath = '/data2/home_backup/lc/code/BYSY/bone_for_ligament/segmentation/Seg/src/TestImage'
    dstpath = '/data2/home_backup/lc/code/BYSY/bone_for_ligament/segmentation/Seg/src/TestMask'
    for patient_path in os.listdir(srcpath):
        img_path = os.path.join(srcpath, patient_path)
        mask_path = os.path.join(dstpath, patient_path)
        for imgname in os.listdir(img_path): 
            if imgname.endswith(".png") and 'mask' not in imgname:
                # ！！！ serialnum自适应
                origin_img = cv2.cvtColor(cv2.imread(os.path.join(img_path, imgname), flags=cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                model = model_sag
                h, w = origin_img.shape[:2]
                img = data_transform(origin_img)
                img = torch.unsqueeze(img, 0).to(device)  # [C, H, W] -> [1, C, H, W]

                with torch.no_grad():
                    # init model
                    img_height, img_width = img.shape[-2:]
                    init_img = torch.zeros((1, 3, img_height, img_width), device=device)
                    model(init_img)

                    pred = model(img)
                    pred = torch.argmax(pred, dim=1)
                    pred = torch.squeeze(pred).to("cpu").numpy().astype(float)  # [1, 1, H, W] -> [H, W]

                    pred = cv2.resize(pred, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
                    pred_mask = np.where(pred > threshold, 255, 0)
                    save_path = mask_path
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    cv2.imwrite(os.path.join(save_path, imgname.replace('image', 'pred_mask')), pred_mask)
        print(img_path, 'done------------')

def pred_sag_cor():

    femur_weights_path = "/data2/home_backup/lc/code/BYSY/bone_for_ligament/segmentation/Seg/save_weights/model_femur_202500625.pth"
    patella_weights_path = "/data2/home_backup/lc/code/BYSY/bone_for_ligament/segmentation/Seg/save_weights/model_patella_202500625.pth"

    threshold = 0.5


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(320),
        # transforms.Normalize(mean=(0.485, 0.456, 0.406),
        #                      std=(0.229, 0.224, 0.225))
    ])

    # 加载模型
    model_sag = u2net_full(3)
    model_sag.to(device)
    model_sag = torch.nn.DataParallel(model_sag)
    weights = torch.load(cor_weights_path, map_location=device)
    if "model" in weights:
        model_sag.load_state_dict(weights["model"])
    else:
        model_sag.load_state_dict(weights)
    model_cor = u2net_full(3)
    model_cor.to(device)
    model_cor = torch.nn.DataParallel(model_cor)
    weights = torch.load(sag_weights_path, map_location=device)
    if "model" in weights:
        model_cor.load_state_dict(weights["model"])
    else:
        model_cor.load_state_dict(weights)

    model_sag.eval()
    model_cor.eval()

    # 逐个文件夹遍历
    srcpath = '/data1/liuchao/dataset/Point_Circle_All/Mask'

    # 读取原始带mask标签的训练集图片
    test_pidpath = '/home/lc/code/BYSY/bone_for_ligament/3D/v0.3_D3D_PCT/utils/label_test.txt'
    test_pid_list = []
    with open(test_pidpath, 'r') as f:
        for line in f:
            test_pid_list.append(line.strip().split(' ')[0])
    diease_list = ['532']
    for diease_id in os.listdir(srcpath):
        diease_pid = diease_id.split('_')[-2]
        if diease_pid not in diease_list:
            continue
        patient_src_path = os.path.join(srcpath, diease_id)
        for patient_path in os.listdir(patient_src_path):
            # if patient_path != '01200302210008':
            #     continue
            img_path = os.path.join(patient_src_path, patient_path)
            # mask_path = os.path.join('/data1/liuchao/dataset/evaluate_15000_multi/vis_sample/sin_predict', diease_id)
            mask_path = '/data1/liuchao/dataset/evaluate_15000/532'
            res = sag_cor_mask(img_path, mask_path, threshold, data_transform, device, model_cor, model_sag)
            if res:
                print(img_path, 'done------------')
            else:
                print(img_path, 'skip------------')


def sag_cor_mask(img_path, mask_path, threshold=0.5, data_transform=None, device=None, model_sag=None, model_cor=None, ori=None):
    # 逐个读取图片
    save_path = os.path.join(mask_path, img_path.split('/')[-1])
    if os.path.exists(save_path):
        if len(os.listdir(save_path)) > 10:
            print('save path exists:', save_path, 'skip')
            return False
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        print('save path exists:', save_path, 'skip')
        return False
    for imgname in os.listdir(img_path):
        if imgname.endswith(".png") and 'mask' not in imgname:
            serialnum = imgname.split('_')[-1].split('.')[0]
            origin_img = cv2.cvtColor(cv2.imread(os.path.join(img_path, imgname), flags=cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            if '冠' in imgname:
                model = model_cor
            elif '矢' in imgname:
                model = model_sag
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
                pred = torch.squeeze(pred).to("cpu").numpy()  # [1, 1, H, W] -> [H, W]

                # for binary-class
                # pred = torch.squeeze(pred).to("cpu").numpy()  # [1, 1, H, W] -> [H, W]
                # pred = np.where(pred > threshold, 255, 0)

                # 转为uint8
                pred = pred.astype(np.uint8)

                pred = cv2.resize(pred, dsize=(w, h), interpolation=cv2.INTER_LINEAR)

                # 大于0的值设为255，小于等于1的值设为0
                pred = np.where(pred > 0, 255, 0)

                # pred_mask = np.where(pred > threshold, 255, 0)
                if '冠' in imgname:
                    cv2.imwrite(os.path.join(save_path, serialnum + '_cor_mask.png'), pred)
                elif '矢' in imgname:
                    cv2.imwrite(os.path.join(save_path, serialnum + '_sag_mask.png'), pred)
    return True

def pred_sag_cor_source_test():

    cor_weights_path = "/home/lc/code/BYSY/bone_for_ligament/segmentation/Seg/save_weights/model_cor_multiclass_2024424_best.pth"
    sag_weights_path = "/home/lc/code/BYSY/bone_for_ligament/segmentation/Seg/save_weights/model_sag_multiclass_2024424_best.pth"


    # assert os.path.exists(img_path), f"image file {img_path} dose not exists."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(320),
        # transforms.Normalize(mean=(0.485, 0.456, 0.406),
        #                      std=(0.229, 0.224, 0.225))
    ])

    # 加载模型
    model_sag = u2net_full(3)
    model_sag.to(device)
    model_sag = torch.nn.DataParallel(model_sag)
    weights = torch.load(cor_weights_path, map_location=device)
    if "model" in weights:
        model_sag.load_state_dict(weights["model"])
    else:
        model_sag.load_state_dict(weights)
    model_cor = u2net_full(3)
    model_cor.to(device)
    model_cor = torch.nn.DataParallel(model_cor)
    weights = torch.load(sag_weights_path, map_location=device)
    if "model" in weights:
        model_cor.load_state_dict(weights["model"])
    else:
        model_cor.load_state_dict(weights)

    model_sag.eval()
    model_cor.eval()

    # 逐个文件夹遍历
    srcpath = '/data1/liuchao/dataset/Point_Circle_All/Mask'

    # 读取原始带mask标签的训练集图片
    test_pidpath = '/home/lc/code/BYSY/bone_for_ligament/3D/v0.3_D3D_PCT/utils/label_test.txt'
    test_pid_list = []
    with open(test_pidpath, 'r') as f:
        for line in f:
            test_pid_list.append(line.strip().split(' ')[0])
    for diease_id in os.listdir(srcpath):
        diease_pid = diease_id.split('_')[-2]
        if diease_pid not in test_pid_list:
            continue
        patient_src_path = os.path.join(srcpath, diease_id)
        patient_save_path = os.path.join('/data2/lc/dataset_data2/segmask/source_test', diease_pid)
        if not os.path.exists(patient_save_path):
            os.makedirs(patient_save_path)
        for img_or_mask in os.listdir(patient_src_path):
            # mask
            if '_mask' in img_or_mask and 'femur' not in img_or_mask and 'tibia' not in img_or_mask:
                source_mask_path = os.path.join(patient_src_path, img_or_mask)
                shutil.copy(source_mask_path, patient_save_path)
                if 'cor' in diease_id:
                    os.rename(os.path.join(patient_save_path, img_or_mask), os.path.join(patient_save_path, img_or_mask.replace('_mask', '_cor_mask')))
                elif 'sag' in diease_id:
                    os.rename(os.path.join(patient_save_path, img_or_mask), os.path.join(patient_save_path, img_or_mask.replace('_mask', '_sag_mask')))
            # img
            if '._image' in img_or_mask:
                img_path = os.path.join(patient_src_path, img_or_mask)
                shutil.copy(img_path, patient_save_path)
                if 'cor' in diease_id:
                    os.rename(os.path.join(patient_save_path, img_or_mask), os.path.join(patient_save_path, img_or_mask.replace('._image', '_cor_image')))
                elif 'sag' in diease_id:
                    os.rename(os.path.join(patient_save_path, img_or_mask), os.path.join(patient_save_path, img_or_mask.replace('._image', '_sag_image')))
                serialnum = img_or_mask.split('_')[0]
                origin_img = cv2.cvtColor(cv2.imread(img_path, flags=cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                if 'cor' in diease_id:
                    model = model_cor
                elif 'sag' in diease_id:
                    model = model_sag
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
                    pred = torch.squeeze(pred).to("cpu").numpy()  # [1, 1, H, W] -> [H, W]

                    # for binary-class
                    # pred = torch.squeeze(pred).to("cpu").numpy()  # [1, 1, H, W] -> [H, W]
                    # pred = np.where(pred > threshold, 255, 0)

                    # 转为uint8
                    pred = pred.astype(np.uint8)

                    pred = cv2.resize(pred, dsize=(w, h), interpolation=cv2.INTER_LINEAR)

                    # 大于0的值设为255，小于等于1的值设为0
                    pred = np.where(pred > 0, 255, 0)

                    # pred_mask = np.where(pred > threshold, 255, 0)
                    if 'cor' in diease_id:
                        cv2.imwrite(os.path.join(patient_save_path, serialnum + '_cor_segmask.png'), pred)
                    elif 'sag' in diease_id:
                        cv2.imwrite(os.path.join(patient_save_path, serialnum + '_sag_segmask.png'), pred)

                    print(img_path, 'done------------')


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

def pred_mask(img_path, mask_path, threshold=0.5, data_transform=None, device=None, model=None, femur=None):
    # 逐个读取图片
    for imgname in os.listdir(img_path):
        if imgname.endswith(".png") and 'mask' not in imgname:
            serialnum = imgname.split('_')[0]
            origin_img = cv2.cvtColor(cv2.imread(os.path.join(img_path, imgname), flags=cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

            h, w = origin_img.shape[:2]
            img = data_transform(origin_img)
            img = torch.unsqueeze(img, 0).to(device)  # [C, H, W] -> [1, C, H, W]

            with torch.no_grad():
                # init model
                img_height, img_width = img.shape[-2:]
                init_img = torch.zeros((1, 3, img_height, img_width), device=device)
                model(init_img)

                pred = model(img)
                pred = torch.squeeze(pred).to("cpu").numpy()  # [1, 1, H, W] -> [H, W]

                pred = cv2.resize(pred, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
                pred_mask = np.where(pred > threshold, 255, 0)
                save_path = mask_path
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                if femur == 'femur':
                    cv2.imwrite(os.path.join(save_path, serialnum + '_fmask.png'), pred_mask)
                elif femur == 'tibia':
                    cv2.imwrite(os.path.join(save_path, serialnum + '_tmask.png'), pred_mask)

# 20250119 用于生成左右腿对应的mask
def pred_left_right(source_path):

    cor_weights_path = "/home/lc/code/BYSY/bone_for_ligament/segmentation/Seg/save_weights/model_cor_multiclass_2024424_best.pth"
    sag_weights_path = "/home/lc/code/BYSY/bone_for_ligament/segmentation/Seg/save_weights/model_sag_multiclass_2024424_best.pth"


    # assert os.path.exists(img_path), f"image file {img_path} dose not exists."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(320),
        # transforms.Normalize(mean=(0.485, 0.456, 0.406),
        #                      std=(0.229, 0.224, 0.225))
    ])

    # 加载模型
    model_sag = u2net_full(3)
    model_sag.to(device)
    model_sag = torch.nn.DataParallel(model_sag)
    weights = torch.load(cor_weights_path, map_location=device)
    if "model" in weights:
        model_sag.load_state_dict(weights["model"])
    else:
        model_sag.load_state_dict(weights)
    model_cor = u2net_full(3)
    model_cor.to(device)
    model_cor = torch.nn.DataParallel(model_cor)
    weights = torch.load(sag_weights_path, map_location=device)
    if "model" in weights:
        model_cor.load_state_dict(weights["model"])
    else:
        model_cor.load_state_dict(weights)

    model_sag.eval()
    model_cor.eval()

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
                if 'cor' in img_or_mask:
                    model = model_cor
                elif 'sag' in img_or_mask:
                    model = model_sag
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
                    pred = torch.squeeze(pred).to("cpu").numpy()  # [1, 1, H, W] -> [H, W]

                    # for binary-class
                    # pred = torch.squeeze(pred).to("cpu").numpy()  # [1, 1, H, W] -> [H, W]
                    # pred = np.where(pred > threshold, 255, 0)

                    # 转为uint8
                    pred = pred.astype(np.uint8)

                    pred = cv2.resize(pred, dsize=(w, h), interpolation=cv2.INTER_LINEAR)

                    # 大于0的值设为255，小于等于1的值设为0
                    pred = np.where(pred > 0, 255, 0)

                    # pred_mask = np.where(pred > threshold, 255, 0)
                    if 'cor' in img_or_mask:
                        cv2.imwrite(img_path.replace('extracted_data', 'segmask_data'), pred)
                    elif 'sag' in img_or_mask:
                        cv2.imwrite(img_path.replace('extracted_data', 'segmask_data'), pred)

                    print(img_path, 'done------------')


def rename(path):
    for diease_id in os.listdir(path):
        patient_path = os.path.join(path, diease_id)
        for img_or_mask in os.listdir(patient_path):
            if 'png' in img_or_mask:
                img_path = os.path.join(patient_path, img_or_mask)
                # 重命名 冠改为_cor 矢改为_sag
                if '超薄' in img_or_mask:
                    os.rename(img_path, img_path.replace('超薄', 'thin_'))
                print(img_path)


if __name__ == '__main__':
    # rename('/data2/lc/dataset_data2/wang_leftright_20250104/extracted_data/doubleknee18k')
    # rename('/data2/lc/dataset_data2/wang_leftright_20250104/extracted_data/normalknee')
    # rename('/data2/lc/dataset_data2/wang_leftright_20250104/extracted_data/normalkneethin')
    # pred_left_right('/data2/lc/dataset_data2/wang_leftright_20250104/extracted_data/doubleknee18k')
    # pred_left_right('/data2/lc/dataset_data2/wang_leftright_20250104/extracted_data/normalknee')
    # pred_left_right('/data2/lc/dataset_data2/wang_leftright_20250104/extracted_data/normalkneethin')
    pred_femur()
