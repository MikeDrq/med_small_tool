import numpy as np
import cv2
import os
import shutil
import pickle



# 获取图片的像素数和轮廓数
def get_pixel_and_contour(img_path):
    img = cv2.imread(img_path, 0)
    femur_pixel_num = np.sum(img == 1)
    tibia_pixel_num = np.sum(img == 2)
    femur_img = np.where(img == 1, 1, 0)
    tibia_img = np.where(img == 2, 1, 0)
    f_contour, _ = cv2.findContours(femur_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    t_contour, _ = cv2.findContours(tibia_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    f_con_num = len(f_contour)
    t_con_num = len(t_contour)
    # 返回胫骨、股骨的像素数和轮廓数
    pixel_num = [femur_pixel_num, tibia_pixel_num]
    contour_num = [f_con_num, t_con_num]
    return pixel_num, contour_num

# 方向、轮廓数量、像素数量获取第一步的列表
def save_to_file(prev_num, pixel_num, contour_num, img_path, save_dict, img_len, orientation):
    pid_path = os.path.dirname(img_path)
    this_img_name = os.path.basename(img_path)
    # 冠位：股骨轮廓小于3，胫骨小于2，像素小于1500；矢位：股骨轮廓小于2，胫骨小于2，像素小于1500
    if (orientation == 0 and (contour_num[0] > 2 or contour_num[1] > 1)) or \
    (orientation == 1 and (contour_num[0] > 1 or contour_num[1] > 1)) or pixel_num[0] < 1500 or pixel_num[1] < 1500:
        return
    if pid_path not in save_dict:
        save_dict[pid_path] = []
    save_dict[pid_path].append(this_img_name)

def chg_img_dict(img_dict):
    # 将字典中的图片路径转换为列表
    new_dict = {}
    return new_dict


def get_all_img_dict(img_dir, orientation=0):
    img_dict = {}
    orientation_dict = {0: 'cor', 1: 'sag'}
    ori = orientation_dict[orientation]
    for pid in os.listdir(img_dir):
        # if not pid == '02180427208035':
        #     continue
        pid_path = os.path.join(img_dir, pid)
        if not os.path.isdir(pid_path):
            continue
        img_list = [i for i in os.listdir(pid_path) if ori in i]
        # 从小到大排序
        img_list.sort(key=lambda x: int(x.split('_')[0]))
        img_len = len(img_list)
        for i in range(img_len):
            img_path = os.path.join(pid_path, img_list[i])
            pixel_num, contour_num = get_pixel_and_contour(img_path)
            if i <= img_len / 2:
                prev_path = os.path.join(pid_path, img_list[i + 1])
            else:
                prev_path = os.path.join(pid_path, img_list[i - 1])
            prev_pixel_num, _ = get_pixel_and_contour(prev_path)
            save_to_file(prev_pixel_num, pixel_num, contour_num, img_path, img_dict, img_len, 0)
        # sample
        # if len(img_dict) > 30:
        #     return img_dict

def copy_by_pickle(pkl_path):
    # 根据pkl记录的信息，拷贝图片到指定目录
    save_base_dir = '/data1/liuchao/dataset/evaluate_15000_filter/6016'
    with open(pkl_path, 'rb') as f:
        save_dict = pickle.load(f)
    cnt = len(save_dict)
    ncnt = 0
    for pid_path in save_dict:
        save_dir = os.path.join(save_base_dir, pid_path.split('/')[-1])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for img_name in save_dict[pid_path]:
            img_path = os.path.join(pid_path, img_name)
            shutil.copy(img_path, save_dir)
        ncnt += 1
        print(ncnt, '/', cnt)


if __name__ == '__main__':
    # 生成pkl
    img_dir = '/data1/liuchao/dataset/evaluate_15000/6016'
    save_dict = get_all_img_dict(img_dir, 0)
    with open('/home/lc/code/BYSY/bone_for_ligament/segmentation/Seg/evaluate_pkl/6016_cor_img.pkl', 'wb') as f:
        pickle.dump(save_dict, f)
    print('cor done')
    save_dict = get_all_img_dict(img_dir, 1)
    with open('/home/lc/code/BYSY/bone_for_ligament/segmentation/Seg/evaluate_pkl/6016_sag_img.pkl', 'wb') as f:
        pickle.dump(save_dict, f)
    print('sag done')

    # 根据pkl拷贝图片
    copy_by_pickle('/home/lc/code/BYSY/bone_for_ligament/segmentation/Seg/evaluate_pkl/6016_cor_img.pkl')
    copy_by_pickle('/home/lc/code/BYSY/bone_for_ligament/segmentation/Seg/evaluate_pkl/6016_sag_img.pkl')