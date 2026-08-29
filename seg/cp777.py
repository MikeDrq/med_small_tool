import os
import shutil
import cv2
import numpy as np

src_img_path = '/home/lc/code/BYSY/bone_for_ligament/segmentation/Seg/777'
dst_img_path = '/data1/liuchao/dataset/segMask/BONE_3D/val_777_cor/image'

# 将100例777验证集的image经过处理成为24*256*256的shape后逐切片放入对应文件夹
def process(src, dst):
    cnt = 0
    for pid in os.listdir(src):
        img_list = [ i for i in os.listdir(os.path.join(src, pid)) if i.endswith('cor_image.png') ]
        img_list.sort(key=lambda x: int(x.split('_')[0]))
        lens = len(img_list)
        if lens < 13:
            continue
        img_array = []
        for img_name in img_list:
            img = cv2.imread(os.path.join(src, pid, img_name), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (256, 256))
            img_array.append(img)
        if lens < 24:
            pad = (24 - lens) // 2
            image = np.pad(img_array, ((pad, pad), (0, 0), (0, 0)), 'constant', constant_values=0)
        elif lens > 24:
            image = np.array(img_array[lens//2-12:lens//2+12])
        else:
            image = np.array(img_array)
        # 判断为24 还是 23
        if len(image) == 23:
            image = np.pad(image, ((1, 0), (0, 0), (0, 0)), 'constant', constant_values=0)
        # 逐个切片保存至对应文件夹
        dst_path = os.path.join(dst, pid)
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        for i in range(24):
            cv2.imwrite(os.path.join(dst_path, str(i)+'_image.png'), image[i])
        print(pid, 'done------------')
        cnt += 1
        if cnt == 100:
            print('process done!')
            break

    
if __name__ == '__main__':
    process(src_img_path, dst_img_path)