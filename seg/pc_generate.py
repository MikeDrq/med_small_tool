import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pickle

# 读取aclnet_test文件夹中的图片,将其中的fmask和tmask合并成一张图片,大小resize为512*512
def merge_f_t_mask():
    for patient_path in os.listdir('aclnet_test'):
        img_path = os.path.join('aclnet_test', patient_path)
        for image in os.listdir(img_path):
            if image.endswith('fmask.png'):
                serialnum = image.split('.')[0]
                fmask = cv2.imread(os.path.join(img_path, image), cv2.IMREAD_GRAYSCALE)
                tmask = cv2.imread(os.path.join(img_path, image.replace('f', 't')), cv2.IMREAD_GRAYSCALE)
            mask = fmask + tmask
            mask = cv2.resize(mask, (512, 512))
            plt.imsave(os.path.join(img_path, serialnum + '_mask.png'), mask, cmap='gray')
        print (img_path, 'done------------')


# 读取aclnet_test文件夹中的图片，获取其中的mask.png，调取cv2的findContours方法，获取mask的轮廓点集并保存
def get_contours():
    contour_list = {}
    for patient_path in os.listdir('aclnet_test_1'):
        contour_list[patient_path] = []
        img_path = os.path.join('aclnet_test_1', patient_path)
        for image in os.listdir(img_path):
            if image.endswith('_mask.png'):
                serialnum = image.split('_')[0]
                mask = cv2.imread(os.path.join(img_path, image), cv2.IMREAD_GRAYSCALE)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # 转为nparray, 并以[point[0][0], point[0][1], serialnum]的形式存储
                for contour in contours:
                    contour = contour.squeeze()
                    contour_wz = np.concatenate((contour, np.ones((contour.shape[0], 1)) * int(serialnum)), axis=1)
                    contour_list[patient_path].append(contour_wz)

        print (img_path, 'contour done------------')
    pickle.dump(contour_list, open('contour_list_1.pkl', 'wb'))
    return contour_list

def transtoxyz(contour, ori_pid, serial, argsinfo):
    newxyz = []
    contour = contour.squeeze()
    pid = ori_pid.split('_')[1]
    try:
        spacing = argsinfo[pid][f'{ori_pid}_{serial}']['spacing']
        orient = argsinfo[pid][f'{ori_pid}_{serial}']['orientation']
        position = argsinfo[pid][f'{ori_pid}_{serial}']['position']
        zlocation = argsinfo[pid][f'{ori_pid}_{serial}']['slicelocation']
    except:
        print('error:', ori_pid, serial)
        return
    rowvector = np.multiply(orient[:3], spacing)
    colvector = np.multiply(orient[3:], spacing)
    zvector = np.cross(rowvector, colvector)
    for point in contour:
        xyz = np.array([point[0], point[1], zlocation, 1])
        # 仿射矩阵
        allvector = np.vstack((np.append(rowvector, 0), np.append(colvector, 0), np.append(zvector, 0), np.append(position, 1))).transpose()
        realxyz = np.dot(allvector, xyz)
        newxyz.append(realxyz)
    return np.array(newxyz)

# 根据z轴的范围，将serialnum归一化至0-224
def get_contours_znor(argsinfo):
    contour_list = {}
    for patient_path in os.listdir('/data2/lc/dataset_data2/segmask/source_test'):
        contour_list[patient_path] = []
        img_path = os.path.join('/data2/lc/dataset_data2/segmask/source_test', patient_path)
        for image in os.listdir(img_path):
            if image.endswith('_segmask.png'):
                serialnum = image.split('_')[0]
                orient = image.split('_')[1]
                ori_pid = f'{orient}_{patient_path}'
                mask = cv2.imread(os.path.join(img_path, image), cv2.IMREAD_GRAYSCALE)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # 转为nparray, 并以[point[0][0], point[0][1], serialnum]的形式存储
                for contour in contours:
                    # 小于20个点的轮廓不要
                    if contour.shape[0] < 20:
                        continue
                    newxyz = transtoxyz(contour, ori_pid, serialnum, argsinfo)
                    if newxyz is not None:
                        contour_list[patient_path].append(newxyz)
                        

        print (img_path, 'contour done------------')
    pickle.dump(contour_list, open('autoseg_test_contour_list.pkl', 'wb'))
    return contour_list

# 计算中心点和平均距离
def calculate_center_and_distance(point_cloud):
    center = point_cloud.get_center()
    distance = 0
    for point in point_cloud.points:
        distance += np.linalg.norm(point - center)
    distance /= len(point_cloud.points)
    return center, distance


def generate_combined_point_cloud(contour_list):
    combined_point_cloud = {}
    for patient in contour_list:
        patient_id = patient
        if patient_id in combined_point_cloud.keys():
            continue
        combined_point_cloud[patient_id] = o3d.geometry.PointCloud()
        for contour in contour_list[patient]:
            combined_point_cloud[patient_id].points.extend(o3d.utility.Vector3dVector(contour[:, :3]))
    pc_center = {}
    pc_distance = {}
    for patient_id in combined_point_cloud.keys():
        pc_center[patient_id], pc_distance[patient_id] = calculate_center_and_distance(combined_point_cloud[patient_id])
        combined_point_cloud[patient_id].translate(-pc_center[patient_id])
        combined_point_cloud[patient_id].scale(1/pc_distance[patient_id], center=[0, 0, 0])
        o3d.io.write_point_cloud(os.path.join('/data2/lc/dataset_data2/segmask','combined_aligned_with_normal', patient_id + '.ply'), combined_point_cloud[patient_id])
        print(patient_id, 'PC done')

# 根据countour_list的点集，对于每个病例分别生成sag和cor的点云
def generate_point_cloud(contour_list):
    sag_point_cloud = {}
    cor_point_cloud = {}
    for patient in contour_list:
        patient_id = patient.split('_')[1]
        if patient_id in sag_point_cloud.keys():
            continue
        sag_point_cloud[patient_id] = o3d.geometry.PointCloud()
        cor_point_cloud[patient_id] = o3d.geometry.PointCloud()
        for contour in contour_list[f'sag_{patient_id}']:
            sag_point_cloud[patient_id].points.extend(o3d.utility.Vector3dVector(contour[:, :3]))
        for contour in contour_list[f'cor_{patient_id}']:
            cor_point_cloud[patient_id].points.extend(o3d.utility.Vector3dVector(contour[:, :3]))

    return sag_point_cloud, cor_point_cloud
        
def normalization(sag_point_cloud, cor_point_cloud):
    # 计算sagittal和coronal两个方向的点云的中心点，进行对齐
    sag_center = {}
    cor_center = {}
    pidlist = []
    for patient_id in sag_point_cloud.keys():
        sag_center[patient_id], sag_distance = calculate_center_and_distance(sag_point_cloud[patient_id])
        # 去掉点云中的异常点，即离中心点距离大于3倍平均距离的点
        flag = 0
        # 有异常点的病例忽略
        for point in sag_point_cloud[patient_id].points:
            if np.linalg.norm(point - sag_center[patient_id]) > 3 * sag_distance:
                pidlist.append(patient_id)
                flag = 1
                break
        if flag == 1:
            continue

        # 如果cor_point_cloud中没有对应的patient_id，说明该病人没有coronal方向的点云，不进行对齐
        if patient_id in cor_point_cloud.keys() and len(cor_point_cloud[patient_id].points) > 0:
            # 计算中心及平均距离
            cor_center[patient_id], cor_distance = calculate_center_and_distance(cor_point_cloud[patient_id])
            # 去掉点云中的异常点，即离中心点距离大于3倍平均距离的点
            flag = 0
            # 有异常点的病例忽略
            for point in cor_point_cloud[patient_id].points:
                if np.linalg.norm(point - cor_center[patient_id]) > 3 * cor_distance:
                    pidlist.append(patient_id)
                    flag = 1
                    break
            if flag == 1:
                continue

            # 对齐
            sag_point_cloud[patient_id].translate(-sag_center[patient_id])
            cor_point_cloud[patient_id].translate(-cor_center[patient_id])

            # 点云normalization
            sag_point_cloud[patient_id].scale(1/sag_distance, center=[0, 0, 0])
            cor_point_cloud[patient_id].scale(1/cor_distance, center=[0, 0, 0])
            # 保存对齐且归一化后的点云
            o3d.io.write_point_cloud(os.path.join('/data2/lc/dataset_data2/segmask/source_test','sag_aligned_with_normal', patient_id + '.ply'), sag_point_cloud[patient_id])
            o3d.io.write_point_cloud(os.path.join('/data2/lc/dataset_data2/segmask/source_test','cor_aligned_with_normal', patient_id + '.ply'), cor_point_cloud[patient_id])

            # 组合点云
            sag_point_cloud[patient_id].points.extend(cor_point_cloud[patient_id].points)
            o3d.io.write_point_cloud(os.path.join('/data2/lc/dataset_data2/segmask/source_test','combined_aligned_with_normal', patient_id + '.ply'), sag_point_cloud[patient_id])
        else:
            sag_point_cloud[patient_id].translate(-sag_center[patient_id])
            sag_point_cloud[patient_id].scale(1/sag_distance, center=[0, 0, 0])
            o3d.io.write_point_cloud(os.path.join('/data2/lc/dataset_data2/segmask/source_test','sag_aligned_with_normal', patient_id + '.ply'), sag_point_cloud[patient_id])

        print(patient_id, 'PC done')

# 采样点云至指定数量
def sample_point_cloud(ply_path,  num_points=2048):
    ply_cloud = o3d.io.read_point_cloud(ply_path)
    point_num = np.asarray(ply_cloud.points).shape[0]

    if point_num < num_points:
        sampled_cloud = ply_cloud.uniform_down_sample(1)
        while np.asarray(sampled_cloud.points).shape[0] < num_points:
            sampled_cloud += ply_cloud.uniform_down_sample(1)
        sampled_cloud = np.asarray(sampled_cloud.points)[:num_points]
        # 转回点云
        sampled_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sampled_cloud))
    else:
        sampled_cloud = ply_cloud.farthest_point_down_sample(num_points)
    o3d.io.write_point_cloud(ply_path.replace(".ply", "_" + str(num_points) + ".ply"), sampled_cloud)

if __name__ == '__main__':
    # merge_f_t_mask()
    # 生成点云列表
    # argsinfo = pickle.load(open('test_args.pkl', 'rb'))
    # contour_list = get_contours_znor(argsinfo)

    # # 从点集列表采样真实点云
    # contour_list = pickle.load(open('autoseg_test_contour_list.pkl', 'rb'))
    # sag_point_cloud, cor_point_cloud = generate_point_cloud(contour_list)
    # normalization(sag_point_cloud, cor_point_cloud)

    # 从点集列表采样真实点云
    contour_list = pickle.load(open('autoseg_test_contour_list.pkl', 'rb'))
    combined_pc = generate_combined_point_cloud(contour_list)

    # 采样点云
    for patient in os.listdir('/data2/lc/dataset_data2/segmask/combined_aligned_with_normal'):
        sample_point_cloud(os.path.join('/data2/lc/dataset_data2/segmask/combined_aligned_with_normal', patient), 2048)