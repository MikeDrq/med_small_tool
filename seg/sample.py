import os
import open3d as o3d
import numpy as np

# 定义输入和输出目录
ply_dir = "aclnettest_ply/sag_aligned_with_normal"
cor_dir = "aclnettest_ply/cor_aligned_with_normal"
output_dir = "aclnettest_ply/sampled_pointclouds_from_test"

# 获取ply目录下的所有文件
ply_files = os.listdir(ply_dir)

# 遍历每个文件
for ply_file in ply_files:
    # 构建输入和输出文件路径
    ply_path = os.path.join(ply_dir, ply_file)
    cor_path = os.path.join(cor_dir, ply_file)
    if not os.path.exists(cor_path):
        continue
    output_path = os.path.join(output_dir, ply_file)

    # 读取ply文件
    ply_cloud = o3d.io.read_point_cloud(ply_path)
    cor_cloud = o3d.io.read_point_cloud(cor_path)

    # 点云长度为0则跳过
    if len(ply_cloud.points) == 0 or len(cor_cloud.points) == 0:
        continue

    # 合并点云
    ply_cloud += cor_cloud

    point_num = np.asarray(ply_cloud.points).shape[0]

    # 分别采样1024、2048、4096、8192、16384个点，用farthest down sample，若采样数量大于点云数量，则进行重复采样
    # 若采样点数不够，则进行重复采样
    # for sample_point in [20000]:
    for sample_point in [1024, 2048, 4096]:
        if point_num < sample_point:
            sampled_cloud = ply_cloud.uniform_down_sample(1)
            while np.asarray(sampled_cloud.points).shape[0] < sample_point:
                sampled_cloud += ply_cloud.uniform_down_sample(1)
            sampled_cloud = np.asarray(sampled_cloud.points)[:sample_point]
            # 转回点云
            sampled_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sampled_cloud))
        else:
            sampled_cloud = ply_cloud.farthest_point_down_sample(sample_point)
        # print(np.asarray(sampled_cloud.points).shape[0])
        o3d.io.write_point_cloud(output_path.replace(".ply", "_" + str(sample_point) + ".ply"), sampled_cloud)

    print (ply_file, "done")
