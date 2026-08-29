import os
import shutil
import pickle

data_path = '/home/lc/code/BYSY/bone_for_ligament/3D/v0.3_D3D_PCT/utils/label_test.txt'

def get_all_mask_list():
    mask_list = [ i for i in os.listdir('/data1/liuchao/dataset/Point_Circle_All/Mask') ]
    return mask_list

def copy_one_pid_mask(pid, mask_list, src, dst):
    for mask in mask_list:
        if pid in mask:
            if not os.path.exists(os.path.join(dst, mask.split('_')[0] + '_' + pid)):
                os.makedirs(os.path.join(dst, mask.split('_')[0] + '_' + pid))
            for png in os.listdir(os.path.join(src, mask)):
                if png.endswith('_image.png'):
                    shutil.copy(os.path.join(src, mask, png), os.path.join(dst, mask.split('_')[0] + '_' + pid, png))

def count_png_num(pid, mask_list, src):
    count = 0
    for mask in mask_list:
        if pid in mask:
            for png in os.listdir(os.path.join(src, mask)):
                if png.endswith('_image.png'):
                    count +=1
    return count

# 从原来的pkl文件中，找到所有验证集对应的病例的spacing、orientation、position、slicelocation等信息, 保存至test_args.pkl
def get_recon_args(pid, sag_pkl, cor_pkl):
    args = {}
    for key in sag_pkl.keys():
        sagkeyargs = {}
        if pid in key:
            instanceid = key.split('_')[2]
            try:
                sagkeyargs['spacing'] = sag_pkl[key]['spacing']
                sagkeyargs['orientation'] = sag_pkl[key]['orientation']
                sagkeyargs['position'] = sag_pkl[key]['position']
                sagkeyargs['slicelocation'] = sag_pkl[key]['slicelocation']
            except:
                print('error:', key)
                continue
            args[f'sag_{pid}_{instanceid}'] = sagkeyargs
    for key in cor_pkl.keys():
        corkeyargs = {}
        if pid in key:
            instanceid = key.split('_')[2]
            try:
                corkeyargs['spacing'] = cor_pkl[key]['spacing']
                corkeyargs['orientation'] = cor_pkl[key]['orientation']
                corkeyargs['position'] = cor_pkl[key]['position']
                corkeyargs['slicelocation'] = cor_pkl[key]['slicelocation']
            except:
                print('error:', key)
                continue
            args[f'cor_{pid}_{instanceid}'] = corkeyargs
    print(pid, 'done')
    return args


# sag_pkl_path = '/data1/liuchao/dataset/Point_Circle_All/Point/Sag_Point_data.pkl'
# cor_pkl_path = '/data1/liuchao/dataset/Point_Circle_All/Point/Cor_Point_data.pkl'
# sag_pkl = pickle.load(open(sag_pkl_path, 'rb'))
# cor_pkl = pickle.load(open(cor_pkl_path, 'rb'))
# allargs = {}
# with open(data_path, 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         pid = line.split(' ')[0]
#         args = get_recon_args(pid, sag_pkl, cor_pkl)
#         allargs[pid] = args
#     pickle.dump(allargs, open('test_args.pkl', 'wb'))

# 读取test_args.pkl, 生成test_args.txt
test_args = pickle.load(open('test_args.pkl', 'rb'))
with open('test_args.txt', 'w') as f:
    for pid in test_args.keys():
        f.write(pid + '\n')
        for key in test_args[pid].keys():
            f.write(key + '\n')
            f.write(str(test_args[pid][key]) + '\n')
        f.write('\n')
