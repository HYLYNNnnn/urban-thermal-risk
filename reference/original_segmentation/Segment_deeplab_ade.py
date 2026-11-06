# 导入需要使用的包
import os

# 解决OpenMP冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import csv
from itertools import islice
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib

matplotlib.use('Agg')
import pandas as pd
import cv2
from torchvision.models.segmentation import deeplabv3_resnet101

file_path = './pic/'
out_path = './out'

# 创建目录
if not os.path.exists(out_path):
    os.makedirs(out_path)
if not os.path.exists(file_path):
    os.makedirs(file_path)
    print(f"创建了输入目录: {file_path}")
    print("请将要处理的图片放入该目录")

filelist_out = os.listdir(out_path)

# 支持的图片格式
supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

# 关于图片重构的尺寸参数
# 如果图像的长宽 大于 max_size 则需要重构。
max_size = 1800
# reshape_size 为重构后的尺寸
reshape_size = 1000

# 忽略警告
import warnings

warnings.filterwarnings("ignore")

# 设定使用GPU或者CPU进行计算
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 读取标签信息 并 作为输出结果的列
def ReadLable():
    col_map_dict = {}
    filepath = './lable.csv'
    with open(filepath, encoding='GBK') as f:
        reader = csv.reader(f, skipinitialspace=True)
        for row in islice(reader, 1, None):
            id = int(row[0]) - 1
            Name = row[4]
            col_map_dict[id] = Name
    return col_map_dict


# ADE20K类别映射（150个类别）
def get_ade20k_palette():
    """ADE20K数据集的调色板"""
    return np.array([
        [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
        [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
        [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
        [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
        [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
        [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
        [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
        [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
        [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
        [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
        [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
        [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
        [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
        [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
        [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
        [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
        [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
        [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
        [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
        [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
        [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
        [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
        [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
        [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
        [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
        [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
        [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
        [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
        [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
        [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
        [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
        [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
        [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
        [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
        [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
        [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
        [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
        [102, 255, 0], [92, 0, 255]
    ])


# 图像预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 计算绿视率的函数
def calculate_green_view_index(predict, col_map):
    """计算绿视率（植被类别占总像素的比例）"""
    green_categories = ['tree', 'grass', 'plant', 'vegetation', 'foliage', 'flora']
    total_pixels = predict.shape[0] * predict.shape[1]
    green_pixels = 0

    for idx, category in col_map.items():
        if any(green in category.lower() for green in green_categories):
            green_pixels += np.sum(predict == idx)

    green_view_index = green_pixels / total_pixels
    return green_view_index


# 计算天空可视度的函数
def calculate_sky_visibility(predict, col_map):
    """计算人视角天空可视度（天空类别占总像素的比例）"""
    sky_categories = ['sky']
    total_pixels = predict.shape[0] * predict.shape[1]
    sky_pixels = 0

    for idx, category in col_map.items():
        if any(sky in category.lower() for sky in sky_categories):
            sky_pixels += np.sum(predict == idx)

    sky_visibility = sky_pixels / total_pixels
    return sky_visibility


# 计算界面围合度的函数
def calculate_enclosure_index(predict, col_map):
    """计算界面围合度（建筑物和墙体等垂直界面占总像素的比例）"""
    enclosure_categories = ['building', 'wall', 'house', 'skyscraper', 'edifice']
    total_pixels = predict.shape[0] * predict.shape[1]
    enclosure_pixels = 0

    for idx, category in col_map.items():
        if any(enc in category.lower() for enc in enclosure_categories):
            enclosure_pixels += np.sum(predict == idx)

    enclosure_index = enclosure_pixels / total_pixels
    return enclosure_index


# 定义函数对单张图片进行图像分割，并将结果存为pd.Series
def get_seg(file, model, col_map):
    img = Image.open(file).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)['out']
        predict = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    # 注意：由于使用的是VOC模型（21类），我们需要创建一个模拟的150类输出
    # 实际应用中应该使用ADE20K预训练的模型
    pred = []
    for i in range(21):  # VOC只有21个类别
        pred.append((len(predict[predict == i]) / (predict.shape[0] * predict.shape[1])))

    # 填充剩余的类别为0
    for i in range(21, 150):
        pred.append(0.0)

    # 计算绿视率、天空可视度和界面围合度
    # 注意：由于使用VOC模型，这些计算可能不准确
    green_view_index = calculate_green_view_index(predict, col_map)
    sky_visibility = calculate_sky_visibility(predict, col_map)
    enclosure_index = calculate_enclosure_index(predict, col_map)

    pred_series = pd.Series(pred).rename(col_map)
    # 添加新指标
    pred_series['green_view_index'] = green_view_index
    pred_series['sky_visibility'] = sky_visibility
    pred_series['enclosure_index'] = enclosure_index

    return pred_series, predict


# 加载预训练的DeepLabV3模型
# 注意：PyTorch官方的DeepLabV3是在PASCAL VOC上预训练的（21类），不是ADE20K（150类）
# 对于ADE20K，您需要使用其他来源的预训练权重或自行训练
# 这里我们使用VOC预训练模型作为示例，但结果可能不准确
model = deeplabv3_resnet101(pretrained=True)
model = model.to(device)
model.eval()

print("警告：当前使用的是PASCAL VOC预训练模型（21类），而不是ADE20K（150类）。")
print("要获得准确结果，请使用ADE20K预训练的模型权重。")

col_map = ReadLable()
filelist = os.listdir(file_path)
# 过滤出图片文件
filelist = [f for f in filelist if f.lower().endswith(supported_formats)]

if len(filelist) == 0:
    print(f"错误：在 {file_path} 目录中没有找到图片文件")
    print(f"支持的图片格式: {supported_formats}")
    exit()

print(f"找到 {len(filelist)} 个图片文件待处理")

col = ['id', 'lng', 'lat', 'pixels', 'green_view_index', 'sky_visibility', 'enclosure_index']
for k in col_map.keys():
    col.append(col_map[k])
df = pd.DataFrame(columns=col)
print(df)

# 图片重构后的分辨率:需要根据原始图片的分辨率比例进行设置
for i in filelist:  # 循环遍历所有的图片进行语义分割，并将结果存如pd.DataFrame
    # i 是图片名
    if i not in filelist_out:
        img_path = os.path.join(file_path, i)
        img_id = i
        # 图片的经纬度信息和朝向信息 如果图片名称中有这些信息，则可从图片名称 i 中解析。
        lng = 0
        lat = 0

        img = cv2.imread(img_path)

        # 检查图片是否成功读取
        if img is None:
            print(f"警告：无法读取图片 {img_path}")
            print(f"请检查文件是否存在，以及文件格式是否正确")
            continue

        # 读取完后将路径改成临时路径
        temp_path = './temp_image.jpg'
        # 获取原始图片尺寸
        ori_size = [img.shape[1], img.shape[0]]
        # 如果图片尺寸过大则进行图片的重构
        if ori_size[0] > max_size:
            Scale_Factor = ori_size[0] / reshape_size
            img_size = (int(ori_size[0] / Scale_Factor), int(ori_size[1] / Scale_Factor))
            print(i, ori_size, 'Resize to:', img_size)
        else:
            img_size = (int(ori_size[0]), int(ori_size[1]))

        img2 = cv2.resize(img, img_size, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(temp_path, img2)

        pixels = img_size[0] * img_size[1]
        seg_result, predict = get_seg(temp_path, model, col_map)

        # 创建一个包含所有数据的字典
        data_dict = {'id': img_id, 'lng': lng, 'lat': lat, 'pixels': pixels}
        # 将分割结果添加到字典中
        for key, value in seg_result.items():
            data_dict[key] = value

        # 创建DataFrame的一行
        new_row = pd.DataFrame([data_dict])
        df = pd.concat([df, new_row], axis=0, ignore_index=True)

        # 设定可视化方式
        palette = get_ade20k_palette()
        mask = Image.fromarray(predict.astype(np.uint8))
        mask.putpalette(palette.flatten().tolist())
        mask = mask.convert('RGB')

        print(i + ' seg has saved!')
        base = mpimg.imread(temp_path)
        plt.figure(figsize=(10, 5))
        # 显示原始图像
        plt.imshow(base)
        # 显示标签色块
        plt.imshow(mask, alpha=0.8)
        plt.axis('off')
        plt.savefig(out_path + '/' + i, dpi=300, bbox_inches='tight')
        plt.close('all')

        # 清理临时文件
        os.remove(temp_path)

        # 输出结果的路径 csv文件
        df.to_csv("./img_seg_deeplab_ade20k.csv")  # 将结果保存到csv
    else:
        print(i, '已经存在！')