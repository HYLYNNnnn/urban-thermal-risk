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

file_path = './pic'
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


# PASCAL VOC调色板
def get_pascal_voc_palette():
    """PASCAL VOC数据集的调色板"""
    return np.array([
        [0, 0, 0],  # background
        [128, 0, 0],  # aeroplane
        [0, 128, 0],  # bicycle
        [128, 128, 0],  # bird
        [0, 0, 128],  # boat
        [128, 0, 128],  # bottle
        [0, 128, 128],  # bus
        [128, 128, 128],  # car
        [64, 0, 0],  # cat
        [192, 0, 0],  # chair
        [64, 128, 0],  # cow
        [192, 128, 0],  # dining table
        [64, 0, 128],  # dog
        [192, 0, 128],  # horse
        [64, 128, 128],  # motorbike
        [192, 128, 128],  # person
        [0, 64, 0],  # potted plant
        [128, 64, 0],  # sheep
        [0, 192, 0],  # sofa
        [128, 192, 0],  # train
        [0, 64, 128],  # tv/monitor
        [224, 224, 192]  # void (边界)
    ])


# 图像预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 定义函数对单张图片进行图像分割，并将结果存为pd.Series
def get_seg(file, model, col_map):
    img = Image.open(file).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)['out']
        predict = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    pred = []
    # 对于PASCAL VOC，只有21个类别
    for i in range(21):
        pred.append((len(predict[predict == i]) / (predict.shape[0] * predict.shape[1])))

    # 补充到150个元素（为了兼容原始代码结构）
    for i in range(21, 150):
        pred.append(0.0)

    pred_series = pd.Series(pred).rename(col_map)
    return pred_series, predict


col_map = ReadLable()

# 加载预训练的DeepLabV3模型（在PASCAL VOC上训练）
model = deeplabv3_resnet101(pretrained=True)  # 默认就是21个类别
model = model.to(device)
model.eval()

filelist = os.listdir(file_path)
# 过滤出图片文件
filelist = [f for f in filelist if f.lower().endswith(supported_formats)]

if len(filelist) == 0:
    print(f"错误：在 {file_path} 目录中没有找到图片文件")
    print(f"支持的图片格式: {supported_formats}")
    exit()

print(f"找到 {len(filelist)} 个图片文件待处理")

col = ['id', 'lng', 'lat', 'pixels']
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
        palette = get_pascal_voc_palette()
        mask_rgb = np.zeros((predict.shape[0], predict.shape[1], 3), dtype=np.uint8)
        for label_id in range(21):
            mask_rgb[predict == label_id] = palette[label_id]

        print(i + ' seg has saved!')
        base = mpimg.imread(temp_path)
        plt.figure(figsize=(10, 5))
        # 显示原始图像
        plt.imshow(base)
        # 显示标签色块
        plt.imshow(mask_rgb, alpha=0.86)
        plt.axis('off')
        plt.savefig(out_path + '/' + i, dpi=150, bbox_inches='tight')
        plt.close('all')

        # 清理临时文件
        os.remove(temp_path)

        # 输出结果的路径 csv文件
        df.to_csv("./00 01 02 03 04 05 06 07 08 09 10 img_seg_ade20k.csv")  # 将结果保存到csv
    else:
        print(i, '已经存在！')