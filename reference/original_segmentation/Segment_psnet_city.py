# 导入需要使用的包
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 解决OpenMP冲突

import csv
from itertools import islice
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib

matplotlib.use('Agg')
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# 避免多线程冲突
torch.set_num_threads(1)

file_path = './pic'
out_path = './out'
metrics_path = './metrics'  # 指标分析结果保存路径

# 创建输出目录
os.makedirs(out_path, exist_ok=True)
os.makedirs(metrics_path, exist_ok=True)

# 关于图片重构的尺寸参数
max_size = 2200
reshape_size = 2000

# 设定使用GPU或者CPU进行计算
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# Cityscapes数据集的类别映射
cityscapes_classes = {
    0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence',
    5: 'pole', 6: 'traffic light', 7: 'traffic sign', 8: 'vegetation',
    9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider', 13: 'car',
    14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle'
}

# Cityscapes调色板
cityscapes_palette = np.array([
    [128, 64, 128],  # road
    [244, 35, 232],  # sidewalk
    [70, 70, 70],  # building
    [102, 102, 156],  # wall
    [190, 153, 153],  # fence
    [153, 153, 153],  # pole
    [250, 170, 30],  # traffic light
    [220, 220, 0],  # traffic sign
    [107, 142, 35],  # vegetation
    [152, 251, 152],  # terrain
    [70, 130, 180],  # sky
    [220, 20, 60],  # person
    [255, 0, 0],  # rider
    [0, 0, 142],  # car
    [0, 0, 70],  # truck
    [0, 60, 100],  # bus
    [0, 80, 100],  # train
    [0, 0, 230],  # motorcycle
    [119, 11, 32],  # bicycle
    [0, 0, 0]  # unlabeled
])

# 定义城市形态分析相关的类别
vegetation_classes = [8, 9]  # vegetation, terrain
sky_classes = [10]  # sky
building_classes = [2, 3, 4]  # building, wall, fence
vertical_elements = [2, 3, 4, 5, 6, 7]  # building, wall, fence, pole, traffic light, traffic sign


def get_color_pallete(predict, dataset='citys'):
    """将预测结果转换为彩色图像"""
    mask = np.zeros((predict.shape[0], predict.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(cityscapes_palette):
        mask[predict == label] = color
    return mask


def calculate_urban_metrics(predict, original_img):
    """
    计算城市形态指标
    predict: 语义分割结果
    original_img: 原始图像（用于颜色辅助判断）
    """
    height, width = predict.shape
    total_pixels = height * width

    # 1. 计算绿视率 (Green View Index)
    # 方法1：基于语义分割
    vegetation_mask = np.isin(predict, vegetation_classes)
    gvi_semantic = np.sum(vegetation_mask) / total_pixels

    # 方法2：基于HSV颜色空间（使用更宽松的范围）
    hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([20, 25, 25])  # 更宽松的范围
    upper_green = np.array([95, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    gvi_color = np.sum(green_mask > 0) / total_pixels

    # 综合两种方法
    if gvi_semantic < 0.01:  # 语义分割几乎没有检测到植被
        gvi = gvi_color
    else:
        gvi = (gvi_semantic * 0.7 + gvi_color * 0.3)  # 更信任语义分割结果

    # 2. 计算天空占比 (Sky Ratio) - 人视角的天空占比
    sky_mask = np.isin(predict, sky_classes)
    sky_ratio = np.sum(sky_mask) / total_pixels

    # 3. 计算界面围合度 (Enclosure Index)
    # 基于垂直元素在图像中部区域的占比（调整区域范围）
    middle_start = height // 4  # 从1/4处开始
    middle_end = 3 * height // 4  # 到3/4处结束
    middle_region = predict[middle_start:middle_end, :]

    # 计算中部区域的垂直元素占比
    vertical_mask = np.isin(middle_region, vertical_elements)
    enclosure_index = np.sum(vertical_mask) / middle_region.size

    # 也可以结合边缘检测
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    middle_edges = edges[middle_start:middle_end, :]
    edge_density = np.sum(middle_edges > 0) / middle_edges.size

    # 综合评分
    enclosure_index = (enclosure_index * 0.7 + edge_density * 0.3)

    return gvi, sky_ratio, enclosure_index, green_mask, vegetation_mask


def visualize_metrics(img_path, predict, gvi, sky_ratio, enclosure_index, green_mask, save_path):
    """
    可视化分析结果
    """
    # 读取原图
    base = cv2.imread(img_path)
    base = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)

    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 原图+语义分割叠加
    mask = get_color_pallete(predict, 'citys')
    axes[0, 0].imshow(base)
    axes[0, 0].imshow(mask, alpha=0.7)
    axes[0, 0].set_title('Semantic Segmentation Result', fontsize=14)
    axes[0, 0].axis('off')

    # 2. 绿视率可视化
    vegetation_mask = np.isin(predict, vegetation_classes)
    axes[0, 1].imshow(base)
    green_overlay = np.zeros_like(base)
    green_overlay[:, :, 1] = vegetation_mask * 255  # 绿色通道
    # 也显示HSV检测的绿色
    green_overlay[:, :, 0] = (green_mask > 0) * 100  # 稍微加点红色区分
    axes[0, 1].imshow(green_overlay, alpha=0.5)
    axes[0, 1].set_title(f'Green View Index: {gvi:.2%}', fontsize=14)
    axes[0, 1].axis('off')

    # 3. 天空占比可视化
    sky_mask = np.isin(predict, sky_classes)
    axes[1, 0].imshow(base)
    sky_overlay = np.zeros_like(base)
    sky_overlay[:, :, 2] = sky_mask * 255  # 蓝色通道
    axes[1, 0].imshow(sky_overlay, alpha=0.5)
    axes[1, 0].set_title(f'Sky Ratio: {sky_ratio:.2%}', fontsize=14)
    axes[1, 0].axis('off')

    # 4. 综合指标展示
    axes[1, 1].imshow(base)
    # 添加文字背景框
    props = dict(boxstyle='round,pad=0.5', alpha=0.8)
    axes[1, 1].text(20, 50, f'GVI: {gvi:.2%}', fontsize=12,
                    bbox=dict(facecolor='lightgreen', **props))
    axes[1, 1].text(20, 100, f'Sky Ratio: {sky_ratio:.2%}', fontsize=12,
                    bbox=dict(facecolor='lightblue', **props))
    axes[1, 1].text(20, 150, f'Enclosure: {enclosure_index:.2%}', fontsize=12,
                    bbox=dict(facecolor='lightyellow', **props))

    # 添加围合度计算区域标识
    height = base.shape[0]
    middle_start = height // 4
    middle_end = 3 * height // 4
    axes[1, 1].axhline(y=middle_start, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].axhline(y=middle_end, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].text(base.shape[1] - 200, middle_start + 10, 'Enclosure Area',
                    fontsize=10, color='red')

    axes[1, 1].set_title('Urban Morphology Metrics', fontsize=14)
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close('all')


# 加载PyTorch模型
print("正在加载DeepLabV3模型...")
model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()
model.to(device)
print("模型加载完成！")

# 定义图像预处理
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_seg_pytorch(img_path, model):
    """使用PyTorch进行语义分割"""
    # 读取图像
    image = Image.open(img_path).convert('RGB')

    # 预处理
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        output = model(input_batch)['out'][0]

    # 获取预测结果
    output_predictions = output.argmax(0).cpu().numpy()

    # 计算每个类别的占比
    pred = []
    total_pixels = output_predictions.shape[0] * output_predictions.shape[1]
    for i in range(19):  # Cityscapes有19个类别
        ratio = np.sum(output_predictions == i) / total_pixels
        pred.append(ratio)

    pred_series = pd.Series(pred, index=list(cityscapes_classes.values())[:19])

    return pred_series, output_predictions


def process_folder(folder_path, folder_name):
    """处理单个文件夹"""
    print(f"\n处理文件夹: {folder_name}")

    # 创建文件夹专属的输出目录
    folder_out_path = os.path.join(out_path, folder_name)
    folder_metrics_path = os.path.join(metrics_path, folder_name)
    os.makedirs(folder_out_path, exist_ok=True)
    os.makedirs(folder_metrics_path, exist_ok=True)

    # 获取文件夹中的图片列表
    filelist = [f for f in os.listdir(folder_path)
                if f.endswith(('.jpg', '.png', '.jpeg'))]

    if not filelist:
        print(f"文件夹 {folder_name} 中没有找到图片")
        return None

    # 检查已处理的文件
    filelist_out = os.listdir(folder_out_path)

    # 初始化DataFrame
    df = pd.DataFrame(columns=['id', 'lng', 'lat', 'pixels', 'road', 'sidewalk', 'building', 'wall', 'fence',
                               'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
                               'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle',
                               'gvi', 'sky_ratio', 'enclosure_index'])

    # 用于存储该文件夹的平均指标
    folder_metrics = {'gvi': [], 'sky_ratio': [], 'enclosure_index': []}

    # 处理每张图片
    for i in tqdm(filelist, desc=f"处理 {folder_name}"):
        if i not in filelist_out:
            img_path = os.path.join(folder_path, i)
            img_id = i

            # 解析图片名称中的信息
            try:
                parts = i[:-4].split('_')
                if len(parts) >= 4:
                    rid = parts[0]
                    lng = float(parts[1])
                    lat = float(parts[2])
                    heading = float(parts[3])
                else:
                    rid = i[:-4]
                    lng = 0
                    lat = 0
                    heading = 0
            except:
                rid = i[:-4]
                lng = 0
                lat = 0
                heading = 0

            try:
                # 读取图片
                img = cv2.imread(img_path)
                if img is None:
                    print(f"无法读取图片: {img_path}")
                    continue

                img_bgr = img.copy()  # 保存BGR格式用于颜色分析

                # 临时路径
                temp_dir = './image_processed'
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, i)

                # 获取原始图片尺寸
                ori_size = [img.shape[1], img.shape[0]]

                # 如果图片尺寸过大则进行图片的重构
                if ori_size[0] > max_size:
                    Scale_Factor = ori_size[0] / reshape_size
                    img_size = (int(ori_size[0] / Scale_Factor), int(ori_size[1] / Scale_Factor))
                else:
                    img_size = (int(ori_size[0]), int(ori_size[1]))

                img2 = cv2.resize(img, img_size, interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(temp_path, img2)

                pixels = img_size[0] * img_size[1]

                # 获取语义分割结果
                seg_result, predict = get_seg_pytorch(temp_path, model)

                # 计算城市形态指标
                img_resized = cv2.resize(img_bgr, img_size, interpolation=cv2.INTER_CUBIC)
                gvi, sky_ratio, enclosure_index, green_mask, vegetation_mask = calculate_urban_metrics(predict,
                                                                                                       img_resized)

                # 构建数据行
                data_dict = {
                    'id': img_id,
                    'lng': lng,
                    'lat': lat,
                    'pixels': pixels,
                    'gvi': gvi,
                    'sky_ratio': sky_ratio,
                    'enclosure_index': enclosure_index
                }
                # 添加语义分割结果
                for class_name, ratio in seg_result.items():
                    data_dict[class_name] = ratio

                new_row = pd.DataFrame([data_dict])
                df = pd.concat([df, new_row], axis=0, ignore_index=True)

                # 可视化并保存结果
                # 1. 保存原始的语义分割结果
                mask = get_color_pallete(predict, 'citys')
                base = cv2.imread(temp_path)
                base = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)

                plt.figure(figsize=(10, 5))
                plt.imshow(base)
                plt.imshow(mask, alpha=0.8)
                plt.axis('off')
                plt.savefig(os.path.join(folder_out_path, i), dpi=300, bbox_inches='tight')
                plt.close('all')

                # 2. 保存城市形态指标可视化
                metrics_save_path = os.path.join(folder_metrics_path, f'metrics_{i}')
                visualize_metrics(temp_path, predict, gvi, sky_ratio, enclosure_index, green_mask, metrics_save_path)

                # 记录指标
                folder_metrics['gvi'].append(gvi)
                folder_metrics['sky_ratio'].append(sky_ratio)
                folder_metrics['enclosure_index'].append(enclosure_index)

                # 删除临时文件
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            except Exception as e:
                print(f"处理图片 {i} 时出错: {str(e)}")
                continue

    # 保存该文件夹的结果
    if len(df) > 0:
        csv_path = f"./img_seg_{folder_name}_with_metrics.csv"
        df.to_csv(csv_path, index=False)
        print(f"文件夹 {folder_name} 的详细结果保存在: {csv_path}")

    # 返回该文件夹的平均指标
    if folder_metrics['gvi']:
        avg_metrics = {
            'folder': folder_name,
            'avg_gvi': np.mean(folder_metrics['gvi']),
            'avg_sky_ratio': np.mean(folder_metrics['sky_ratio']),
            'avg_enclosure_index': np.mean(folder_metrics['enclosure_index']),
            'std_gvi': np.std(folder_metrics['gvi']),
            'std_sky_ratio': np.std(folder_metrics['sky_ratio']),
            'std_enclosure_index': np.std(folder_metrics['enclosure_index']),
            'sample_count': len(folder_metrics['gvi'])
        }
        return avg_metrics
    else:
        return None


# 主处理流程
print("开始处理图片...")

# 检查是否有子文件夹
subfolders = [f for f in os.listdir(file_path)
              if os.path.isdir(os.path.join(file_path, f))]

all_folder_metrics = []

if subfolders:
    # 如果有子文件夹，分别处理每个文件夹
    print(f"找到 {len(subfolders)} 个子文件夹")
    for folder_name in subfolders:
        folder_path = os.path.join(file_path, folder_name)
        avg_metrics = process_folder(folder_path, folder_name)
        if avg_metrics:
            all_folder_metrics.append(avg_metrics)
            print(f"\n文件夹 {folder_name} 平均指标:")
            print(f"  平均GVI: {avg_metrics['avg_gvi']:.2%}")
            print(f"  平均天空占比: {avg_metrics['avg_sky_ratio']:.2%}")
            print(f"  平均围合度: {avg_metrics['avg_enclosure_index']:.2%}")
else:
    # 如果没有子文件夹，直接处理pic目录下的图片
    print("直接处理pic目录下的图片")
    avg_metrics = process_folder(file_path, 'main')
    if avg_metrics:
        all_folder_metrics.append(avg_metrics)

# 保存所有文件夹的平均指标
if all_folder_metrics:
    df_all_avg = pd.DataFrame(all_folder_metrics)
    df_all_avg.to_csv("./all_folders_average_metrics.csv", index=False, encoding='utf-8-sig')
    print(f"\n所有文件夹的平均指标保存在: ./all_folders_average_metrics.csv")

    # 打印总体统计
    print("\n总体统计:")
    print(f"所有文件夹平均GVI: {df_all_avg['avg_gvi'].mean():.2%}")
    print(f"所有文件夹平均天空占比: {df_all_avg['avg_sky_ratio'].mean():.2%}")
    print(f"所有文件夹平均围合度: {df_all_avg['avg_enclosure_index'].mean():.2%}")

print("\n处理完成！")