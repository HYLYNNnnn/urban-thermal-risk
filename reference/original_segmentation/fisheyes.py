# 鱼眼图像语义分割与可视因子计算
# 计算天空可视因子(SVF)、树木可视因子(TVF)、建筑可视因子(BVF)
# 安装依赖：pip install transformers torch torchvision pillow opencv-python matplotlib pandas numpy

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import numpy as np
import cv2
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import time
from pathlib import Path

warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
# 鱼眼图像目录
FISHEYE_DIR = './fisheye'
OUTPUT_DIR = './fisheye_output'
RESULTS_CSV = './fisheye_view_factors.csv'

# 模型选择
MODEL_NAME = 'nvidia/segformer-b1-finetuned-cityscapes-1024-1024'  # 可以根据需要调整
MAX_IMAGE_SIZE = 1024  # 鱼眼图像处理尺寸

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 初始化模型 ====================
print("=" * 60)
print("鱼眼图像可视因子分析程序")
print("=" * 60)
print(f"\n正在加载模型...")

processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForSemanticSegmentation.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

print(f"模型加载完成！使用设备: {device}")

# Cityscapes类别定义
CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
    'motorcycle', 'bicycle'
]

# 定义用于计算可视因子的类别
SKY_CLASSES = ['sky']
TREE_CLASSES = ['vegetation', 'terrain']  # 树木和植被
BUILDING_CLASSES = ['building', 'wall']  # 建筑物和墙体


# ==================== 鱼眼图像处理函数 ====================
def create_fisheye_mask(image_shape):
    """创建鱼眼图像的圆形掩码，只分析圆形区域内的像素"""
    h, w = image_shape[:2]
    center = (w // 2, h // 2)
    radius = min(center[0], center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius

    return mask


def apply_zenith_angle_weight(image_shape, mask):
    """计算天顶角权重，用于准确计算可视因子"""
    h, w = image_shape[:2]
    center = (w // 2, h // 2)
    radius = min(center[0], center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    # 计算天顶角 (0 at center, π/2 at edge)
    zenith_angle = (dist_from_center / radius) * (np.pi / 2)

    # 立体角权重 (考虑球面投影)
    weight = np.sin(zenith_angle)
    weight[~mask] = 0  # 圆形区域外的权重为0

    return weight


def process_fisheye_image(image_path):
    """处理单张鱼眼图像"""
    # 读取图像
    image = Image.open(image_path).convert('RGB')
    original_size = image.size

    # 调整大小以提高处理速度
    if max(image.size) > MAX_IMAGE_SIZE:
        ratio = MAX_IMAGE_SIZE / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    # 预处理
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 推理
    with torch.no_grad():
        outputs = model(**inputs)

    # 后处理 - 调整回原始尺寸
    logits = outputs.logits
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=original_size[::-1],  # (height, width)
        mode="bilinear",
        align_corners=False,
    )

    seg_map = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
    original_image = np.array(Image.open(image_path).convert('RGB'))

    return seg_map, original_image


def calculate_view_factors(seg_map):
    """计算鱼眼图像的可视因子"""
    # 创建鱼眼掩码
    fisheye_mask = create_fisheye_mask(seg_map.shape)

    # 计算天顶角权重
    zenith_weights = apply_zenith_angle_weight(seg_map.shape, fisheye_mask)

    # 计算总权重
    total_weight = np.sum(zenith_weights)

    if total_weight == 0:
        return 0, 0, 0

    # 初始化可视因子
    svf = 0  # 天空可视因子
    tvf = 0  # 树木可视因子
    bvf = 0  # 建筑可视因子

    # 计算各类别的加权面积
    for i, class_name in enumerate(CITYSCAPES_CLASSES):
        class_mask = (seg_map == i) & fisheye_mask
        weighted_area = np.sum(zenith_weights[class_mask])

        if class_name in SKY_CLASSES:
            svf += weighted_area / total_weight
        elif class_name in TREE_CLASSES:
            tvf += weighted_area / total_weight
        elif class_name in BUILDING_CLASSES:
            bvf += weighted_area / total_weight

    return svf, tvf, bvf


def visualize_fisheye_segmentation(image, seg_map, svf, tvf, bvf, save_path):
    """可视化鱼眼图像分割结果"""
    # 创建彩色分割图
    h, w = seg_map.shape
    color_seg = np.zeros((h, w, 3), dtype=np.uint8)

    # 简化的调色板 - 只显示主要类别
    for i, class_name in enumerate(CITYSCAPES_CLASSES):
        if class_name in SKY_CLASSES:
            color_seg[seg_map == i] = [135, 206, 235]  # 天蓝色
        elif class_name in TREE_CLASSES:
            color_seg[seg_map == i] = [34, 139, 34]  # 森林绿
        elif class_name in BUILDING_CLASSES:
            color_seg[seg_map == i] = [139, 69, 19]  # 褐色
        else:
            color_seg[seg_map == i] = [128, 128, 128]  # 灰色（其他）

    # 创建鱼眼掩码用于显示
    fisheye_mask = create_fisheye_mask(seg_map.shape)
    color_seg[~fisheye_mask] = [0, 0, 0]  # 圆形外部设为黑色

    # 创建可视化
    fig = plt.figure(figsize=(12, 8))

    # 原始图像
    plt.subplot(2, 3, 1)
    plt.imshow(image)
    plt.title('原始鱼眼图像', fontsize=12)
    plt.axis('off')

    # 分割结果
    plt.subplot(2, 3, 2)
    plt.imshow(color_seg)
    plt.title('语义分割结果', fontsize=12)
    plt.axis('off')

    # 叠加显示
    plt.subplot(2, 3, 3)
    overlay = cv2.addWeighted(image, 0.6, color_seg, 0.4, 0)
    plt.imshow(overlay)
    plt.title('叠加显示', fontsize=12)
    plt.axis('off')

    # 可视因子显示
    plt.subplot(2, 3, 4)
    factors = ['SVF\n(天空)', 'TVF\n(树木)', 'BVF\n(建筑)']
    values = [svf, tvf, bvf]
    colors = ['skyblue', 'forestgreen', 'saddlebrown']

    bars = plt.bar(factors, values, color=colors)
    plt.ylim(0, 1)
    plt.ylabel('可视因子值', fontsize=12)
    plt.title('可视因子分析', fontsize=12)

    # 在柱状图上显示数值
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.3f}', ha='center', va='bottom')

    # 饼图
    plt.subplot(2, 3, 5)
    sizes = [svf, tvf, bvf, 1 - svf - tvf - bvf]
    labels = ['天空', '树木', '建筑', '其他']
    colors_pie = ['skyblue', 'forestgreen', 'saddlebrown', 'gray']

    plt.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
    plt.title('场景构成比例', fontsize=12)

    # 图例和说明
    plt.subplot(2, 3, 6)
    plt.text(0.1, 0.8, f'天空可视因子 (SVF): {svf:.3f}', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.6, f'树木可视因子 (TVF): {tvf:.3f}', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.4, f'建筑可视因子 (BVF): {bvf:.3f}', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.2, f'总计: {svf + tvf + bvf:.3f}', fontsize=12, transform=plt.gca().transAxes)
    plt.axis('off')
    plt.title('可视因子数值', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


# ==================== 主处理流程 ====================
def main():
    print(f"\n扫描目录: {FISHEYE_DIR}")

    # 获取所有子文件夹
    folders = sorted([f for f in os.listdir(FISHEYE_DIR)
                      if os.path.isdir(os.path.join(FISHEYE_DIR, f))
                      and f.isdigit()])

    if not folders:
        print("错误：未找到数字命名的文件夹")
        return

    print(f"找到 {len(folders)} 个文件夹待处理")
    print("-" * 60)

    results = []

    # 处理每个文件夹
    for folder in folders:
        folder_path = os.path.join(FISHEYE_DIR, folder)
        print(f"\n处理文件夹: {folder}")

        # 查找图像文件
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(Path(folder_path).glob(f'*{ext}'))
            image_files.extend(Path(folder_path).glob(f'*{ext.upper()}'))

        if not image_files:
            print(f"  警告：文件夹 {folder} 中未找到图像文件")
            continue

        # 处理第一张找到的图像
        image_path = str(image_files[0])
        image_name = os.path.basename(image_path)

        try:
            print(f"  处理图像: {image_name}")
            start_time = time.time()

            # 处理图像
            seg_map, original_image = process_fisheye_image(image_path)

            # 计算可视因子
            svf, tvf, bvf = calculate_view_factors(seg_map)

            # 保存可视化结果
            output_path = os.path.join(OUTPUT_DIR, f'{folder}_analysis.png')
            visualize_fisheye_segmentation(original_image, seg_map, svf, tvf, bvf, output_path)

            # 记录结果
            result = {
                'folder_id': int(folder),
                'image_name': image_name,
                'SVF': svf,
                'TVF': tvf,
                'BVF': bvf,
                'total_VF': svf + tvf + bvf
            }
            results.append(result)

            process_time = time.time() - start_time
            print(f"  ✓ 处理完成！用时: {process_time:.2f}秒")
            print(f"    SVF: {svf:.3f}, TVF: {tvf:.3f}, BVF: {bvf:.3f}")

        except Exception as e:
            print(f"  ✗ 处理失败: {e}")
            import traceback
            traceback.print_exc()

    # 保存结果
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('folder_id')  # 按文件夹ID排序
        df.to_csv(RESULTS_CSV, index=False, encoding='utf-8-sig')

        print("\n" + "=" * 60)
        print("处理完成！")
        print(f"结果已保存到: {RESULTS_CSV}")
        print(f"可视化图像保存在: {OUTPUT_DIR}")

        # 显示统计信息
        print("\n=== 统计信息 ===")
        print(f"成功处理: {len(results)} 张图像")
        print(f"\n可视因子统计:")
        print(f"  天空可视因子(SVF): 平均={df['SVF'].mean():.3f}, 标准差={df['SVF'].std():.3f}")
        print(f"  树木可视因子(TVF): 平均={df['TVF'].mean():.3f}, 标准差={df['TVF'].std():.3f}")
        print(f"  建筑可视因子(BVF): 平均={df['BVF'].mean():.3f}, 标准差={df['BVF'].std():.3f}")
        print(f"\n可视因子范围:")
        print(f"  SVF: {df['SVF'].min():.3f} ~ {df['SVF'].max():.3f}")
        print(f"  TVF: {df['TVF'].min():.3f} ~ {df['TVF'].max():.3f}")
        print(f"  BVF: {df['BVF'].min():.3f} ~ {df['BVF'].max():.3f}")
    else:
        print("\n没有成功处理的图像")


if __name__ == "__main__":
    main()