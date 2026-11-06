# 使用Hugging Face的SegFormer模型 - 多文件夹处理
# 安装：pip install transformers

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# 禁用Hugging Face的symlinks警告
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import numpy as np
import cv2
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
import matplotlib.pyplot as plt
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# 配置路径
base_path = './pic'
out_path = './out'
if not os.path.exists(out_path):
    os.makedirs(out_path)

print("正在加载模型...")
# 选择模型 - 这里使用Cityscapes预训练的SegFormer
model_name = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"

processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForSemanticSegmentation.from_pretrained(model_name)

# 使用GPU如果可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
model = model.to(device)
model.eval()

# Cityscapes类别
cityscapes_classes = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
                      'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider',
                      'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

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
    [119, 11, 32]  # bicycle
])


def process_image(image_path, max_size=1024):
    """处理单张图片并返回分割结果"""
    # 读取图片
    image = Image.open(image_path).convert('RGB')
    original_size = image.size

    # 如果图片太大，先缩放以提高处理速度
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        print(f"  图片缩放: {original_size} -> {new_size}")

    # 预处理
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 推理
    with torch.no_grad():
        outputs = model(**inputs)

    # 后处理 - 将结果调整回原始尺寸
    logits = outputs.logits
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=original_size[::-1],  # (height, width)
        mode="bilinear",
        align_corners=False,
    )

    # 获取预测结果
    seg_map = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

    # 重新读取原始尺寸的图片
    original_image = np.array(Image.open(image_path).convert('RGB'))

    return seg_map, original_image


def calculate_metrics(seg_map):
    """计算各种指标"""
    total_pixels = seg_map.shape[0] * seg_map.shape[1]

    # 计算各类别比例
    class_ratios = {}
    for i, class_name in enumerate(cityscapes_classes):
        pixels = np.sum(seg_map == i)
        ratio = pixels / total_pixels
        class_ratios[class_name] = ratio

    # 计算特殊指标
    green_view = class_ratios.get('vegetation', 0) + class_ratios.get('terrain', 0)
    sky_visibility = class_ratios.get('sky', 0)
    enclosure = class_ratios.get('building', 0) + class_ratios.get('wall', 0) + class_ratios.get('fence', 0)

    return class_ratios, green_view, sky_visibility, enclosure


def visualize_segmentation(image, seg_map, save_path):
    """创建并保存分割可视化"""
    # 创建彩色分割图
    color_seg = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    for label_id in range(len(cityscapes_classes)):
        color_seg[seg_map == label_id] = cityscapes_palette[label_id]

    # 创建叠加图像
    overlay = cv2.addWeighted(image, 0.6, color_seg, 0.4, 0)

    # 保存结果
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('原始图片')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(color_seg)
    plt.title('语义分割结果')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title('叠加显示')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# 获取所有子文件夹
subfolders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
folder_results = []
all_image_results = []

print(f"\n找到 {len(subfolders)} 个文件夹待处理")
print("=" * 50)

for folder_idx, folder_name in enumerate(subfolders, 1):
    folder_path = os.path.join(base_path, folder_name)
    print(f"\n[{folder_idx}/{len(subfolders)}] 处理文件夹: {folder_name}")

    # 获取文件夹中的图片
    filelist = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(filelist) == 0:
        print(f"  ✗ 文件夹中没有图片，跳过")
        continue

    # 为每个文件夹创建输出子目录
    folder_out_path = os.path.join(out_path, folder_name)
    if not os.path.exists(folder_out_path):
        os.makedirs(folder_out_path)

    folder_class_ratios = {cls: [] for cls in cityscapes_classes}
    folder_green_views = []
    folder_sky_visibilities = []
    folder_enclosures = []

    print(f"  文件夹中包含 {len(filelist)} 张图片")

    for img_idx, img_file in enumerate(filelist, 1):
        img_path = os.path.join(folder_path, img_file)
        print(f"  [{img_idx}/{len(filelist)}] 处理图片: {img_file}")

        try:
            # 处理图片
            seg_map, original_image = process_image(img_path)

            # 计算指标
            class_ratios, green_view, sky_vis, enclosure = calculate_metrics(seg_map)

            # 保存可视化结果
            out_file = os.path.join(folder_out_path, img_file.replace('.', '_seg.'))
            visualize_segmentation(original_image, seg_map, out_file)

            # 记录单张图片结果
            image_result = {
                'folder': folder_name,
                'image': img_file,
                'green_view_index': green_view,
                'sky_visibility': sky_vis,
                'enclosure_index': enclosure
            }
            image_result.update(class_ratios)
            all_image_results.append(image_result)

            # 收集文件夹级别的数据
            folder_green_views.append(green_view)
            folder_sky_visibilities.append(sky_vis)
            folder_enclosures.append(enclosure)

            for cls in cityscapes_classes:
                folder_class_ratios[cls].append(class_ratios.get(cls, 0))

            # 打印结果
            print(f"    ✓ 处理成功!")
            print(f"    - 绿视率: {green_view:.1%}")
            print(f"    - 天空可视度: {sky_vis:.1%}")
            print(f"    - 建筑围合度: {enclosure:.1%}")

        except Exception as e:
            print(f"    ✗ 处理失败: {e}")

    # 计算文件夹的平均值
    if folder_green_views:  # 确保有成功处理的图片
        avg_green = np.mean(folder_green_views)
        avg_sky = np.mean(folder_sky_visibilities)
        avg_enclosure = np.mean(folder_enclosures)

        avg_class_ratios = {cls: np.mean(ratios) for cls, ratios in folder_class_ratios.items()}

        # 记录文件夹结果
        folder_result = {
            'folder': folder_name,
            'image_count': len(folder_green_views),
            'avg_green_view_index': avg_green,
            'avg_sky_visibility': avg_sky,
            'avg_enclosure_index': avg_enclosure
        }
        folder_result.update(avg_class_ratios)
        folder_results.append(folder_result)

        # 打印文件夹汇总
        print(f"\n  === 文件夹 {folder_name} 汇总 ===")
        print(f"  平均绿视率: {avg_green:.1%}")
        print(f"  平均天空可视度: {avg_sky:.1%}")
        print(f"  平均建筑围合度: {avg_enclosure:.1%}")

        # 显示主要类别
        sorted_classes = sorted(avg_class_ratios.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  主要类别: ", end="")
        print(", ".join([f"{name} ({ratio:.1%})" for name, ratio in sorted_classes if ratio > 0.01]))

# 保存所有图片结果到CSV
if all_image_results:
    df_images = pd.DataFrame(all_image_results)
    csv_file_images = './all_image_results.csv'
    df_images.to_csv(csv_file_images, index=False)
    print(f"\n所有图片结果已保存到: {csv_file_images}")

# 保存文件夹平均结果到CSV
if folder_results:
    df_folders = pd.DataFrame(folder_results)
    csv_file_folders = './folder_average_results.csv'
    df_folders.to_csv(csv_file_folders, index=False)
    print(f"文件夹平均结果已保存到: {csv_file_folders}")

    # 打印总体汇总统计
    print("\n" + "=" * 50)
    print("=== 总体汇总统计 ===")
    print(f"处理文件夹: {len(folder_results)}/{len(subfolders)}")
    print(f"总平均绿视率: {df_folders['avg_green_view_index'].mean():.1%}")
    print(f"总平均天空可视度: {df_folders['avg_sky_visibility'].mean():.1%}")
    print(f"总平均建筑围合度: {df_folders['avg_enclosure_index'].mean():.1%}")
else:
    print("\n没有成功处理的文件夹数据")