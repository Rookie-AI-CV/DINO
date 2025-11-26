#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DINO 推理脚本
支持批量推理，输入为图片路径（单个文件或目录）

使用示例:
    # 单张图片推理
    python inference.py \
        --model_config config/DINO/DINO_4scale.py \
        --model_checkpoint ckpts/88830c8c6ab6a5dd006fc00191fa01ed.pth \
        --image_path ./figs/idea.jpg \
        --batch_size 1

    # 批量推理（目录）
    python inference.py \
        --model_config config/DINO/DINO_4scale.py \
        --model_checkpoint ckpts/88830c8c6ab6a5dd006fc00191fa01ed.pth \
        --image_path ./images/ \
        --batch_size 4 \
        --threshold 0.3 \
        --output results.json \
        --save_vis ./visualizations/
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import glob

from main import build_model_main
from util.slconfig import SLConfig
from util.visualizer import COCOVisualizer
from util import box_ops
import datasets.transforms as T


def load_id2name(json_path='util/coco_id2name.json'):
    """加载类别ID到名称的映射"""
    with open(json_path) as f:
        id2name = json.load(f)
        id2name = {int(k): v for k, v in id2name.items()}
    return id2name


def get_image_paths(input_path):
    """获取所有图片路径"""
    input_path = Path(input_path)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    if input_path.is_file():
        # 单个文件
        if input_path.suffix.lower() in image_extensions:
            return [str(input_path)]
        else:
            raise ValueError(f"不支持的文件格式: {input_path.suffix}")
    elif input_path.is_dir():
        # 目录，查找所有图片
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(str(input_path / f'*{ext}')))
            image_paths.extend(glob.glob(str(input_path / f'*{ext.upper()}')))
        if not image_paths:
            raise ValueError(f"在目录 {input_path} 中未找到图片文件")
        return sorted(image_paths)
    else:
        raise ValueError(f"路径不存在: {input_path}")


class ImageDataset(torch.utils.data.Dataset):
    """图片数据集，用于批量加载"""
    def __init__(self, image_paths, max_size=800):
        self.image_paths = image_paths
        # 使用配置的 max_size，如果没有则使用默认值 1333
        max_size_value = max_size if max_size else 800
        max_size_max = int(max_size_value * 1.2) if max_size_value else 1333
        self.transform = T.Compose([
            T.RandomResize([max_size_value], max_size=max_size_max),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        original_size = image.size  # (width, height)
        image, _ = self.transform(image, None)
        return image, original_size, img_path


def load_model(model_config_path, model_checkpoint_path, device='cuda'):
    """加载模型"""
    print(f"加载配置文件: {model_config_path}")
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    
    print(f"构建模型...")
    model, criterion, postprocessors = build_model_main(args)
    
    print(f"加载模型权重: {model_checkpoint_path}")
    checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.eval()
    model = model.to(device)
    
    return model, postprocessors, args


def inference_batch(model, postprocessors, dataloader, id2name, threshold=0.3, 
                   device='cuda', save_dir=None, visualize=False):
    """批量推理"""
    model.eval()
    all_results = []
    
    vslzr = COCOVisualizer() if visualize else None
    
    with torch.no_grad():
        for batch_images, batch_sizes, batch_paths in tqdm(dataloader, desc="推理中"):
            batch_images = batch_images.to(device)
            batch_sizes = batch_sizes  # list of (width, height) tuples
            
            # 推理
            outputs = model(batch_images)
            
            # 后处理：使用归一化后的尺寸 [1.0, 1.0]（与notebook一致）
            batch_size = len(batch_images)
            target_sizes = torch.Tensor([[1.0, 1.0]] * batch_size).to(device)
            results = postprocessors['bbox'](outputs, target_sizes)
            
            # 处理每个图片的结果
            for idx, (output, orig_size, img_path) in enumerate(zip(results, batch_sizes, batch_paths)):
                scores = output['scores']
                labels = output['labels']
                boxes = output['boxes']  # 已经是 xyxy 格式，且是归一化的 [0, 1]
                
                # 应用阈值过滤
                select_mask = scores > threshold
                
                if select_mask.sum() == 0:
                    # 没有检测到目标
                    all_results.append({
                        'image_path': img_path,
                        'detections': []
                    })
                    continue
                
                # 转换坐标格式用于可视化
                boxes_cxcywh = box_ops.box_xyxy_to_cxcywh(boxes[select_mask])
                scores_filtered = scores[select_mask]
                labels_filtered = labels[select_mask]
                
                # 转换为原始图片尺寸（boxes 是归一化的 [0, 1]，需要乘以原始尺寸）
                w, h = orig_size
                boxes_xyxy = boxes[select_mask] * torch.Tensor([w, h, w, h]).to(device)
                
                # 构建检测结果
                detections = []
                box_labels = [id2name[int(item)] for item in labels_filtered]
                
                for i in range(len(scores_filtered)):
                    x1, y1, x2, y2 = boxes_xyxy[i].cpu().tolist()
                    detections.append({
                        'class': box_labels[i],
                        'confidence': float(scores_filtered[i].cpu().item()),
                        'bbox': {
                            'x1': float(x1),
                            'y1': float(y1),
                            'x2': float(x2),
                            'y2': float(y2)
                        }
                    })
                
                all_results.append({
                    'image_path': img_path,
                    'detections': detections
                })
                
                # 可视化（如果需要）
                if visualize and vslzr:
                    # 获取变换后的图片尺寸（用于可视化）
                    img_h, img_w = batch_images[idx].shape[1], batch_images[idx].shape[2]
                    # 生成 image_id（使用图片路径的哈希值）
                    import hashlib
                    image_id = int(hashlib.md5(img_path.encode()).hexdigest()[:8], 16) % (10**8)
                    pred_dict = {
                        'image_id': image_id,
                        'boxes': boxes_cxcywh.cpu(),
                        'size': torch.Tensor([img_h, img_w]),
                        'box_label': box_labels
                    }
                    # 保存可视化结果
                    # visualizer 期望 savedir 是目录路径，它会自动生成文件名
                    if save_dir:
                        os.makedirs(save_dir, exist_ok=True)
                        # 使用图片名称作为 caption，这样文件名会更友好
                        img_name = Path(img_path).stem
                        savedir = save_dir
                        caption = img_name
                    else:
                        savedir = None
                        caption = None
                    
                    # 使用变换后的图片进行可视化
                    image_vis = batch_images[idx].cpu()
                    vslzr.visualize(image_vis, pred_dict, savedir=savedir, caption=caption, dpi=100)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='DINO 推理脚本')
    parser.add_argument('--model_config', type=str, required=True,
                        help='模型配置文件路径 (例如: config/DINO/DINO_4scale.py)')
    parser.add_argument('--model_checkpoint', type=str, required=True,
                        help='模型权重文件路径 (例如: ckpts/xxx.pth)')
    parser.add_argument('--image_path', type=str, required=True,
                        help='图片路径（可以是单个文件或目录）')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='批量大小 (默认: 1)')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='检测阈值 (默认: 0.3)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出结果JSON文件路径 (可选)')
    parser.add_argument('--save_vis', type=str, default=None,
                        help='保存可视化结果的目录 (可选)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda 或 cpu, 默认: cuda)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数 (默认: 4)')
    
    args = parser.parse_args()
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA 不可用，使用 CPU")
        args.device = 'cpu'
    
    # 加载模型
    model, postprocessors, model_args = load_model(
        args.model_config, 
        args.model_checkpoint, 
        device=args.device
    )
    
    # 加载类别映射
    id2name = load_id2name()
    
    # 获取图片路径
    print(f"查找图片: {args.image_path}")
    image_paths = get_image_paths(args.image_path)
    print(f"找到 {len(image_paths)} 张图片")
    
    # 创建数据集和数据加载器
    max_size = getattr(model_args, 'data_aug_max_size', 800)
    dataset = ImageDataset(image_paths, max_size=max_size)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=lambda batch: (
            torch.stack([item[0] for item in batch]),
            [item[1] for item in batch],
            [item[2] for item in batch]
        )
    )
    
    # 执行推理
    visualize = args.save_vis is not None
    results = inference_batch(
        model, 
        postprocessors, 
        dataloader, 
        id2name, 
        threshold=args.threshold,
        device=args.device,
        save_dir=args.save_vis,
        visualize=visualize
    )
    
    # 保存结果
    if args.output:
        print(f"保存结果到: {args.output}")
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    else:
        # 打印结果摘要
        print("\n推理结果摘要:")
        print("=" * 60)
        for result in results:
            print(f"\n图片: {result['image_path']}")
            print(f"检测到 {len(result['detections'])} 个目标:")
            for det in result['detections']:
                print(f"  - {det['class']}: {det['confidence']:.3f}")
    
    print("\n推理完成！")


if __name__ == '__main__':
    main()

