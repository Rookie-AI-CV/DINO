#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DINO 推理性能测试脚本
计算单张图片推理时间（ms）和 FPS

使用示例:
    # 使用真实图片测试
    python benchmark_inference.py \
        --model_config config/DINO/DINO_4scale.py \
        --model_checkpoint ckpts/88830c8c6ab6a5dd006fc00191fa01ed.pth \
        --image_path ./figs/idea.jpg \
        --batch_size 1 \
        --num_iterations 100

    # 使用随机张量测试（不依赖图片）
    python benchmark_inference.py \
        --model_config config/DINO/DINO_4scale.py \
        --model_checkpoint ckpts/88830c8c6ab6a5dd006fc00191fa01ed.pth \
        --batch_size 1 \
        --num_iterations 100 \
        --use_dummy_input

    # 使用 torch.compile 编译模型测试
    python benchmark_inference.py \
        --model_config config/DINO/DINO_4scale.py \
        --model_checkpoint ckpts/88830c8c6ab6a5dd006fc00191fa01ed.pth \
        --use_dummy_input \
        --batch_size 1 \
        --use_compile \
        --compile_mode max-autotune \
        --num_iterations 100
"""

import os
import sys
import argparse
import time
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import statistics

from main import build_model_main
from util.slconfig import SLConfig
import datasets.transforms as T


def load_model(model_config_path, model_checkpoint_path, device='cuda', use_compile=False, compile_mode='default'):
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
    
    # 应用 torch.compile（如果启用）
    if use_compile:
        
        print(f"使用 torch.compile 编译模型 (模式: {compile_mode})...")
        try:
            model = torch.compile(model, mode=compile_mode)
            print("模型编译成功")
        except Exception as e:
            print(f"警告: 模型编译失败: {e}")
            print("继续使用未编译的模型")
    
    return model, postprocessors, args


def prepare_dummy_input(batch_size, device='cuda', fixed_size=1024):
    """准备随机输入张量用于测试，固定尺寸"""
    # 创建固定尺寸的输入张量 1024x1024
    h = fixed_size
    w = fixed_size
    # 归一化到 [0, 1] 范围，然后应用 ImageNet 归一化
    dummy_input = torch.rand(batch_size, 3, h, w).to(device)
    # 应用 ImageNet 归一化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    dummy_input = (dummy_input - mean) / std
    return dummy_input


def prepare_image_input(image_path, batch_size, model_args, device='cuda', fixed_size=1024):
    """从图片路径准备输入，统一裁剪到固定尺寸"""
    # 先 resize 保持宽高比，然后 center crop 到固定尺寸
    image = Image.open(image_path).convert("RGB")
    
    # 计算 resize 尺寸（保持宽高比，短边缩放到 fixed_size）
    w, h = image.size
    if w < h:
        new_w = fixed_size
        new_h = int(h * fixed_size / w)
    else:
        new_h = fixed_size
        new_w = int(w * fixed_size / h)
    
    # Resize 保持宽高比
    image = image.resize((new_w, new_h), Image.BILINEAR)
    
    # Center crop 到 fixed_size x fixed_size
    left = (new_w - fixed_size) // 2
    top = (new_h - fixed_size) // 2
    right = left + fixed_size
    bottom = top + fixed_size
    image = image.crop((left, top, right, bottom))
    
    # 转换为 tensor 并归一化
    import torchvision.transforms.functional as F
    image_tensor = F.to_tensor(image)
    image_tensor = F.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # 扩展到 batch_size
    image = image_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
    return image


def warmup(model, inputs, postprocessors, num_warmup=10, device='cuda', use_compile=False):
    """预热模型"""
    # 如果使用编译，需要更多预热次数以确保编译完成
    if use_compile:
        num_warmup = max(num_warmup, 20)  # 至少20次预热以确保编译完成
    
    print(f"预热模型 ({num_warmup} 次)...")
    model.eval()
    with torch.inference_mode():
        for i in range(num_warmup):
            outputs = model(inputs)
            # 也测试后处理
            batch_size = inputs.shape[0]
            target_sizes = torch.Tensor([[1.0, 1.0]] * batch_size).to(device)
            _ = postprocessors['bbox'](outputs, target_sizes)
            
            # 对于编译模式，前几次迭代可能较慢（编译过程）
            if use_compile and i == 0 and device == 'cuda':
                torch.cuda.synchronize()  # 确保第一次编译完成
    
    if device == 'cuda':
        torch.cuda.synchronize()
    print("预热完成")


def benchmark_forward(model, inputs, postprocessors, num_iterations=100, 
                      device='cuda', include_postprocess=False):
    """测试 forward 推理性能"""
    model.eval()
    times = []
    batch_size = inputs.shape[0]
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    print(f"开始性能测试 ({num_iterations} 次迭代, batch_size={batch_size})...")
    
    with torch.inference_mode():
        for i in range(num_iterations):
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            # Forward pass
            outputs = model(inputs)
            
            if include_postprocess:
                # 包含后处理
                target_sizes = torch.Tensor([[1.0, 1.0]] * batch_size).to(device)
                _ = postprocessors['bbox'](outputs, target_sizes)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            elapsed_time = (end_time - start_time) * 1000  # 转换为毫秒
            times.append(elapsed_time)
    
    return times


def print_statistics(times, batch_size, test_name="推理"):
    """打印统计信息"""
    times_per_image = [t / batch_size for t in times]  # 每张图片的时间
    
    mean_time = statistics.mean(times)
    mean_time_per_image = statistics.mean(times_per_image)
    median_time = statistics.median(times)
    median_time_per_image = statistics.median(times_per_image)
    std_time = statistics.stdev(times) if len(times) > 1 else 0
    min_time = min(times)
    max_time = max(times)
    
    # 计算 FPS
    fps_batch = 1000.0 / mean_time if mean_time > 0 else 0
    fps_per_image = 1000.0 / mean_time_per_image if mean_time_per_image > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"{test_name} 性能统计 (batch_size={batch_size})")
    print(f"{'='*60}")
    print(f"批次级别:")
    print(f"  平均时间: {mean_time:.2f} ms/batch")
    print(f"  中位数时间: {median_time:.2f} ms/batch")
    print(f"  标准差: {std_time:.2f} ms")
    print(f"  最小时间: {min_time:.2f} ms")
    print(f"  最大时间: {max_time:.2f} ms")
    print(f"  批次 FPS: {fps_batch:.2f} batch/s")
    print(f"\n单张图片级别:")
    print(f"  平均时间: {mean_time_per_image:.2f} ms/image")
    print(f"  中位数时间: {median_time_per_image:.2f} ms/image")
    print(f"  单张 FPS: {fps_per_image:.2f} FPS")
    print(f"{'='*60}\n")
    
    return {
        'batch_size': batch_size,
        'mean_time_batch_ms': mean_time,
        'mean_time_per_image_ms': mean_time_per_image,
        'fps_batch': fps_batch,
        'fps_per_image': fps_per_image,
        'std_ms': std_time,
        'min_ms': min_time,
        'max_ms': max_time
    }


def main():
    parser = argparse.ArgumentParser(description='DINO 推理性能测试脚本')
    parser.add_argument('--model_config', type=str, required=True,
                        help='模型配置文件路径')
    parser.add_argument('--model_checkpoint', type=str, required=True,
                        help='模型权重文件路径')
    parser.add_argument('--image_path', type=str, default=None,
                        help='测试图片路径（可选，如果不提供则使用随机张量）')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='批量大小 (默认: 1)')
    parser.add_argument('--num_iterations', type=int, default=100,
                        help='测试迭代次数 (默认: 100)')
    parser.add_argument('--num_warmup', type=int, default=10,
                        help='预热次数 (默认: 10)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda 或 cpu, 默认: cuda)')
    parser.add_argument('--use_dummy_input', action='store_true',
                        help='使用随机张量而不是真实图片')
    parser.add_argument('--include_postprocess', action='store_true',
                        help='包含后处理时间（默认只测试 forward）')
    parser.add_argument('--test_multiple_batch_sizes', action='store_true',
                        help='测试多个 batch_size (1, 2, 4, 8)')
    parser.add_argument('--use_compile', action='store_true',
                        help='使用 torch.compile 编译模型以提升性能')
    parser.add_argument('--compile_mode', type=str, default='default',
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='torch.compile 的编译模式 (默认: default)')
    parser.add_argument('--compare_compile', action='store_true',
                        help='对比编译和未编译的性能（会测试两次）')
    parser.add_argument('--fixed_size', type=int, default=1024,
                        help='固定输入图片尺寸 (默认: 1024，即 1024x1024)')
    
    args = parser.parse_args()
    
    # 如果启用对比模式，需要重新加载未编译的模型
    if args.compare_compile:
        args.use_compile = False  # 先测试未编译版本
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA 不可用，使用 CPU")
        args.device = 'cpu'
    
    # 加载模型
    model, postprocessors, model_args = load_model(
        args.model_config, 
        args.model_checkpoint, 
        device=args.device,
        use_compile=args.use_compile,
        compile_mode=args.compile_mode
    )
    
    # 显示编译状态
    compile_status = "已编译" if args.use_compile else "未编译"
    print(f"\n模型状态: {compile_status}")
    if args.use_compile:
        print(f"编译模式: {args.compile_mode}")
    print()
    
    # 准备输入
    if args.use_dummy_input or args.image_path is None:
        print(f"使用随机张量作为输入 (固定尺寸: {args.fixed_size}x{args.fixed_size})")
        inputs = prepare_dummy_input(args.batch_size, device=args.device, fixed_size=args.fixed_size)
    else:
        if not os.path.exists(args.image_path):
            raise ValueError(f"图片路径不存在: {args.image_path}")
        print(f"使用图片: {args.image_path} (裁剪到: {args.fixed_size}x{args.fixed_size})")
        inputs = prepare_image_input(args.image_path, args.batch_size, model_args, 
                                    device=args.device, fixed_size=args.fixed_size)
    
    print(f"输入形状: {inputs.shape}")
    
    # 预热
    warmup(model, inputs, postprocessors, num_warmup=args.num_warmup, 
           device=args.device, use_compile=args.use_compile)
    
    # 测试不同的 batch_size
    if args.test_multiple_batch_sizes:
        batch_sizes = [1, 2, 4, 8]
        all_results = []
        
        for bs in batch_sizes:
            print(f"\n测试 batch_size={bs}")
            # 调整输入大小
            if args.use_dummy_input or args.image_path is None:
                test_inputs = prepare_dummy_input(bs, device=args.device, fixed_size=args.fixed_size)
            else:
                test_inputs = prepare_image_input(args.image_path, bs, model_args, 
                                                  device=args.device, fixed_size=args.fixed_size)
            
            # 预热
            warmup(model, test_inputs, postprocessors, num_warmup=5, 
                   device=args.device, use_compile=args.use_compile)
            
            # 测试
            times = benchmark_forward(
                model, test_inputs, postprocessors,
                num_iterations=args.num_iterations,
                device=args.device,
                include_postprocess=args.include_postprocess
            )
            
            compile_suffix = f" (Compiled: {args.compile_mode})" if args.use_compile else ""
            test_name = "Forward" + (" + Postprocess" if args.include_postprocess else "") + compile_suffix
            result = print_statistics(times, bs, test_name=test_name)
            all_results.append(result)
        
        # 打印汇总
        print(f"\n{'='*60}")
        print("性能汇总")
        print(f"{'='*60}")
        print(f"{'Batch Size':<12} {'Time/Image (ms)':<18} {'FPS':<10}")
        print(f"{'-'*60}")
        for r in all_results:
            print(f"{r['batch_size']:<12} {r['mean_time_per_image_ms']:<18.2f} {r['fps_per_image']:<10.2f}")
        print(f"{'='*60}\n")
    else:
        # 只测试指定的 batch_size
        compile_suffix = f" (Compiled: {args.compile_mode})" if args.use_compile else ""
        test_name = "Forward" + (" + Postprocess" if args.include_postprocess else "") + compile_suffix
        times = benchmark_forward(
            model, inputs, postprocessors,
            num_iterations=args.num_iterations,
            device=args.device,
            include_postprocess=args.include_postprocess
        )
        
        result = print_statistics(times, args.batch_size, test_name=test_name)
    
    # 如果启用对比模式，再测试编译版本（仅支持单个 batch_size）
    if args.compare_compile and not args.test_multiple_batch_sizes:
        print("\n" + "="*60)
        print("开始测试编译版本...")
        print("="*60 + "\n")
        
        # 重新加载模型（编译版本）
        model_compiled, postprocessors_compiled, _ = load_model(
            args.model_config, 
            args.model_checkpoint, 
            device=args.device,
            use_compile=True,
            compile_mode=args.compile_mode
        )
        
        # 准备输入（与之前相同）
        if args.use_dummy_input or args.image_path is None:
            inputs_compiled = prepare_dummy_input(args.batch_size, device=args.device, fixed_size=args.fixed_size)
        else:
            inputs_compiled = prepare_image_input(args.image_path, args.batch_size, model_args, 
                                                 device=args.device, fixed_size=args.fixed_size)
        
        # 预热编译版本
        warmup(model_compiled, inputs_compiled, postprocessors_compiled, 
               num_warmup=max(args.num_warmup, 20), device=args.device, use_compile=True)
        
        # 测试编译版本
        compile_suffix = f" (Compiled: {args.compile_mode})"
        test_name_compiled = "Forward" + (" + Postprocess" if args.include_postprocess else "") + compile_suffix
        times_compiled = benchmark_forward(
            model_compiled, inputs_compiled, postprocessors_compiled,
            num_iterations=args.num_iterations,
            device=args.device,
            include_postprocess=args.include_postprocess
        )
        
        result_compiled = print_statistics(times_compiled, args.batch_size, test_name=test_name_compiled)
        
        # 打印对比结果
        print(f"\n{'='*60}")
        print("性能对比 (未编译 vs 编译)")
        print(f"{'='*60}")
        print(f"{'版本':<25} {'Time/Image (ms)':<20} {'FPS':<15} {'加速比':<10}")
        print(f"{'-'*70}")
        
        time_uncompiled = result['mean_time_per_image_ms']
        fps_uncompiled = result['fps_per_image']
        time_compiled = result_compiled['mean_time_per_image_ms']
        fps_compiled = result_compiled['fps_per_image']
        speedup = time_uncompiled / time_compiled if time_compiled > 0 else 0
        
        print(f"{'未编译':<25} {time_uncompiled:<20.2f} {fps_uncompiled:<15.2f} {'1.00x':<10}")
        print(f"{'编译 (' + args.compile_mode + ')':<25} {time_compiled:<20.2f} {fps_compiled:<15.2f} {f'{speedup:.2f}x':<10}")
        print(f"{'='*70}\n")
        
        if speedup > 1:
            print(f"✅ 编译版本加速: {speedup:.2f}x ({((speedup-1)*100):.1f}% 提升)")
        elif speedup < 1:
            print(f"⚠️  编译版本较慢: {1/speedup:.2f}x ({((1-speedup)*100):.1f}% 下降)")
        else:
            print("➡️  性能相同")
        print()
    elif args.compare_compile and args.test_multiple_batch_sizes:
        print("\n⚠️  警告: 对比模式不支持多 batch_size 测试，请使用单个 batch_size")
    
    print("性能测试完成！")


if __name__ == '__main__':
    main()

