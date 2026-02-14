#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import argparse
import numpy as np
from PIL import Image
import imageio
from tqdm import tqdm
import torch
from einops import rearrange
import subprocess
import tempfile
import shutil
import json
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

# 添加 pymediainfo 导入
try:
    from pymediainfo import MediaInfo
    PYMEDIAINFO_AVAILABLE = True
except ImportError:
    print("警告: 未安装 pymediainfo，将使用备用方法")
    PYMEDIAINFO_AVAILABLE = False

from diffsynth import ModelManager, FlashVSRTinyLongPipeline
from utils.utils import Causal_LQ4x_Proj
from utils.TCDecoder import build_tcdecoder

# 全局变量用于记录日志信息
log_context = {"current_task": 0, "total_tasks": 0, "parallel_tasks": 0}

def log_message(message, task_id=None, parallel_tasks=None):
    """统一的日志输出函数，支持任务序号和并行任务数标识"""
    if task_id is None:
        task_id = log_context.get("current_task", 0)
    if parallel_tasks is None:
        parallel_tasks = log_context.get("parallel_tasks", 0)
    
    prefix = f"[{task_id}/{parallel_tasks}]" if task_id > 0 else "[0/0]"
    print(f"{prefix} {message}")

def get_video_info_accurate(path):
    """使用 pymediainfo 获取精确的视频信息"""
    if not PYMEDIAINFO_AVAILABLE:
        return get_video_info_fallback(path)
    
    try:
        media_info = MediaInfo.parse(path)
        video_track = None
        general_track = None
        
        for track in media_info.tracks:
            if track.track_type == 'Video':
                video_track = track
            elif track.track_type == 'General':
                general_track = track
        
        if not video_track:
            raise ValueError("未找到视频流")
        
        # 获取精确的帧率
        frame_rate = None
        if hasattr(video_track, 'frame_rate') and video_track.frame_rate:
            frame_rate_str = str(video_track.frame_rate)
            # 处理分数形式的帧率 (如 30000/1001)
            if '/' in frame_rate_str:
                try:
                    numerator, denominator = map(float, frame_rate_str.split('/'))
                    frame_rate = numerator / denominator
                except:
                    frame_rate = float(frame_rate_str)
            else:
                frame_rate = float(frame_rate_str)
        
        # 获取总帧数（优先使用精确统计）
        frame_count = None
        if hasattr(video_track, 'frame_count') and video_track.frame_count:
            frame_count = int(video_track.frame_count)
        
        # 获取时长（毫秒）- 优先使用视频流时长
        duration_ms = None
        if hasattr(video_track, 'duration') and video_track.duration:
            duration_ms = float(video_track.duration)
            log_message(f"使用视频流时长: {duration_ms}ms")
        elif hasattr(general_track, 'duration') and general_track.duration:
            duration_ms = float(general_track.duration)
            log_message(f"使用容器时长: {duration_ms}ms")
        
        # 计算时长（秒）
        duration_seconds = duration_ms / 1000.0 if duration_ms else None
        
        # 如果帧数未知但时长和帧率已知，计算帧数
        if frame_count is None and duration_seconds and frame_rate:
            frame_count = int(round(duration_seconds * frame_rate))
        
        # 如果时长未知但帧数和帧率已知，计算时长
        if duration_seconds is None and frame_count and frame_rate:
            duration_seconds = frame_count / frame_rate
        
        # 默认值
        if frame_rate is None:
            frame_rate = 25.0
        if frame_count is None:
            # 使用备用方法
            return get_video_info_fallback(path)
        if duration_seconds is None:
            duration_seconds = frame_count / frame_rate
        
        # 获取分辨率
        width = int(video_track.width) if hasattr(video_track, 'width') else 0
        height = int(video_track.height) if hasattr(video_track, 'height') else 0
        
        # 验证数据一致性
        calculated_frames = int(round(duration_seconds * frame_rate))
        if abs(calculated_frames - frame_count) > 2:  # 允许2帧误差
            log_message(f"帧数不一致: 统计={frame_count}, 计算={calculated_frames}, 使用统计值")
        
        log_message(f"pymediainfo 精确信息: {frame_count}帧, {frame_rate:.6f}fps, {duration_seconds:.6f}秒")
        
        return {
            'frame_count': frame_count,
            'frame_rate': frame_rate,
            'duration': duration_seconds,
            'width': width,
            'height': height,
            'is_accurate': True
        }
        
    except Exception as e:
        log_message(f"pymediainfo 解析失败: {e}，使用备用方法")
        return get_video_info_fallback(path)

def get_video_info_fallback(path):
    """备用方法：使用 ffprobe 获取视频信息"""
    try:
        # 使用 ffprobe 获取精确帧数
        frame_count_cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-count_frames',
            '-show_entries', 'stream=nb_read_frames',
            '-of', 'csv=p=0',
            path
        ]
        
        result = subprocess.run(frame_count_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            frame_count = int(result.stdout.strip())
        else:
            frame_count = 0
        
        # 获取时长和帧率
        info_cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate,duration,width,height',
            '-of', 'json',
            path
        ]
        
        result = subprocess.run(info_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            info = json.loads(result.stdout)
            stream = info['streams'][0] if info['streams'] else {}
            
            # 解析帧率
            frame_rate_str = stream.get('r_frame_rate', '25/1')
            if '/' in frame_rate_str:
                num, den = map(int, frame_rate_str.split('/'))
                frame_rate = num / den
            else:
                frame_rate = float(frame_rate_str)
            
            # 获取时长
            duration = float(stream.get('duration', 0))
            
            # 如果时长未知，使用帧数和帧率计算
            if duration <= 0 and frame_count > 0:
                duration = frame_count / frame_rate
            
            # 如果帧数未知，使用时長和帧率计算
            if frame_count <= 0 and duration > 0:
                frame_count = int(round(duration * frame_rate))
            
            width = int(stream.get('width', 0))
            height = int(stream.get('height', 0))
            
            log_message(f"ffprobe 信息: {frame_count}帧, {frame_rate:.6f}fps, {duration:.6f}秒")
            
            return {
                'frame_count': frame_count,
                'frame_rate': frame_rate,
                'duration': duration,
                'width': width,
                'height': height,
                'is_accurate': False
            }
    
    except Exception as e:
        log_message(f"备用方法也失败: {e}")
    
    # 最终备用值
    return {
        'frame_count': 250,  # 默认值
        'frame_rate': 25.0,
        'duration': 10.0,
        'width': 720,
        'height': 540,
        'is_accurate': False
    }

def tensor2video(frames):
    frames = rearrange(frames, "C T H W -> T H W C")
    frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    frames = [Image.fromarray(frame) for frame in frames]
    return frames

def natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'([0-9]+)', os.path.basename(name))]

def list_images_natural(folder: str):
    exts = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    fs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(exts)]
    fs.sort(key=natural_key)
    return fs

def list_videos_natural(folder: str):
    exts = ('.mp4', '.mov', '.avi', '.mkv', '.MP4', '.MOV', '.AVI', '.MKV')
    fs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(exts)]
    fs.sort(key=natural_key)
    return fs

def get_input_files(input_path):
    input_files = []
    
    if os.path.isfile(input_path):
        input_files.append(input_path)
    elif os.path.isdir(input_path):
        video_files = list_videos_natural(input_path)
        input_files.extend(video_files)
        
        for item in os.listdir(input_path):
            item_path = os.path.join(input_path, item)
            if os.path.isdir(item_path):
                image_files = list_images_natural(item_path)
                if image_files:
                    input_files.append(item_path)
    else:
        raise ValueError(f"输入路径不存在: {input_path}")
    
    return input_files

def largest_8n1_leq(n):
    return 0 if n < 1 else ((n - 1)//8)*8 + 1

def is_video(path):
    return os.path.isfile(path) and path.lower().endswith(('.mp4','.mov','.avi','.mkv'))

def pil_to_tensor_neg1_1(img: Image.Image, dtype=torch.bfloat16, device='cuda'):
    t = torch.from_numpy(np.asarray(img, np.uint8)).to(device=device, dtype=torch.float32)
    t = t.permute(2,0,1) / 255.0 * 2.0 - 1.0
    return t.to(dtype)

def save_video_directly_from_tensor(frames, save_path, fps=30, quality=5):
    """直接从张量创建视频，跳过PNG中间步骤"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    w = imageio.get_writer(save_path, fps=fps, quality=quality)
    
    # 将张量转换为numpy数组并保存
    frames_array = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    frames_array = rearrange(frames_array, "C T H W -> T H W C")
    
    for i in tqdm(range(frames_array.shape[0]), desc=f"Creating video directly {os.path.basename(save_path)}"):
        frame = frames_array[i]
        w.append_data(frame)
    w.close()

def compute_scaled_and_target_dims(w0: int, h0: int, scale: float = 4.0, multiple: int = 128):
    if w0 <= 0 or h0 <= 0:
        raise ValueError("Invalid original size")
    if scale <= 0:
        raise ValueError("scale must be > 0")

    sW = int(round(w0 * scale))
    sH = int(round(h0 * scale))
    tW = (sW // multiple) * multiple
    tH = (sH // multiple) * multiple

    if tW == 0 or tH == 0:
        raise ValueError(
            f"Scaled size too small ({sW}x{sH}) for multiple={multiple}. "
            f"Increase scale (got {scale})."
        )
    return sW, sH, tW, tH

def upscale_then_center_crop(img: Image.Image, scale: float, tW: int, tH: int) -> Image.Image:
    w0, h0 = img.size
    sW = int(round(w0 * scale))
    sH = int(round(h0 * scale))

    if tW > sW or tH > sH:
        raise ValueError(
            f"Target crop ({tW}x{tH}) exceeds scaled size ({sW}x{sH}). "
            f"Increase scale."
        )

    up = img.resize((sW, sH), Image.BICUBIC)
    l = (sW - tW) // 2
    t = (sH - tH) // 2
    return up.crop((l, t, l + tW, t + tH))

def prepare_input_tensor(path: str, scale: float = 4, dtype=torch.bfloat16, device='cuda'):
    """准备输入张量，使用精确的视频信息"""
    # 使用精确的视频信息获取
    if is_video(path):
        video_info = get_video_info_accurate(path)
        original_frame_count = video_info['frame_count']
        original_fps = video_info['frame_rate']
        original_duration = video_info['duration']
        w0 = video_info['width']
        h0 = video_info['height']
        
        is_video_input = True
        
        log_message(f"精确视频信息: {original_frame_count}帧, {original_fps:.6f}fps, {original_duration:.6f}秒, {w0}x{h0}")
        
        # 验证数据一致性
        calculated_frames = int(round(original_duration * original_fps))
        if abs(calculated_frames - original_frame_count) > 1:
            log_message(f"警告: 帧数不一致，使用统计值 {original_frame_count} 而非计算值 {calculated_frames}")
        
    else:
        # 图像序列处理（保持原逻辑）
        paths0 = list_images_natural(path)
        if not paths0:
            raise FileNotFoundError(f"No images in {path}")

        with Image.open(paths0[0]) as _img0:
            w0, h0 = _img0.size
            original_frame_count = len(paths0)
            original_fps = 30.0
            original_duration = original_frame_count / original_fps

        is_video_input = False
        log_message(f"图像序列: {original_frame_count}帧, {original_fps}fps, {original_duration:.2f}秒, {w0}x{h0}")

    # 计算目标尺寸
    sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
    log_message(f"缩放目标: {w0}x{h0} -> {sW}x{sH} -> {tW}x{tH} (x{scale:.2f})")

    # 帧处理逻辑
    if is_video(path):
        # 视频文件处理
        rdr = imageio.get_reader(path)
        
        # 使用精确的帧数
        total = original_frame_count
        
        idx = list(range(total)) + [total-1]*4
        F = largest_8n1_leq(len(idx))
        if F == 0:
            rdr.close()
            raise RuntimeError(f"帧数不足: {path}, 得到 {len(idx)} 帧")
        
        idx = idx[:F]
        processed_frame_count = F - 4  # 实际处理的帧数（减去填充）
        
        log_message(f"帧处理: 原始{total}帧 -> 填充后{F}帧 -> 实际处理{processed_frame_count}帧")

        frames = []
        try:
            for i in idx:
                img = Image.fromarray(rdr.get_data(i)).convert('RGB')
                img_out = upscale_then_center_crop(img, scale=scale, tW=tW, tH=tH)
                frames.append(pil_to_tensor_neg1_1(img_out, dtype, 'cpu'))
        finally:
            try: 
                rdr.close()
            except Exception: 
                pass

        vid = torch.stack(frames, 0).permute(1,0,2,3).unsqueeze(0)
        fps = original_fps
        
    else:
        # 图像序列处理
        paths0 = list_images_natural(path)
        paths = paths0 + [paths0[-1]] * 4
        F = largest_8n1_leq(len(paths))
        if F == 0:
            raise RuntimeError(f"帧数不足: {path}, 得到 {len(paths)} 帧")
        
        paths = paths[:F]
        processed_frame_count = F - 4
        
        log_message(f"帧处理: 原始{len(paths0)}帧 -> 填充后{F}帧 -> 实际处理{processed_frame_count}帧")

        frames = []
        for p in paths:
            with Image.open(p).convert('RGB') as img:
                img_out = upscale_then_center_crop(img, scale=scale, tW=tW, tH=tH)
                frames.append(pil_to_tensor_neg1_1(img_out, dtype, 'cpu'))
        
        vid = torch.stack(frames, 0).permute(1,0,2,3).unsqueeze(0)
        fps = original_fps

    return (vid, tH, tW, F, fps, is_video_input, original_fps, original_duration, original_frame_count, processed_frame_count)

def init_pipeline(gpu_id=0):
    # 首先检查CUDA是否可用
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用")
    
    # 检查GPU数量
    gpu_count = torch.cuda.device_count()
    log_message(f"系统检测到 {gpu_count} 个GPU")
    
    for i in range(gpu_count):
        log_message(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 检查请求的GPU是否存在
    if gpu_id >= gpu_count:
        raise RuntimeError(f"GPU {gpu_id} 不存在，可用GPU: 0-{gpu_count-1}")
    
    # 设置当前设备
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(gpu_id)
    
    log_message(f"正在使用GPU: {gpu_id} ({torch.cuda.get_device_name(gpu_id)})")
    log_message(f"GPU内存: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.1f} GB")
    
    # 初始化模型管理器
    mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    mm.load_models([
        "./FlashVSR-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors",
    ])
    
    # 创建管道
    pipe = FlashVSRTinyLongPipeline.from_model_manager(mm, device=device)
    
    # 加载LQ投影器
    pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=torch.bfloat16)
    LQ_proj_in_path = "./FlashVSR-v1.1/LQ_proj_in.ckpt"
    if os.path.exists(LQ_proj_in_path):
        pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(LQ_proj_in_path, map_location="cpu"), strict=True)
        pipe.denoising_model().LQ_proj_in.to(device)

    # 加载TC解码器
    multi_scale_channels = [512, 256, 128, 128]
    pipe.TCDecoder = build_tcdecoder(new_channels=multi_scale_channels, new_latent_channels=16+768)
    mis = pipe.TCDecoder.load_state_dict(torch.load("./FlashVSR-v1.1/TCDecoder.ckpt"), strict=False)
    log_message(f"TC解码器加载状态: {mis}")

    # 将管道移动到GPU
    pipe.to(device)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv()
    pipe.load_models_to_device(["dit","vae"])
    
    return pipe, device

def process_video_finalization(args):
    """处理视频最终化任务 - 直接方法版"""
    task_id, parallel_tasks, temp_dir, video_tensor, temp_video_path, final_video_path, original_fps, original_frame_count, original_duration, is_video_file, input_path, _ = args
    
    try:
        log_message(f"开始创建纯视频文件: {os.path.basename(final_video_path)}", task_id, parallel_tasks)
        log_message(f"  使用帧率: {original_fps:.6f} FPS, 处理帧数: {video_tensor.shape[2]}", task_id, parallel_tasks)
        
        # 直接方法：从张量直接创建视频
        log_message("使用直接方法：从张量直接创建视频（跳过PNG中转）", task_id, parallel_tasks)
        
        # 创建视频目录
        os.makedirs(os.path.dirname(temp_video_path), exist_ok=True)
        
        # 从张量直接创建视频
        save_video_directly_from_tensor(video_tensor, temp_video_path, fps=original_fps, quality=5)
        
        log_message(f"视频创建完成: {os.path.basename(final_video_path)}", task_id, parallel_tasks)
        
        # 复制临时视频到最终路径
        shutil.copy(temp_video_path, final_video_path)
        log_message(f"完成纯视频输出: {os.path.basename(final_video_path)}", task_id, parallel_tasks)
        
        # 验证最终文件参数
        try:
            final_info = get_video_info_accurate(final_video_path)
            log_message(f"最终视频参数:", task_id, parallel_tasks)
            log_message(f"  帧数: {final_info['frame_count']} (原始: {original_frame_count})", task_id, parallel_tasks)
            log_message(f"  帧率: {final_info['frame_rate']:.6f} (原始: {original_fps:.6f})", task_id, parallel_tasks)
            log_message(f"  时长: {final_info['duration']:.6f}秒", task_id, parallel_tasks)
            
            # 检查参数一致性
            fps_match = abs(final_info['frame_rate'] - original_fps) < 0.01
            
            if fps_match:
                log_message("帧率一致性: ✅ 匹配", task_id, parallel_tasks)
            else:
                log_message(f"帧率差异: {final_info['frame_rate']:.6f} vs {original_fps:.6f}", task_id, parallel_tasks)
        except Exception as e:
            log_message(f"无法验证最终文件参数: {e}", task_id, parallel_tasks)
        
        # 清理临时文件
        try:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
        except:
            pass
            
        return True, final_video_path
        
    except Exception as e:
        log_message(f"视频最终化失败 {os.path.basename(final_video_path)}: {e}", task_id, parallel_tasks)
        return False, final_video_path

def main():
    parser = argparse.ArgumentParser(description='FlashVSR视频超分辨率处理')
    parser.add_argument('--input', type=str, required=True, help='输入文件路径或目录路径')
    parser.add_argument('--output', type=str, default='./results', help='输出目录路径')
    parser.add_argument('--preprocess', action='store_true', help='已移除：参数保留但无效')
    parser.add_argument('--gpu', type=int, default=0, help='GPU设备ID (0, 1, 2, 3)')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--scale', type=float, default=4.0, help='缩放比例')
    parser.add_argument('--sparse_ratio', type=float, default=2.0, help='稀疏比率')
    parser.add_argument('--local_range', type=int, default=11, help='局部范围')
    parser.add_argument('--max_workers', type=int, default=2, help='并行处理的最大工作线程数')
    
    args = parser.parse_args()
    
    log_message("=== FlashVSR GPU设置 ===")
    log_message(f"请求使用GPU: {args.gpu}")
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        log_message("错误: CUDA不可用，请检查CUDA安装")
        return
    
    gpu_count = torch.cuda.device_count()
    log_message(f"系统检测到 {gpu_count} 个GPU")
    
    if args.gpu >= gpu_count:
        log_message(f"错误: GPU {args.gpu} 不存在，可用GPU: 0-{gpu_count-1}")
        return
    
    # 提示预处理参数已移除
    if args.preprocess:
        log_message("注意: --preprocess 参数功能已移除，将跳过视频预处理")
    
    RESULT_ROOT = args.output
    os.makedirs(RESULT_ROOT, exist_ok=True)
    
    # 创建临时目录在输出目录下
    TEMP_ROOT = os.path.join(RESULT_ROOT, "temp")
    os.makedirs(TEMP_ROOT, exist_ok=True)
    log_message(f"临时文件目录: {TEMP_ROOT}")
    
    # 获取输入文件列表
    try:
        input_files = get_input_files(args.input)
    except Exception as e:
        log_message(f"输入文件错误: {e}")
        return
    
    if not input_files:
        log_message(f"在路径 {args.input} 中没有找到支持的视频文件或图像序列")
        return
    
    log_message(f"找到 {len(input_files)} 个文件需要处理")
    
    try:
        # 初始化管道
        pipe, device = init_pipeline(args.gpu)
    except Exception as e:
        log_message(f"管道初始化失败: {e}")
        return
    
    # 创建线程池用于并行处理
    executor = ThreadPoolExecutor(max_workers=args.max_workers)
    futures = []
    
    # 处理每个文件
    for i, input_path in enumerate(input_files, 1):
        log_message(f"=== 处理文件 {i}/{len(input_files)} ===")
        log_message(f"输入: {input_path}")
        
        # 更新全局日志上下文
        log_context["current_task"] = i
        log_context["parallel_tasks"] = len(futures) + 1
        
        # 清理GPU内存
        torch.cuda.empty_cache()
        
        name = os.path.basename(input_path.rstrip('/'))
        if name.startswith('.'):
            continue
        
        # 检查是否为视频文件
        is_video_file = is_video(input_path)
        
        # 注意：FFmpeg预处理功能已移除
        
        # 使用精确的视频信息获取
        try:
            LQ, th, tw, F, fps, from_video, original_fps, original_duration, original_frame_count, processed_frame_count = prepare_input_tensor(
                input_path, scale=args.scale, dtype=torch.bfloat16, device=device)
        except Exception as e:
            log_message(f"[错误] 准备输入张量失败: {e}", i, len(futures)+1)
            continue

        # 对于视频文件，显示信息
        if is_video_file:
            log_message(f"视频文件信息: {original_frame_count}帧, {original_fps:.6f}fps, {original_duration:.6f}秒", i, len(futures)+1)
            log_message(f"处理后帧数: {processed_frame_count}帧", i, len(futures)+1)

        try:
            log_message("开始FlashVSR处理...", i, len(futures)+1)
            video = pipe(
                prompt="", 
                negative_prompt="", 
                cfg_scale=1.0, 
                num_inference_steps=1, 
                seed=args.seed,
                LQ_video=LQ, 
                num_frames=F, 
                height=th, 
                width=tw, 
                is_full_block=False, 
                if_buffer=True,
                topk_ratio=args.sparse_ratio*768 * 1280/(th*tw), 
                kv_ratio=3.0,
                local_range=args.local_range,
                color_fix=True,
            )

            # 计算处理后的实际帧数（减去填充的帧）
            processed_frame_count = F - 4
            
            # 总是使用直接方法创建视频
            log_message("使用直接方法：从张量直接创建视频（跳过PNG中转）", i, len(futures)+1)
            
            # 生成输出文件名
            if os.path.isdir(input_path):
                base_name = os.path.basename(input_path.rstrip('/'))
            else:
                base_name = os.path.splitext(os.path.basename(input_path))[0]
            
            # 创建临时目录
            temp_dir = tempfile.mkdtemp(dir=TEMP_ROOT, prefix=f"temp_{base_name}_")
            log_message(f"临时目录: {temp_dir}", i, len(futures)+1)
            
            # 准备最终视频路径
            temp_video_path = os.path.join(temp_dir, "temp_video.mp4")
            final_video_filename = f"FlashVSR_v1.1_Tiny_Long_{base_name}_gpu{args.gpu}_seed{args.seed}.mp4"
            final_video_path = os.path.join(RESULT_ROOT, final_video_filename)
            
            # 提交并行处理任务
            log_message(f"提交并行处理任务: {os.path.basename(final_video_path)}", i, len(futures)+1)
            future = executor.submit(process_video_finalization, (
                i, len(futures)+1, temp_dir, video, temp_video_path, final_video_path, 
                original_fps, original_frame_count, original_duration, is_video_file, input_path, True
            ))
            futures.append((future, final_video_path, temp_dir, i))
            
            log_message(f"当前并行任务数: {len(futures)}", i, len(futures)+1)
            
            # 如果并行任务达到上限，等待部分任务完成
            if len(futures) >= args.max_workers * 2:
                log_message("达到并行任务上限，等待部分任务完成...", i, len(futures)+1)
                completed_count = 0
                for f, path, temp_dir, task_id in futures[:]:
                    if f.done():
                        try:
                            success, result_path = f.result(timeout=1)
                            if success:
                                log_message(f"并行任务完成: {os.path.basename(result_path)}", task_id, len(futures))
                            else:
                                log_message(f"并行任务失败: {os.path.basename(result_path)}", task_id, len(futures))
                            # 清理临时目录
                            try:
                                shutil.rmtree(temp_dir)
                                log_message(f"清理临时目录: {os.path.basename(temp_dir)}", task_id, len(futures))
                            except:
                                pass
                            futures.remove((f, path, temp_dir, task_id))
                            completed_count += 1
                        except:
                            pass
                
                if completed_count > 0:
                    log_message(f"已完成 {completed_count} 个任务，继续处理...", i, len(futures)+1)
            
        except Exception as e:
            log_message(f"[处理错误] {name}: {e}", i, len(futures)+1)
            # 清理临时目录
            try:
                shutil.rmtree(temp_dir)
                log_message(f"清理临时目录（错误时）: {os.path.basename(temp_dir)}", i, len(futures)+1)
            except:
                pass
            continue

    log_message(f"等待所有并行任务完成...")
    
    # 等待所有剩余任务完成
    completed_count = 0
    failed_count = 0
    
    for future, final_video_path, temp_dir, task_id in futures:
        try:
            success, result_path = future.result(timeout=300)  # 5分钟超时
            if success:
                log_message(f"任务完成: {os.path.basename(result_path)}", task_id, len(futures))
                completed_count += 1
            else:
                log_message(f"任务失败: {os.path.basename(result_path)}", task_id, len(futures))
                failed_count += 1
        except Exception as e:
            log_message(f"任务超时或失败 {os.path.basename(final_video_path)}: {e}", task_id, len(futures))
            failed_count += 1
        finally:
            # 清理临时目录
            try:
                shutil.rmtree(temp_dir)
                log_message(f"清理临时目录: {os.path.basename(temp_dir)}", task_id, len(futures))
            except:
                pass
    
    # 关闭线程池
    executor.shutdown(wait=True)
    
    # 尝试清理整个临时目录（如果为空）
    try:
        if os.path.exists(TEMP_ROOT) and not os.listdir(TEMP_ROOT):
            shutil.rmtree(TEMP_ROOT)
            log_message(f"清理临时根目录: {TEMP_ROOT}")
    except:
        pass
    
    log_message(f"\n=== 所有文件处理完成 ===")
    log_message(f"✅ 成功: {completed_count} 个文件")
    log_message(f"❌ 失败: {failed_count} 个文件")
    log_message(f"📁 输出目录: {RESULT_ROOT}")
    log_message(f"🗑️ 临时文件已清理")
    log_message("🎯 输出文件为纯视频流（无音频）")

if __name__ == "__main__":
    main()
