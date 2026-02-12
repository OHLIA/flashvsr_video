#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, time, argparse
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
            print(f"✅ 使用视频流时长: {duration_ms}ms")
        elif hasattr(general_track, 'duration') and general_track.duration:
            duration_ms = float(general_track.duration)
            print(f"⚠️ 使用容器时长: {duration_ms}ms")
        
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
            print(f"⚠️ 帧数不一致: 统计={frame_count}, 计算={calculated_frames}, 使用统计值")
        
        print(f"📊📊 pymediainfo 精确信息: {frame_count}帧, {frame_rate:.6f}fps, {duration_seconds:.6f}秒")
        
        return {
            'frame_count': frame_count,
            'frame_rate': frame_rate,
            'duration': duration_seconds,
            'width': width,
            'height': height,
            'is_accurate': True
        }
        
    except Exception as e:
        print(f"❌❌ pymediainfo 解析失败: {e}，使用备用方法")
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
            
            print(f"📊📊 ffprobe 信息: {frame_count}帧, {frame_rate:.6f}fps, {duration:.6f}秒")
            
            return {
                'frame_count': frame_count,
                'frame_rate': frame_rate,
                'duration': duration,
                'width': width,
                'height': height,
                'is_accurate': False
            }
    
    except Exception as e:
        print(f"❌❌ 备用方法也失败: {e}")
    
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

def save_frames_as_png(frames, output_dir, base_name="frame"):
    """保存帧序列为PNG图片"""
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    
    for i, frame in enumerate(tqdm(frames, desc=f"Saving PNG frames")):
        filename = f"{base_name}_{i:06d}.png"
        filepath = os.path.join(output_dir, filename)
        frame.save(filepath, 'PNG')
        saved_paths.append(filepath)
    
    return saved_paths

def save_video_from_frames(frame_paths, save_path, fps=30, quality=5):
    """从帧路径列表创建视频"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    w = imageio.get_writer(save_path, fps=fps, quality=quality)
    
    for frame_path in tqdm(frame_paths, desc=f"Creating video {os.path.basename(save_path)}"):
        frame = Image.open(frame_path)
        w.append_data(np.array(frame))
    w.close()

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

def pad_frames_to_match_original_uniform(processed_frames_dir, original_frame_count, output_dir):
    """
    改进的帧填补方法：将缺失帧均匀插入到视频序列中
    方案：每间隔N帧插入一帧，确保最终帧数等于原始帧数
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取处理后的帧
    processed_frames = [f for f in os.listdir(processed_frames_dir) if f.endswith('.png')]
    processed_frames.sort()
    processed_count = len(processed_frames)
    
    frames_needed = original_frame_count - processed_count
    print(f"📊📊 均匀填补: 处理{processed_count}帧, 需要{original_frame_count}帧, 需填补{frames_needed}帧")
    
    if frames_needed <= 0:
        # 如果处理后的帧数多于或等于原始帧数，直接复制前N帧
        for i in range(original_frame_count):
            src_path = os.path.join(processed_frames_dir, processed_frames[i])
            dst_path = os.path.join(output_dir, f"frame_{i:06d}.png")
            shutil.copy2(src_path, dst_path)
        final_count_actual = original_frame_count
    else:
        # === 核心改进：计算均匀插入的间隔 ===
        # 计算理论插入间隔（向下取整）
        interval = max(1, processed_count // (frames_needed + 1))
        print(f"🎯 均匀填补: 每 {interval} 帧后插入1帧，共需插入 {frames_needed} 帧")
        
        # 构建新的帧序列
        new_frame_list = []
        processed_idx = 0
        insertions_made = 0
        
        # 遍历所有处理后的帧
        for i in range(processed_count):
            # 添加当前处理帧
            src_path = os.path.join(processed_frames_dir, processed_frames[i])
            dst_filename = f"frame_{len(new_frame_list):06d}.png"
            dst_path = os.path.join(output_dir, dst_filename)
            shutil.copy2(src_path, dst_path)
            new_frame_list.append(dst_path)
            
            # 判断是否需要在此处插入一帧（在特定间隔后插入，且不是最后一帧）
            # 条件：(i+1)能被间隔整除，还有剩余帧需要插入，且不是处理序列的最后一帧
            if ((i + 1) % interval == 0 and 
                insertions_made < frames_needed and 
                i < processed_count - 1):
                
                # 选择要插入的帧：使用当前帧或前一个帧，以达到平滑过渡
                # 这里使用当前帧进行复制（效果类似于短暂停留）
                insert_src_path = os.path.join(processed_frames_dir, processed_frames[i])
                insert_dst_filename = f"frame_{len(new_frame_list):06d}.png"
                insert_dst_path = os.path.join(output_dir, insert_dst_filename)
                shutil.copy2(insert_src_path, insert_dst_path)
                
                new_frame_list.append(insert_dst_path)
                insertions_made += 1
                print(f"  → 插入第 {insertions_made}/{frames_needed} 帧，位置在原始帧 {i+1} 之后")
        
        # 如果还有未插入的帧（通常因为末尾不足一个间隔），在末尾补齐
        while insertions_made < frames_needed:
            # 使用最后一帧进行补齐
            src_path = os.path.join(processed_frames_dir, processed_frames[-1])
            dst_filename = f"frame_{len(new_frame_list):06d}.png"
            dst_path = os.path.join(output_dir, dst_filename)
            shutil.copy2(src_path, dst_path)
            
            new_frame_list.append(dst_path)
            insertions_made += 1
            print(f"  → 末尾补齐第 {insertions_made}/{frames_needed} 帧")
        
        final_count_actual = len(new_frame_list)
    
    # 验证最终帧数
    final_frames = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    final_frames.sort()
    final_count_actual = len(final_frames)
    
    if final_count_actual != original_frame_count:
        print(f"❌❌ 帧数验证失败: 期望{original_frame_count}, 实际{final_count_actual}")
        # 强制调整到正确帧数
        if final_count_actual < original_frame_count:
            # 如果还是少了，用最后一帧补齐
            last_frame_path = os.path.join(output_dir, f"frame_{final_count_actual-1:06d}.png")
            for i in range(final_count_actual, original_frame_count):
                dst_path = os.path.join(output_dir, f"frame_{i:06d}.png")
                shutil.copy2(last_frame_path, dst_path)
                final_frames.append(dst_path)
                print(f"  → 强制补齐第 {i-final_count_actual+1} 帧")
            final_count_actual = original_frame_count
    else:
        print(f"✅ 帧数验证成功: {final_count_actual}帧，使用均匀填补法")
    
    return [os.path.join(output_dir, f) for f in final_frames]

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
        
        print(f"🎯🎯 精确视频信息: {original_frame_count}帧, {original_fps:.6f}fps, {original_duration:.6f}秒, {w0}x{h0}")
        
        # 验证数据一致性
        calculated_frames = int(round(original_duration * original_fps))
        if abs(calculated_frames - original_frame_count) > 1:
            print(f"⚠️ 警告: 帧数不一致，使用统计值 {original_frame_count} 而非计算值 {calculated_frames}")
        
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
        print(f"📁📁 图像序列: {original_frame_count}帧, {original_fps}fps, {original_duration:.2f}秒, {w0}x{h0}")

    # 计算目标尺寸
    sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
    print(f"📐📐 缩放目标: {w0}x{h0} -> {sW}x{sH} -> {tW}x{tH} (x{scale:.2f})")

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
        
        print(f"🔄🔄 帧处理: 原始{total}帧 -> 填充后{F}帧 -> 实际处理{processed_frame_count}帧")

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
        
        print(f"🔄🔄 帧处理: 原始{len(paths0)}帧 -> 填充后{F}帧 -> 实际处理{processed_frame_count}帧")

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
    print(f"系统检测到 {gpu_count} 个GPU")
    
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 检查请求的GPU是否存在
    if gpu_id >= gpu_count:
        raise RuntimeError(f"GPU {gpu_id} 不存在，可用GPU: 0-{gpu_count-1}")
    
    # 设置当前设备
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(gpu_id)
    
    print(f"正在使用GPU: {gpu_id} ({torch.cuda.get_device_name(gpu_id)})")
    print(f"GPU内存: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.1f} GB")
    
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
    print("TC解码器加载状态:", mis)

    # 将管道移动到GPU
    pipe.to(device)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv()
    pipe.load_models_to_device(["dit","vae"])
    
    return pipe, device

def process_video_finalization(args):
    """处理视频最终化任务（使用改进的均匀填补法）"""
    temp_dir, video_tensor, temp_video_path, final_video_path, original_fps, original_frame_count, original_duration, is_video_file, input_path, use_direct_method = args
    
    try:
        print(f"开始创建纯视频文件: {os.path.basename(final_video_path)}")
        print(f"  使用帧率: {original_fps:.6f} FPS, 期望帧数: {original_frame_count}")
        
        if use_direct_method:
            # 直接方法：从张量直接创建视频（仅在帧数完全匹配时使用）
            print("🎯 使用直接方法：从张量直接创建视频（帧数完全匹配）")
            save_video_directly_from_tensor(video_tensor, temp_video_path, fps=original_fps, quality=5)
        else:
            # 传统方法：通过PNG序列创建视频，使用均匀填补法
            print("📁 使用改进方法：通过PNG序列创建视频，使用均匀填补法")
            video_frames = tensor2video(video_tensor)
            
            # 1. 先保存为PNG序列帧
            processed_frames_dir = os.path.join(temp_dir, "processed_frames")
            saved_frame_paths = save_frames_as_png(video_frames, processed_frames_dir, "frame")
            print(f"✓ PNG序列帧保存完成: {len(saved_frame_paths)}帧")
            
            # 2. 使用均匀填补法填补到原始视频的帧数
            final_frames_dir = os.path.join(temp_dir, "final_frames")
            if is_video_file and original_frame_count > 0 and len(saved_frame_paths) < original_frame_count:
                print(f"🔧 使用均匀填补法: {len(saved_frame_paths)} -> {original_frame_count}")
                final_frame_paths = pad_frames_to_match_original_uniform(
                    processed_frames_dir, original_frame_count, final_frames_dir)
            else:
                # 对于图像序列或帧数已匹配的情况，直接使用处理后的帧
                final_frame_paths = saved_frame_paths
                original_frame_count = len(saved_frame_paths)
            
            # 3. 从帧路径创建视频
            save_video_from_frames(final_frame_paths, temp_video_path, fps=original_fps, quality=5)
        
        print(f"✅ 视频创建完成: {os.path.basename(final_video_path)}")
        
        # 复制临时视频到最终路径
        shutil.copy(temp_video_path, final_video_path)
        print(f"✅ 完成纯视频输出: {os.path.basename(final_video_path)}")
        
        # 验证最终文件参数
        try:
            final_info = get_video_info_accurate(final_video_path)
            print(f"📊📊 最终视频参数:")
            print(f"  帧数: {final_info['frame_count']} (原始: {original_frame_count})")
            print(f"  帧率: {final_info['frame_rate']:.6f} (原始: {original_fps:.6f})")
            print(f"  时长: {final_info['duration']:.6f}秒 (原始: {original_duration:.6f}秒)")
            
            # 检查参数一致性
            frame_match = abs(final_info['frame_count'] - original_frame_count) <= 1
            fps_match = abs(final_info['frame_rate'] - original_fps) < 0.01
            duration_match = abs(final_info['duration'] - original_duration) < 0.1
            
            if frame_match and fps_match and duration_match:
                print("🎯🎯 参数一致性: ✅ 完美匹配")
            else:
                print("⚠️ 参数一致性: 部分参数有差异")
                if not frame_match:
                    print(f"  帧数差异: {final_info['frame_count']} vs {original_frame_count}")
                if not fps_match:
                    print(f"  帧率差异: {final_info['frame_rate']:.6f} vs {original_fps:.6f}")
                if not duration_match:
                    print(f"  时长差异: {final_info['duration']:.6f} vs {original_duration:.6f}")
        except Exception as e:
            print(f"⚠️ 无法验证最终文件参数: {e}")
        
        # 清理临时文件
        try:
            if os.path.exists(os.path.join(temp_dir, "processed_frames")):
                shutil.rmtree(os.path.join(temp_dir, "processed_frames"))
            if os.path.exists(os.path.join(temp_dir, "final_frames")):
                shutil.rmtree(os.path.join(temp_dir, "final_frames"))
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
        except:
            pass
            
        return True, final_video_path
        
    except Exception as e:
        print(f"❌❌ 视频最终化失败 {os.path.basename(final_video_path)}: {e}")
        return False, final_video_path

def main():
    parser = argparse.ArgumentParser(description='FlashVSR视频超分辨率处理（改进版均匀填补法）')
    parser.add_argument('--input', type=str, required=True, help='输入文件路径或目录路径')
    parser.add_argument('--output', type=str, default='./results', help='输出目录路径')
    parser.add_argument('--gpu', type=int, default=0, help='GPU设备ID (0, 1, 2, 3)')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--scale', type=float, default=4.0, help='缩放比例')
    parser.add_argument('--sparse_ratio', type=float, default=2.0, help='稀疏比率')
    parser.add_argument('--local_range', type=int, default=11, help='局部范围')
    parser.add_argument('--max_workers', type=int, default=2, help='并行处理的最大工作线程数')
    
    args = parser.parse_args()
    
    print("=== FlashVSR GPU设置（改进版均匀填补法） ===")
    print(f"请求使用GPU: {args.gpu}")
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("错误: CUDA不可用，请检查CUDA安装")
        return
    
    gpu_count = torch.cuda.device_count()
    print(f"系统检测到 {gpu_count} 个GPU")
    
    if args.gpu >= gpu_count:
        print(f"错误: GPU {args.gpu} 不存在，可用GPU: 0-{gpu_count-1}")
        return
    
    RESULT_ROOT = args.output
    os.makedirs(RESULT_ROOT, exist_ok=True)
    
    # 创建临时目录在输出目录下
    TEMP_ROOT = os.path.join(RESULT_ROOT, "temp")
    os.makedirs(TEMP_ROOT, exist_ok=True)
    print(f"临时文件目录: {TEMP_ROOT}")
    
    # 获取输入文件列表
    try:
        input_files = get_input_files(args.input)
    except Exception as e:
        print(f"输入文件错误: {e}")
        return
    
    if not input_files:
        print(f"在路径 {args.input} 中没有找到支持的视频文件或图像序列")
        return
    
    print(f"找到 {len(input_files)} 个文件需要处理")
    
    try:
        # 初始化管道
        pipe, device = init_pipeline(args.gpu)
    except Exception as e:
        print(f"管道初始化失败: {e}")
        return
    
    # 创建线程池用于并行处理
    executor = ThreadPoolExecutor(max_workers=args.max_workers)
    futures = []
    
    # 处理每个文件
    for i, input_path in enumerate(input_files, 1):
        print(f"\n=== 处理文件 {i}/{len(input_files)} ===")
        print(f"输入: {input_path}")
        
        # 清理GPU内存
        torch.cuda.empty_cache()
        
        name = os.path.basename(input_path.rstrip('/'))
        if name.startswith('.'):
            continue
        
        # 检查是否为视频文件
        is_video_file = is_video(input_path)
        
        # 使用精确的视频信息获取
        try:
            LQ, th, tw, F, fps, from_video, original_fps, original_duration, original_frame_count, processed_frame_count = prepare_input_tensor(
                input_path, scale=args.scale, dtype=torch.bfloat16, device=device)
        except Exception as e:
            print(f"[错误] 准备输入张量失败: {e}")
            continue

        # 对于视频文件，显示信息
        if is_video_file:
            print(f"✓ 视频文件信息: {original_frame_count}帧, {original_fps:.6f}fps, {original_duration:.6f}秒")
            print(f"  处理后帧数: {processed_frame_count}帧 (需要填补 {original_frame_count - processed_frame_count} 帧)")

        try:
            print("开始FlashVSR处理...")
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
            
            # 判断是否可以使用直接方法
            # 如果处理后的帧数等于原始帧数，使用直接方法
            use_direct_method = (processed_frame_count == original_frame_count)
            
            if use_direct_method:
                print("🎯 帧数匹配，使用直接方法创建视频（跳过PNG转换）")
            else:
                missing_frames = original_frame_count - processed_frame_count
                print(f"📁 使用改进的均匀填补法: 处理{processed_frame_count}帧，需要{original_frame_count}帧，填补{missing_frames}帧")
                print(f"  计算插入间隔: 每 {max(1, processed_frame_count // (missing_frames + 1))} 帧后插入1帧")
            
            # 生成输出文件名
            if os.path.isdir(input_path):
                base_name = os.path.basename(input_path.rstrip('/'))
            else:
                base_name = os.path.splitext(os.path.basename(input_path))[0]
            
            # 创建临时目录
            temp_dir = tempfile.mkdtemp(dir=TEMP_ROOT, prefix=f"temp_{base_name}_")
            print(f"📁📁 临时目录: {temp_dir}")
            
            # 准备最终视频路径
            temp_video_path = os.path.join(temp_dir, "temp_video.mp4")
            final_video_filename = f"FlashVSR_v1.1_Tiny_Long_{base_name}_gpu{args.gpu}_seed{args.seed}_uniform.mp4"
            final_video_path = os.path.join(RESULT_ROOT, final_video_filename)
            
            # 提交并行处理任务
            print(f"🚀🚀 提交并行处理任务: {os.path.basename(final_video_path)}")
            future = executor.submit(process_video_finalization, (
                temp_dir, video, temp_video_path, final_video_path, 
                original_fps, original_frame_count, original_duration, is_video_file, input_path, use_direct_method
            ))
            futures.append((future, final_video_path, temp_dir))
            
            print(f"📊📊 当前并行任务数: {len(futures)}")
            
            # 如果并行任务达到上限，等待部分任务完成
            if len(futures) >= args.max_workers * 2:
                print("🔄🔄 达到并行任务上限，等待部分任务完成...")
                completed_count = 0
                for f, path, temp_dir in futures[:]:
                    if f.done():
                        try:
                            success, result_path = f.result(timeout=1)
                            if success:
                                print(f"✅ 并行任务完成: {os.path.basename(result_path)}")
                            else:
                                print(f"❌❌ 并行任务失败: {os.path.basename(result_path)}")
                            # 清理临时目录
                            try:
                                shutil.rmtree(temp_dir)
                                print(f"🗑🗑️ 清理临时目录: {os.path.basename(temp_dir)}")
                            except:
                                pass
                            futures.remove((f, path, temp_dir))
                            completed_count += 1
                        except:
                            pass
                
                if completed_count > 0:
                    print(f"🔄🔄 已完成 {completed_count} 个任务，继续处理...")
            
        except Exception as e:
            print(f"[处理错误] {name}: {e}")
            # 清理临时目录
            try:
                shutil.rmtree(temp_dir)
                print(f"🗑🗑️ 清理临时目录（错误时）: {os.path.basename(temp_dir)}")
            except:
                pass
            continue

    print(f"\n🔄🔄 等待所有并行任务完成...")
    
    # 等待所有剩余任务完成
    completed_count = 0
    failed_count = 0
    
    for future, final_video_path, temp_dir in futures:
        try:
            success, result_path = future.result(timeout=300)  # 5分钟超时
            if success:
                print(f"✅ 任务完成: {os.path.basename(result_path)}")
                completed_count += 1
            else:
                print(f"❌❌ 任务失败: {os.path.basename(result_path)}")
                failed_count += 1
        except Exception as e:
            print(f"❌❌ 任务超时或失败 {os.path.basename(final_video_path)}: {e}")
            failed_count += 1
        finally:
            # 清理临时目录
            try:
                shutil.rmtree(temp_dir)
                print(f"🗑🗑️ 清理临时目录: {os.path.basename(temp_dir)}")
            except:
                pass
    
    # 关闭线程池
    executor.shutdown(wait=True)
    
    # 尝试清理整个临时目录（如果为空）
    try:
        if os.path.exists(TEMP_ROOT) and not os.listdir(TEMP_ROOT):
            shutil.rmtree(TEMP_ROOT)
            print(f"🗑🗑️ 清理临时根目录: {TEMP_ROOT}")
    except:
        pass
    
    print(f"\n=== 所有文件处理完成 ===")
    print(f"✅ 成功: {completed_count} 个文件")
    print(f"❌❌ 失败: {failed_count} 个文件")
    print(f"📁📁 输出目录: {RESULT_ROOT}")
    print(f"🗑🗑️ 临时文件已清理")
    print("🎯🎯 输出文件为纯视频流（无音频），使用均匀填补法优化帧数")

if __name__ == "__main__":
    main()
