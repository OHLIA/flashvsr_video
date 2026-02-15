#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ComfyUI FlashVSR 批量视频处理工具 - 增强版（支持自动重启和断点续传）
支持动态参数传递、自动检测视频帧数、GPU设备选择
自动处理连接失败和进程重启
输出文件按照 ComfyUI 默认方法存储
"""

import json
import requests
import os
import time
import sys
import math
from glob import glob
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import subprocess
import signal
import psutil
import traceback

# 尝试导入 pymediainfo
try:
    from pymediainfo import MediaInfo
    PYMEDIAINFO_AVAILABLE = True
except ImportError:
    PYMEDIAINFO_AVAILABLE = False
    print("⚠️  pymediainfo 未安装，将使用备用方法获取视频信息")

class ComfyUI_FlashVSR_BatchProcessor:
    def __init__(self, comfyui_url: str = "http://127.0.0.1:8188"):
        """
        初始化 ComfyUI FlashVSR 批量处理器
        
        参数:
            comfyui_url: ComfyUI 服务器地址
        """
        self.comfyui_url = comfyui_url.rstrip('/')
        self.api_prompt = f"{comfyui_url}/prompt"
        self.api_history = f"{comfyui_url}/history"
        self.api_view = f"{comfyui_url}/view"
        
        # 添加状态跟踪
        self.comfyui_process = None
        self.comfyui_path = r"F:\AI\ComfyUI_Mie_V7.0"
        self.comfyui_script = r"F:\AI\ComfyUI_Mie_V7.0\run_nvidia_gpu_fast_fp16_accumulation_hf_mirror.bat"
        self.output_dir = r"F:\AI\ComfyUI_Mie_V7.0\comfyui\output"
    
    def kill_comfyui_processes(self):
        """关闭所有ComfyUI相关进程"""
        print("🔪 正在关闭ComfyUI进程...")
        try:
            # 查找并终止ComfyUI相关进程
            killed_count = 0
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline and any('comfy' in part.lower() or 'python' in part.lower() for part in cmdline):
                        # 检查是否是ComfyUI进程
                        is_comfyui = any('main.py' in ' '.join(cmdline) or 'run_nvidia_gpu' in ' '.join(cmdline) for part in cmdline)
                        if is_comfyui:
                            print(f"  终止进程 PID={proc.info['pid']}, 命令={cmdline}")
                            proc.kill()
                            killed_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if killed_count > 0:
                print(f"✅ 已终止 {killed_count} 个ComfyUI进程")
            else:
                print("ℹ️ 未找到运行的ComfyUI进程")
            
            # 等待进程完全终止
            time.sleep(3)
            
        except Exception as e:
            print(f"⚠️ 终止进程时出错: {e}")
    
    def start_comfyui(self):
        """启动ComfyUI进程"""
        print(f"🚀 正在启动ComfyUI: {self.comfyui_script}")
        
        try:
            # 切换到ComfyUI目录
            os.chdir(self.comfyui_path)
            
            # 启动新进程
            self.comfyui_process = subprocess.Popen(
                [self.comfyui_script],
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                cwd=self.comfyui_path
            )
            
            print(f"✅ ComfyUI进程已启动，PID: {self.comfyui_process.pid}")
            
            # 等待启动完成
            wait_time = 120
            for i in range(wait_time):
                print(f"⏳ 等待ComfyUI启动 ({i+1}/{wait_time})...")
                if self.check_comfyui_server(timeout=5):
                    print("✅ ComfyUI服务器已准备就绪")
                    return True
                time.sleep(1)
            
            print("❌ ComfyUI启动超时")
            return False
            
        except Exception as e:
            print(f"❌ 启动ComfyUI失败: {e}")
            return False
    
    def restart_comfyui(self):
        """重启ComfyUI服务"""
        print("🔄 正在重启ComfyUI服务...")
        
        # 1. 关闭现有进程
        self.kill_comfyui_processes()
        
        # 2. 启动新进程
        if self.start_comfyui():
            print("✅ ComfyUI重启成功")
            return True
        else:
            print("❌ ComfyUI重启失败")
            return False
    
    def check_comfyui_server(self, timeout: int = 5) -> bool:
        """
        检查ComfyUI服务是否可用
        
        参数:
            timeout: 超时时间（秒）
        
        返回:
            True 如果服务可用，False 如果不可用
        """
        try:
            response = requests.get(f"{self.comfyui_url}/", timeout=timeout)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException as e:
            print(f"❌ 无法连接到 ComfyUI 服务: {e}")
        return False
    
    def wait_for_completion(self, prompt_id: str, poll_interval: int = 5) -> bool:
        """
        等待任务完成（增强版，支持连接失败检测和自动重启）
        
        参数:
            prompt_id: 任务ID
            poll_interval: 轮询间隔（秒）
        
        返回:
            True 如果成功完成，False 如果失败或超时
        """
        print(f"⏳ 等待任务 {prompt_id} 完成...")
        
        start_time = time.time()
        max_wait_time = 3600  # 最长等待1小时
        poll_failures = 0
        max_poll_failures = 5  # 最大轮询失败次数
        
        while time.time() - start_time < max_wait_time:
            try:
                response = requests.get(f"{self.api_history}/{prompt_id}", timeout=10)
                
                if response.status_code == 200:
                    history = response.json()
                    
                    if history and len(history) > 0:
                        status = history[prompt_id]
                        
                        if status.get("status", {}).get("completed", False):
                            print(f"✅ 任务 {prompt_id} 已完成")
                            return True
                        
                        if status.get("status", {}).get("has_error", False):
                            print(f"❌ 任务 {prompt_id} 执行失败")
                            return False
                
                # 任务仍在进行中
                poll_failures = 0  # 重置失败计数
                time.sleep(poll_interval)
                
            except requests.exceptions.RequestException as e:
                print(f"⚠️ 轮询失败: {e}")
                poll_failures += 1
                
                if poll_failures >= max_poll_failures:
                    print(f"❌ 轮询失败超过 {max_poll_failures} 次，认为任务失败")
                    return False
                
                time.sleep(poll_interval)
        
        print(f"❌ 任务 {prompt_id} 超时")
        return False
    
    def wait_for_completion_with_restart(self, prompt_id: str, video_name: str, poll_interval: int = 5) -> Tuple[bool, bool]:
        """
        等待任务完成，支持连接失败时的自动重启
        
        参数:
            prompt_id: 任务ID
            video_name: 当前处理的视频名称（用于清理输出文件）
            poll_interval: 轮询间隔（秒）
        
        返回:
            (success: bool, was_restarted: bool) - 是否成功，是否重启了ComfyUI
        """
        print(f"⏳ 等待任务 {prompt_id} 完成...")
        
        start_time = time.time()
        max_wait_time = 3600
        poll_failures = 0
        max_poll_failures = 10
        
        while time.time() - start_time < max_wait_time:
            try:
                response = requests.get(f"{self.api_history}/{prompt_id}", timeout=10)
                
                if response.status_code == 200:
                    history = response.json()
                    
                    if history and len(history) > 0:
                        status = history[prompt_id]
                        
                        if status.get("status", {}).get("completed", False):
                            print(f"✅ 任务 {prompt_id} 已完成")
                            return True, False
                        
                        if status.get("status", {}).get("has_error", False):
                            print(f"❌ 任务 {prompt_id} 执行失败")
                            return False, False
                
                # 任务仍在进行中
                poll_failures = 0
                time.sleep(poll_interval)
                
            except requests.exceptions.RequestException as e:
                print(f"⚠️ 轮询失败: {e}")
                poll_failures += 1
                
                if poll_failures >= max_poll_failures:
                    print(f"❌ 轮询失败超过 {max_poll_failures} 次，将重启ComfyUI")
                    
                    # 清理当前视频的输出文件
                    self.clean_output_files(video_name)
                    
                    # 重启ComfyUI
                    if self.restart_comfyui():
                        print(f"🔄 ComfyUI已重启，返回失败状态以便重试")
                        return False, True
                    else:
                        print(f"❌ ComfyUI重启失败，任务中止")
                        return False, False
                
                time.sleep(poll_interval)
        
        print(f"❌ 任务 {prompt_id} 超时")
        return False, False
    
    def clean_output_files(self, video_name: str):
        """
        清理指定视频的输出文件
        
        参数:
            video_name: 视频文件名（用于匹配输出文件）
        """
        print(f"🧹 清理 {video_name} 的输出文件...")
        
        try:
            # 获取视频基本名称（不含扩展名）
            base_name = os.path.splitext(video_name)[0]
            
            # 构建文件匹配模式
            patterns = [
                f"flashvsr_*{base_name}*",
                f"*{base_name}*enhanced*",
                f"*{base_name}*_batch*"
            ]
            
            deleted_count = 0
            for pattern in patterns:
                for file_path in glob(os.path.join(self.output_dir, pattern)):
                    try:
                        os.remove(file_path)
                        print(f"  删除: {os.path.basename(file_path)}")
                        deleted_count += 1
                    except Exception as e:
                        print(f"  删除失败 {file_path}: {e}")
            
            if deleted_count > 0:
                print(f"✅ 已清理 {deleted_count} 个输出文件")
            else:
                print("ℹ️ 未找到需要清理的输出文件")
                
        except Exception as e:
            print(f"⚠️ 清理输出文件时出错: {e}")
    
    def get_video_frame_count(self, video_path: str) -> Tuple[int, float, str]:
        """
        获取视频的总帧数、帧率和检测方法
        """
        try:
            if PYMEDIAINFO_AVAILABLE:
                media_info = MediaInfo.parse(video_path)
                video_track = None
                
                for track in media_info.tracks:
                    if track.track_type == 'Video':
                        video_track = track
                        break
                
                if video_track:
                    frame_count = 0
                    if hasattr(video_track, 'frame_count') and video_track.frame_count:
                        frame_count = int(video_track.frame_count)
                    
                    frame_rate = 25.0
                    if hasattr(video_track, 'frame_rate') and video_track.frame_rate:
                        try:
                            frame_rate_str = str(video_track.frame_rate)
                            if '/' in frame_rate_str:
                                numerator, denominator = map(float, frame_rate_str.split('/'))
                                frame_rate = numerator / denominator
                            else:
                                frame_rate = float(frame_rate_str)
                        except:
                            frame_rate = 25.0
                    
                    if frame_count > 0:
                        return frame_count, frame_rate, "pymediainfo"
            
            # 备用方法：使用 OpenCV
            try:
                import cv2
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()
                    
                    if frame_count > 0 and fps > 0:
                        return frame_count, fps, "OpenCV"
            except ImportError:
                pass
            
            print(f"⚠️  无法获取 {os.path.basename(video_path)} 的准确帧数，使用默认值")
            return 100, 25.0, "默认值"
            
        except Exception as e:
            print(f"⚠️  获取视频信息失败 {os.path.basename(video_path)}: {e}")
            return 100, 25.0, "错误-默认值"
    
    def load_workflow_template(self, template_path: str) -> Dict:
        """加载工作流 JSON 模板"""
        with open(template_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def update_workflow_parameters(
        self, 
        workflow: Dict, 
        video_path: str, 
        output_prefix: Optional[str] = None,
        scale: float = 4.0,
        tile_size: int = 256,
        tile_overlap: int = 24,
        total_frames: Optional[int] = None,
        frames_per_batch: int = 201,
        gpu_device: str = "auto"
    ) -> Dict:
        """更新工作流中的所有参数"""
        modified_workflow = json.loads(json.dumps(workflow))
        
        # 1. 设置输入视频路径
        for node_id, node_data in modified_workflow.items():
            if node_data.get("class_type") == "VHS_LoadVideo":
                node_data["inputs"]["video"] = video_path
        
        # 2. 设置 FlashVSR 核心参数
        for node_id, node_data in modified_workflow.items():
            if node_data.get("class_type") == "FlashVSRNodeAdv":
                if "{{scale}}" in str(node_data["inputs"].get("scale", "")):
                    node_data["inputs"]["scale"] = scale
                if "{{t_z}}" in str(node_data["inputs"].get("tile_size", "")):
                    node_data["inputs"]["tile_size"] = tile_size
                if "{{t_o}}" in str(node_data["inputs"].get("tile_overlap", "")):
                    node_data["inputs"]["tile_overlap"] = tile_overlap
        
        # 3. 设置 GPU 设备
        for node_id, node_data in modified_workflow.items():
            if node_id == "5" and node_data.get("class_type") == "FlashVSRInitPipe":
                if "{{gpu}}" in str(node_data["inputs"].get("device", "")):
                    if gpu_device == "auto":
                        device_value = "auto"
                    elif gpu_device.isdigit():
                        device_value = f"cuda:{gpu_device}"
                    else:
                        device_value = gpu_device
                    node_data["inputs"]["device"] = device_value
                    print(f"✅ 已将GPU设备设置为: {device_value}")
                elif isinstance(node_data["inputs"].get("device"), str):
                    if gpu_device.isdigit():
                        device_value = f"cuda:{gpu_device}"
                    else:
                        device_value = gpu_device
                    node_data["inputs"]["device"] = device_value
                    print(f"✅ 已将GPU设备设置为: {device_value} (直接赋值)")
        
        # 4. 设置总帧数
        if total_frames is None:
            total_frames, _, _ = self.get_video_frame_count(video_path)
        
        for node_id, node_data in modified_workflow.items():
            if node_id == "50" and node_data.get("class_type") == "PrimitiveInt":
                current_value = str(node_data["inputs"].get("value", ""))
                if "{{TOTAL_FRAMES}}" in current_value or "{{TATAL_FRAMES}}" in current_value:
                    node_data["inputs"]["value"] = total_frames
                    print(f"✅ 已将总帧数 {total_frames} 设置到节点 50")
                elif isinstance(node_data["inputs"].get("value"), (int, float)):
                    node_data["inputs"]["value"] = total_frames
                    print(f"✅ 已将总帧数 {total_frames} 设置到节点 50 (直接赋值)")
                else:
                    print(f"⚠️  节点 50 的值既不是占位符也不是数字: {current_value}")
        
        # 5. 设置每批帧数
        for node_id, node_data in modified_workflow.items():
            if node_id == "8" and node_data.get("class_type") == "PrimitiveInt":
                if "{{FRAMES_PER_BATCH}}" in str(node_data["inputs"].get("value", "")):
                    node_data["inputs"]["value"] = frames_per_batch
                    print(f"✅ 已将每批帧数 {frames_per_batch} 设置到节点 8")
                elif isinstance(node_data["inputs"].get("value"), (int, float)):
                    node_data["inputs"]["value"] = frames_per_batch
                    print(f"✅ 已将每批帧数 {frames_per_batch} 设置到节点 8 (直接赋值)")
        
        # 6. 设置输出文件名前缀
        if output_prefix is None:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_prefix = f"flashvsr_{base_name}_enhanced"
        
        for node_id, node_data in modified_workflow.items():
            if node_data.get("class_type") == "VHS_VideoCombine":
                if "{{OUTPUT_PREFIX}}" in str(node_data["inputs"].get("filename_prefix", "")):
                    node_data["inputs"]["filename_prefix"] = output_prefix
                elif isinstance(node_data["inputs"].get("filename_prefix"), str):
                    node_data["inputs"]["filename_prefix"] = output_prefix
        
        return modified_workflow
    
    def queue_prompt(self, workflow: Dict, timeout: int = 300) -> Optional[str]:
        """将工作流发送到 ComfyUI 执行"""
        if not self.check_comfyui_server():
            print("❌ ComfyUI 服务不可用，无法提交任务")
            return None
        
        print("=== 工作流参数验证 ===")
        for node_id, node_data in workflow.items():
            if node_id in ["5", "8", "50"]:
                print(f"节点 {node_id} ({node_data.get('class_type')}): {node_data['inputs']}")

        try:
            response = requests.post(
                self.api_prompt, 
                json={"prompt": workflow}, 
                timeout=timeout
            )
            response.raise_for_status()
            
            data = response.json()
            prompt_id = data.get('prompt_id')
            
            if prompt_id:
                print(f"✅ 任务已提交，ID: {prompt_id}")
                return prompt_id
            else:
                print(f"❌ 未收到任务ID，响应: {data}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 请求失败: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"❌ JSON 解析失败: {e}")
            return None
    
    def get_output_files(self, prompt_id: str) -> List[str]:
        """获取任务生成的输出文件列表"""
        try:
            response = requests.get(f"{self.api_view}/{prompt_id}", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                outputs = data.get("outputs", {})
                file_list = []
                
                for node_id, node_output in outputs.items():
                    if "images" in node_output:
                        for img in node_output["images"]:
                            if "filename" in img:
                                file_list.append(img["filename"])
                
                return file_list
                
        except Exception as e:
            print(f"⚠️ 获取输出文件失败: {e}")
        
        return []
    
    def process_single_video(
        self, 
        workflow_template: Dict, 
        video_path: str, 
        output_prefix: Optional[str] = None,
        scale: float = 4.0,
        tile_size: int = 64,
        tile_overlap: int = 8,
        total_frames: Optional[int] = None,
        frames_per_batch: int = 125,
        gpu_device: str = "auto"
    ) -> Tuple[bool, int]:
        """
        处理单个视频文件，支持重试
        
        返回:
            (success: bool, retry_count: int) - 是否成功，重试次数
        """
        video_name = os.path.basename(video_path)
        print(f"\n{'='*60}")
        print(f"处理视频: {video_path}")
        print(f"{'='*60}")
        
        if not os.path.exists(video_path):
            print(f"❌ 文件不存在: {video_path}")
            return False, 0
        
        # 获取视频信息
        detected_total_frames, fps, detection_method = self.get_video_frame_count(video_path)
        
        if total_frames is None:
            total_frames = detected_total_frames
            print(f"📊 视频信息: {total_frames} 帧, {fps:.2f} FPS (自动检测)")
            print(f"🔍 检测方法: {detection_method}")
        else:
            print(f"📊 视频信息: {total_frames} 帧 (手动指定), {fps:.2f} FPS (自动检测)")
        
        if frames_per_batch > 0 and total_frames > 0:
            batch_count = math.ceil(total_frames / frames_per_batch)
            print(f"📦 处理批次: {batch_count} 批（每批 {frames_per_batch} 帧）")
        
        if fps > 0 and total_frames > 0:
            duration_seconds = total_frames / fps
            minutes = int(duration_seconds // 60)
            seconds = int(duration_seconds % 60)
            print(f"⏱️  视频时长: {minutes}:{seconds:02d} (mm:ss)")
        
        if gpu_device == "auto":
            print(f"🎮 GPU设备: auto (自动选择)")
        elif gpu_device.isdigit():
            print(f"🎮 GPU设备: cuda:{gpu_device}")
        else:
            print(f"🎮 GPU设备: {gpu_device}")
        
        # 更新工作流参数
        workflow = self.update_workflow_parameters(
            workflow_template, 
            video_path, 
            output_prefix,
            scale=scale,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            total_frames=total_frames,
            frames_per_batch=frames_per_batch,
            gpu_device=gpu_device
        )
        
        # 尝试提交任务，最多重试3次
        max_retries = 3
        for retry_count in range(max_retries):
            print(f"\n🔄 尝试 {retry_count + 1}/{max_retries}")
            
            # 提交任务
            prompt_id = self.queue_prompt(workflow)
            if not prompt_id:
                if retry_count < max_retries - 1:
                    print("⏳ 任务提交失败，5秒后重试...")
                    time.sleep(5)
                    continue
                return False, retry_count + 1
            
            # 等待任务完成，支持自动重启
            success, was_restarted = self.wait_for_completion_with_restart(
                prompt_id, 
                os.path.splitext(video_name)[0]  # 只传基本名称
            )
            
            if was_restarted:
                # 如果ComfyUI被重启，需要重试
                if retry_count < max_retries - 1:
                    print("⏳ ComfyUI已重启，5秒后重试此任务...")
                    time.sleep(5)
                    continue
                else:
                    return False, retry_count + 1
            elif not success:
                if retry_count < max_retries - 1:
                    print("⏳ 任务处理失败，5秒后重试...")
                    time.sleep(5)
                    continue
                return False, retry_count + 1
            else:
                # 成功
                output_files = self.get_output_files(prompt_id)
                if output_files:
                    print(f"📁 生成的文件（保存在 ComfyUI 默认输出目录）:")
                    for file in output_files:
                        print(f"  - {file}")
                else:
                    print("ℹ️  任务完成，但未获取到输出文件列表")
                return True, retry_count + 1
        
        return False, max_retries
    
    def batch_process_videos(
        self, 
        workflow_template_path: str, 
        video_files: List[str], 
        output_prefix_base: Optional[str] = None,
        scale: float = 4.0,
        tile_size: int = 256,
        tile_overlap: int = 24,
        total_frames: Optional[int] = None,
        frames_per_batch: int = 201,
        gpu_device: str = "auto"
    ) -> Dict[str, Dict]:
        """
        批量处理多个视频文件，支持断点续传和重试
        
        返回:
            字典：{视频文件: {成功: bool, 重试次数: int, 信息: str}}
        """
        # 加载工作流模板
        try:
            workflow_template = self.load_workflow_template(workflow_template_path)
        except Exception as e:
            print(f"❌ 加载工作流模板失败: {e}")
            return {}
        
        # 先检查ComfyUI服务
        if not self.check_comfyui_server():
            print("❌ ComfyUI 服务未运行，正在启动...")
            if not self.start_comfyui():
                print("❌ 无法启动ComfyUI，程序退出")
                return {}
        
        results = {}
        total_videos = len(video_files)
        
        print(f"🎬 开始批量处理 {total_videos} 个视频")
        print(f"⚙️  参数: scale={scale}, tile_size={tile_size}, tile_overlap={tile_overlap}")
        print(f"🎮 GPU设备: {gpu_device}")
        print(f"🔄 每个任务最多重试: 3次")
        print(f"💾 输出目录: {self.output_dir}")
        
        for i, video_path in enumerate(video_files, 1):
            print(f"\n📊 进度: {i}/{total_videos}")
            
            # 设置输出前缀
            output_prefix = None
            if output_prefix_base:
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                output_prefix = f"{output_prefix_base}_{base_name}"
            
            # 处理单个视频
            success, retry_count = self.process_single_video(
                workflow_template, 
                video_path, 
                output_prefix,
                scale=scale,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                total_frames=total_frames,
                frames_per_batch=frames_per_batch,
                gpu_device=gpu_device
            )
            
            results[video_path] = {
                "success": success,
                "retry_count": retry_count,
                "message": "成功" if success else f"失败（重试{retry_count}次）"
            }
        
        # 输出统计信息
        print(f"\n{'='*60}")
        print("批量处理完成")
        print(f"{'='*60}")
        
        success_count = sum(1 for r in results.values() if r["success"])
        total_retries = sum(r["retry_count"] for r in results.values())
        
        print(f"✅ 成功: {success_count}/{total_videos}")
        print(f"❌ 失败: {total_videos - success_count}/{total_videos}")
        print(f"🔄 总重试次数: {total_retries}")
        print(f"🎮 GPU设备: {gpu_device}")
        print(f"💾 输出位置: {self.output_dir}")
        
        if success_count > 0:
            print(f"\n✅ 成功文件列表:")
            for video_path, result in results.items():
                if result["success"]:
                    print(f"  ✓ {os.path.basename(video_path)} (重试: {result['retry_count']})")
        
        if success_count < total_videos:
            print(f"\n❌ 失败文件列表:")
            for video_path, result in results.items():
                if not result["success"]:
                    print(f"  ✗ {os.path.basename(video_path)} (重试: {result['retry_count']})")
        
        return results

def collect_video_files(input_path: str, pattern: str = '*.mp4') -> List[str]:
    """根据输入路径收集视频文件"""
    video_files = []
    
    if os.path.isfile(input_path):
        if input_path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv')):
            video_files.append(input_path)
            print(f"✅ 添加单个文件: {input_path}")
        else:
            print(f"❌ 文件格式不支持: {input_path}")
    elif os.path.isdir(input_path):
        search_pattern = os.path.join(input_path, pattern)
        found_files = glob(search_pattern)
        
        video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.MP4', '.MOV', '.AVI', '.MKV']
        for ext in video_extensions:
            if ext not in pattern:
                additional_pattern = os.path.join(input_path, f"*{ext}")
                additional_files = glob(additional_pattern)
                found_files.extend(additional_files)
        
        video_files = sorted(list(set(found_files)))
        
        if not video_files:
            print(f"❌ 目录 {input_path} 中未找到任何视频文件")
        else:
            print(f"✅ 从目录 {input_path} 找到 {len(video_files)} 个视频文件")
    else:
        print(f"❌ 路径不存在: {input_path}")
    
    return video_files

def main():
    """主函数：批量处理视频示例"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='ComfyUI FlashVSR 批量视频处理工具 - 增强版（支持自动重启和断点续传）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 处理单个视频文件，使用GPU 0
  python batch_process_videos.py --input video.mp4 --gpu 0
  
  # 处理目录下的所有视频文件，使用GPU 1
  python batch_process_videos.py --input ./videos --gpu 1
  
  # 自动选择GPU
  python batch_process_videos.py --input ./videos --gpu auto
  
  # 自定义参数
  python batch_process_videos.py --input ./videos --scale 2.0 --tile-size 128 --gpu 0
  
  # 手动指定总帧数
  python batch_process_videos.py --input ./videos --total-frames 300 --gpu 0
  
  # 设置输出文件前缀
  python batch_process_videos.py --input ./videos --output-prefix batch_001 --gpu 0

自动重启功能:
  - 当轮询失败超过10次时，自动重启ComfyUI
  - 重启前会清理当前视频的输出文件
  - 重启后自动重新处理当前视频
  - 每个视频最多重试3次
        """
    )
    
    # 输入参数
    input_group = parser.add_argument_group('输入选项')
    input_group.add_argument('--template', type=str, default='flashvsr_template.json',
                           help='工作流模板 JSON 文件路径 (默认: flashvsr_template.json)')
    input_group.add_argument('--input', type=str, required=True,
                           help='输入路径（可以是视频文件路径或包含视频文件的目录）')
    input_group.add_argument('--pattern', type=str, default='*.mp4',
                           help='视频文件匹配模式，当输入是目录时使用 (默认: *.mp4)')
    
    # 输出参数
    output_group = parser.add_argument_group('输出选项')
    output_group.add_argument('--output-prefix', type=str, 
                            help='输出文件名前缀（可选，用于区分批次）')
    
    # FlashVSR 处理参数
    processing_group = parser.add_argument_group('处理参数')
    processing_group.add_argument('--scale', type=float, default=4.0,
                                help='放大倍数 (默认: 4.0)')
    processing_group.add_argument('--tile-size', type=int, default=256,
                                help='分块大小 (默认: 256)')
    processing_group.add_argument('--tile-overlap', type=int, default=24,
                                help='分块重叠像素 (默认: 24)')
    processing_group.add_argument('--frames-per-batch', type=int, default=201,
                                help='每批处理的帧数 (默认: 201)')
    processing_group.add_argument('--total-frames', type=int,
                                help='视频总帧数 (如不提供则自动检测)')
    
    # GPU参数
    gpu_group = parser.add_argument_group('GPU选项')
    gpu_group.add_argument('--gpu', type=str, default='auto',
                         help='GPU设备选择: auto, 0, 1, 2, cuda:0, cuda:1等 (默认: auto)')
    
    # 系统参数
    system_group = parser.add_argument_group('系统参数')
    system_group.add_argument('--server', type=str, default='http://127.0.0.1:8188',
                            help='ComfyUI 服务器地址 (默认: http://127.0.0.1:8188)')
    system_group.add_argument('--skip-pymedia-check', action='store_true',
                            help='跳过 pymediainfo 检查')
    
    args = parser.parse_args()
    
    # 检查 pymediainfo
    if not PYMEDIAINFO_AVAILABLE and not args.skip_pymedia_check:
        print("⚠️  未检测到 pymediainfo 库")
        print("   建议安装以获得更准确的视频帧数检测:")
        print("   pip install pymediainfo")
        print("   或添加 --skip-pymedia-check 参数跳过此警告")
        
        response = input("是否继续? (y/n): ")
        if response.lower() != 'y':
            print("退出程序")
            return
    
    # 准备视频文件列表
    video_files = collect_video_files(args.input, args.pattern)
    
    if not video_files:
        print("❌ 未找到任何视频文件")
        return
    
    print(f"\n找到 {len(video_files)} 个视频文件:")
    for vf in video_files:
        print(f"  - {vf}")
    
    # 显示处理参数
    print(f"\n⚙️  处理参数:")
    print(f"  scale: {args.scale}")
    print(f"  tile_size: {args.tile_size}")
    print(f"  tile_overlap: {args.tile_overlap}")
    print(f"  frames_per_batch: {args.frames_per_batch}")
    
    if args.total_frames:
        print(f"  total_frames: {args.total_frames} (手动指定)")
    else:
        print(f"  total_frames: 自动检测")
    
    if args.output_prefix:
        print(f"  output_prefix: {args.output_prefix} (输出文件名前缀)")
    
    if args.gpu == "auto":
        print(f"🎮 GPU设备: auto (自动选择)")
    elif args.gpu.isdigit():
        print(f"🎮 GPU设备: cuda:{args.gpu}")
    else:
        print(f"🎮 GPU设备: {args.gpu}")
    
    print(f"💾 输出目录: F:\\AI\\ComfyUI_Mie_V7.0\\comfyui\\output")
    print(f"🔄 每个任务最多重试: 3次")
    print(f"🔄 轮询失败处理: 超过10次自动重启ComfyUI")
    print(f"🔧 断点续传: 支持 (失败时自动清理并重试)")
    
    # 初始化处理器
    processor = ComfyUI_FlashVSR_BatchProcessor(comfyui_url=args.server)
    
    # 批量处理
    start_time = time.time()
    
    results = processor.batch_process_videos(
        workflow_template_path=args.template,
        video_files=video_files,
        output_prefix_base=args.output_prefix,
        scale=args.scale,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        total_frames=args.total_frames,
        frames_per_batch=args.frames_per_batch,
        gpu_device=args.gpu
    )
    
    # 计算总耗时
    total_time = time.time() - start_time
    print(f"\n⏱️  总耗时: {total_time:.2f} 秒")
    print(f"💾 所有输出文件已保存到: F:\\AI\\ComfyUI_Mie_V7.0\\comfyui\\output")
    print(f"   您可以在 ComfyUI 的 output 文件夹中找到生成的文件")
    
    # 最后关闭ComfyUI进程
    print(f"\n🔧 处理完成，正在关闭ComfyUI进程...")
    processor.kill_comfyui_processes()
    print("✅ 所有处理完成！")

if __name__ == "__main__":
    main()
