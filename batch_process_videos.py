#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ComfyUI FlashVSR 批量视频处理工具 - 增强任务监控版（修正版）
改进的任务状态监控逻辑，支持多API协同监控和智能重试
修复占位符替换问题
"""

import json
import requests
import os
import time
import sys
import math
import re
from glob import glob
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import subprocess
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
        self.api_queue = f"{comfyui_url}/queue"  # 新增：队列API
        
        # 添加状态跟踪
        self.comfyui_process = None
        self.comfyui_path = r"F:\AI\ComfyUI_Mie_V7.0"
        self.comfyui_script = r"F:\AI\ComfyUI_Mie_V7.0\run_nvidia_gpu_fast_fp16_accumulation_hf_mirror.bat"
        self.output_dir = r"F:\AI\ComfyUI_Mie_V7.0\comfyui\output"
        
        # 创建日志文件
        self.log_file = os.path.join(self.comfyui_path, "batch_processing.log")
        
    def save_processing_status(self, video_name: str, batch_number: int = None, action: str = None):
        """
        保存处理状态到日志文件
        """
        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_file, 'a', encoding='utf-8') as f:
                if batch_number is not None and action is not None:
                    f.write(f"[{timestamp}] 视频: {video_name}, 批次: {batch_number}, 操作: {action}\n")
                elif batch_number is not None:
                    f.write(f"[{timestamp}] 视频: {video_name}, 批次: {batch_number}\n")
                else:
                    f.write(f"[{timestamp}] 视频: {video_name}\n")
        except Exception as e:
            print(f"⚠️ 保存处理状态失败: {e}")
    
    def kill_comfyui_processes(self):
        """关闭所有ComfyUI相关进程"""
        print("🔪 正在关闭ComfyUI进程...")
        self.save_processing_status("系统", action="关闭ComfyUI进程")
        
        try:
            killed_count = 0
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info.get('cmdline', []))
                    if 'comfy' in cmdline.lower() or 'main.py' in cmdline:
                        print(f"  终止进程 PID={proc.info['pid']}")
                        proc.kill()
                        killed_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if killed_count > 0:
                print(f"✅ 已终止 {killed_count} 个ComfyUI进程")
            else:
                print("ℹ️ 未找到运行的ComfyUI进程")
            
            time.sleep(3)
            
        except Exception as e:
            print(f"⚠️ 终止进程时出错: {e}")
    
    def start_comfyui(self):
        """启动ComfyUI进程"""
        print(f"🚀 正在启动ComfyUI: {self.comfyui_script}")
        self.save_processing_status("系统", action="启动ComfyUI")
        
        try:
            os.chdir(self.comfyui_path)
            
            self.comfyui_process = subprocess.Popen(
                [self.comfyui_script],
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                cwd=self.comfyui_path
            )
            
            print(f"✅ ComfyUI进程已启动，PID: {self.comfyui_process.pid}")
            
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
        self.save_processing_status("系统", action="重启ComfyUI")
        
        self.kill_comfyui_processes()
        
        if self.start_comfyui():
            print("✅ ComfyUI重启成功")
            return True
        else:
            print("❌ ComfyUI重启失败")
            return False
    
    def check_comfyui_server(self, timeout: int = 5) -> bool:
        """检查ComfyUI服务是否可用"""
        try:
            response = requests.get(f"{self.comfyui_url}/", timeout=timeout)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def get_queue_status(self) -> Optional[Dict]:
        """
        获取队列状态
        
        返回:
            队列状态字典或None
        """
        try:
            response = requests.get(self.api_queue, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"⚠️ 获取队列状态失败: {e}")
        return None
    
    def check_task_in_queue(self, prompt_id: str) -> str:
        """
        检查任务是否在队列中
        
        返回:
            "running" - 正在执行
            "pending" - 等待中
            "not_found" - 不在队列中
            "error" - 检查失败
        """
        try:
            queue_data = self.get_queue_status()
            if not queue_data:
                return "error"
            
            # 检查正在运行的任务
            for task in queue_data.get("queue_running", []):
                if len(task) > 1 and task[1] == prompt_id:
                    return "running"
            
            # 检查等待中的任务
            for task in queue_data.get("queue_pending", []):
                if len(task) > 1 and task[1] == prompt_id:
                    return "pending"
            
            return "not_found"
            
        except Exception as e:
            print(f"⚠️ 检查队列失败: {e}")
            return "error"
    
    def check_history_api(self, prompt_id: str, max_items: int = 5) -> Dict:
        """
        检查历史记录中的任务状态
        
        返回状态字典:
        {
            "status": "success" | "interrupted" | "error" | "not_found",
            "completed": bool,
            "has_error": bool,
            "message": str
        }
        """
        try:
            response = requests.get(f"{self.api_history}?max_items={max_items}", timeout=10)
            if response.status_code == 200:
                history_data = response.json()
                
                # 查找特定任务
                for task_id, task_info in history_data.items():
                    if task_id == prompt_id:
                        status_info = task_info.get("status", {})
                        
                        # 1. 成功完成
                        if status_info.get("status_str") == "success" and status_info.get("completed", False):
                            return {
                                "status": "success",
                                "completed": True,
                                "has_error": False,
                                "message": "任务成功完成"
                            }
                        
                        # 2. 中断
                        messages = status_info.get("messages", [])
                        is_interrupted = any(msg[0] == "execution_interrupted" for msg in messages)
                        
                        if is_interrupted or (status_info.get("status_str") == "error" and not status_info.get("completed", False)):
                            return {
                                "status": "interrupted",
                                "completed": False,
                                "has_error": True,
                                "message": "任务被中断"
                            }
                        
                        # 3. 错误
                        if status_info.get("status_str") == "error":
                            return {
                                "status": "error",
                                "completed": status_info.get("completed", False),
                                "has_error": True,
                                "message": "任务执行错误"
                            }
                
                # 任务不在历史记录中
                return {
                    "status": "not_found",
                    "completed": False,
                    "has_error": False,
                    "message": "任务不在历史记录中"
                }
        except Exception as e:
            print(f"⚠️ 检查历史API失败: {e}")
        
        return {
            "status": "error",
            "completed": False,
            "has_error": True,
            "message": "检查历史记录失败"
        }
    
    def get_task_status(self, prompt_id: str) -> Dict:
        """
        综合检查任务状态
        
        返回状态字典:
        {
            "status": "success" | "interrupted" | "error" | "running" | "pending" | "unknown",
            "in_queue": bool,
            "in_history": bool,
            "is_completed": bool,
            "message": str
        }
        """
        # 1. 先检查队列
        queue_status = self.check_task_in_queue(prompt_id)
        
        if queue_status == "running":
            return {
                "status": "running",
                "in_queue": True,
                "in_history": False,
                "is_completed": False,
                "message": "任务正在执行中"
            }
        elif queue_status == "pending":
            return {
                "status": "pending",
                "in_queue": True,
                "in_history": False,
                "is_completed": False,
                "message": "任务在队列中等待"
            }
        elif queue_status == "not_found":
            # 2. 不在队列中，检查历史记录
            history_result = self.check_history_api(prompt_id)
            
            if history_result["status"] == "success":
                return {
                    "status": "success",
                    "in_queue": False,
                    "in_history": True,
                    "is_completed": True,
                    "message": history_result["message"]
                }
            elif history_result["status"] in ["interrupted", "error"]:
                return {
                    "status": history_result["status"],
                    "in_queue": False,
                    "in_history": True,
                    "is_completed": history_result["completed"],
                    "message": history_result["message"]
                }
            else:
                # 既不在队列也不在历史记录
                return {
                    "status": "unknown",
                    "in_queue": False,
                    "in_history": False,
                    "is_completed": False,
                    "message": "任务状态未知"
                }
        
        # 队列检查失败
        return {
            "status": "unknown",
            "in_queue": False,
            "in_history": False,
            "is_completed": False,
            "message": "无法获取任务状态"
        }
    
    def smart_wait_for_completion(self, prompt_id: str, video_path: str, max_retries: int = 3) -> Tuple[bool, bool, int]:
        """
        智能等待任务完成
        
        返回:
            (success: bool, need_restart: bool, retry_count: int)
        """
        video_name = os.path.basename(video_path)
        print(f"⏳ 等待任务 {prompt_id} 完成...")
        
        start_time = time.time()
        max_wait_time = 3600  # 1小时
        poll_interval = 5
        
        retry_count = 0
        last_status = ""
        
        while time.time() - start_time < max_wait_time:
            # 检查ComfyUI服务是否可用
            if not self.check_comfyui_server():
                print("❌ ComfyUI服务不可用，需要重启")
                return False, True, retry_count
            
            # 获取任务状态
            task_status = self.get_task_status(prompt_id)
            current_status = task_status["status"]
            
            # 输出状态变化
            if current_status != last_status:
                status_messages = {
                    "running": "▶️ 任务执行中",
                    "pending": "⏳ 任务排队中", 
                    "success": "✅ 任务成功完成",
                    "interrupted": "⏹️ 任务被中断",
                    "error": "❌ 任务执行错误",
                    "unknown": "❓ 任务状态未知"
                }
                message = status_messages.get(current_status, current_status)
                print(f"[{time.strftime('%H:%M:%S')}] {message}")
                last_status = current_status
            
            # 处理不同状态
            if current_status == "success":
                print(f"✅ 任务 {prompt_id} 成功完成")
                return True, False, retry_count
            
            elif current_status == "interrupted":
                print(f"⏹️ 任务被中断，将重试 (重试 {retry_count + 1}/{max_retries})")
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"❌ 已达到最大重试次数 ({max_retries})")
                    return False, False, retry_count
                else:
                    # 清理输出文件后重试
                    self.clean_output_files(video_path)
                    return False, False, retry_count
            
            elif current_status == "error":
                print(f"❌ 任务执行错误，将重试 (重试 {retry_count + 1}/{max_retries})")
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"❌ 已达到最大重试次数 ({max_retries})")
                    return False, False, retry_count
                else:
                    self.clean_output_files(video_path)
                    return False, False, retry_count
            
            elif current_status in ["running", "pending"]:
                # 任务正在进行中，继续等待
                time.sleep(poll_interval)
                continue
            
            elif current_status == "unknown":
                # 状态未知，可能是网络问题或任务被系统移除
                print("⚠️ 任务状态未知，等待后重新检查...")
                time.sleep(poll_interval * 2)
                continue
        
        # 超时
        print(f"⏰ 任务 {prompt_id} 等待超时")
        retry_count += 1
        return False, False, retry_count
    
    def clean_output_files(self, video_path: str):
        """清理指定视频的输出文件"""
        video_name = os.path.basename(video_path)
        base_name = os.path.splitext(video_name)[0]
        print(f"🧹 清理 {video_name} 的输出文件...")
        
        try:
            patterns = [
                f"flashvsr_{base_name}_enhanced*.mp4",
                f"flashvsr_{base_name}_enhanced*.png",
                f"*{base_name}*.mp4",
                f"*{base_name}*.png",
            ]
            
            deleted_count = 0
            for pattern in patterns:
                full_pattern = os.path.join(self.output_dir, pattern)
                for file_path in glob(full_pattern):
                    try:
                        filename = os.path.basename(file_path)
                        if base_name in filename:
                            os.remove(file_path)
                            deleted_count += 1
                    except Exception:
                        pass
            
            if deleted_count > 0:
                print(f"✅ 已清理 {deleted_count} 个输出文件")
            else:
                print("ℹ️ 未找到需要清理的输出文件")
                
        except Exception as e:
            print(f"⚠️ 清理输出文件时出错: {e}")
    
    def get_video_frame_count(self, video_path: str) -> Tuple[int, float, str]:
        """获取视频的总帧数、帧率和检测方法"""
        try:
            if PYMEDIAINFO_AVAILABLE:
                media_info = MediaInfo.parse(video_path)
                for track in media_info.tracks:
                    if track.track_type == 'Video':
                        frame_count = 0
                        if hasattr(track, 'frame_count') and track.frame_count:
                            frame_count = int(track.frame_count)
                        
                        frame_rate = 25.0
                        if hasattr(track, 'frame_rate') and track.frame_rate:
                            try:
                                frame_rate_str = str(track.frame_rate)
                                if '/' in frame_rate_str:
                                    numerator, denominator = map(float, frame_rate_str.split('/'))
                                    frame_rate = numerator / denominator
                                else:
                                    frame_rate = float(frame_rate_str)
                            except:
                                frame_rate = 25.0
                        
                        if frame_count > 0:
                            return frame_count, frame_rate, "pymediainfo"
            
            # 备用方法
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
        """
        更新工作流中的所有参数 - 兼容性修正版
        支持占位符替换和直接赋值
        """
        modified_workflow = json.loads(json.dumps(workflow))
        
        print("=== 工作流参数更新开始 ===")
        
        # 1. 设置输入视频路径
        for node_id, node_data in modified_workflow.items():
            if node_data.get("class_type") == "VHS_LoadVideo":
                current_video = node_data["inputs"].get("video", "")
                if "{{VIDEO_PATH}}" in str(current_video):
                    node_data["inputs"]["video"] = video_path
                    print(f"✅ 已设置视频路径: {video_path}")
                elif isinstance(node_data["inputs"].get("video"), str):
                    node_data["inputs"]["video"] = video_path
                    print(f"✅ 已设置视频路径: {video_path} (直接赋值)")
        
        # 2. 设置 FlashVSR 核心参数 - 检查占位符
        for node_id, node_data in modified_workflow.items():
            if node_data.get("class_type") == "FlashVSRNodeAdv":
                # 检查scale参数
                current_scale = str(node_data["inputs"].get("scale", ""))
                if "{{scale}}" in current_scale:
                    node_data["inputs"]["scale"] = scale
                    print(f"✅ 已设置缩放比例: {scale}")
                elif isinstance(node_data["inputs"].get("scale"), (int, float, str)):
                    try:
                        node_data["inputs"]["scale"] = float(scale)
                        print(f"✅ 已设置缩放比例: {scale} (直接赋值)")
                    except:
                        pass
                
                # 检查tile_size参数
                current_tile_size = str(node_data["inputs"].get("tile_size", ""))
                if "{{t_z}}" in current_tile_size or "{{tile_size}}" in current_tile_size:
                    node_data["inputs"]["tile_size"] = tile_size
                    print(f"✅ 已设置分块大小: {tile_size}")
                elif isinstance(node_data["inputs"].get("tile_size"), (int, float, str)):
                    try:
                        node_data["inputs"]["tile_size"] = int(tile_size)
                        print(f"✅ 已设置分块大小: {tile_size} (直接赋值)")
                    except:
                        pass
                
                # 检查tile_overlap参数
                current_tile_overlap = str(node_data["inputs"].get("tile_overlap", ""))
                if "{{t_o}}" in current_tile_overlap or "{{tile_overlap}}" in current_tile_overlap:
                    node_data["inputs"]["tile_overlap"] = tile_overlap
                    print(f"✅ 已设置分块重叠: {tile_overlap}")
                elif isinstance(node_data["inputs"].get("tile_overlap"), (int, float, str)):
                    try:
                        node_data["inputs"]["tile_overlap"] = int(tile_overlap)
                        print(f"✅ 已设置分块重叠: {tile_overlap} (直接赋值)")
                    except:
                        pass
        
        # 3. 设置 GPU 设备 - 检查占位符
        for node_id, node_data in modified_workflow.items():
            if node_data.get("class_type") == "FlashVSRInitPipe":
                current_device = str(node_data["inputs"].get("device", ""))
                if "{{gpu}}" in current_device:
                    if gpu_device == "auto":
                        device_value = "auto"
                    elif gpu_device.isdigit():
                        device_value = f"cuda:{gpu_device}"
                    else:
                        device_value = gpu_device
                    node_data["inputs"]["device"] = device_value
                    print(f"✅ 已设置GPU设备: {device_value}")
                elif isinstance(node_data["inputs"].get("device"), str):
                    if gpu_device.isdigit():
                        device_value = f"cuda:{gpu_device}"
                    else:
                        device_value = gpu_device
                    node_data["inputs"]["device"] = device_value
                    print(f"✅ 已设置GPU设备: {device_value} (直接赋值)")
        
        # 4. 设置总帧数 - 检查占位符
        if total_frames is None:
            total_frames, _, _ = self.get_video_frame_count(video_path)
            print(f"📊 自动检测到视频总帧数: {total_frames}")
        
        for node_id, node_data in modified_workflow.items():
            if node_data.get("class_type") == "PrimitiveInt":
                # 检查是否是总帧数节点（节点50）
                if node_id == "50":
                    current_value = str(node_data["inputs"].get("value", ""))
                    if "{{TOTAL_FRAMES}}" in current_value or "{{TATAL_FRAMES}}" in current_value:
                        node_data["inputs"]["value"] = total_frames
                        print(f"✅ 已设置总帧数: {total_frames} 到节点 50")
                    elif isinstance(node_data["inputs"].get("value"), (int, float, str)):
                        try:
                            node_data["inputs"]["value"] = int(total_frames)
                            print(f"✅ 已设置总帧数: {total_frames} 到节点 50 (直接赋值)")
                        except:
                            pass
                    else:
                        print(f"⚠️  节点 50 的值格式异常: {current_value}")
                
                # 检查是否是每批帧数节点（节点8）
                elif node_id == "8":
                    current_value = str(node_data["inputs"].get("value", ""))
                    if "{{FRAMES_PER_BATCH}}" in current_value:
                        node_data["inputs"]["value"] = frames_per_batch
                        print(f"✅ 已设置每批帧数: {frames_per_batch} 到节点 8")
                    elif isinstance(node_data["inputs"].get("value"), (int, float, str)):
                        try:
                            node_data["inputs"]["value"] = int(frames_per_batch)
                            print(f"✅ 已设置每批帧数: {frames_per_batch} 到节点 8 (直接赋值)")
                        except:
                            pass
                    else:
                        print(f"⚠️  节点 8 的值格式异常: {current_value}")
        
        # 5. 设置输出文件名前缀 - 检查占位符
        if output_prefix is None:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_prefix = f"flashvsr_{base_name}_enhanced"
        
        for node_id, node_data in modified_workflow.items():
            if node_data.get("class_type") == "VHS_VideoCombine":
                current_prefix = str(node_data["inputs"].get("filename_prefix", ""))
                if "{{OUTPUT_PREFIX}}" in current_prefix:
                    node_data["inputs"]["filename_prefix"] = output_prefix
                    print(f"✅ 已设置输出前缀: {output_prefix}")
                elif isinstance(node_data["inputs"].get("filename_prefix"), str):
                    node_data["inputs"]["filename_prefix"] = output_prefix
                    print(f"✅ 已设置输出前缀: {output_prefix} (直接赋值)")
        
        print("=== 工作流参数更新完成 ===")
        return modified_workflow
    
    def queue_prompt(self, workflow: Dict, timeout: int = 300) -> Optional[str]:
        """将工作流发送到 ComfyUI 执行 - 带详细调试信息"""
        if not self.check_comfyui_server():
            print("❌ ComfyUI 服务不可用，无法提交任务")
            return None

        print("=== 工作流参数验证 ===")
        # 验证关键节点
        key_nodes = ["5", "8", "50", "49", "62"]
        for node_id in key_nodes:
            if node_id in workflow:
                node_data = workflow[node_id]
                node_type = node_data.get("class_type", "Unknown")
                inputs = node_data.get("inputs", {})
                print(f"节点 {node_id} ({node_type}):")
                
                for key in ["device", "value", "video", "filename_prefix", "scale", "tile_size", "tile_overlap"]:
                    if key in inputs:
                        print(f"  {key}: {inputs[key]}")
        
        try:
            print(f"📤 正在提交任务到: {self.api_prompt}")
            print(f"📦 工作流大小: {len(str(workflow))} 字符")
            
            response = requests.post(
                self.api_prompt, 
                json={"prompt": workflow}, 
                timeout=timeout,
                headers={'Content-Type': 'application/json'}
            )
            
            print(f"📥 响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                prompt_id = data.get('prompt_id')
                
                if prompt_id:
                    print(f"✅ 任务已提交，ID: {prompt_id}")
                    return prompt_id
                else:
                    print(f"❌ 未收到任务ID，完整响应: {data}")
                    return None
            else:
                print(f"❌ 请求失败，状态码: {response.status_code}")
                print(f"📄 错误详情: {response.text[:500]}")
                
                # 如果收到400错误，可能是工作流格式问题
                if response.status_code == 400:
                    print("🔍 分析400错误可能的原因:")
                    print("  1. 工作流中存在未替换的占位符（如{{gpu}}、{{TOTAL_FRAMES}}）")
                    print("  2. 工作流格式不符合ComfyUI要求")
                    print("  3. 某些节点参数类型不正确")
                    
                    # 检查工作流中是否还有占位符
                    workflow_str = json.dumps(workflow)
                    placeholder_patterns = ["{{gpu}}", "{{TOTAL_FRAMES}}", "{{FRAMES_PER_BATCH}}", 
                                          "{{OUTPUT_PREFIX}}", "{{VIDEO_PATH}}", "{{scale}}"]
                    
                    for pattern in placeholder_patterns:
                        if pattern in workflow_str:
                            print(f"⚠️  发现未替换的占位符: {pattern}")
                
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
    
    def process_video_with_retry(
        self,
        workflow_template: Dict,
        video_path: str,
        output_prefix: Optional[str] = None,
        scale: float = 4.0,
        tile_size: int = 256,
        tile_overlap: int = 24,
        total_frames: Optional[int] = None,
        frames_per_batch: int = 125,
        gpu_device: str = "auto",
        max_retries: int = 3
    ) -> Tuple[bool, int, str]:
        """
        处理单个视频文件，支持智能重试
        
        返回:
            (success: bool, retry_count: int, final_prompt_id: str)
        """
        video_name = os.path.basename(video_path)
        print(f"\n{'='*60}")
        print(f"处理视频: {video_path}")
        print(f"{'='*60}")
        
        retry_count = 0
        current_prompt_id = None
        
        while retry_count < max_retries:
            retry_count += 1
            print(f"\n🔄 尝试 {retry_count}/{max_retries}")
            
            # 1. 检查并确保ComfyUI正在运行
            if not self.check_comfyui_server():
                print("❌ ComfyUI服务未运行，正在启动...")
                if not self.start_comfyui():
                    print("❌ 无法启动ComfyUI，跳过此视频")
                    return False, retry_count, ""
            
            # 2. 更新工作流参数并提交任务
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
            
            current_prompt_id = self.queue_prompt(workflow)
            if not current_prompt_id:
                print(f"❌ 任务提交失败，将重试")
                if retry_count < max_retries:
                    print("⏳ 5秒后重试...")
                    time.sleep(5)
                    continue
                return False, retry_count, ""
            
            # 3. 智能等待任务完成
            success, need_restart, wait_retries = self.smart_wait_for_completion(
                current_prompt_id, 
                video_path,
                max_retries - retry_count + 1  # 剩余重试次数
            )
            
            if need_restart:
                # ComfyUI需要重启
                print("🔄 ComfyUI需要重启...")
                if not self.restart_comfyui():
                    print("❌ ComfyUI重启失败")
                    return False, retry_count, current_prompt_id
                
                # 清理输出文件后继续重试
                self.clean_output_files(video_path)
                continue
            
            elif success:
                # 任务成功完成
                print(f"✅ 视频 {video_name} 处理成功")
                output_files = self.get_output_files(current_prompt_id)
                if output_files:
                    print(f"📁 生成的文件:")
                    for file in output_files:
                        print(f"  - {file}")
                return True, retry_count, current_prompt_id
            
            else:
                # 任务失败但不是因为ComfyUI需要重启
                print(f"❌ 视频 {video_name} 处理失败")
                
                if retry_count < max_retries:
                    print(f"⏳ 等待5秒后重试...")
                    time.sleep(5)
                    self.clean_output_files(video_path)
                    continue
                else:
                    print(f"❌ 已达到最大重试次数 ({max_retries})")
                    return False, retry_count, current_prompt_id
        
        return False, retry_count, current_prompt_id if current_prompt_id else ""
    
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
        gpu_device: str = "auto",
        max_retries: int = 3
    ) -> Dict[str, Dict]:
        """
        批量处理多个视频文件
        
        返回:
            字典：{视频文件: {成功: bool, 重试次数: int, prompt_id: str}}
        """
        # 加载工作流模板
        try:
            workflow_template = self.load_workflow_template(workflow_template_path)
            print(f"✅ 已加载工作流模板: {workflow_template_path}")
        except Exception as e:
            print(f"❌ 加载工作流模板失败: {e}")
            return {}
        
        # 检查ComfyUI服务
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
        print(f"🔄 每个任务最多重试: {max_retries}次")
        print(f"💾 输出目录: {self.output_dir}")
        print(f"📋 工作流模板: {workflow_template_path}")
        
        for i, video_path in enumerate(video_files, 1):
            print(f"\n📊 进度: {i}/{total_videos}")
            
            # 设置输出前缀
            output_prefix = None
            if output_prefix_base:
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                output_prefix = f"{output_prefix_base}_{base_name}"
            
            # 处理单个视频
            success, retry_count, prompt_id = self.process_video_with_retry(
                workflow_template, 
                video_path, 
                output_prefix,
                scale=scale,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                total_frames=total_frames,
                frames_per_batch=frames_per_batch,
                gpu_device=gpu_device,
                max_retries=max_retries
            )
            
            results[video_path] = {
                "success": success,
                "retry_count": retry_count,
                "prompt_id": prompt_id,
                "message": "成功" if success else f"失败（重试{retry_count}次）"
            }
            
            if not success:
                print(f"⚠️ 视频 {os.path.basename(video_path)} 处理失败，继续下一个")
        
        # 输出统计信息
        print(f"\n{'='*60}")
        print("批量处理完成")
        print(f"{'='*60}")
        
        success_count = sum(1 for r in results.values() if r["success"])
        total_retries = sum(r["retry_count"] for r in results.values())
        
        print(f"✅ 成功: {success_count}/{total_videos}")
        print(f"❌ 失败: {total_videos - success_count}/{total_videos}")
        print(f"🔄 总重试次数: {total_retries}")
        
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
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='ComfyUI FlashVSR 批量视频处理工具 - 增强任务监控版（修正版）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 处理单个视频文件，使用GPU 0
  python batch_process_videos.py --input video.mp4 --gpu 0
  
  # 处理目录下的所有视频文件，使用GPU 1
  python batch_process_videos.py --input ./videos --gpu 1
  
  # 自动选择GPU
  python batch_process_videos.py --input ./videos --gpu auto
  
  # 自定义重试次数
  python batch_process_videos.py --input ./videos --max-retries 5 --gpu 0

主要改进:
  - 修复占位符替换问题（支持 {{gpu}}, {{TOTAL_FRAMES}}, {{FRAMES_PER_BATCH}} 等）
  - 增强调试信息输出
  - 智能任务状态监控（使用队列API和历史API）
  - 自动区分任务状态：运行中、排队中、成功、中断、错误
  - 失败时自动重试，最多重试3次（可配置）
  - 当ComfyUI服务不可用时自动重启
  - 重启前清理当前视频的输出文件
  - 重启后继续处理当前视频
        """
    )
    
    # 输入参数
    parser.add_argument('--template', type=str, default='flashvsr_template.json',
                       help='工作流模板 JSON 文件路径 (默认: flashvsr_template.json)')
    parser.add_argument('--input', type=str, required=True,
                       help='输入路径（可以是视频文件路径或包含视频文件的目录）')
    parser.add_argument('--pattern', type=str, default='*.mp4',
                       help='视频文件匹配模式，当输入是目录时使用 (默认: *.mp4)')
    
    # 输出参数
    parser.add_argument('--output-prefix', type=str, 
                       help='输出文件名前缀（可选，用于区分批次）')
    
    # FlashVSR 处理参数
    parser.add_argument('--scale', type=float, default=4.0,
                       help='放大倍数 (默认: 4.0)')
    parser.add_argument('--tile-size', type=int, default=256,
                       help='分块大小 (默认: 256)')
    parser.add_argument('--tile-overlap', type=int, default=24,
                       help='分块重叠像素 (默认: 24)')
    parser.add_argument('--frames-per-batch', type=int, default=201,
                       help='每批处理的帧数 (默认: 201)')
    parser.add_argument('--total-frames', type=int,
                       help='视频总帧数 (如不提供则自动检测)')
    
    # 重试参数
    parser.add_argument('--max-retries', type=int, default=3,
                       help='每个任务的最大重试次数 (默认: 3)')
    
    # GPU参数
    parser.add_argument('--gpu', type=str, default='auto',
                       help='GPU设备选择: auto, 0, 1, 2, cuda:0, cuda:1等 (默认: auto)')
    
    # 系统参数
    parser.add_argument('--server', type=str, default='http://127.0.0.1:8188',
                       help='ComfyUI 服务器地址 (默认: http://127.0.0.1:8188)')
    parser.add_argument('--skip-pymedia-check', action='store_true',
                       help='跳过 pymediainfo 检查')
    
    args = parser.parse_args()
    
    # 检查 pymediainfo
    if not PYMEDIAINFO_AVAILABLE and not args.skip_pymedia_check:
        print("⚠️  未检测到 pymediainfo 库")
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
    print(f"  max_retries: {args.max_retries}")
    
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
        gpu_device=args.gpu,
        max_retries=args.max_retries
    )
    
    # 计算总耗时
    total_time = time.time() - start_time
    print(f"\n⏱️  总耗时: {total_time:.2f} 秒")
    
    # 最后关闭ComfyUI进程
    print(f"\n🔧 处理完成，正在关闭ComfyUI进程...")
    processor.kill_comfyui_processes()
    print("✅ 所有处理完成！")

if __name__ == "__main__":
    main()
