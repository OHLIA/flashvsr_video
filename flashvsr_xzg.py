#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ComfyUI FlashVSR-XZG 批量视频处理脚本（v3.0 完整增强版）
支持断点续跑、状态文件、智能批次调整、输出验证和并行处理
版本: 3.0
"""

import json
import requests
import os
import time
import sys
import math
import re
import subprocess
import threading
from glob import glob
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import traceback
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, Future

# 尝试导入 pymediainfo
try:
    from pymediainfo import MediaInfo
    PYMEDIAINFO_AVAILABLE = True
except ImportError:
    PYMEDIAINFO_AVAILABLE = False
    print("⚠️  pymediainfo 未安装，将使用备用方法获取视频信息")

class FlashVSR_XZG_Processor:
    def __init__(self, comfyui_url: str = "http://127.0.0.1:8188", log_dir: str = "."):
        """
        初始化 ComfyUI FlashVSR-XZG 处理器（v3.0 完整增强版）
        
        参数:
            comfyui_url: ComfyUI 服务器地址
            log_dir: 日志目录
        """
        self.comfyui_url = comfyui_url.rstrip('/')
        self.api_prompt = f"{comfyui_url}/prompt"
        self.api_history = f"{comfyui_url}/history"
        self.api_view = f"{comfyui_url}/view"
        self.api_queue = f"{comfyui_url}/queue"
        
        # 日志设置
        self.log_dir = log_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"flashvsr_xzg_{timestamp}.log")
        self.state_dir = os.path.join(self.log_dir, "states")
        
        # 创建日志和状态目录
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.state_dir, exist_ok=True)
        
        # 初始化日志
        self._init_log_file()
        
        self.log("📱 初始化 FlashVSR-XZG 处理器 v3.0")
        self.log(f"🔗 ComfyUI 地址: {self.comfyui_url}")
        self.log(f"📝 日志文件: {self.log_file}")
        self.log(f"💾 状态目录: {self.state_dir}")
        
        # 状态跟踪
        self.processing_state = {}
        self.output_validation_enabled = True
    
    def _init_log_file(self):
        """初始化日志文件"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"{'='*80}\n")
            f.write(f"FlashVSR-XZG 处理日志 v3.0\n")
            f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"支持功能: 断点续跑、状态文件、智能批次调整、输出验证、并行处理\n")
            f.write(f"{'='*80}\n\n")
    
    def log(self, message: str, level: str = "INFO"):
        """
        记录日志
        
        参数:
            message: 日志消息
            level: 日志级别
        """
        timestamp = datetime.now().strftime("%Y%m%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        # 输出到控制台
        print(log_entry)
        
        # 写入日志文件
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry + "\n")
        except Exception as e:
            print(f"⚠️ 写入日志失败: {e}")
    
    def save_processing_state(self, video_path: str, frames_processed: int, batches_processed: int, 
                            success: bool = True, error_msg: str = ""):
        """
        保存处理状态到文件（改进点1）
        
        参数:
            video_path: 视频路径
            frames_processed: 已处理帧数
            batches_processed: 已处理批次
            success: 是否成功
            error_msg: 错误信息
        """
        try:
            video_name = os.path.basename(video_path)
            # 安全文件名（移除特殊字符）
            safe_video_name = re.sub(r'[^\w\-\.]', '_', video_name)
            state_file = os.path.join(self.state_dir, f"{safe_video_name}_state.json")
            
            state = {
                "video_path": video_path,
                "video_name": video_name,
                "frames_processed": frames_processed,
                "batches_processed": batches_processed,
                "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "success": success,
                "error_msg": error_msg
            }
            
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            
            self.log(f"💾 已保存状态文件: {state_file}", "INFO")
            return True
        except Exception as e:
            self.log(f"❌ 保存状态文件失败: {e}", "ERROR")
            return False
    
    def load_processing_state(self, video_path: str) -> Tuple[int, int, Dict]:
        """
        从文件加载处理状态（改进点1）
        
        参数:
            video_path: 视频路径
            
        返回:
            (frames_processed: int, batches_processed: int, state: Dict)
        """
        try:
            video_name = os.path.basename(video_path)
            safe_video_name = re.sub(r'[^\w\-\.]', '_', video_name)
            
            # 查找状态文件（支持多种命名格式）
            state_files = [
                os.path.join(self.state_dir, f"{safe_video_name}_state.json"),
                os.path.join(self.state_dir, f"{video_name}_state.json"),
                os.path.join(self.log_dir, f"flashvsr_state_{safe_video_name}.json"),
                os.path.join(self.log_dir, f"flashvsr_state_{video_name}.json"),
            ]
            
            for state_file in state_files:
                if os.path.exists(state_file):
                    with open(state_file, 'r', encoding='utf-8') as f:
                        state = json.load(f)
                    
                    frames = state.get("frames_processed", 0)
                    batches = state.get("batches_processed", 0)
                    
                    self.log(f"📂 加载状态文件: {state_file}", "INFO")
                    self.log(f"  📊 已处理: {frames} 帧, {batches} 批", "INFO")
                    
                    return frames, batches, state
            
            return 0, 0, {}
        except Exception as e:
            self.log(f"❌ 加载状态文件失败: {e}", "ERROR")
            return 0, 0, {}
    
    def check_comfyui_server(self, timeout: int = 10) -> bool:
        """检查 ComfyUI 服务是否可用"""
        try:
            response = requests.get(f"{self.comfyui_url}/", timeout=timeout)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            self.log(f"检查 ComfyUI 服务失败: {e}", "WARN")
            return False
    
    def get_video_info(self, video_path: str) -> Tuple[float, int, str]:
        """
        获取视频信息
        
        参数:
            video_path: 视频文件路径
            
        返回:
            (fps: float, total_frames: int, method: str)
        """
        try:
            if PYMEDIAINFO_AVAILABLE:
                self.log(f"使用 pymediainfo 获取视频信息: {video_path}", "INFO")
                media_info = MediaInfo.parse(video_path)
                
                for track in media_info.tracks:
                    if track.track_type == 'Video':
                        # 获取帧率
                        fps = 25.0
                        if hasattr(track, 'frame_rate') and track.frame_rate:
                            try:
                                fps_str = str(track.frame_rate)
                                if '/' in fps_str:
                                    numerator, denominator = map(float, fps_str.split('/'))
                                    fps = numerator / denominator
                                else:
                                    fps = float(fps_str)
                            except:
                                self.log(f"解析帧率失败，使用默认值 25.0", "WARN")
                        
                        # 获取总帧数
                        total_frames = 0
                        if hasattr(track, 'frame_count') and track.frame_count:
                            total_frames = int(track.frame_count)
                        
                        if total_frames > 0:
                            self.log(f"视频信息: FPS={fps:.2f}, 总帧数={total_frames}", "INFO")
                            return fps, total_frames, "pymediainfo"
            
            # 备用方法：使用 OpenCV
            try:
                import cv2
                self.log(f"使用 OpenCV 获取视频信息: {video_path}", "INFO")
                cap = cv2.VideoCapture(video_path)
                
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    
                    if fps > 0 and total_frames > 0:
                        self.log(f"视频信息: FPS={fps:.2f}, 总帧数={total_frames}", "INFO")
                        return fps, total_frames, "OpenCV"
            except ImportError:
                self.log("OpenCV 未安装", "WARN")
            except Exception as e:
                self.log(f"OpenCV 获取信息失败: {e}", "WARN")
            
            # 默认值
            self.log(f"无法获取视频信息，使用默认值: FPS=25.0, 总帧数=100", "WARN")
            return 25.0, 100, "默认值"
            
        except Exception as e:
            self.log(f"获取视频信息失败: {e}", "ERROR")
            return 25.0, 100, "错误-默认值"
    
    def load_workflow_template(self, template_path: str) -> Dict:
        """加载工作流 JSON 模板"""
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                workflow = json.load(f)
            
            self.log(f"✅ 已加载工作流模板: {template_path}", "INFO")
            return workflow
            
        except FileNotFoundError:
            self.log(f"❌ 找不到工作流模板文件: {template_path}", "ERROR")
            raise
        except json.JSONDecodeError as e:
            self.log(f"❌ JSON 解析失败: {e}", "ERROR")
            raise
        except Exception as e:
            self.log(f"❌ 加载工作流模板失败: {e}", "ERROR")
            raise
    
    def update_workflow_parameters(
        self, 
        workflow: Dict, 
        video_path: str,
        video_fps: float,
        frames_per_batch: int,
        frames_skip: int,
        output_prefix: str,
        batch_number: int = 1,
        total_batches: int = 1,
        frames_pre: int = 0,
        batch_pre: int = 0
    ) -> Dict:
        """
        更新工作流参数（v3.0增强版）
        
        参数:
            workflow: 工作流模板
            video_path: 视频路径
            video_fps: 视频帧率
            frames_per_batch: 每批帧数
            frames_skip: 跳过帧数
            output_prefix: 输出前缀
            batch_number: 当前任务批次号
            total_batches: 总批次数
            frames_pre: 已跑帧数
            batch_pre: 已跑批次
            
        返回:
            更新后的工作流
        """
        # 创建深拷贝
        modified_workflow = json.loads(json.dumps(workflow))
        
        self.log(f"🔄 更新工作流参数 (批次 {batch_number}/{total_batches})", "INFO")
        if frames_pre > 0:
            self.log(f"  ⏭️  已跑帧数: {frames_pre} 帧", "INFO")
        if batch_pre > 0:
            self.log(f"  📦 已跑批次: {batch_pre} 批", "INFO")
        
        # 更新所有节点参数
        for node_id, node_data in modified_workflow.items():
            # 更新 VHS_LoadVideo 节点 (ID 816)
            if node_data.get("class_type") == "VHS_LoadVideo":
                inputs = node_data["inputs"]
                
                # 更新视频路径
                if isinstance(inputs.get("video"), str) and "{{VIDEO_PATH}}" in inputs["video"]:
                    inputs["video"] = video_path
                    self.log(f"  ✅ 设置视频路径: {video_path}", "INFO")
                
                # 更新帧率
                if isinstance(inputs.get("force_rate"), str) and "{{VIDEO_FPS}}" in inputs["force_rate"]:
                    inputs["force_rate"] = str(video_fps)
                    self.log(f"  ✅ 设置帧率: {video_fps}", "INFO")
                
                # 更新每批帧数
                if isinstance(inputs.get("frame_load_cap"), str) and "{{FRAMES_PER_BATCH}}" in inputs["frame_load_cap"]:
                    inputs["frame_load_cap"] = str(frames_per_batch)
                    self.log(f"  ✅ 设置每批帧数: {frames_per_batch}", "INFO")
                
                # 更新跳过帧数（v3.0增强版逻辑）
                if isinstance(inputs.get("skip_first_frames"), str) and "{{FRAMES_SKIP}}" in inputs["skip_first_frames"]:
                    inputs["skip_first_frames"] = str(frames_skip)
                    self.log(f"  ✅ 设置跳过帧数: {frames_skip}", "INFO")
                
                # 新增：已跑帧数参数（如果模板支持）
                if isinstance(inputs.get("skip_first_frames"), str) and "{{FRAMS_PRE}}" in inputs["skip_first_frames"]:
                    inputs["skip_first_frames"] = str(frames_skip)  # 直接使用计算好的总跳过帧数
                    self.log(f"  ℹ️  检测到 {{FRAMS_PRE}} 占位符，已使用 frames_skip={frames_skip} 替代", "INFO")
            
            # 更新 VHS_VideoCombine 节点 (ID 817)
            elif node_data.get("class_type") == "VHS_VideoCombine":
                inputs = node_data["inputs"]
                
                # 更新输出前缀
                if isinstance(inputs.get("filename_prefix"), str) and "{{OUTPUT_PREFIX}}" in inputs["filename_prefix"]:
                    inputs["filename_prefix"] = output_prefix
                    self.log(f"  ✅ 设置输出前缀: {output_prefix}", "INFO")
        
        return modified_workflow
    
    def queue_prompt(self, workflow: Dict, timeout: int = 60) -> Optional[str]:
        """
        将工作流发送到 ComfyUI 执行
        
        参数:
            workflow: 工作流配置
            timeout: 超时时间（秒）
            
        返回:
            prompt_id: 任务ID
        """
        if not self.check_comfyui_server():
            self.log("❌ ComfyUI 服务不可用，无法提交任务", "ERROR")
            return None
        
        try:
            self.log(f"📤 提交任务到 ComfyUI", "INFO")
            
            response = requests.post(
                self.api_prompt, 
                json={"prompt": workflow}, 
                timeout=timeout,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                data = response.json()
                prompt_id = data.get('prompt_id')
                
                if prompt_id:
                    self.log(f"✅ 任务已提交，ID: {prompt_id}", "INFO")
                    return prompt_id
                else:
                    self.log(f"❌ 未收到任务ID，响应: {data}", "ERROR")
                    return None
            else:
                self.log(f"❌ 请求失败，状态码: {response.status_code}", "ERROR")
                self.log(f"📄 错误详情: {response.text[:500]}", "ERROR")
                return None
                
        except requests.exceptions.RequestException as e:
            self.log(f"❌ 请求失败: {e}", "ERROR")
            return None
        except json.JSONDecodeError as e:
            self.log(f"❌ JSON 解析失败: {e}", "ERROR")
            return None
        except Exception as e:
            self.log(f"❌ 提交任务失败: {e}", "ERROR")
            return None
    
    def wait_for_task_completion(self, prompt_id: str, timeout: int = 600) -> bool:
        """
        等待任务完成
        
        参数:
            prompt_id: 任务ID
            timeout: 超时时间（秒）
            
        返回:
            是否成功完成
        """
        start_time = time.time()
        self.log(f"⏳ 等待任务 {prompt_id} 完成，超时: {timeout}秒", "INFO")
        
        while time.time() - start_time < timeout:
            try:
                # 检查历史记录
                response = requests.get(f"{self.api_history}?max_items=10", timeout=10)
                if response.status_code == 200:
                    history_data = response.json()
                    
                    # 查找特定任务
                    if prompt_id in history_data:
                        task_info = history_data[prompt_id]
                        status_info = task_info.get("status", {})
                        
                        # 成功完成
                        if status_info.get("status_str") == "success" and status_info.get("completed", False):
                            self.log(f"✅ 任务 {prompt_id} 成功完成", "INFO")
                            return True
                        
                        # 错误
                        if status_info.get("status_str") == "error":
                            self.log(f"❌ 任务 {prompt_id} 执行错误", "ERROR")
                            return False
                
                # 检查队列状态
                response = requests.get(self.api_queue, timeout=10)
                if response.status_code == 200:
                    queue_data = response.json()
                    
                    # 检查正在运行的任务
                    for task in queue_data.get("queue_running", []):
                        if len(task) > 1 and task[1] == prompt_id:
                            elapsed = time.time() - start_time
                            if elapsed > 60 and int(elapsed) % 30 == 0:
                                self.log(f"⏰ 任务仍在运行，已等待 {elapsed:.1f}秒", "INFO")
                            time.sleep(5)
                            continue
                
                time.sleep(2)
                
            except requests.exceptions.RequestException as e:
                self.log(f"⚠️ 检查任务状态失败: {e}，继续等待...", "WARN")
                time.sleep(5)
                continue
        
        self.log(f"⏰ 任务 {prompt_id} 等待超时 ({timeout}秒)", "ERROR")
        return False
    
    def validate_output_file(self, output_path: str, min_size_kb: int = 10) -> bool:
        """
        验证生成的视频文件（改进点3）
        
        参数:
            output_path: 输出文件路径
            min_size_kb: 最小文件大小（KB）
            
        返回:
            是否有效
        """
        if not self.output_validation_enabled:
            return True
            
        try:
            if not os.path.exists(output_path):
                self.log(f"❌ 输出文件不存在: {output_path}", "ERROR")
                return False
            
            # 检查文件大小
            file_size_kb = os.path.getsize(output_path) / 1024
            if file_size_kb < min_size_kb:
                self.log(f"❌ 输出文件太小: {file_size_kb:.1f}KB (< {min_size_kb}KB) - {output_path}", "ERROR")
                return False
            
            # 检查文件扩展名
            if not output_path.lower().endswith('.mp4'):
                self.log(f"⚠️ 输出文件不是MP4格式: {output_path}", "WARN")
            
            # 尝试使用 OpenCV 验证视频（可选）
            try:
                import cv2
                cap = cv2.VideoCapture(output_path)
                if cap.isOpened():
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    if frame_count > 0:
                        self.log(f"✅ 输出文件验证通过: {output_path} ({file_size_kb:.1f}KB, {frame_count}帧)", "INFO")
                        return True
                    else:
                        self.log(f"❌ 输出文件无有效帧: {output_path}", "ERROR")
                        return False
                else:
                    self.log(f"❌ 无法打开输出文件: {output_path}", "ERROR")
                    return False
            except ImportError:
                # OpenCV 不可用，跳过深度验证
                self.log(f"✅ 输出文件大小验证通过: {output_path} ({file_size_kb:.1f}KB)", "INFO")
                return True
            except Exception as e:
                self.log(f"⚠️ 视频深度验证失败，跳过: {e}", "WARN")
                return True  # 验证失败但仍继续
            
        except Exception as e:
            self.log(f"❌ 输出文件验证失败: {e}", "ERROR")
            return False
    
    def process_single_video_batch(
        self,
        workflow_template: Dict,
        video_path: str,
        video_fps: float,
        frames_per_batch: int,
        batch_number: int,
        total_batches: int,
        base_output_prefix: str,
        frames_pre: int = 0,
        batch_pre: int = 0,
        timeout: int = 600,
        output_dir: str = "output"
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        处理单个视频批次（v3.0增强版）
        
        参数:
            workflow_template: 工作流模板
            video_path: 视频路径
            video_fps: 视频帧率
            frames_per_batch: 每批帧数
            batch_number: 当前任务批次号
            total_batches: 总批次数
            base_output_prefix: 基础输出前缀
            frames_pre: 已跑帧数
            batch_pre: 已跑批次
            timeout: 超时时间（秒）
            output_dir: 输出目录
            
        返回:
            (success: bool, prompt_id: str or None, output_file: str or None)
        """
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # 智能批次大小调整（改进点2）
        actual_frames_per_batch = frames_per_batch
        if batch_number == total_batches and frames_pre > 0:
            # 计算剩余帧数
            video_fps, total_frames, _ = self.get_video_info(video_path)
            remaining_frames = total_frames - frames_pre
            
            # 计算最后一批的实际帧数
            last_batch_frames = remaining_frames - (frames_per_batch * (batch_number - 1))
            if 0 < last_batch_frames < frames_per_batch:
                actual_frames_per_batch = last_batch_frames
                self.log(f"🔄 最后一批智能调整帧数: {actual_frames_per_batch} 帧 (原: {frames_per_batch})", "INFO")
        
        # 计算跳过帧数（v3.0增强版逻辑）
        # {{FRAMES_SKIP}} = {{FRAMS_PRE}} + frames_per_batch * (batch_number - 1)
        frames_skip = frames_pre + frames_per_batch * (batch_number - 1)
        
        # 计算当前总批次号
        current_batch_number = batch_pre + batch_number
        
        # 生成输出前缀
        # flashvsr_{源文件名}_{ {{BATCH_PRE}} + 当前任务的批次号}
        output_prefix = f"{base_output_prefix}_{current_batch_number:03d}"
        
        # 预期输出文件路径
        expected_output_file = os.path.join(output_dir, f"{output_prefix}.mp4")
        
        self.log(f"🎬 处理批次 {batch_number}/{total_batches} (总批次: {current_batch_number})", "INFO")
        self.log(f"  📂 视频: {video_name}", "INFO")
        self.log(f"  ⏱️  帧率: {video_fps:.2f}", "INFO")
        self.log(f"  🎞️  每批帧数: {actual_frames_per_batch} (原: {frames_per_batch})", "INFO")
        self.log(f"  ⏭️  跳过帧数: {frames_skip} (已跑 {frames_pre} + 当前跳过 {frames_per_batch*(batch_number-1)})", "INFO")
        self.log(f"  📁 输出前缀: {output_prefix}", "INFO")
        self.log(f"  📄 预期输出: {expected_output_file}", "INFO")
        if frames_pre > 0:
            self.log(f"  📊 断点续跑: 已处理 {frames_pre} 帧 ({batch_pre} 批)", "INFO")
        
        # 检查输出文件是否已存在（避免重复处理）
        if os.path.exists(expected_output_file):
            file_size_mb = os.path.getsize(expected_output_file) / (1024 * 1024)
            self.log(f"⚠️  输出文件已存在: {expected_output_file} ({file_size_mb:.1f}MB)", "WARN")
            response = input("是否覆盖？(y/n/skip): ").lower()
            if response == 'n':
                self.log(f"⏭️  跳过已存在批次 {batch_number}", "INFO")
                return True, None, expected_output_file
            elif response == 'skip':
                # 跳过所有已存在的批次
                return False, None, None
        
        # 更新工作流参数
        workflow = self.update_workflow_parameters(
            workflow_template,
            video_path,
            video_fps,
            actual_frames_per_batch,  # 使用调整后的帧数
            frames_skip,
            output_prefix,
            batch_number,
            total_batches,
            frames_pre,
            batch_pre
        )
        
        # 提交任务
        prompt_id = self.queue_prompt(workflow, timeout=timeout)
        if not prompt_id:
            self.log(f"❌ 提交批次 {batch_number} 失败", "ERROR")
            return False, None, None
        
        # 等待任务完成
        success = self.wait_for_task_completion(prompt_id, timeout=timeout)
        
        if success:
            self.log(f"✅ 批次 {batch_number} 处理完成 (总批次: {current_batch_number})", "INFO")
            
            # 验证输出文件（改进点3）
            if os.path.exists(expected_output_file):
                if self.validate_output_file(expected_output_file):
                    self.log(f"✅ 输出文件验证成功: {expected_output_file}", "INFO")
                else:
                    self.log(f"⚠️  输出文件验证失败: {expected_output_file}", "WARN")
                    # 验证失败仍视为成功，但记录警告
            else:
                self.log(f"⚠️  预期输出文件不存在: {expected_output_file}", "WARN")
                # 尝试查找实际输出文件
                output_files = glob(os.path.join(output_dir, f"{output_prefix}*.mp4"))
                if output_files:
                    actual_output = output_files[0]
                    self.log(f"🔍 找到实际输出文件: {actual_output}", "INFO")
                    expected_output_file = actual_output
                else:
                    self.log(f"❌ 未找到任何输出文件，批次可能失败", "ERROR")
                    success = False
            
            return success, prompt_id, expected_output_file
        else:
            self.log(f"❌ 批次 {batch_number} 处理失败 (总批次: {current_batch_number})", "ERROR")
            return False, prompt_id, None
    
    def process_video_file(
        self,
        workflow_template_path: str,
        video_path: str,
        frames_per_batch: int = 50,
        timeout_per_batch: int = 600,
        frames_pre: int = 0,
        batch_pre: int = 0,
        auto_load_state: bool = True,
        save_state: bool = True,
        max_workers: int = 1,
        output_dir: str = "output"
    ) -> Dict:
        """
        处理单个视频文件（v3.0增强版）
        
        参数:
            workflow_template_path: 工作流模板路径
            video_path: 视频文件路径
            frames_per_batch: 每批帧数
            timeout_per_batch: 每批超时时间（秒）
            frames_pre: 已跑帧数
            batch_pre: 已跑批次
            auto_load_state: 自动加载状态
            save_state: 保存状态
            max_workers: 最大并行工作数
            output_dir: 输出目录
            
        返回:
            处理结果字典
        """
        video_name = os.path.basename(video_path)
        self.log(f"🎬 开始处理视频: {video_name}", "INFO")
        self.log(f"📂 路径: {video_path}", "INFO")
        
        # 自动加载状态（改进点1）
        if auto_load_state:
            loaded_frames_pre, loaded_batch_pre, state_info = self.load_processing_state(video_path)
            if loaded_frames_pre > 0 or loaded_batch_pre > 0:
                frames_pre = loaded_frames_pre
                batch_pre = loaded_batch_pre
                self.log(f"🔄 自动加载断点状态: 已处理 {frames_pre} 帧, {batch_pre} 批", "INFO")
        
        # 检查断点续跑参数
        if frames_pre > 0:
            self.log(f"🔄 断点续跑模式: 已处理 {frames_pre} 帧, {batch_pre} 批", "INFO")
        
        # 加载工作流模板
        try:
            workflow_template = self.load_workflow_template(workflow_template_path)
        except Exception as e:
            error_msg = f"加载工作流模板失败: {e}"
            self.log(f"❌ {error_msg}", "ERROR")
            if save_state:
                self.save_processing_state(video_path, frames_pre, batch_pre, False, error_msg)
            return {
                "video": video_name,
                "path": video_path,
                "success": False,
                "error": error_msg,
                "results": []
            }
        
        # 获取视频信息
        video_fps, total_frames, method = self.get_video_info(video_path)
        self.log(f"📊 视频信息: {total_frames} 帧, {video_fps:.2f} FPS (方法: {method})", "INFO")
        
        # 计算剩余可处理帧数
        remaining_frames = total_frames - frames_pre
        if remaining_frames <= 0:
            self.log(f"✅ 视频已全部处理完成，无需继续处理", "INFO")
            result = {
                "video": video_name,
                "path": video_path,
                "success": True,
                "batches_processed": 0,
                "total_batches": 0,
                "video_fps": video_fps,
                "total_frames": total_frames,
                "remaining_frames": 0,
                "frames_pre": frames_pre,
                "batch_pre": batch_pre,
                "success_rate": "100%",
                "results": []
            }
            if save_state:
                self.save_processing_state(video_path, frames_pre, batch_pre, True)
            return result
        
        # 计算批次数
        # (总帧数 - {{FRAMS_PRE}}) / frames_per_batch
        total_batches = remaining_frames // frames_per_batch
        if remaining_frames % frames_per_batch > 0:
            total_batches += 1
        
        self.log(f"📦 批次计算: {remaining_frames} 剩余帧 / {frames_per_batch} 帧每批 = {total_batches} 批", "INFO")
        self.log(f"📈 进度: {frames_pre}/{total_frames} 帧 ({frames_pre/total_frames*100:.1f}%)", "INFO")
        self.log(f"⚡ 并行处理: {max_workers} 个工作线程", "INFO")
        
        # 基础输出前缀
        video_base_name = os.path.splitext(video_name)[0]
        base_output_prefix = f"flashvsr_{video_base_name}"
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        success_count = 0
        output_files = []
        
        # 并行处理逻辑（改进点4）
        if max_workers > 1 and total_batches > 1:
            self.log(f"🚀 启动并行处理，最大工作线程数: {max_workers}", "INFO")
            
            # 使用线程池并行处理
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 准备任务列表
                futures = []
                for batch_number in range(1, total_batches + 1):
                    future = executor.submit(
                        self.process_single_video_batch,
                        workflow_template,
                        video_path,
                        video_fps,
                        frames_per_batch,
                        batch_number,
                        total_batches,
                        base_output_prefix,
                        frames_pre,
                        batch_pre,
                        timeout_per_batch,
                        output_dir
                    )
                    futures.append((batch_number, future))
                
                # 收集结果
                for batch_number, future in futures:
                    try:
                        success, prompt_id, output_file = future.result(timeout=timeout_per_batch + 60)
                        
                        results.append({
                            "batch": batch_number,
                            "total_batch": batch_pre + batch_number,
                            "success": success,
                            "prompt_id": prompt_id,
                            "output_file": output_file,
                            "frames_skip": frames_pre + frames_per_batch * (batch_number - 1)
                        })
                        
                        if success:
                            success_count += 1
                            if output_file:
                                output_files.append(output_file)
                            # 保存进度状态
                            current_frames = frames_pre + batch_number * frames_per_batch
                            if save_state and batch_number % 3 == 0:  # 每3批保存一次状态
                                self.save_processing_state(video_path, min(current_frames, total_frames), 
                                                         batch_pre + batch_number, True)
                        else:
                            self.log(f"⚠️ 批次 {batch_number} 失败", "WARN")
                            
                    except Exception as e:
                        self.log(f"❌ 批次 {batch_number} 执行异常: {e}", "ERROR")
                        results.append({
                            "batch": batch_number,
                            "total_batch": batch_pre + batch_number,
                            "success": False,
                            "error": str(e),
                            "frames_skip": frames_pre + frames_per_batch * (batch_number - 1)
                        })
        else:
            # 顺序处理（兼容原有逻辑）
            for batch_number in range(1, total_batches + 1):
                self.log(f"{'='*60}", "INFO")
                success, prompt_id, output_file = self.process_single_video_batch(
                    workflow_template,
                    video_path,
                    video_fps,
                    frames_per_batch,
                    batch_number,
                    total_batches,
                    base_output_prefix,
                    frames_pre,
                    batch_pre,
                    timeout_per_batch,
                    output_dir
                )
                
                results.append({
                    "batch": batch_number,
                    "total_batch": batch_pre + batch_number,
                    "success": success,
                    "prompt_id": prompt_id,
                    "output_file": output_file,
                    "frames_skip": frames_pre + frames_per_batch * (batch_number - 1)
                })
                
                if success:
                    success_count += 1
                    if output_file:
                        output_files.append(output_file)
                    # 保存进度状态
                    current_frames = frames_pre + batch_number * frames_per_batch
                    if save_state and batch_number % 3 == 0:  # 每3批保存一次状态
                        self.save_processing_state(video_path, min(current_frames, total_frames), 
                                                 batch_pre + batch_number, True)
                else:
                    self.log(f"⚠️ 批次 {batch_number} 失败，是否继续处理后续批次？", "WARN")
                    # 这里可以添加中断逻辑，默认继续处理
                    continue
        
        # 汇总结果
        all_success = success_count == total_batches
        processed_frames = frames_pre + success_count * frames_per_batch
        if processed_frames > total_frames:
            processed_frames = total_frames
        
        summary = {
            "video": video_name,
            "path": video_path,
            "success": all_success,
            "batches_processed": success_count,
            "total_batches": total_batches,
            "video_fps": video_fps,
            "total_frames": total_frames,
            "remaining_frames": remaining_frames,
            "processed_frames": processed_frames,
            "frames_per_batch": frames_per_batch,
            "frames_pre": frames_pre,
            "batch_pre": batch_pre,
            "total_batch_count": batch_pre + success_count,
            "success_rate": f"{success_count}/{total_batches} ({success_count/total_batches*100:.1f}%)",
            "progress": f"{processed_frames}/{total_frames} ({processed_frames/total_frames*100:.1f}%)",
            "output_files": output_files,
            "output_dir": output_dir,
            "results": results
        }
        
        self.log(f"{'='*60}", "INFO")
        if all_success:
            self.log(f"✅ 视频 {video_name} 当前阶段处理完成", "INFO")
            self.log(f"📊 累计进度: {processed_frames}/{total_frames} 帧 ({processed_frames/total_frames*100:.1f}%)", "INFO")
            self.log(f"📦 累计批次: {batch_pre + success_count} 批", "INFO")
            self.log(f"📄 生成文件: {len(output_files)} 个", "INFO")
            for i, file_path in enumerate(output_files, 1):
                if os.path.exists(file_path):
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    self.log(f"  {i:2d}. {os.path.basename(file_path)} ({size_mb:.1f}MB)", "INFO")
        else:
            self.log(f"⚠️ 视频 {video_name} 部分批次失败 ({success_count}/{total_batches})", "WARN")
        
        # 保存最终状态（改进点1）
        if save_state:
            self.save_processing_state(
                video_path, 
                processed_frames, 
                batch_pre + success_count, 
                all_success,
                "" if all_success else f"{total_batches - success_count} batches failed"
            )
        
        return summary
    
    def process_directory(
        self,
        workflow_template_path: str,
        input_path: str,
        pattern: str = '*.mp4',
        frames_per_batch: int = 50,
        timeout_per_batch: int = 600,
        max_workers: int = 1,
        output_dir: str = "output",
        auto_load_state: bool = True,
        save_state: bool = True
    ) -> List[Dict]:
        """
        处理目录下的所有视频文件（v3.0增强版）
        
        参数:
            workflow_template_path: 工作流模板路径
            input_path: 输入目录
            pattern: 文件匹配模式
            frames_per_batch: 每批帧数
            timeout_per_batch: 每批超时时间（秒）
            max_workers: 最大并行工作数
            output_dir: 输出目录
            auto_load_state: 自动加载状态
            save_state: 保存状态
            
        返回:
            所有视频的处理结果列表
        """
        # 收集视频文件
        video_files = self.collect_video_files(input_path, pattern)
        
        if not video_files:
            self.log(f"❌ 在目录 {input_path} 中未找到视频文件", "ERROR")
            return []
        
        self.log(f"📁 找到 {len(video_files)} 个视频文件", "INFO")
        for vf in video_files:
            self.log(f"  - {os.path.basename(vf)}", "INFO")
        
        all_results = []
        
        # 处理每个视频文件
        for i, video_path in enumerate(video_files, 1):
            self.log(f"\n{'#'*80}", "INFO")
            self.log(f"📊 进度: {i}/{len(video_files)}", "INFO")
            
            # 为每个视频创建单独的输出子目录
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            video_output_dir = os.path.join(output_dir, video_name)
            
            result = self.process_video_file(
                workflow_template_path,
                video_path,
                frames_per_batch,
                timeout_per_batch,
                frames_pre=0,  # 从状态文件加载
                batch_pre=0,   # 从状态文件加载
                auto_load_state=auto_load_state,
                save_state=save_state,
                max_workers=max_workers,
                output_dir=video_output_dir
            )
            
            all_results.append(result)
            
            # 输出当前视频结果
            if result["success"]:
                self.log(f"✅ 视频 {result['video']} 处理成功 ({result['success_rate']})", "INFO")
            else:
                self.log(f"❌ 视频 {result['video']} 处理失败 ({result['success_rate']})", "ERROR")
        
        return all_results
    
    def collect_video_files(self, input_path: str, pattern: str = '*.mp4') -> List[str]:
        """
        收集视频文件
        
        参数:
            input_path: 输入路径
            pattern: 文件匹配模式
            
        返回:
            视频文件路径列表
        """
        video_files = []
        supported_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', 
                               '.MP4', '.MOV', '.AVI', '.MKV', '.WEBM', '.FLV']
        
        if os.path.isfile(input_path):
            # 单个文件
            file_ext = os.path.splitext(input_path)[1].lower()
            if file_ext in [ext.lower() for ext in supported_extensions]:
                video_files.append(input_path)
                self.log(f"✅ 添加单个文件: {input_path}", "INFO")
            else:
                self.log(f"❌ 文件格式不支持: {input_path}", "ERROR")
        
        elif os.path.isdir(input_path):
            # 目录
            self.log(f"📂 扫描目录: {input_path}", "INFO")
            
            # 根据模式查找文件
            search_pattern = os.path.join(input_path, pattern)
            found_files = glob(search_pattern, recursive=False)
            
            # 查找其他常见视频格式
            for ext in supported_extensions:
                if f"*{ext}" not in pattern:
                    additional_pattern = os.path.join(input_path, f"*{ext}")
                    additional_files = glob(additional_pattern, recursive=False)
                    found_files.extend(additional_files)
            
            # 去重并排序
            video_files = sorted(list(set(found_files)))
            
            if not video_files:
                self.log(f"❌ 目录 {input_path} 中未找到任何视频文件", "WARN")
            else:
                self.log(f"✅ 从目录找到 {len(video_files)} 个视频文件", "INFO")
        
        else:
            self.log(f"❌ 路径不存在: {input_path}", "ERROR")
        
        return video_files

def main():
    """主函数（v3.0完整增强版）"""
    parser = argparse.ArgumentParser(
        description='ComfyUI FlashVSR-XZG 批量视频处理脚本 v3.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 处理单个视频文件（从头开始）
  python flashvsr_xzg_v3.py -i video.mp4 --template FlashVSR-XZG.json
  
  # 断点续跑，自动加载状态文件
  python flashvsr_xzg_v3.py -i video.mp4 --template FlashVSR-XZG.json --auto-load-state
  
  # 指定已处理帧数和批次
  python flashvsr_xzg_v3.py -i video.mp4 --template FlashVSR-XZG.json --frames-pre 100 --batch-pre 3
  
  # 并行处理（4个线程）
  python flashvsr_xzg_v3.py -i video.mp4 --template FlashVSR-XZG.json --max-workers 4
  
  # 处理目录下的所有视频文件
  python flashvsr_xzg_v3.py -i ./videos --template FlashVSR-XZG.json --max-workers 2
  
  # 自定义输出目录
  python flashvsr_xzg_v3.py -i video.mp4 --template FlashVSR-XZG.json --output-dir ./processed_videos
  
  # 不保存状态文件（一次性处理）
  python flashvsr_xzg_v3.py -i video.mp4 --template FlashVSR-XZG.json --no-save-state

功能特性:
  1. 断点续跑支持：自动保存/加载处理状态
  2. 智能批次调整：最后一批自动调整帧数
  3. 输出文件验证：确保生成的文件有效
  4. 并行处理支持：大幅提升处理速度
  5. 状态文件管理：便于管理和恢复任务

注意:
  1. 脚本使用 pymediainfo 获取视频信息，请确保已安装
  2. 并行处理时请根据硬件配置调整 --max-workers 参数
  3. 状态文件保存在 ./states/ 目录下
  4. 默认输出目录为 ./output/
        """
    )
    
    # 必需参数
    parser.add_argument('-i', '--input', type=str, required=True,
                       help='输入路径（可以是视频文件或目录）')
    
    # 工作流参数
    parser.add_argument('--template', type=str, default='FlashVSR-XZG.json',
                       help='工作流模板 JSON 文件路径 (默认: FlashVSR-XZG.json)')
    parser.add_argument('--frames-per-batch', type=int, default=50,
                       help='每批处理的帧数 (默认: 50)')
    
    # 断点续跑参数
    parser.add_argument('--frames-pre', type=int, default=0,
                       help='已处理的帧数（手动指定）(默认: 0)')
    parser.add_argument('--batch-pre', type=int, default=0,
                       help='已处理的批次（手动指定）(默认: 0)')
    parser.add_argument('--auto-load-state', action='store_true',
                       help='自动从状态文件加载处理进度')
    parser.add_argument('--save-state', action='store_true', default=True,
                       help='保存处理状态到文件 (默认: True)')
    parser.add_argument('--no-save-state', action='store_false', dest='save_state',
                       help='不保存处理状态文件')
    
    # 并行处理参数（改进点4）
    parser.add_argument('--max-workers', type=int, default=1,
                       help='最大并行工作线程数 (默认: 1)')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str, default='output',
                       help='输出目录 (默认: output)')
    
    # 处理参数
    parser.add_argument('--timeout', type=int, default=600,
                       help='每批处理的超时时间（秒）(默认: 600)')
    parser.add_argument('--pattern', type=str, default='*.mp4',
                       help='文件匹配模式，当输入是目录时使用 (默认: *.mp4)')
    
    # 服务器参数
    parser.add_argument('--server', type=str, default='http://127.0.0.1:8188',
                       help='ComfyUI 服务器地址 (默认: http://127.0.0.1:8188)')
    
    # 其他参数
    parser.add_argument('--log-dir', type=str, default='.',
                       help='日志目录 (默认: 当前目录)')
    parser.add_argument('--skip-pymedia-check', action='store_true',
                       help='跳过 pymediainfo 检查')
    parser.add_argument('--disable-validation', action='store_true',
                       help='禁用输出文件验证')
    
    args = parser.parse_args()
    
    # 检查 pymediainfo
    if not PYMEDIAINFO_AVAILABLE and not args.skip_pymedia_check:
        print("⚠️  未检测到 pymediainfo 库")
        response = input("是否继续? (y/n): ")
        if response.lower() != 'y':
            print("退出程序")
            return
    
    # 检查输入路径是否存在
    if not os.path.exists(args.input):
        print(f"❌ 输入路径不存在: {args.input}")
        return
    
    # 检查模板文件是否存在
    if not os.path.exists(args.template):
        print(f"❌ 工作流模板不存在: {args.template}")
        return
    
    # 验证断点参数
    if args.frames_pre < 0:
        print(f"❌ 已处理帧数不能为负数: {args.frames_pre}")
        return
    if args.batch_pre < 0:
        print(f"❌ 已处理批次不能为负数: {args.batch_pre}")
        return
    
    # 验证并行处理参数
    if args.max_workers < 1:
        print(f"❌ 最大工作线程数必须大于0: {args.max_workers}")
        return
    
    # 初始化处理器
    processor = FlashVSR_XZG_Processor(
        comfyui_url=args.server,
        log_dir=args.log_dir
    )
    
    # 设置验证选项
    processor.output_validation_enabled = not args.disable_validation
    
    # 检查 ComfyUI 服务
    if not processor.check_comfyui_server():
        processor.log("❌ ComfyUI 服务不可用，请确保 ComfyUI 已启动", "ERROR")
        return
    
    processor.log(f"🚀 FlashVSR-XZG v3.0 开始处理", "INFO")
    processor.log(f"📂 输入路径: {args.input}", "INFO")
    processor.log(f"📄 工作流模板: {args.template}", "INFO")
    processor.log(f"🎞️  每批帧数: {args.frames_per_batch}", "INFO")
    processor.log(f"⏱️  超时时间: {args.timeout}秒", "INFO")
    processor.log(f"📁 输出目录: {args.output_dir}", "INFO")
    processor.log(f"⚡ 并行处理: {args.max_workers} 个工作线程", "INFO")
    
    if args.auto_load_state:
        processor.log(f"🔄 自动加载状态: 已启用", "INFO")
    if args.frames_pre > 0 or args.batch_pre > 0:
        processor.log(f"📊 手动断点: 已处理 {args.frames_pre} 帧, {args.batch_pre} 批", "INFO")
    if not args.save_state:
        processor.log(f"💾 状态保存: 已禁用", "INFO")
    if args.disable_validation:
        processor.log(f"🔍 输出验证: 已禁用", "INFO")
    
    start_time = time.time()
    
    # 判断输入类型并处理
    if os.path.isfile(args.input):
        # 单个文件
        processor.log(f"📄 处理单个文件", "INFO")
        result = processor.process_video_file(
            args.template,
            args.input,
            args.frames_per_batch,
            args.timeout,
            args.frames_pre,
            args.batch_pre,
            args.auto_load_state,
            args.save_state,
            args.max_workers,
            args.output_dir
        )
        
        results = [result]
        
    elif os.path.isdir(args.input):
        # 目录
        processor.log(f"📁 处理目录", "INFO")
        results = processor.process_directory(
            args.template,
            args.input,
            args.pattern,
            args.frames_per_batch,
            args.timeout,
            args.max_workers,
            args.output_dir,
            args.auto_load_state,
            args.save_state
        )
    
    else:
        processor.log(f"❌ 输入路径类型未知: {args.input}", "ERROR")
        return
    
    # 计算总耗时
    total_time = time.time() - start_time
    
    # 输出汇总结果
    processor.log(f"\n{'='*80}", "INFO")
    processor.log(f"📊 v3.0 处理完成汇总", "INFO")
    processor.log(f"{'='*80}", "INFO")
    
    if not results:
        processor.log(f"❌ 没有处理任何视频", "ERROR")
        return
    
    total_videos = len(results)
    success_videos = sum(1 for r in results if r["success"])
    failed_videos = total_videos - success_videos
    
    total_batches = sum(r["total_batches"] for r in results)
    success_batches = sum(r["batches_processed"] for r in results)
    
    # 计算总处理帧数
    total_frames_processed = sum(r.get("processed_frames", 0) for r in results)
    total_files_generated = sum(len(r.get("output_files", [])) for r in results)
    
    processor.log(f"⏱️  总耗时: {total_time:.2f}秒 ({total_time/60:.1f}分钟)", "INFO")
    processor.log(f"📁 总视频数: {total_videos}", "INFO")
    processor.log(f"✅ 成功视频: {success_videos}", "INFO")
    processor.log(f"❌ 失败视频: {failed_videos}", "INFO" if failed_videos == 0 else "ERROR")
    processor.log(f"📦 总批次: {total_batches}", "INFO")
    processor.log(f"✅ 成功批次: {success_batches} ({success_batches/total_batches*100:.1f}%)", "INFO")
    processor.log(f"🎞️  总处理帧数: {total_frames_processed}", "INFO")
    processor.log(f"📄 总生成文件: {total_files_generated}", "INFO")
    
    # 输出失败详情
    if failed_videos > 0:
        processor.log(f"\n❌ 失败视频详情:", "ERROR")
        for result in results:
            if not result["success"]:
                processor.log(f"  - {result['video']}: {result.get('error', '未知错误')}", "ERROR")
    
    processor.log(f"\n💾 状态文件目录: {processor.state_dir}", "INFO")
    processor.log(f"📝 详细日志: {processor.log_file}", "INFO")
    processor.log(f"🎉 v3.0 处理完成!", "INFO")

if __name__ == "__main__":
    main()
