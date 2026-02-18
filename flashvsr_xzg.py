#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ComfyUI FlashVSR-XZG 批量视频处理脚本
配合 FlashVSR-XZG.json 工作流模板使用
版本: 1.0
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
        初始化 ComfyUI FlashVSR-XZG 处理器
        
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
        
        # 创建日志目录
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 初始化日志
        self._init_log_file()
        
        self.log("📱 初始化 FlashVSR-XZG 处理器")
        self.log(f"🔗 ComfyUI 地址: {self.comfyui_url}")
        self.log(f"📝 日志文件: {self.log_file}")
    
    def _init_log_file(self):
        """初始化日志文件"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"{'='*80}\n")
            f.write(f"FlashVSR-XZG 处理日志\n")
            f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
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
        total_batches: int = 1
    ) -> Dict:
        """
        更新工作流参数
        
        参数:
            workflow: 工作流模板
            video_path: 视频路径
            video_fps: 视频帧率
            frames_per_batch: 每批帧数
            frames_skip: 跳过帧数
            output_prefix: 输出前缀
            batch_number: 批次号
            total_batches: 总批次数
            
        返回:
            更新后的工作流
        """
        # 创建深拷贝
        modified_workflow = json.loads(json.dumps(workflow))
        
        self.log(f"🔄 更新工作流参数 (批次 {batch_number}/{total_batches})", "INFO")
        
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
                
                # 更新跳过帧数
                if isinstance(inputs.get("skip_first_frames"), str) and "{{FRAMES_SKIP}}" in inputs["skip_first_frames"]:
                    inputs["skip_first_frames"] = str(frames_skip)
                    self.log(f"  ✅ 设置跳过帧数: {frames_skip}", "INFO")
            
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
    
    def process_single_video_batch(
        self,
        workflow_template: Dict,
        video_path: str,
        video_fps: float,
        frames_per_batch: int,
        batch_number: int,
        total_batches: int,
        base_output_prefix: str,
        timeout: int = 600
    ) -> Tuple[bool, Optional[str]]:
        """
        处理单个视频批次
        
        参数:
            workflow_template: 工作流模板
            video_path: 视频路径
            video_fps: 视频帧率
            frames_per_batch: 每批帧数
            batch_number: 批次号
            total_batches: 总批次数
            base_output_prefix: 基础输出前缀
            timeout: 超时时间（秒）
            
        返回:
            (success: bool, prompt_id: str or None)
        """
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # 计算跳过帧数
        frames_skip = frames_per_batch * (batch_number - 1)
        
        # 生成输出前缀
        output_prefix = f"{base_output_prefix}_{batch_number:03d}"
        
        self.log(f"🎬 处理批次 {batch_number}/{total_batches}", "INFO")
        self.log(f"  📂 视频: {video_name}", "INFO")
        self.log(f"  ⏱️  帧率: {video_fps:.2f}", "INFO")
        self.log(f"  🎞️  每批帧数: {frames_per_batch}", "INFO")
        self.log(f"  ⏭️  跳过帧数: {frames_skip}", "INFO")
        self.log(f"  📁 输出前缀: {output_prefix}", "INFO")
        
        # 更新工作流参数
        workflow = self.update_workflow_parameters(
            workflow_template,
            video_path,
            video_fps,
            frames_per_batch,
            frames_skip,
            output_prefix,
            batch_number,
            total_batches
        )
        
        # 提交任务
        prompt_id = self.queue_prompt(workflow, timeout=timeout)
        if not prompt_id:
            self.log(f"❌ 提交批次 {batch_number} 失败", "ERROR")
            return False, None
        
        # 等待任务完成
        success = self.wait_for_task_completion(prompt_id, timeout=timeout)
        
        if success:
            self.log(f"✅ 批次 {batch_number} 处理完成", "INFO")
        else:
            self.log(f"❌ 批次 {batch_number} 处理失败", "ERROR")
        
        return success, prompt_id
    
    def process_video_file(
        self,
        workflow_template_path: str,
        video_path: str,
        frames_per_batch: int = 50,
        timeout_per_batch: int = 600
    ) -> Dict:
        """
        处理单个视频文件
        
        参数:
            workflow_template_path: 工作流模板路径
            video_path: 视频文件路径
            frames_per_batch: 每批帧数
            timeout_per_batch: 每批超时时间（秒）
            
        返回:
            处理结果字典
        """
        video_name = os.path.basename(video_path)
        self.log(f"🎬 开始处理视频: {video_name}", "INFO")
        self.log(f"📂 路径: {video_path}", "INFO")
        
        # 加载工作流模板
        try:
            workflow_template = self.load_workflow_template(workflow_template_path)
        except Exception as e:
            return {
                "video": video_name,
                "path": video_path,
                "success": False,
                "batches_processed": 0,
                "total_batches": 0,
                "error": f"加载工作流模板失败: {e}",
                "results": []
            }
        
        # 获取视频信息
        video_fps, total_frames, method = self.get_video_info(video_path)
        self.log(f"📊 视频信息: {total_frames} 帧, {video_fps:.2f} FPS (方法: {method})", "INFO")
        
        # 计算批次数
        total_batches = total_frames // frames_per_batch
        if total_frames % frames_per_batch > 0:
            total_batches += 1
        
        self.log(f"📦 批次计算: {total_frames} 帧 / {frames_per_batch} 帧每批 = {total_batches} 批", "INFO")
        
        # 基础输出前缀
        video_base_name = os.path.splitext(video_name)[0]
        base_output_prefix = f"flashvsr_{video_base_name}"
        
        results = []
        success_count = 0
        
        # 处理每个批次
        for batch_number in range(1, total_batches + 1):
            self.log(f"{'='*60}", "INFO")
            success, prompt_id = self.process_single_video_batch(
                workflow_template,
                video_path,
                video_fps,
                frames_per_batch,
                batch_number,
                total_batches,
                base_output_prefix,
                timeout=timeout_per_batch
            )
            
            results.append({
                "batch": batch_number,
                "success": success,
                "prompt_id": prompt_id
            })
            
            if success:
                success_count += 1
            else:
                # 批次失败，是否继续处理后续批次
                self.log(f"⚠️ 批次 {batch_number} 失败，是否继续处理后续批次？", "WARN")
                # 这里可以添加中断逻辑，默认继续处理
                continue
        
        # 汇总结果
        all_success = success_count == total_batches
        summary = {
            "video": video_name,
            "path": video_path,
            "success": all_success,
            "batches_processed": success_count,
            "total_batches": total_batches,
            "video_fps": video_fps,
            "total_frames": total_frames,
            "frames_per_batch": frames_per_batch,
            "success_rate": f"{success_count}/{total_batches} ({success_count/total_batches*100:.1f}%)",
            "results": results
        }
        
        self.log(f"{'='*60}", "INFO")
        if all_success:
            self.log(f"✅ 视频 {video_name} 全部批次处理完成 ({success_count}/{total_batches})", "INFO")
        else:
            self.log(f"⚠️ 视频 {video_name} 部分批次失败 ({success_count}/{total_batches})", "WARN")
        
        return summary
    
    def process_directory(
        self,
        workflow_template_path: str,
        input_path: str,
        pattern: str = '*.mp4',
        frames_per_batch: int = 50,
        timeout_per_batch: int = 600,
        skip_existing: bool = False
    ) -> List[Dict]:
        """
        处理目录下的所有视频文件
        
        参数:
            workflow_template_path: 工作流模板路径
            input_path: 输入目录
            pattern: 文件匹配模式
            frames_per_batch: 每批帧数
            timeout_per_batch: 每批超时时间（秒）
            skip_existing: 是否跳过已处理文件
            
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
            
            result = self.process_video_file(
                workflow_template_path,
                video_path,
                frames_per_batch,
                timeout_per_batch
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
    """主函数"""
    parser = argparse.ArgumentParser(
        description='ComfyUI FlashVSR-XZG 批量视频处理脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 处理单个视频文件
  python flashvsr_xzg.py -i video.mp4 --template FlashVSR-XZG.json
  
  # 处理目录下的所有视频文件
  python flashvsr_xzg.py -i ./videos --template FlashVSR-XZG.json
  
  # 自定义每批帧数
  python flashvsr_xzg.py -i video.mp4 --template FlashVSR-XZG.json --frames-per-batch 100
  
  # 自定义超时时间
  python flashvsr_xzg.py -i video.mp4 --template FlashVSR-XZG.json --timeout 300
  
  # 使用特定 ComfyUI 地址
  python flashvsr_xzg.py -i video.mp4 --template FlashVSR-XZG.json --server http://localhost:8189

注意:
  1. 脚本使用 pymediainfo 获取视频信息，请确保已安装
  2. 工作流模板中的占位符会被自动替换
  3. 日志文件保存在当前目录
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
    
    # 初始化处理器
    processor = FlashVSR_XZG_Processor(
        comfyui_url=args.server,
        log_dir=args.log_dir
    )
    
    # 检查 ComfyUI 服务
    if not processor.check_comfyui_server():
        processor.log("❌ ComfyUI 服务不可用，请确保 ComfyUI 已启动", "ERROR")
        return
    
    processor.log(f"🚀 开始处理", "INFO")
    processor.log(f"📂 输入路径: {args.input}", "INFO")
    processor.log(f"📄 工作流模板: {args.template}", "INFO")
    processor.log(f"🎞️  每批帧数: {args.frames_per_batch}", "INFO")
    processor.log(f"⏱️  超时时间: {args.timeout}秒", "INFO")
    
    start_time = time.time()
    
    # 判断输入类型并处理
    if os.path.isfile(args.input):
        # 单个文件
        processor.log(f"📄 处理单个文件", "INFO")
        result = processor.process_video_file(
            args.template,
            args.input,
            args.frames_per_batch,
            args.timeout
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
            args.timeout
        )
    
    else:
        processor.log(f"❌ 输入路径类型未知: {args.input}", "ERROR")
        return
    
    # 计算总耗时
    total_time = time.time() - start_time
    
    # 输出汇总结果
    processor.log(f"\n{'='*80}", "INFO")
    processor.log(f"📊 处理完成汇总", "INFO")
    processor.log(f"{'='*80}", "INFO")
    
    if not results:
        processor.log(f"❌ 没有处理任何视频", "ERROR")
        return
    
    total_videos = len(results)
    success_videos = sum(1 for r in results if r["success"])
    failed_videos = total_videos - success_videos
    
    total_batches = sum(r["total_batches"] for r in results)
    success_batches = sum(r["batches_processed"] for r in results)
    
    processor.log(f"⏱️  总耗时: {total_time:.2f}秒 ({total_time/60:.1f}分钟)", "INFO")
    processor.log(f"📁 总视频数: {total_videos}", "INFO")
    processor.log(f"✅ 成功视频: {success_videos}", "INFO")
    processor.log(f"❌ 失败视频: {failed_videos}", "INFO" if failed_videos == 0 else "ERROR")
    processor.log(f"📦 总批次: {total_batches}", "INFO")
    processor.log(f"✅ 成功批次: {success_batches} ({success_batches/total_batches*100:.1f}%)", "INFO")
    
    # 输出失败详情
    if failed_videos > 0:
        processor.log(f"\n❌ 失败视频详情:", "ERROR")
        for result in results:
            if not result["success"]:
                processor.log(f"  - {result['video']}: {result.get('error', '未知错误')}", "ERROR")
    
    processor.log(f"\n📝 详细日志: {processor.log_file}", "INFO")
    processor.log(f"🎉 处理完成!", "INFO")

if __name__ == "__main__":
    main()
