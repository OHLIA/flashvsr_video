#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ComfyUI FlashVSR-XZG 高级批量视频处理脚本（混合版）
专为 api_flashvsr_mix.json 工作流模板设计
版本: 1.2
修复tiled_dit变量名错误问题
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
import signal
import psutil

# 尝试导入 pymediainfo
try:
    from pymediainfo import MediaInfo
    PYMEDIAINFO_AVAILABLE = True
except ImportError:
    PYMEDIAINFO_AVAILABLE = False
    print("⚠️  pymediainfo 未安装，将使用备用方法获取视频信息")

class FlashVSR_XZG_MIX_Processor:
    def __init__(self, comfyui_url: str = "http://127.0.0.1:8188", log_dir: str = "."):
        """
        初始化 ComfyUI FlashVSR-XZG MIX 处理器
        
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
        self.log_file = os.path.join(self.log_dir, f"flashvsr_mix_{timestamp}.log")
        self.state_dir = os.path.join(self.log_dir, "states_mix")
        
        # 创建日志和状态目录
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.state_dir, exist_ok=True)
        
        # 初始化日志
        self._init_log_file()
        
        self.log("📱 初始化 FlashVSR-XZG MIX 处理器 v1.2", "INFO")
        self.log(f"🔗 ComfyUI 地址: {self.comfyui_url}", "INFO")
        self.log(f"📝 日志文件: {self.log_file}", "INFO")
        self.log(f"💾 状态目录: {self.state_dir}", "INFO")
        
        # 状态跟踪
        self.processing_state = {}
    
    def _init_log_file(self):
        """初始化日志文件"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"{'='*80}\n")
            f.write(f"FlashVSR-XZG MIX 处理日志 v1.2\n")
            f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"工作流模板: api_flashvsr_mix.json\n")
            f.write(f"{'='*80}\n\n")
    
    def log(self, message: str, level: str = "INFO"):
        """
        记录日志（改进格式，与flashvsr_xzg.py保持一致）
        
        参数:
            message: 日志消息
            level: 日志级别
        """
        # 使用统一的时间戳格式 [YYYY-MM-DD HH:MM:SS]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        # 输出到控制台
        print(log_entry)
        
        # 写入日志文件
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry + "\n")
        except Exception as e:
            print(f"写入日志失败: {e}")
    
    def save_processing_state(self, video_path: str, frames_processed: int, batches_processed: int, 
                            success: bool = True, error_msg: str = ""):
        """
        保存处理状态到文件
        
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
            self.log(f"保存状态文件失败: {e}", "ERROR")
            return False
    
    def load_processing_state(self, video_path: str) -> Tuple[int, int, Dict]:
        """
        从文件加载处理状态
        
        参数:
            video_path: 视频路径
            
        返回:
            (frames_processed: int, batches_processed: int, state: Dict)
        """
        try:
            video_name = os.path.basename(video_path)
            safe_video_name = re.sub(r'[^\w\-\.]', '_', video_name)
            
            # 查找状态文件
            state_files = [
                os.path.join(self.state_dir, f"{safe_video_name}_state.json"),
                os.path.join(self.state_dir, f"{video_name}_state.json"),
                os.path.join(self.log_dir, f"flashvsr_mix_state_{safe_video_name}.json"),
                os.path.join(self.log_dir, f"flashvsr_mix_state_{video_name}.json"),
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
            self.log(f"加载状态文件失败: {e}", "ERROR")
            return 0, 0, {}
    
    def check_comfyui_server(self, timeout: int = 10) -> bool:
        """检查 ComfyUI 服务是否可用"""
        try:
            response = requests.get(f"{self.comfyui_url}/", timeout=timeout)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            self.log(f"检查 ComfyUI 服务失败: {e}", "WARN")
            return False
    
    def get_video_info(self, video_path: str) -> Tuple[float, int, int, int, str]:
        """
        获取视频信息（增强版，包含宽度和高度）
        
        参数:
            video_path: 视频文件路径
            
        返回:
            (fps: float, total_frames: int, width: int, height: int, method: str)
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
                        
                        # 获取宽度和高度
                        width = 0
                        height = 0
                        if hasattr(track, 'width') and track.width:
                            width = int(track.width)
                        if hasattr(track, 'height') and track.height:
                            height = int(track.height)
                        
                        if total_frames > 0 and width > 0 and height > 0:
                            self.log(f"视频信息: FPS={fps:.2f}, 总帧数={total_frames}, 分辨率={width}x{height}", "INFO")
                            return fps, total_frames, width, height, "pymediainfo"
            
            # 备用方法：使用 OpenCV
            try:
                import cv2
                self.log(f"使用 OpenCV 获取视频信息: {video_path}", "INFO")
                cap = cv2.VideoCapture(video_path)
                
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                    
                    if fps > 0 and total_frames > 0 and width > 0 and height > 0:
                        self.log(f"视频信息: FPS={fps:.2f}, 总帧数={total_frames}, 分辨率={width}x{height}", "INFO")
                        return fps, total_frames, width, height, "OpenCV"
            except ImportError:
                self.log("OpenCV 未安装", "WARN")
            except Exception as e:
                self.log(f"OpenCV 获取信息失败: {e}", "WARN")
            
            # 默认值
            self.log(f"无法获取视频信息，使用默认值: FPS=25.0, 总帧数=100, 分辨率=1280x720", "WARN")
            return 25.0, 100, 1280, 720, "默认值"
            
        except Exception as e:
            self.log(f"获取视频信息失败: {e}", "ERROR")
            return 25.0, 100, 1280, 720, "错误-默认值"
    
    def calculate_aligned_dimension(self, dimension: int, alignment: int = 128) -> int:
        """
        计算对齐到指定倍数的尺寸
        例如：720/128=5.625 → 向上取整为6 → 6 * 128=768
        
        参数:
            dimension: 原始尺寸
            alignment: 对齐倍数（默认128）
            
        返回:
            对齐后的尺寸
        """
        if dimension <= 0:
            return alignment
        
        # 计算需要多少个对齐单位
        units = math.ceil(dimension / alignment)
        
        # 返回对齐后的尺寸
        aligned_dimension = units * alignment
        self.log(f"  尺寸对齐: {dimension} -> {aligned_dimension} (单位: {alignment})", "INFO")
        
        return aligned_dimension
    
    def load_workflow_template(self, template_path: str) -> Dict:
        """加载工作流 JSON 模板"""
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                workflow = json.load(f)
            
            self.log(f"已加载工作流模板: {template_path}", "INFO")
            return workflow
            
        except FileNotFoundError:
            self.log(f"找不到工作流模板文件: {template_path}", "ERROR")
            raise
        except json.JSONDecodeError as e:
            self.log(f"JSON 解析失败: {e}", "ERROR")
            raise
        except Exception as e:
            self.log(f"加载工作流模板失败: {e}", "ERROR")
            raise
    
    def update_workflow_parameters(
        self, 
        workflow: Dict, 
        video_path: str,
        video_fps: float,
        frames_per_batch: int,
        frames_skip: int,
        output_prefix: str,
        attn_mode: str = "block_sparse_attention",
        tiled_dit: bool = False,  # 修复：改为 bool 类型
        tile_size: int = 256,
        tile_overlap: int = 24,
        scale: int = 2,
        in_width: int = 768,
        out_width: int = 3072,
        batch_number: int = 1,
        total_batches: int = 1,
        frames_pre: int = 0,
        batch_pre: int = 0,
        gpu_device: str = "auto"
    ) -> Dict:
        """
        更新工作流参数（针对 api_flashvsr_mix.json）
        
        参数:
            workflow: 工作流模板
            video_path: 视频路径
            video_fps: 视频帧率
            frames_per_batch: 每批帧数
            frames_skip: 跳过帧数
            output_prefix: 输出前缀
            attn_mode: 稀疏模式 ("block_sparse_attention" 或 "sparse_sage_attention")
            tiled_dit: 分块开关 (布尔值)
            tile_size: 分块大小
            tile_overlap: 分块重叠
            scale: 放大倍数
            in_width: 输入宽度（128对齐）
            out_width: 输出宽度（128对齐）
            batch_number: 当前任务批次号
            total_batches: 总批次数
            frames_pre: 已跑帧数
            batch_pre: 已跑批次
            gpu_device: GPU设备选择
            
        返回:
            更新后的工作流
        """
        # 创建深拷贝
        modified_workflow = json.loads(json.dumps(workflow))
        
        self.log(f"更新工作流参数 (批次 {batch_number}/{total_batches})", "INFO")
        if frames_pre > 0:
            self.log(f"  ⏭️  已跑帧数: {frames_pre} 帧", "INFO")
        if batch_pre > 0:
            self.log(f"  📦 已跑批次: {batch_pre} 批", "INFO")
        
        # 更新所有节点参数
        for node_id, node_data in modified_workflow.items():
            node_class = node_data.get("class_type", "")
            inputs = node_data.get("inputs", {})
            
            # 1. VHS_LoadVideo 节点 (ID 25)
            if node_class == "VHS_LoadVideo":
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
            
            # 2. FlashVSRInitPipe 节点 (ID 29)
            elif node_class == "FlashVSRInitPipe":
                # 设置GPU设备
                if isinstance(inputs.get("device"), str) and "{{gpu}}" in inputs["device"]:
                    if gpu_device == "auto":
                        device_value = "auto"
                    elif gpu_device.isdigit():
                        device_value = f"cuda:{gpu_device}"
                    else:
                        device_value = gpu_device
                    inputs["device"] = device_value
                    self.log(f"  ✅ 设置GPU设备: {device_value}", "INFO")
                elif isinstance(inputs.get("device"), str):
                    if gpu_device.isdigit():
                        device_value = f"cuda:{gpu_device}"
                    else:
                        device_value = gpu_device
                    inputs["device"] = device_value
                    self.log(f"  ✅ 设置GPU设备: {device_value} (直接赋值)", "INFO")
                
                # 设置稀疏模式
                if isinstance(inputs.get("attention_mode"), str) and "{{attn_mode}}" in inputs["attention_mode"]:
                    inputs["attention_mode"] = attn_mode
                    self.log(f"  ✅ 设置稀疏模式: {attn_mode}", "INFO")
                elif isinstance(inputs.get("attention_mode"), str):
                    inputs["attention_mode"] = attn_mode
                    self.log(f"  ✅ 设置稀疏模式: {attn_mode} (直接赋值)", "INFO")
            
            # 3. FlashVSRNodeAdv 节点 (ID 28)
            elif node_class == "FlashVSRNodeAdv":
                # 设置缩放比例
                if isinstance(inputs.get("scale"), str) and "{{scale}}" in inputs["scale"]:
                    inputs["scale"] = str(scale)
                    self.log(f"  ✅ 设置缩放比例: {scale}", "INFO")
                elif isinstance(inputs.get("scale"), (int, float, str)):
                    try:
                        inputs["scale"] = int(scale)  # 改为 int 类型
                        self.log(f"  ✅ 设置缩放比例: {scale} (直接赋值，转为整型)", "INFO")
                    except:
                        pass
                
                # 设置分块开关 - 修复：正确处理布尔值
                if "tiled_dit" in inputs:
                    # 直接赋布尔值
                    inputs["tiled_dit"] = tiled_dit  
                    self.log(f"  ✅ 设置分块开关: {tiled_dit} (原始值: {tiled_dit})", "INFO")
                
                # 设置分块大小
                if isinstance(inputs.get("tile_size"), str) and "{{t_z}}" in inputs["tile_size"]:
                    inputs["tile_size"] = str(tile_size)
                    self.log(f"  ✅ 设置分块大小: {tile_size}", "INFO")
                elif isinstance(inputs.get("tile_size"), (int, float, str)):
                    try:
                        inputs["tile_size"] = int(tile_size)
                        self.log(f"  ✅ 设置分块大小: {tile_size} (直接赋值)", "INFO")
                    except:
                        pass
                
                # 设置分块重叠
                if isinstance(inputs.get("tile_overlap"), str) and "{{t_o}}" in inputs["tile_overlap"]:
                    inputs["tile_overlap"] = str(tile_overlap)
                    self.log(f"  ✅ 设置分块重叠: {tile_overlap}", "INFO")
                elif isinstance(inputs.get("tile_overlap"), (int, float, str)):
                    try:
                        inputs["tile_overlap"] = int(tile_overlap)
                        self.log(f"  ✅ 设置分块重叠: {tile_overlap} (直接赋值)", "INFO")
                    except:
                        pass
            
            # 4. 图像缩放节点 - 输入 (ID 26)
            elif node_class == "LayerUtility: ImageScaleByAspectRatio V2" and node_id == "26":
                # 设置输入宽度
                if isinstance(inputs.get("scale_to_length"), str) and "{{IN_WIDTH}}" in inputs["scale_to_length"]:
                    inputs["scale_to_length"] = str(in_width)
                    self.log(f"  ✅ 设置输入宽度: {in_width}", "INFO")
                elif isinstance(inputs.get("scale_to_length"), (int, float, str)):
                    inputs["scale_to_length"] = str(in_width)
                    self.log(f"  ✅ 设置输入宽度: {in_width} (直接赋值)", "INFO")
            
            # 5. 图像缩放节点 - 输出 (ID 19)
            elif node_class == "LayerUtility: ImageScaleByAspectRatio V2" and node_id == "19":
                # 设置输出宽度
                if isinstance(inputs.get("scale_to_length"), str) and "{{OUT_WIDTH}}" in inputs["scale_to_length"]:
                    inputs["scale_to_length"] = str(out_width)
                    self.log(f"  ✅ 设置输出宽度: {out_width}", "INFO")
                elif isinstance(inputs.get("scale_to_length"), (int, float, str)):
                    inputs["scale_to_length"] = str(out_width)
                    self.log(f"  ✅ 设置输出宽度: {out_width} (直接赋值)", "INFO")
            
            # 6. VHS_VideoCombine 节点 (ID 34)
            elif node_class == "VHS_VideoCombine":
                # 更新输出前缀
                if isinstance(inputs.get("filename_prefix"), str) and "{{OUTPUT_PREFIX}}" in inputs["filename_prefix"]:
                    inputs["filename_prefix"] = output_prefix
                    self.log(f"  ✅ 设置输出前缀: {output_prefix}", "INFO")
                elif isinstance(inputs.get("filename_prefix"), str):
                    inputs["filename_prefix"] = output_prefix
                    self.log(f"  ✅ 设置输出前缀: {output_prefix} (直接赋值)", "INFO")
                
                # 设置 trim_to_audio 为 false
                if "trim_to_audio" in inputs:
                    inputs["trim_to_audio"] = False
                    self.log("  ✅ 设置 trim_to_audio 为 false", "INFO")
        
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
            self.log("ComfyUI 服务不可用，无法提交任务", "ERROR")
            return None
        
        try:
            self.log(f"提交任务到 ComfyUI", "INFO")
            
            # 验证关键参数
            self.log(f"=== 工作流关键参数验证 ===", "INFO")
            key_nodes = ["25", "28", "29", "34", "26", "19"]
            for node_id in key_nodes:
                if node_id in workflow:
                    node_data = workflow[node_id]
                    node_type = node_data.get("class_type", "Unknown")
                    inputs = node_data.get("inputs", {})
                    self.log(f"节点 {node_id} ({node_type}):", "INFO")
                    
                    for key in ["video", "force_rate", "frame_load_cap", "skip_first_frames", 
                               "device", "attention_mode", "scale", "tiled_dit", "tile_size", 
                               "tile_overlap", "filename_prefix", "scale_to_length"]:
                        if key in inputs:
                            self.log(f"  {key}: {inputs[key]}", "INFO")
            
            response = requests.post(
                self.api_prompt, 
                json={"prompt": workflow}, 
                timeout=timeout,
                headers={'Content-Type': 'application/json'}
            )
            
            self.log(f"响应状态码: {response.status_code}", "INFO")
            
            if response.status_code == 200:
                data = response.json()
                prompt_id = data.get('prompt_id')
                
                if prompt_id:
                    self.log(f"任务已提交，ID: {prompt_id}", "INFO")
                    return prompt_id
                else:
                    self.log(f"未收到任务ID，响应: {data}", "ERROR")
                    return None
            else:
                self.log(f"请求失败，状态码: {response.status_code}", "ERROR")
                self.log(f"错误详情: {response.text[:500]}", "ERROR")
                
                if response.status_code == 400:
                    self.log("分析400错误可能的原因:", "INFO")
                    self.log("  1. 工作流中存在未替换的占位符", "INFO")
                    self.log("  2. 工作流格式不符合ComfyUI要求", "INFO")
                    self.log("  3. 某些节点参数类型不正确", "INFO")
                
                return None
                
        except requests.exceptions.RequestException as e:
            self.log(f"请求失败: {e}", "ERROR")
            return None
        except json.JSONDecodeError as e:
            self.log(f"JSON 解析失败: {e}", "ERROR")
            return None
        except Exception as e:
            self.log(f"提交任务失败: {e}", "ERROR")
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
        self.log(f"等待任务 {prompt_id} 完成，超时: {timeout}秒", "INFO")
        
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
                            self.log(f"任务 {prompt_id} 成功完成", "INFO")
                            return True
                        
                        # 错误
                        if status_info.get("status_str") == "error":
                            self.log(f"任务 {prompt_id} 执行错误", "ERROR")
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
                                self.log(f"任务仍在运行，已等待 {elapsed:.1f}秒", "INFO")
                            time.sleep(5)
                            continue
                
                time.sleep(2)
                
            except requests.exceptions.RequestException as e:
                self.log(f"检查任务状态失败: {e}，继续等待...", "WARN")
                time.sleep(5)
                continue
        
        self.log(f"任务 {prompt_id} 等待超时 ({timeout}秒)", "ERROR")
        return False
    
    def process_single_video_batch(
        self,
        workflow_template: Dict,
        video_path: str,
        video_fps: float,
        video_width: int,
        video_height: int,
        frames_per_batch: int,
        batch_number: int,
        total_batches: int,
        base_output_prefix: str,
        attn_mode: str = "block_sparse_attention",
        tiled_dit: bool = False,  # 修复：改为 bool 类型
        tile_size: int = 256,
        tile_overlap: int = 24,
        scale: int = 2,
        frames_pre: int = 0,
        batch_pre: int = 0,
        gpu_device: str = "auto",
        timeout: int = 600,
        output_dir: str = "output"
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        处理单个视频批次
        
        参数:
            workflow_template: 工作流模板
            video_path: 视频路径
            video_fps: 视频帧率
            video_width: 视频宽度
            video_height: 视频高度
            frames_per_batch: 每批帧数
            batch_number: 当前任务批次号
            total_batches: 总批次数
            base_output_prefix: 基础输出前缀
            attn_mode: 稀疏模式
            tiled_dit: 分块开关 (布尔值)
            tile_size: 分块大小
            tile_overlap: 分块重叠
            scale: 放大倍数
            frames_pre: 已跑帧数
            batch_pre: 已跑批次
            gpu_device: GPU设备选择
            timeout: 超时时间（秒）
            output_dir: 输出目录
            
        返回:
            (success: bool, prompt_id: str or None, output_file: str or None)
        """
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # 智能批次大小调整
        actual_frames_per_batch = frames_per_batch
        if batch_number == total_batches and frames_pre > 0:
            # 计算剩余帧数
            video_fps, total_frames, _, _, _ = self.get_video_info(video_path)
            remaining_frames = total_frames - frames_pre
            
            # 计算最后一批的实际帧数
            last_batch_frames = remaining_frames - (frames_per_batch * (batch_number - 1))
            if 0 < last_batch_frames < frames_per_batch:
                actual_frames_per_batch = last_batch_frames
                self.log(f"最后一批智能调整帧数: {actual_frames_per_batch} 帧 (原: {frames_per_batch})", "INFO")
        
        # 计算跳过帧数
        frames_skip = frames_pre + frames_per_batch * (batch_number - 1)
        
        # 计算当前总批次号
        current_batch_number = batch_pre + batch_number
        
        # 生成输出前缀
        output_prefix = f"{base_output_prefix}_{current_batch_number:03d}"
        
        # 计算输入和输出宽度（对齐到128）
        in_width_aligned = self.calculate_aligned_dimension(video_width, 128)
        out_width_aligned = self.calculate_aligned_dimension(int(video_width * scale), 128)
        
        # 预期输出文件路径
        expected_output_file = os.path.join(output_dir, f"{output_prefix}.mp4")
        
        self.log(f"处理批次 {batch_number}/{total_batches} (总批次: {current_batch_number})", "INFO")
        self.log(f"  📂 视频: {video_name}", "INFO")
        self.log(f"  📏 分辨率: {video_width}x{video_height}", "INFO")
        self.log(f"  ⏱️  帧率: {video_fps:.2f}", "INFO")
        self.log(f"  🎞️  每批帧数: {actual_frames_per_batch} (原: {frames_per_batch})", "INFO")
        self.log(f"  ⏭️  跳过帧数: {frames_skip} (已跑 {frames_pre} + 当前跳过 {frames_per_batch*(batch_number-1)})", "INFO")
        self.log(f"  📁 输出前缀: {output_prefix}", "INFO")
        self.log(f"  ⚙️  稀疏模式: {attn_mode}", "INFO")
        self.log(f"  🧱 分块开关: {'启用' if tiled_dit else '禁用'}", "INFO")
        self.log(f"  🧩 分块大小: {tile_size}", "INFO")
        self.log(f"  🔗 分块重叠: {tile_overlap}", "INFO")
        self.log(f"  🔍 缩放倍数: {scale}", "INFO")
        self.log(f"  📏 输入宽度: {video_width} -> {in_width_aligned} (128对齐)", "INFO")
        self.log(f"  📏 输出宽度: {int(video_width * scale)} -> {out_width_aligned} (128对齐)", "INFO")
        self.log(f"  📄 预期输出: {expected_output_file}", "INFO")
        if frames_pre > 0:
            self.log(f"  📊 断点续跑: 已处理 {frames_pre} 帧 ({batch_pre} 批)", "INFO")
        
        # 检查输出文件是否已存在
        if os.path.exists(expected_output_file):
            file_size_mb = os.path.getsize(expected_output_file) / (1024 * 1024)
            self.log(f"输出文件已存在: {expected_output_file} ({file_size_mb:.1f}MB)", "WARN")
            response = input("是否覆盖？(y/n/skip): ").lower()
            if response == 'n':
                self.log(f"跳过已存在批次 {batch_number}", "INFO")
                return True, None, expected_output_file
            elif response == 'skip':
                return False, None, None
        
        # 更新工作流参数
        workflow = self.update_workflow_parameters(
            workflow_template,
            video_path,
            video_fps,
            actual_frames_per_batch,  # 使用调整后的帧数
            frames_skip,
            output_prefix,
            attn_mode=attn_mode,
            tiled_dit=tiled_dit,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            scale=scale,
            in_width=in_width_aligned,
            out_width=out_width_aligned,
            batch_number=batch_number,
            total_batches=total_batches,
            frames_pre=frames_pre,
            batch_pre=batch_pre,
            gpu_device=gpu_device
        )
        
        # 提交任务
        prompt_id = self.queue_prompt(workflow, timeout=timeout)
        if not prompt_id:
            self.log(f"提交批次 {batch_number} 失败", "ERROR")
            return False, None, None
        
        # 等待任务完成
        success = self.wait_for_task_completion(prompt_id, timeout=timeout)
        
        if success:
            self.log(f"批次 {batch_number} 处理完成 (总批次: {current_batch_number})", "INFO")
            
            # 检查输出文件
            if os.path.exists(expected_output_file):
                file_size_mb = os.path.getsize(expected_output_file) / (1024 * 1024)
                self.log(f"输出文件生成成功: {expected_output_file} ({file_size_mb:.1f}MB)", "INFO")
                return success, prompt_id, expected_output_file
            else:
                # 尝试查找实际输出文件
                import glob
                output_files = glob.glob(os.path.join(output_dir, f"{output_prefix}*.mp4"))
                if output_files:
                    actual_output = output_files[0]
                    file_size_mb = os.path.getsize(actual_output) / (1024 * 1024)
                    self.log(f"找到实际输出文件: {actual_output} ({file_size_mb:.1f}MB)", "INFO")
                    return success, prompt_id, actual_output
                else:
                    self.log(f"预期输出文件不存在，但任务显示成功", "WARN")
                    return success, prompt_id, None
        else:
            self.log(f"批次 {batch_number} 处理失败 (总批次: {current_batch_number})", "ERROR")
            return False, prompt_id, None
    
    def process_video_file(
        self,
        workflow_template_path: str,
        video_path: str,
        frames_per_batch: int = 50,
        attn_mode: str = "block_sparse_attention",
        tiled_dit: bool = False,  # 修复：改为 bool 类型
        tile_size: int = 256,
        tile_overlap: int = 24,
        scale: int = 2,
        gpu_device: str = "auto",
        timeout_per_batch: int = 600,
        frames_pre: int = 0,
        batch_pre: int = 0,
        auto_load_state: bool = True,  # 修复：改为 bool 类型
        save_state: bool = True,  # 修复：改为 bool 类型
        max_workers: int = 1,
        output_dir: str = "output"
    ) -> Dict:
        """
        处理单个视频文件
        
        参数:
            workflow_template_path: 工作流模板路径
            video_path: 视频文件路径
            frames_per_batch: 每批帧数
            attn_mode: 稀疏模式
            tiled_dit: 分块开关 (布尔值)
            tile_size: 分块大小
            tile_overlap: 分块重叠
            scale: 放大倍数
            gpu_device: GPU设备选择
            timeout_per_batch: 每批超时时间（秒）
            frames_pre: 已跑帧数
            batch_pre: 已跑批次
            auto_load_state: 自动加载状态 (布尔值)
            save_state: 保存状态 (布尔值)
            max_workers: 最大并行工作数
            output_dir: 输出目录
            
        返回:
            处理结果字典
        """
        video_name = os.path.basename(video_path)
        self.log(f"开始处理视频: {video_name}", "INFO")
        self.log(f"路径: {video_path}", "INFO")
        
        # 自动加载状态
        if auto_load_state:
            loaded_frames_pre, loaded_batch_pre, state_info = self.load_processing_state(video_path)
            if loaded_frames_pre > 0 or loaded_batch_pre > 0:
                frames_pre = loaded_frames_pre
                batch_pre = loaded_batch_pre
                self.log(f"自动加载断点状态: 已处理 {frames_pre} 帧, {batch_pre} 批", "INFO")
        
        # 检查断点续跑参数
        if frames_pre > 0:
            self.log(f"断点续跑模式: 已处理 {frames_pre} 帧, {batch_pre} 批", "INFO")
        
        # 加载工作流模板
        try:
            workflow_template = self.load_workflow_template(workflow_template_path)
        except Exception as e:
            error_msg = f"加载工作流模板失败: {e}"
            self.log(error_msg, "ERROR")
            if save_state:
                self.save_processing_state(video_path, frames_pre, batch_pre, False, error_msg)
            return {
                "video": video_name,
                "path": video_path,
                "success": False,
                "error": error_msg,
                "results": []
            }
        
        # 获取视频信息（包含分辨率）
        video_fps, total_frames, video_width, video_height, method = self.get_video_info(video_path)
        self.log(f"视频信息: {total_frames} 帧, {video_fps:.2f} FPS, 分辨率: {video_width}x{video_height} (方法: {method})", "INFO")
        
        # 计算对齐后的输入输出宽度
        in_width_aligned = self.calculate_aligned_dimension(video_width, 128)
        out_width_aligned = self.calculate_aligned_dimension(int(video_width * scale), 128)
        self.log(f"尺寸对齐: 输入 {video_width} -> {in_width_aligned}, 输出 {int(video_width * scale)} -> {out_width_aligned}", "INFO")
        
        # 计算剩余可处理帧数
        remaining_frames = total_frames - frames_pre
        if remaining_frames <= 0:
            self.log(f"视频已全部处理完成，无需继续处理", "INFO")
            result = {
                "video": video_name,
                "path": video_path,
                "success": True,
                "batches_processed": 0,
                "total_batches": 0,
                "video_fps": video_fps,
                "video_width": video_width,
                "video_height": video_height,
                "in_width_aligned": in_width_aligned,
                "out_width_aligned": out_width_aligned,
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
        
        # 计算批次数（与flashvsr_xzg.py保持一致的逻辑）
        # (总帧数 - {{FRAMS_PRE}}) / frames_per_batch
        total_batches = remaining_frames // frames_per_batch
        if remaining_frames % frames_per_batch > 0:
            total_batches += 1
        
        self.log(f"批次计算: {remaining_frames} 剩余帧 / {frames_per_batch} 帧每批 = {total_batches} 批", "INFO")
        self.log(f"进度: {frames_pre}/{total_frames} 帧 ({frames_pre/total_frames*100:.1f}%)", "INFO")
        self.log(f"并行处理: {max_workers} 个工作线程", "INFO")
        
        # 基础输出前缀
        video_base_name = os.path.splitext(video_name)[0]
        base_output_prefix = f"flashvsr_{video_base_name}"
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        success_count = 0
        output_files = []
        
        # 并行处理逻辑
        if max_workers > 1 and total_batches > 1:
            self.log(f"启动并行处理，最大工作线程数: {max_workers}", "INFO")
            
            # 使用线程池并行处理
            from concurrent.futures import ThreadPoolExecutor
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for batch_number in range(1, total_batches + 1):
                    future = executor.submit(
                        self.process_single_video_batch,
                        workflow_template,
                        video_path,
                        video_fps,
                        video_width,
                        video_height,
                        frames_per_batch,
                        batch_number,
                        total_batches,
                        base_output_prefix,
                        attn_mode,
                        tiled_dit,
                        tile_size,
                        tile_overlap,
                        scale,
                        frames_pre,
                        batch_pre,
                        gpu_device,
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
                            if save_state and batch_number % 3 == 0:
                                self.save_processing_state(video_path, min(current_frames, total_frames), 
                                                         batch_pre + batch_number, True)
                        else:
                            self.log(f"批次 {batch_number} 失败", "WARN")
                            
                    except Exception as e:
                        self.log(f"批次 {batch_number} 执行异常: {e}", "ERROR")
                        results.append({
                            "batch": batch_number,
                            "total_batch": batch_pre + batch_number,
                            "success": False,
                            "error": str(e),
                            "frames_skip": frames_pre + frames_per_batch * (batch_number - 1)
                        })
        else:
            # 顺序处理
            for batch_number in range(1, total_batches + 1):
                self.log(f"{'='*60}", "INFO")
                success, prompt_id, output_file = self.process_single_video_batch(
                    workflow_template,
                    video_path,
                    video_fps,
                    video_width,
                    video_height,
                    frames_per_batch,
                    batch_number,
                    total_batches,
                    base_output_prefix,
                    attn_mode,
                    tiled_dit,
                    tile_size,
                    tile_overlap,
                    scale,
                    frames_pre,
                    batch_pre,
                    gpu_device,
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
                    if save_state and batch_number % 3 == 0:
                        self.save_processing_state(video_path, min(current_frames, total_frames), 
                                                 batch_pre + batch_number, True)
                else:
                    self.log(f"批次 {batch_number} 失败，是否继续处理后续批次？", "WARN")
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
            "video_width": video_width,
            "video_height": video_height,
            "in_width_aligned": in_width_aligned,
            "out_width_aligned": out_width_aligned,
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
            "results": results,
            "parameters": {
                "attn_mode": attn_mode,
                "tiled_dit": tiled_dit,
                "tile_size": tile_size,
                "tile_overlap": tile_overlap,
                "scale": scale,
                "gpu_device": gpu_device
            }
        }
        
        self.log(f"{'='*60}", "INFO")
        if all_success:
            self.log(f"视频 {video_name} 当前阶段处理完成", "INFO")
            self.log(f"累计进度: {processed_frames}/{total_frames} 帧 ({processed_frames/total_frames*100:.1f}%)", "INFO")
            self.log(f"累计批次: {batch_pre + success_count} 批", "INFO")
            self.log(f"生成文件: {len(output_files)} 个", "INFO")
            for i, file_path in enumerate(output_files, 1):
                if os.path.exists(file_path):
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    self.log(f"  {i:2d}. {os.path.basename(file_path)} ({size_mb:.1f}MB)", "INFO")
        else:
            self.log(f"视频 {video_name} 部分批次失败 ({success_count}/{total_batches})", "WARN")
        
        # 保存最终状态
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
        attn_mode: str = "block_sparse_attention",
        tiled_dit: bool = False,
        tile_size: int = 256,
        tile_overlap: int = 24,
        scale: int = 2,
        gpu_device: str = "auto",
        timeout_per_batch: int = 600,
        max_workers: int = 1,
        output_dir: str = "output",
        auto_load_state: bool = True,
        save_state: bool = True
    ) -> List[Dict]:
        """
        处理目录下的所有视频文件
        
        参数:
            workflow_template_path: 工作流模板路径
            input_path: 输入目录
            pattern: 文件匹配模式
            frames_per_batch: 每批帧数
            attn_mode: 稀疏模式
            tiled_dit: 分块开关 (布尔值)
            tile_size: 分块大小
            tile_overlap: 分块重叠
            scale: 放大倍数
            gpu_device: GPU设备选择
            timeout_per_batch: 每批超时时间（秒）
            max_workers: 最大并行工作数
            output_dir: 输出目录
            auto_load_state: 自动加载状态 (布尔值)
            save_state: 保存状态 (布尔值)
            
        返回:
            所有视频的处理结果列表
        """
        # 收集视频文件
        video_files = self.collect_video_files(input_path, pattern)
        
        if not video_files:
            self.log(f"在目录 {input_path} 中未找到视频文件", "ERROR")
            return []
        
        self.log(f"找到 {len(video_files)} 个视频文件", "INFO")
        for vf in video_files:
            self.log(f"  - {os.path.basename(vf)}", "INFO")
        
        all_results = []
        
        # 处理每个视频文件
        for i, video_path in enumerate(video_files, 1):
            self.log(f"\n{'#'*80}", "INFO")
            self.log(f"进度: {i}/{len(video_files)}", "INFO")
            
            # 为每个视频创建单独的输出子目录
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            video_output_dir = os.path.join(output_dir, video_name)
            
            result = self.process_video_file(
                workflow_template_path,
                video_path,
                frames_per_batch,
                attn_mode,
                tiled_dit,
                tile_size,
                tile_overlap,
                scale,
                gpu_device,
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
                self.log(f"视频 {result['video']} 处理成功 ({result['success_rate']})", "INFO")
            else:
                self.log(f"视频 {result['video']} 处理失败 ({result['success_rate']})", "ERROR")
        
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
                self.log(f"添加单个文件: {input_path}", "INFO")
            else:
                self.log(f"文件格式不支持: {input_path}", "ERROR")
        
        elif os.path.isdir(input_path):
            # 目录
            self.log(f"扫描目录: {input_path}", "INFO")
            
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
                self.log(f"目录 {input_path} 中未找到任何视频文件", "WARN")
            else:
                self.log(f"从目录找到 {len(video_files)} 个视频文件", "INFO")
        
        else:
            self.log(f"路径不存在: {input_path}", "ERROR")
        
        return video_files

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='ComfyUI FlashVSR-XZG MIX 批量视频处理脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 处理单个视频文件（从头开始）
  python flashvsr_mix.py -i video.mp4 --template api_flashvsr_mix.json
  
  # 指定稀疏模式和分块开关
  python flashvsr_mix.py -i video.mp4 --template api_flashvsr_mix.json --attn-mode sparse_sage_attention --tiled-dit
  
  # 自定义分块参数和放大倍数
  python flashvsr_mix.py -i video.mp4 --template api_flashvsr_mix.json --tile-size 512 --tile-overlap 32 --scale 2
  
  # 断点续跑，自动加载状态文件
  python flashvsr_mix.py -i video.mp4 --template api_flashvsr_mix.json --auto-load-state
  
  # 指定已处理帧数和批次
  python flashvsr_mix.py -i video.mp4 --template api_flashvsr_mix.json --frames-pre 100 --batch-pre 3
  
  # 指定GPU设备
  python flashvsr_mix.py -i video.mp4 --template api_flashvsr_mix.json --gpu 0
  
  # 处理目录下的所有视频文件
  python flashvsr_mix.py -i ./videos --template api_flashvsr_mix.json --max-workers 2

功能特性:
  1. 自动128对齐：自动计算输入和输出宽度，对齐到128的倍数
  2. 多种稀疏模式：支持 block_sparse_attention 和 sparse_sage_attention
  3. 分块开关：可控制是否启用分块处理
  4. 断点续跑：支持自动加载和保存处理状态
  5. 并行处理：支持多线程并行处理

注意:
  1. 脚本使用 pymediainfo 获取视频信息，请确保已安装
  2. 输入输出宽度会自动对齐到128的倍数
  3. 状态文件保存在 ./states_mix/ 目录下
        """
    )
    
    # 必需参数
    parser.add_argument('-i', '--input', type=str, required=True,
                       help='输入路径（可以是视频文件或目录）')
    
    # 工作流参数
    parser.add_argument('--template', type=str, default='api_flashvsr_mix.json',
                       help='工作流模板 JSON 文件路径 (默认: api_flashvsr_mix.json)')
    parser.add_argument('--frames-per-batch', type=int, default=50,
                       help='每批处理的帧数 (默认: 50)')
    
    # FlashVSR 处理参数
    parser.add_argument('--attn-mode', type=str, default='block_sparse_attention',
                       choices=['block_sparse_attention', 'sparse_sage_attention'],
                       help='稀疏模式: block_sparse_attention (默认) 或 sparse_sage_attention')
    parser.add_argument('--tiled-dit', action='store_true',
                       help='启用分块处理 (默认: 禁用)')
    parser.add_argument('--tile-size', type=int, default=256,
                       help='分块大小 (默认: 256)')
    parser.add_argument('--tile-overlap', type=int, default=24,
                       help='分块重叠像素 (默认: 24)')
    parser.add_argument('--scale', type=int, default=2,
                       help='放大倍数 (默认: 2)')
    
    # GPU参数
    parser.add_argument('--gpu', type=str, default='auto',
                       help='GPU设备选择: auto, 0, 1, 2, cuda:0, cuda:1等 (默认: auto)')
    
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
    
    # 并行处理参数
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
        print(f"输入路径不存在: {args.input}")
        return
    
    # 检查模板文件是否存在
    if not os.path.exists(args.template):
        print(f"工作流模板不存在: {args.template}")
        return
    
    # 验证断点参数
    if args.frames_pre < 0:
        print(f"已处理帧数不能为负数: {args.frames_pre}")
        return
    if args.batch_pre < 0:
        print(f"已处理批次不能为负数: {args.batch_pre}")
        return
    
    # 验证并行处理参数
    if args.max_workers < 1:
        print(f"最大工作线程数必须大于0: {args.max_workers}")
        return
    
    # 初始化处理器
    processor = FlashVSR_XZG_MIX_Processor(
        comfyui_url=args.server,
        log_dir=args.log_dir
    )
    
    # 检查 ComfyUI 服务
    if not processor.check_comfyui_server():
        processor.log("ComfyUI 服务不可用，请确保 ComfyUI 已启动", "ERROR")
        return
    
    processor.log(f"FlashVSR-XZG MIX 开始处理", "INFO")
    processor.log(f"输入路径: {args.input}", "INFO")
    processor.log(f"工作流模板: {args.template}", "INFO")
    processor.log(f"每批帧数: {args.frames_per_batch}", "INFO")
    processor.log(f"稀疏模式: {args.attn_mode}", "INFO")
    processor.log(f"分块开关: {'启用' if args.tiled_dit else '禁用'}", "INFO")
    processor.log(f"分块大小: {args.tile_size}", "INFO")
    processor.log(f"分块重叠: {args.tile_overlap}", "INFO")
    processor.log(f"缩放倍数: {args.scale}", "INFO")
    processor.log(f"超时时间: {args.timeout}秒", "INFO")
    processor.log(f"输出目录: {args.output_dir}", "INFO")
    processor.log(f"并行处理: {args.max_workers} 个工作线程", "INFO")
    
    if args.auto_load_state:
        processor.log(f"自动加载状态: 已启用", "INFO")
    if args.frames_pre > 0 or args.batch_pre > 0:
        processor.log(f"手动断点: 已处理 {args.frames_pre} 帧, {args.batch_pre} 批", "INFO")
    if not args.save_state:
        processor.log(f"状态保存: 已禁用", "INFO")
    
    if args.gpu == "auto":
        processor.log(f"GPU设备: auto (自动选择)", "INFO")
    elif args.gpu.isdigit():
        processor.log(f"GPU设备: cuda:{args.gpu}", "INFO")
    else:
        processor.log(f"GPU设备: {args.gpu}", "INFO")
    
    start_time = time.time()
    
    # 判断输入类型并处理
    if os.path.isfile(args.input):
        # 单个文件
        processor.log(f"处理单个文件", "INFO")
        result = processor.process_video_file(
            args.template,
            args.input,
            args.frames_per_batch,
            args.attn_mode,
            args.tiled_dit,
            args.tile_size,
            args.tile_overlap,
            args.scale,
            args.gpu,
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
        processor.log(f"处理目录", "INFO")
        results = processor.process_directory(
            args.template,
            args.input,
            args.pattern,
            args.frames_per_batch,
            args.attn_mode,
            args.tiled_dit,
            args.tile_size,
            args.tile_overlap,
            args.scale,
            args.gpu,
            args.timeout,
            args.max_workers,
            args.output_dir,
            args.auto_load_state,
            args.save_state
        )
    
    else:
        processor.log(f"输入路径类型未知: {args.input}", "ERROR")
        return
    
    # 计算总耗时
    total_time = time.time() - start_time
    
    # 输出汇总结果
    processor.log(f"\n{'='*80}", "INFO")
    processor.log(f"FlashVSR-XZG MIX 处理完成汇总", "INFO")
    processor.log(f"{'='*80}", "INFO")
    
    if not results:
        processor.log(f"没有处理任何视频", "ERROR")
        return
    
    total_videos = len(results)
    success_videos = sum(1 for r in results if r["success"])
    failed_videos = total_videos - success_videos
    
    total_batches = sum(r["total_batches"] for r in results)
    success_batches = sum(r["batches_processed"] for r in results)
    
    # 计算总处理帧数
    total_frames_processed = sum(r.get("processed_frames", 0) for r in results)
    total_files_generated = sum(len(r.get("output_files", [])) for r in results)
    
    # 输出参数摘要
    for i, result in enumerate(results, 1):
        if result.get("parameters"):
            params = result["parameters"]
            processor.log(f"视频 {i} 参数:", "INFO")
            processor.log(f"  稀疏模式: {params.get('attn_mode')}", "INFO")
            processor.log(f"  分块开关: {params.get('tiled_dit')}", "INFO")
            processor.log(f"  分块大小: {params.get('tile_size')}", "INFO")
            processor.log(f"  分块重叠: {params.get('tile_overlap')}", "INFO")
            processor.log(f"  缩放倍数: {params.get('scale')}", "INFO")
            processor.log(f"  输入宽度: {result.get('in_width_aligned', 'N/A')}", "INFO")
            processor.log(f"  输出宽度: {result.get('out_width_aligned', 'N/A')}", "INFO")
    
    processor.log(f"总耗时: {total_time:.2f}秒 ({total_time/60:.1f}分钟)", "INFO")
    processor.log(f"总视频数: {total_videos}", "INFO")
    processor.log(f"成功视频: {success_videos}", "INFO")
    processor.log(f"失败视频: {failed_videos}", "INFO" if failed_videos == 0 else "ERROR")
    processor.log(f"总批次: {total_batches}", "INFO")
    processor.log(f"成功批次: {success_batches} ({success_batches/total_batches*100:.1f}%)", "INFO")
    processor.log(f"总处理帧数: {total_frames_processed}", "INFO")
    processor.log(f"总生成文件: {total_files_generated}", "INFO")
    
    # 输出失败详情
    if failed_videos > 0:
        processor.log(f"\n失败视频详情:", "ERROR")
        for result in results:
            if not result["success"]:
                processor.log(f"  - {result['video']}: {result.get('error', '未知错误')}", "ERROR")
    
    processor.log(f"\n状态文件目录: {processor.state_dir}", "INFO")
    processor.log(f"详细日志: {processor.log_file}", "INFO")
    processor.log(f"FlashVSR-XZG MIX 处理完成!", "INFO")

if __name__ == "__main__":
    main()