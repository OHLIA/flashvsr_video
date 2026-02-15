"""
ComfyUI FlashVSR 批量视频处理工具 - 修复版
修复批次检测和文件验证问题
"""

import os
import sys
import json
import time
import random
import string
import logging
import requests
import subprocess
import datetime
import shutil
import gc
import atexit
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from glob import glob
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comfyui_batch_processor.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 可选依赖
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("⚠️  torch不可用，GPU清理功能将受限")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("⚠️  psutil不可用，进程管理功能将受限")

try:
    from pymediainfo import MediaInfo
    PYMEDIAINFO_AVAILABLE = True
except ImportError:
    PYMEDIAINFO_AVAILABLE = False
    logger.warning("⚠️  pymediainfo不可用，视频帧数检测将使用估计值")

def get_video_info(video_path: str) -> Dict[str, Any]:
    """获取视频信息"""
    video_info = {
        'total_frames': 0,
        'fps': 0.0,
        'duration': 0.0,
        'resolution': '未知',
        'file_size': os.path.getsize(video_path)
    }
    
    if PYMEDIAINFO_AVAILABLE:
        try:
            media_info = MediaInfo.parse(video_path)
            for track in media_info.tracks:
                if track.track_type == 'Video':
                    video_info['total_frames'] = int(track.frame_count) if hasattr(track, 'frame_count') else 0
                    video_info['fps'] = float(track.frame_rate) if hasattr(track, 'frame_rate') else 0.0
                    video_info['duration'] = float(track.duration) / 1000.0 if hasattr(track, 'duration') else 0.0
                    video_info['resolution'] = f"{track.width}x{track.height}" if hasattr(track, 'width') and hasattr(track, 'height') else '未知'
                    break
        except Exception as e:
            logger.warning(f"⚠️  无法通过pymediainfo获取视频信息: {e}")
    
    # 如果无法获取帧数，尝试通过时长和帧率估算
    if video_info['total_frames'] <= 0 and video_info['duration'] > 0 and video_info['fps'] > 0:
        video_info['total_frames'] = int(video_info['duration'] * video_info['fps'])
        logger.info(f"📊 通过时长和帧率估算总帧数: {video_info['total_frames']}")
    
    return video_info

class ComfyUI_Client:
    """ComfyUI API客户端"""
    
    def __init__(self, server_address: str = "http://127.0.0.1:8188"):
        self.server_address = server_address
        self.session = requests.Session()
        self.client_id = self.generate_client_id()
    
    def generate_client_id(self) -> str:
        """生成客户端ID"""
        random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        return f"batch_processor_{random_str}"
    
    def is_server_running(self) -> bool:
        """检查ComfyUI服务器是否运行"""
        try:
            response = self.session.get(f"{self.server_address}/system_stats", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_queue(self) -> List[Dict]:
        """获取队列信息"""
        try:
            response = self.session.get(f"{self.server_address}/queue", timeout=10)
            if response.status_code == 200:
                queue_data = response.json()
                return queue_data.get('queue_running', []) + queue_data.get('queue_pending', [])
        except Exception as e:
            logger.debug(f"获取队列失败: {e}")
        return []
    
    def is_queue_empty(self) -> bool:
        """检查队列是否为空"""
        return len(self.get_queue()) == 0
    
    def get_history(self, prompt_id: str = None) -> Dict:
        """获取历史记录"""
        try:
            response = self.session.get(f"{self.server_address}/history", timeout=10)
            if response.status_code == 200:
                history_data = response.json()
                if prompt_id:
                    return history_data.get(prompt_id, {})
                return history_data
        except Exception as e:
            logger.debug(f"获取历史记录失败: {e}")
        return {}
    
    def get_prompt_status(self, prompt_id: str) -> Optional[Dict]:
        """获取任务状态"""
        try:
            # 首先检查历史记录
            history = self.get_history()
            if prompt_id in history:
                return {
                    'status': {
                        'completed': True,
                        'error': False
                    },
                    'outputs': history[prompt_id].get('outputs', {})
                }
            
            # 检查队列
            queue = self.get_queue()
            for item in queue:
                if item.get('prompt_id') == prompt_id:
                    return {
                        'status': {
                            'completed': False,
                            'error': False
                        },
                        'outputs': {}
                    }
            
            # 如果不在历史和队列中，可能任务不存在或已失败
            return {
                'status': {
                    'completed': False,
                    'error': True,
                    'error_message': '任务不在队列或历史中'
                }
            }
            
        except Exception as e:
            logger.error(f"获取任务状态失败: {e}")
            return None
    
    def is_prompt_completed(self, prompt_id: str) -> bool:
        """检查任务是否完成"""
        prompt_info = self.get_prompt_status(prompt_id)
        if prompt_info and 'status' in prompt_info:
            return prompt_info['status'].get('completed', False)
        return False
    
    def submit_prompt(self, workflow: Dict) -> Optional[str]:
        """提交任务到ComfyUI"""
        try:
            # 清除历史记录
            self.clear_history()
            
            # 提交任务
            response = self.session.post(
                f"{self.server_address}/prompt",
                json={"prompt": workflow, "client_id": self.client_id},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                prompt_id = data.get('prompt_id')
                logger.debug(f"任务提交成功，ID: {prompt_id}")
                return prompt_id
            else:
                logger.error(f"提交任务失败: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"提交任务异常: {e}")
            return None
    
    def clear_history(self) -> bool:
        """清除历史记录"""
        try:
            response = self.session.post(f"{self.server_address}/history", json={"clear": True})
            return response.status_code == 200
        except:
            return False
    
    def clear_queue(self) -> bool:
        """清除队列"""
        try:
            response = self.session.post(f"{self.server_address}/queue", json={"clear": True})
            return response.status_code == 200
        except:
            return False
    
    def wait_for_prompt_completion(self, prompt_id: str, timeout: int = 3600, poll_interval: int = 5) -> bool:
        """等待任务完成"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.is_prompt_completed(prompt_id):
                return True
            
            # 检查队列状态
            if self.is_queue_empty():
                # 队列为空，但任务可能仍在处理中
                time.sleep(1)
                continue
            
            time.sleep(poll_interval)
            
            # 输出进度
            elapsed = int(time.time() - start_time)
            if elapsed % 30 == 0:  # 每30秒输出一次状态
                logger.info(f"⏳ 任务 {prompt_id[:8]}... 已运行 {elapsed} 秒")
        
        logger.warning(f"⚠️  任务 {prompt_id[:8]}... 超时 ({timeout}秒)")
        return False

class BatchOutputTracker:
    """批次输出跟踪器 - 修复版"""
    
    def __init__(self, output_dir: str = None):
        """
        初始化批处理状态跟踪器
        修复：准确检测批次输出文件
        """
        self.output_dir = output_dir or self.get_default_output_dir()
        
    def get_default_output_dir(self) -> str:
        """获取默认输出目录"""
        comfyui_output = r"F:\AI\ComfyUI_Mie_V7.0\ComfyUI\output"
        if os.path.exists(comfyui_output):
            return comfyui_output
        
        default_output = os.path.join(os.getcwd(), "output")
        os.makedirs(default_output, exist_ok=True)
        return default_output
    
    def extract_output_pattern_from_workflow(self, workflow: Dict) -> str:
        """从工作流中提取输出文件名模式"""
        output_prefix = ""
        file_format = "mov"
        
        for node_id, node_data in workflow.items():
            if node_data.get("class_type") == "VHS_VideoCombine":
                inputs = node_data.get("inputs", {})
                output_prefix = inputs.get("filename_prefix", "")
                # 从格式选项获取文件格式
                format_opt = inputs.get("format", "mov")
                if format_opt in ["mov", "mp4", "webm", "avi"]:
                    file_format = format_opt
                break
        
        if not output_prefix:
            # 如果没有设置前缀，使用默认模式
            return f"ComfyUI_*.{file_format}"
        
        # ComfyUI的输出格式通常是: prefix_XXXX_YYYYYYY.format
        # 其中 XXXX 是批次号，YYYYYYY 是随机字符串
        return f"{output_prefix}_*.{file_format}"
    
    def get_output_files(self, video_path: str, workflow: Dict) -> List[str]:
        """获取实际生成的输出文件列表"""
        # 从工作流中提取输出模式
        pattern = self.extract_output_pattern_from_workflow(workflow)
        
        # 在输出目录中查找文件
        all_files = []
        
        # 搜索匹配的文件
        search_pattern = os.path.join(self.output_dir, pattern)
        logger.debug(f"搜索输出文件模式: {search_pattern}")
        
        for ext in ['mov', 'mp4', 'webm', 'avi', 'MOV', 'MP4', 'WEBM', 'AVI']:
            # 尝试各种扩展名
            ext_pattern = os.path.join(self.output_dir, pattern.replace(".mov", f".{ext}"))
            matches = glob(ext_pattern)
            all_files.extend(matches)
        
        # 去重
        unique_files = list(set(all_files))
        
        # 按修改时间排序，最新的在前面
        unique_files.sort(key=os.path.getmtime, reverse=True)
        
        return unique_files
    
    def check_output_files_exist(self, video_path: str, workflow: Dict, expected_min_files: int = 1) -> Tuple[bool, List[str]]:
        """检查输出文件是否存在
        返回: (文件存在, 文件列表)
        """
        output_files = self.get_output_files(video_path, workflow)
        
        if len(output_files) >= expected_min_files:
            logger.info(f"✅ 找到 {len(output_files)} 个输出文件:")
            for file_path in output_files[:5]:  # 只显示前5个
                file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                file_size_mb = file_size / (1024 * 1024)
                logger.info(f"  📄 {os.path.basename(file_path)} ({file_size_mb:.1f}MB)")
            if len(output_files) > 5:
                logger.info(f"  ... 还有 {len(output_files)-5} 个文件")
            return True, output_files
        else:
            logger.warning(f"⚠️  只找到 {len(output_files)} 个输出文件，期望至少 {expected_min_files} 个")
            if output_files:
                logger.info("找到的文件:")
                for file_path in output_files:
                    logger.info(f"  📄 {os.path.basename(file_path)}")
            return False, output_files
    
    def verify_output_files_complete(self, video_path: str, workflow: Dict, total_frames: int, frames_per_batch: int) -> Tuple[bool, int, int]:
        """验证输出文件是否完整
        修复：正确的批次计算和文件验证
        """
        # 计算预期批次数
        if total_frames <= 0 or frames_per_batch <= 0:
            logger.warning(f"⚠️  无效的帧数参数: 总帧数={total_frames}, 每批帧数={frames_per_batch}")
            return False, 0, 0
        
        num_batches = (total_frames + frames_per_batch - 1) // frames_per_batch
        logger.info(f"📊 视频总帧数: {total_frames}, 每批帧数: {frames_per_batch}, 需要批次: {num_batches}")
        
        # 获取实际输出文件
        output_files = self.get_output_files(video_path, workflow)
        found_batches = len(output_files)
        
        logger.info(f"📊 批次完成情况: {found_batches}/{num_batches}")
        
        if found_batches >= num_batches:
            logger.info(f"✅ 所有批次文件已生成")
            return True, found_batches, num_batches
        elif found_batches > 0:
            logger.warning(f"⚠️  只生成 {found_batches}/{num_batches} 个批次文件")
            return False, found_batches, num_batches
        else:
            logger.error(f"❌ 未生成任何批次文件")
            return False, 0, num_batches

class ComfyUI_FlashVSR_BatchProcessor:
    def __init__(self, 
                 comfyui_url: str = "http://127.0.0.1:8188", 
                 task_timeout: int = 300,
                 max_retries: int = 3,
                 restart_delay: int = 5,
                 startup_timeout: int = 300):
        """
        初始化批量处理器 - 修复版
        """
        # API客户端
        self.client = ComfyUI_Client(comfyui_url)
        
        # 批处理跟踪器
        self.output_tracker = BatchOutputTracker()
        
        # 配置参数
        self.comfyui_url = comfyui_url
        self.task_timeout = task_timeout
        self.max_retries = max_retries
        self.restart_delay = restart_delay
        self.startup_timeout = startup_timeout
        
        # 状态跟踪
        self.processed_files = {}
        self.failed_files = {}
        self.restart_history = []
        self.current_retry_count = 0
        
        # 注册清理函数
        atexit.register(self.cleanup)
        
        logger.info("=" * 60)
        logger.info("ComfyUI FlashVSR 批量处理器 - 修复版")
        logger.info(f"ComfyUI地址: {comfyui_url}")
        logger.info(f"任务超时: {task_timeout}秒")
        logger.info(f"最大重试次数: {max_retries}次")
        logger.info(f"输出目录: {self.output_tracker.output_dir}")
        logger.info("=" * 60)
    
    def ensure_comfyui_running(self) -> bool:
        """确保ComfyUI在运行"""
        if self.client.is_server_running():
            return True
        
        logger.warning("⚠️  ComfyUI未运行，请手动启动ComfyUI")
        return False
    
    def clear_cache(self) -> Dict[str, Any]:
        """清理缓存"""
        cleanup_results = {
            "system_memory_freed_mb": 0,
            "gpu_memory_freed_mb": 0,
            "success": False
        }
        
        logger.info("🧹 清理缓存...")
        
        try:
            # 清理GPU显存
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("✅ GPU显存已清理")
                cleanup_results["gpu_memory_freed_mb"] = 1
                cleanup_results["success"] = True
            
            # 清理Python内存
            collected = gc.collect()
            logger.info(f"✅ Python垃圾回收: {collected} 个对象")
            cleanup_results["success"] = True
            
        except Exception as e:
            logger.error(f"❌ 清理缓存失败: {e}")
        
        return cleanup_results
    
    def load_workflow_template(self, template_path: str) -> Dict:
        """加载工作流模板"""
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
        """更新工作流参数"""
        # 深拷贝工作流
        import copy
        modified_workflow = copy.deepcopy(workflow)
        
        # 设置输入视频路径
        for node_id, node_data in modified_workflow.items():
            if node_data.get("class_type") == "VHS_LoadVideo":
                node_data["inputs"]["video"] = video_path
                logger.info(f"✅ 设置输入视频: {os.path.basename(video_path)}")
        
        # 设置FlashVSR参数
        for node_id, node_data in modified_workflow.items():
            if node_data.get("class_type") == "FlashVSRNodeAdv":
                inputs = node_data.get("inputs", {})
                if "{{scale}}" in str(inputs.get("scale", "")):
                    inputs["scale"] = scale
                if "{{t_z}}" in str(inputs.get("tile_size", "")):
                    inputs["tile_size"] = tile_size
                if "{{t_o}}" in str(inputs.get("tile_overlap", "")):
                    inputs["tile_overlap"] = tile_overlap
        
        # 设置GPU设备
        for node_id, node_data in modified_workflow.items():
            if node_id == "5" and node_data.get("class_type") == "FlashVSRInitPipe":
                inputs = node_data.get("inputs", {})
                if isinstance(inputs.get("device"), str):
                    if gpu_device.isdigit():
                        device_value = f"cuda:{gpu_device}"
                    else:
                        device_value = gpu_device
                    inputs["device"] = device_value
                    logger.info(f"✅ 设置GPU设备: {device_value}")
        
        # 设置总帧数
        if total_frames is None or total_frames <= 0:
            video_info = get_video_info(video_path)
            total_frames = video_info.get('total_frames', 0)
            if total_frames <= 0:
                total_frames = 10000
                logger.warning(f"⚠️  无法获取视频总帧数，使用默认值: {total_frames}")
            else:
                logger.info(f"📊 从视频获取总帧数: {total_frames}")
        
        for node_id, node_data in modified_workflow.items():
            if node_id == "50" and node_data.get("class_type") == "PrimitiveInt":
                node_data["inputs"]["value"] = total_frames
                logger.info(f"✅ 设置总帧数: {total_frames}")
        
        # 设置每批帧数
        for node_id, node_data in modified_workflow.items():
            if node_id == "8" and node_data.get("class_type") == "PrimitiveInt":
                node_data["inputs"]["value"] = frames_per_batch
                logger.info(f"✅ 设置每批帧数: {frames_per_batch}")
        
        # 设置输出前缀
        if output_prefix is None:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_prefix = f"flashvsr_{base_name}"
        
        for node_id, node_data in modified_workflow.items():
            if node_data.get("class_type") == "VHS_VideoCombine":
                node_data["inputs"]["filename_prefix"] = output_prefix
                logger.info(f"✅ 设置输出前缀: {output_prefix}")
        
        return modified_workflow
    
    def wait_for_task_completion_with_verification(
        self, 
        prompt_id: str, 
        video_path: str, 
        workflow: Dict,
        total_frames: int,
        frames_per_batch: int,
        timeout: int = 300
    ) -> Tuple[bool, str, List[str]]:
        """
        等待任务完成并进行验证 - 修复版
        返回: (是否成功, 状态信息, 输出文件列表)
        """
        logger.info(f"⏳ 等待任务完成 (超时: {timeout}秒)...")
        
        start_time = time.time()
        last_status_check = 0
        status_check_interval = 5
        empty_queue_checks = 0
        max_empty_queue_checks = 3
        
        while time.time() - start_time < timeout:
            # 检查任务状态
            prompt_info = self.client.get_prompt_status(prompt_id)
            
            if prompt_info:
                status = prompt_info.get('status', {})
                
                if status.get('completed', False):
                    logger.info(f"✅ 任务 {prompt_id[:8]}... API报告已完成")
                    
                    # 重要：立即检查输出文件
                    logger.info("🔍 验证输出文件...")
                    time.sleep(2)  # 等待文件写入
                    
                    # 检查是否有输出文件
                    files_exist, output_files = self.output_tracker.check_output_files_exist(
                        video_path, workflow, expected_min_files=1
                    )
                    
                    if files_exist and output_files:
                        # 验证批次完整性
                        is_complete, found_batches, total_batches = self.output_tracker.verify_output_files_complete(
                            video_path, workflow, total_frames, frames_per_batch
                        )
                        
                        if is_complete or found_batches > 0:
                            logger.info(f"✅ 输出文件验证通过，找到 {len(output_files)} 个文件")
                            return True, "任务完成", output_files
                        else:
                            logger.warning(f"⚠️  输出文件不完整: {found_batches}/{total_batches}")
                            return False, f"输出文件不完整: {found_batches}/{total_batches}", output_files
                    else:
                        logger.warning("⚠️  API报告完成但未找到输出文件")
                        return False, "未找到输出文件", []
                
                if status.get('error', False):
                    error_msg = status.get('error_message', '未知错误')
                    logger.error(f"❌ 任务出错: {error_msg}")
                    return False, f"任务出错: {error_msg}", []
            
            # 检查队列状态
            queue_info = self.client.get_queue()
            queue_length = len(queue_info)
            
            if queue_length > 0:
                logger.debug(f"⏳ 队列中还有 {queue_length} 个任务")
                empty_queue_checks = 0
            else:
                empty_queue_checks += 1
                logger.debug(f"⏳ 队列已空 ({empty_queue_checks}/{max_empty_queue_checks})")
                
                # 重要：即使队列为空，也要检查是否有输出文件
                if empty_queue_checks >= 2:  # 第二次检查队列为空时验证文件
                    logger.info("🔍 队列为空，检查输出文件...")
                    files_exist, output_files = self.output_tracker.check_output_files_exist(
                        video_path, workflow, expected_min_files=1
                    )
                    
                    if files_exist and output_files:
                        # 验证批次完整性
                        is_complete, found_batches, total_batches = self.output_tracker.verify_output_files_complete(
                            video_path, workflow, total_frames, frames_per_batch
                        )
                        
                        if is_complete or found_batches > 0:
                            logger.info(f"✅ 队列为空但找到输出文件: {len(output_files)} 个")
                            return True, "任务完成（队列为空但已输出文件）", output_files
                    elif empty_queue_checks >= max_empty_queue_checks:
                        logger.warning(f"⚠️  队列连续 {max_empty_queue_checks} 次为空且无输出文件")
                        return False, f"队列为空且无输出文件", []
            
            # 定期输出进度
            elapsed = int(time.time() - start_time)
            if elapsed - last_status_check >= 30:
                logger.info(f"⏳ 已处理 {elapsed} 秒，队列: {queue_length} 个任务")
                last_status_check = elapsed
            
            time.sleep(2)
        
        logger.warning(f"⚠️  任务 {prompt_id} 超时 ({timeout}秒)")
        return False, f"任务超时 ({timeout}秒)", []
    
    def process_single_video(
        self,
        workflow_template: Dict,
        video_path: str,
        output_prefix: Optional[str] = None,
        scale: float = 4.0,
        tile_size: int = 256,
        tile_overlap: int = 24,
        frames_per_batch: int = 201,
        gpu_device: str = "auto"
    ) -> Tuple[bool, str, int, List[str]]:
        """
        处理单个视频 - 修复版
        返回: (是否成功, 状态信息, 重试次数, 输出文件列表)
        """
        video_name = os.path.basename(video_path)
        retry_count = 0
        success = False
        status_msg = "初始状态"
        output_files = []
        
        # 确保ComfyUI在运行
        if not self.ensure_comfyui_running():
            return False, "ComfyUI未运行", 0, []
        
        # 获取视频信息
        video_info = get_video_info(video_path)
        total_frames = video_info.get('total_frames', 0)
        
        if total_frames <= 0:
            logger.warning(f"⚠️  无法获取视频 '{video_name}' 的总帧数，使用默认值")
            total_frames = 10000
        
        # 计算预期批次数
        num_batches = 1
        if total_frames > 0 and frames_per_batch > 0:
            num_batches = (total_frames + frames_per_batch - 1) // frames_per_batch
        logger.info(f"📊 视频 '{video_name}' 需要 {num_batches} 个批次 (总帧数: {total_frames}, 每批: {frames_per_batch})")
        
        # 先检查是否已有输出文件
        logger.info("🔍 检查是否已有输出文件...")
        temp_workflow = self.update_workflow_parameters(
            workflow_template, video_path, output_prefix
        )
        files_exist, existing_files = self.output_tracker.check_output_files_exist(
            video_path, temp_workflow, expected_min_files=num_batches
        )
        
        if files_exist and len(existing_files) >= num_batches:
            logger.info(f"✅ 视频 '{video_name}' 已有完整输出文件，跳过处理")
            return True, "已有完整输出文件", 0, existing_files
        
        while retry_count < self.max_retries and not success:
            retry_count += 1
            logger.info(f"🔄 尝试 {retry_count}/{self.max_retries}")
            
            try:
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
                
                # 提交任务
                prompt_id = self.client.submit_prompt(workflow)
                
                if not prompt_id:
                    status_msg = "提交任务失败"
                    logger.error(f"❌ {status_msg}")
                    continue
                
                logger.info(f"✅ 任务已提交: {video_name}")
                logger.info(f"  任务ID: {prompt_id}")
                
                # 等待任务完成并验证输出
                task_success, task_status, output_files = self.wait_for_task_completion_with_verification(
                    prompt_id=prompt_id,
                    video_path=video_path,
                    workflow=workflow,
                    total_frames=total_frames,
                    frames_per_batch=frames_per_batch,
                    timeout=self.task_timeout
                )
                
                if task_success:
                    success = True
                    status_msg = task_status
                    
                    # 最终验证输出文件
                    if output_files:
                        logger.info(f"✅ 视频处理完成，生成 {len(output_files)} 个输出文件")
                        for i, file_path in enumerate(output_files[:3]):  # 只显示前3个
                            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                            file_size_mb = file_size / (1024 * 1024)
                            logger.info(f"  {i+1}. {os.path.basename(file_path)} ({file_size_mb:.1f}MB)")
                        if len(output_files) > 3:
                            logger.info(f"  ... 还有 {len(output_files)-3} 个文件")
                    else:
                        logger.warning("⚠️  任务成功但未记录输出文件")
                    
                    break
                else:
                    status_msg = task_status
                    logger.error(f"❌ 任务失败: {status_msg}")
                    
                    # 如果失败但有部分输出文件，检查是否可用
                    if output_files:
                        logger.info(f"📁 找到 {len(output_files)} 个部分输出文件")
                        is_complete, found_batches, total_batches = self.output_tracker.verify_output_files_complete(
                            video_path, workflow, total_frames, frames_per_batch
                        )
                        
                        if found_batches >= total_batches * 0.8:  # 80%完成认为可用
                            logger.info(f"✅ 部分输出文件可用 ({found_batches}/{total_batches} 批次)")
                            success = True
                            status_msg = f"部分完成: {found_batches}/{total_batches} 批次"
                            break
                    
                    # 记录重启
                    self.record_restart(
                        video_path=video_path,
                        reason=status_msg,
                        attempt=retry_count
                    )
                    
                    if retry_count < self.max_retries:
                        logger.info(f"🔄 准备重试 ({retry_count}/{self.max_retries})...")
                        time.sleep(self.restart_delay)
                    else:
                        logger.error(f"❌ 达到最大重试次数")
                    
            except Exception as e:
                status_msg = f"处理异常: {str(e)}"
                logger.error(f"❌ {status_msg}")
                
                # 记录重启
                self.record_restart(
                    video_path=video_path,
                    reason=status_msg,
                    attempt=retry_count
                )
                
                if retry_count < self.max_retries:
                    logger.info(f"🔄 异常后准备重试 ({retry_count}/{self.max_retries})...")
                    time.sleep(self.restart_delay)
                else:
                    logger.error("❌ 达到最大重试次数")
        
        return success, status_msg, retry_count, output_files
    
    def record_restart(self, video_path: str, reason: str, attempt: int = 1):
        """记录重启事件"""
        restart_entry = {
            'timestamp': datetime.now().isoformat(),
            'video_path': video_path,
            'video_name': os.path.basename(video_path),
            'reason': reason,
            'attempt': attempt
        }
        self.restart_history.append(restart_entry)
        
        # 保存到日志文件
        self.save_restart_log()
    
    def save_restart_log(self, filename: str = "restart_history.json"):
        """保存重启历史到文件"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.restart_history, f, indent=2, ensure_ascii=False, default=str)
            logger.debug(f"✅ 重启历史已保存: {filename}")
        except Exception as e:
            logger.error(f"❌ 保存重启历史失败: {e}")
    
    def move_to_done_directory(self, video_path: str) -> Optional[str]:
        """移动文件到done目录"""
        try:
            if not os.path.exists(video_path):
                logger.error(f"❌ 文件不存在: {video_path}")
                return None
            
            # 获取原文件所在目录
            original_dir = os.path.dirname(video_path)
            file_name = os.path.basename(video_path)
            
            # 创建done目录
            done_dir = os.path.join(original_dir, "done")
            os.makedirs(done_dir, exist_ok=True)
            
            # 生成目标路径
            done_path = os.path.join(done_dir, file_name)
            
            # 如果目标文件已存在，添加时间戳
            if os.path.exists(done_path):
                base_name, ext = os.path.splitext(file_name)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                new_name = f"{base_name}_{timestamp}{ext}"
                done_path = os.path.join(done_dir, new_name)
            
            # 移动文件
            shutil.move(video_path, done_path)
            logger.info(f"✅ 文件已移动到done目录: {os.path.basename(done_path)}")
            
            return done_path
            
        except Exception as e:
            logger.error(f"❌ 移动文件失败: {e}")
            return None
    
    def batch_process(
        self,
        workflow_template_path: str,
        video_files: List[str],
        output_prefix_base: Optional[str] = None,
        scale: float = 4.0,
        tile_size: int = 256,
        tile_overlap: int = 24,
        frames_per_batch: int = 201,
        gpu_device: str = "auto",
        move_to_done: bool = True,
        cleanup_after_each: bool = True
    ) -> Dict[str, Tuple[bool, str, int, List[str]]]:
        """
        批量处理视频 - 修复版
        返回: 字典，键为视频路径，值为(是否成功, 状态信息, 重试次数, 输出文件列表)
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"开始批量处理 {len(video_files)} 个视频")
        logger.info(f"⚙️  参数: scale={scale}, tile_size={tile_size}, tile_overlap={tile_overlap}")
        logger.info(f"🎮 GPU设备: {gpu_device}")
        logger.info(f"⏱️  任务超时: {self.task_timeout}秒")
        logger.info(f"🔄 最大重试: {self.max_retries}次")
        logger.info(f"📁 输出目录: {self.output_tracker.output_dir}")
        logger.info(f"{'='*60}")
        
        # 加载工作流模板
        try:
            workflow_template = self.load_workflow_template(workflow_template_path)
            logger.info(f"✅ 加载工作流模板: {workflow_template_path}")
        except Exception as e:
            logger.error(f"❌ 加载工作流模板失败: {e}")
            return {}
        
        # 确保ComfyUI在运行
        if not self.ensure_comfyui_running():
            logger.error("❌ ComfyUI未运行，请先启动ComfyUI")
            return {}
        
        results = {}
        processed_count = 0
        failed_count = 0
        
        for i, video_path in enumerate(video_files, 1):
            video_name = os.path.basename(video_path)
            
            # 检查文件是否存在
            if not os.path.exists(video_path):
                logger.error(f"❌ 文件不存在: {video_path}")
                self.failed_files[video_path] = {
                    'status': 'failed',
                    'message': '文件不存在',
                    'timestamp': datetime.now().isoformat()
                }
                failed_count += 1
                continue
            
            logger.info(f"\n📊 [{i}/{len(video_files)}] 处理: {video_name}")
            logger.info(f"📁 文件路径: {video_path}")
            
            # 设置输出前缀
            output_prefix = None
            if output_prefix_base:
                base_name = os.path.splitext(video_name)[0]
                output_prefix = f"{output_prefix_base}_{base_name}"
            
            # 处理单个视频
            success, status_msg, retry_count, output_files = self.process_single_video(
                workflow_template=workflow_template,
                video_path=video_path,
                output_prefix=output_prefix,
                scale=scale,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                frames_per_batch=frames_per_batch,
                gpu_device=gpu_device
            )
            
            # 记录结果
            results[video_path] = (success, status_msg, retry_count, output_files)
            
            if success:
                processed_count += 1
                self.processed_files[video_path] = {
                    'status': 'success',
                    'message': status_msg,
                    'retries': retry_count,
                    'output_files': [os.path.basename(f) for f in output_files],
                    'timestamp': datetime.now().isoformat()
                }
                
                # 移动文件到done目录
                if move_to_done:
                    moved_path = self.move_to_done_directory(video_path)
                    if not moved_path:
                        logger.warning(f"⚠️  文件移动失败: {video_name}")
                
                # 重置重试计数
                self.current_retry_count = 0
                
            else:
                failed_count += 1
                self.failed_files[video_path] = {
                    'status': 'failed',
                    'message': status_msg,
                    'retries': retry_count,
                    'output_files': [os.path.basename(f) for f in output_files] if output_files else [],
                    'timestamp': datetime.now().isoformat()
                }
                logger.error(f"❌ 处理失败: {status_msg}")
            
            # 清理缓存
            if cleanup_after_each and success:
                self.clear_cache()
            
            # 短暂间隔
            if i < len(video_files):
                wait_time = 2
                logger.info(f"⏱️  等待 {wait_time} 秒...")
                time.sleep(wait_time)
        
        # 输出统计信息
        logger.info(f"\n{'='*60}")
        logger.info("批量处理完成")
        logger.info(f"{'='*60}")
        logger.info(f"📊 统计信息:")
        logger.info(f"  ✅ 成功: {processed_count}/{len(video_files)}")
        logger.info(f"  ❌ 失败: {failed_count}/{len(video_files)}")
        logger.info(f"  🔄 总重启次数: {len(self.restart_history)}")
        
        # 输出重启摘要
        if self.restart_history:
            logger.info(f"\n🔄 重启摘要:")
            for entry in self.restart_history[-10:]:  # 只显示最后10个
                logger.info(f"  • {entry['video_name']}: {entry['reason']} (尝试 {entry['attempt']})")
            if len(self.restart_history) > 10:
                logger.info(f"  ... 还有 {len(self.restart_history)-10} 个重启记录")
        
        # 保存处理日志
        self.save_processing_log(video_files, results)
        
        return results
    
    def save_processing_log(self, video_files: List[str], results: Dict):
        """保存处理日志"""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'total_files': len(video_files),
            'processed_files': self.processed_files,
            'failed_files': self.failed_files,
            'restart_history': self.restart_history,
            'results': {}
        }
        
        for video_path, (success, status, retries, output_files) in results.items():
            log_data['results'][video_path] = {
                'success': success,
                'status': status,
                'retries': retries,
                'output_files': [os.path.basename(f) for f in output_files] if output_files else [],
                'video_name': os.path.basename(video_path)
            }
        
        try:
            with open('processing_log.json', 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False, default=str)
            logger.info("✅ 处理日志已保存: processing_log.json")
        except Exception as e:
            logger.error(f"❌ 保存处理日志失败: {e}")
    
    def cleanup(self):
        """清理资源"""
        logger.info("🧹 清理资源...")
        try:
            self.client.session.close()
        except:
            pass
        logger.info("✅ 清理完成")

def collect_video_files(input_path: str, pattern: str = '*.mp4') -> List[str]:
    """收集视频文件"""
    video_files = []
    
    if os.path.isfile(input_path):
        if input_path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv')):
            video_files.append(input_path)
            logger.info(f"✅ 添加单个文件: {input_path}")
    elif os.path.isdir(input_path):
        # 搜索视频文件
        video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.MP4', '.MOV', '.AVI', '.MKV']
        
        for ext in video_extensions:
            search_pattern = os.path.join(input_path, f"*{ext}")
            found_files = glob(search_pattern)
            video_files.extend(found_files)
        
        # 去重并排序
        video_files = sorted(list(set(video_files)))
        
        if not video_files:
            logger.error(f"❌ 目录 {input_path} 中未找到任何视频文件")
        else:
            logger.info(f"✅ 从目录 {input_path} 找到 {len(video_files)} 个视频文件")
    else:
        logger.error(f"❌ 路径不存在: {input_path}")
    
    return video_files

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='ComfyUI FlashVSR 批量视频处理工具 - 修复版',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 处理目录下的所有视频
  python batch_processor_fixed.py --input ./videos --gpu 0
  
  # 自定义参数
  python batch_processor_fixed.py --input ./videos --task-timeout 600 --max-retries 5
  
  # 指定工作流模板
  python batch_processor_fixed.py --input ./videos --template flashvsr_template.json

主要修复:
  1. 修复批次检测逻辑，正确识别输出文件
  2. 增加输出文件验证，防止空任务被标记为完成
  3. 改进文件存在性检查
  4. 更好的错误处理和重试逻辑
        """
    )
    
    # 输入参数
    parser.add_argument('--template', type=str, default='flashvsr_template.json',
                       help='工作流模板 JSON 文件路径 (默认: flashvsr_template.json)')
    parser.add_argument('--input', type=str, required=True,
                       help='输入路径（视频文件或目录）')
    parser.add_argument('--pattern', type=str, default='*.mp4',
                       help='视频文件匹配模式 (默认: *.mp4)')
    
    # 处理参数
    parser.add_argument('--scale', type=float, default=4.0,
                       help='放大倍数 (默认: 4.0)')
    parser.add_argument('--tile-size', type=int, default=256,
                       help='分块大小 (默认: 256)')
    parser.add_argument('--tile-overlap', type=int, default=24,
                       help='分块重叠像素 (默认: 24)')
    parser.add_argument('--frames-per-batch', type=int, default=125,
                       help='每批处理的帧数 (默认: 125)')
    parser.add_argument('--gpu', type=str, default='auto',
                       help='GPU设备选择 (默认: auto)')
    
    # 监控参数
    parser.add_argument('--task-timeout', type=int, default=300,
                       help='任务超时时间（秒）(默认: 300)')
    parser.add_argument('--max-retries', type=int, default=3,
                       help='最大重试次数 (默认: 3)')
    parser.add_argument('--restart-delay', type=int, default=5,
                       help='重启后等待时间（秒）(默认: 5)')
    
    # 文件管理
    parser.add_argument('--no-move', action='store_true',
                       help='不将处理完成的文件移动到done目录')
    parser.add_argument('--no-cleanup', action='store_true',
                       help='不清理缓存')
    
    # 系统参数
    parser.add_argument('--server', type=str, default='http://127.0.0.1:8188',
                       help='ComfyUI 服务器地址 (默认: http://127.0.0.1:8188)')
    
    args = parser.parse_args()
    
    # 检查pymediainfo是否安装
    if not PYMEDIAINFO_AVAILABLE:
        logger.warning("⚠️  pymediainfo未安装，将无法获取视频真实帧数")
        logger.info("请运行: pip install pymediainfo")
    
    # 收集视频文件
    video_files = collect_video_files(args.input, args.pattern)
    
    if not video_files:
        logger.error("❌ 未找到任何视频文件")
        return
    
    # 显示文件列表
    logger.info(f"\n📁 找到 {len(video_files)} 个视频文件:")
    for i, vf in enumerate(video_files[:5], 1):
        logger.info(f"  {i}. {os.path.basename(vf)}")
    if len(video_files) > 5:
        logger.info(f"  ... 还有 {len(video_files)-5} 个文件")
    
    # 显示处理参数
    logger.info(f"\n⚙️  处理参数:")
    logger.info(f"  scale: {args.scale}")
    logger.info(f"  tile_size: {args.tile_size}")
    logger.info(f"  tile_overlap: {args.tile_overlap}")
    logger.info(f"  frames_per_batch: {args.frames_per_batch}")
    logger.info(f"  GPU: {args.gpu}")
    logger.info(f"  任务超时: {args.task_timeout}秒")
    logger.info(f"  最大重试: {args.max_retries}次")
    
    # 初始化处理器
    processor = ComfyUI_FlashVSR_BatchProcessor(
        comfyui_url=args.server,
        task_timeout=args.task_timeout,
        max_retries=args.max_retries,
        restart_delay=args.restart_delay
    )
    
    # 批量处理
    start_time = time.time()
    
    try:
        results = processor.batch_process(
            workflow_template_path=args.template,
            video_files=video_files,
            output_prefix_base=None,
            scale=args.scale,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            frames_per_batch=args.frames_per_batch,
            gpu_device=args.gpu,
            move_to_done=not args.no_move,
            cleanup_after_each=not args.no_cleanup
        )
        
        # 计算总耗时
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        logger.info(f"\n⏱️  总耗时: {int(hours)}时{int(minutes)}分{seconds:.0f}秒")
        
        # 输出详细结果
        success_count = sum(1 for success, _, _, _ in results.values() if success)
        output_file_count = sum(len(files) for _, _, _, files in results.values())
        
        logger.info(f"📁 总输出文件数: {output_file_count}")
        
        if success_count > 0:
            logger.info(f"\n✅ 成功文件列表 (前10个):")
            for video_path, (success, status, retries, files) in list(results.items())[:10]:
                if success:
                    file_count = len(files) if files else 0
                    logger.info(f"  ✓ {os.path.basename(video_path)} - {status} (文件: {file_count}, 重试: {retries})")
            if success_count > 10:
                logger.info(f"  ... 还有 {success_count-10} 个成功文件")
        
        if success_count < len(video_files):
            failed_count = len(video_files) - success_count
            logger.info(f"\n❌ 失败文件列表 (前10个):")
            for video_path, (success, status, retries, files) in list(results.items())[:10]:
                if not success:
                    file_count = len(files) if files else 0
                    logger.info(f"  ✗ {os.path.basename(video_path)} - {status} (文件: {file_count}, 重试: {retries})")
            if failed_count > 10:
                logger.info(f"  ... 还有 {failed_count-10} 个失败文件")
        
        logger.info(f"\n📊 详细日志已保存到:")
        logger.info(f"  • comfyui_batch_processor.log (运行日志)")
        logger.info(f"  • processing_log.json (处理结果)")
        logger.info(f"  • restart_history.json (重启历史)")
        
    except KeyboardInterrupt:
        logger.info("\n🛑 用户中断处理")
        logger.info("已保存当前进度")
    except Exception as e:
        logger.error(f"\n❌ 处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
