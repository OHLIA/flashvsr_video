"""
ComfyUI FlashVSR 批量处理器 v22
功能：批处理视频超分，支持断点续传、智能重启、状态监控
作者：智能视频处理助手
版本：v22 (2024-01-15)
主要改进：
1. 简化超时逻辑：只保留批次超时，去除视频总超时
2. 增强状态检查：增加重试机制，避免单次检查失败
3. 任务提交间隔：避免队列冲击
4. 详细状态追踪：区分运行队列和等待队列
5. 批次时间追踪：记录每个批次处理时间用于智能判断
6. 更稳健的任务提交和监控
"""

import os
import json
import time
import logging
import requests
import glob
import shutil
import subprocess
import random
import string
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import atexit
import gc

# 配置日志
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
    """ComfyUI API客户端 - 增强版"""
    
    def __init__(self, server_address: str = "http://127.0.0.1:8188"):
        self.server_address = server_address
        self.session = requests.Session()
        self.client_id = self.generate_client_id()
        # 添加重试机制
        self.max_retries = 3
        self.retry_delay = 2
    
    def generate_client_id(self) -> str:
        """生成客户端ID"""
        random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        return f"batch_processor_{random_str}"
    
    def is_server_running(self) -> bool:
        """检查ComfyUI服务器是否运行"""
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(f"{self.server_address}/system_stats", timeout=5)
                if response.status_code == 200:
                    return True
            except requests.exceptions.ConnectionError:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                return False
            except:
                return False
        return False
    
    def get_queue(self, max_retries: int = 3) -> List[Dict]:
        """获取队列信息（带重试）"""
        for attempt in range(max_retries):
            try:
                response = self.session.get(f"{self.server_address}/queue", timeout=10)
                if response.status_code == 200:
                    queue_data = response.json()
                    # 合并运行中和等待中的队列
                    queue_running = queue_data.get('queue_running', [])
                    queue_pending = queue_data.get('queue_pending', [])
                    return queue_running + queue_pending
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.debug(f"获取队列失败，重试 {attempt+1}/{max_retries}: {e}")
                    time.sleep(self.retry_delay)
                else:
                    logger.debug(f"获取队列最终失败: {e}")
        return []
    
    def is_queue_empty(self) -> bool:
        """检查队列是否为空"""
        return len(self.get_queue()) == 0
    
    def get_queue_load(self) -> Tuple[int, int]:
        """获取队列负载（运行中数量，等待中数量）"""
        try:
            queue_data = self.session.get(f"{self.server_address}/queue", timeout=10).json()
            running = len(queue_data.get('queue_running', []))
            pending = len(queue_data.get('queue_pending', []))
            return running, pending
        except:
            return 0, 0
    
    def wait_for_queue_available(self, max_retries: int = 10, delay: int = 5) -> bool:
        """等待队列可用（不繁忙）"""
        for attempt in range(max_retries):
            running, pending = self.get_queue_load()
            total = running + pending
            
            if total < 2:  # 队列中任务少于2个时认为可用
                logger.debug(f"✅ 队列可用: 运行中={running}, 等待中={pending}")
                return True
            
            logger.info(f"⏳ 队列繁忙: 运行中={running}, 等待中={pending}, 等待中... ({attempt+1}/{max_retries})")
            time.sleep(delay)
        
        logger.warning("⚠️  队列持续繁忙，将继续提交")
        return False
    
    def get_history(self, max_retries: int = 3) -> Dict[str, Dict]:
        """获取历史记录（带重试）"""
        for attempt in range(max_retries):
            try:
                response = self.session.get(f"{self.server_address}/history", timeout=10)
                if response.status_code == 200:
                    history_data = response.json()
                    if isinstance(history_data, dict):
                        return history_data
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.debug(f"获取历史记录失败，重试 {attempt+1}/{max_retries}: {e}")
                    time.sleep(self.retry_delay)
                else:
                    logger.debug(f"获取历史记录最终失败: {e}")
        return {}
    
    def get_prompt_by_id(self, prompt_id: str) -> Optional[Dict]:
        """根据ID获取特定任务的详细信息"""
        for attempt in range(3):
            try:
                history = self.get_history()
                if prompt_id in history:
                    return history[prompt_id]
            except Exception as e:
                if attempt < 2:
                    time.sleep(1)
                    continue
        return None
    
    def get_prompt_status(self, prompt_id: str, max_retries: int = 3) -> Optional[Dict]:
        """获取任务状态（带重试）"""
        for attempt in range(max_retries):
            try:
                # 1. 检查历史记录（已完成的任务）
                prompt_info = self.get_prompt_by_id(prompt_id)
                if prompt_info:
                    return {
                        'status': {
                            'completed': True,
                            'error': False
                        },
                        'outputs': prompt_info.get('outputs', {}),
                        'prompt_id': prompt_id
                    }
                
                # 2. 检查队列（运行中/等待中的任务）
                queue = self.get_queue()
                for item in queue:
                    if isinstance(item, dict) and item.get('prompt_id') == prompt_id:
                        # 判断是在运行队列还是等待队列
                        running_queue = self.session.get(f"{self.server_address}/queue", timeout=10).json().get('queue_running', [])
                        is_running = any(r_item.get('prompt_id') == prompt_id for r_item in running_queue)
                        
                        return {
                            'status': {
                                'completed': False,
                                'error': False,
                                'running': is_running,
                                'pending': not is_running
                            },
                            'outputs': {},
                            'prompt_id': prompt_id
                        }
                
                # 3. 如果不在历史和队列中，可能任务不存在或已失败
                return {
                    'status': {
                        'completed': False,
                        'error': True,
                        'error_message': '任务不在队列或历史中'
                    },
                    'outputs': {},
                    'prompt_id': prompt_id
                }
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.debug(f"获取任务状态失败，重试 {attempt+1}/{max_retries}: {e}")
                    time.sleep(self.retry_delay)
                else:
                    logger.debug(f"获取任务状态最终失败: {e}")
                    return {
                        'status': {
                            'completed': False,
                            'error': True,
                            'error_message': f'获取状态异常: {str(e)}'
                        },
                        'outputs': {},
                        'prompt_id': prompt_id
                    }
    
    def is_prompt_completed(self, prompt_id: str) -> bool:
        """检查任务是否完成"""
        prompt_info = self.get_prompt_status(prompt_id)
        if prompt_info and 'status' in prompt_info:
            return prompt_info['status'].get('completed', False)
        return False
    
    def get_prompt_outputs(self, prompt_id: str) -> Dict:
        """获取任务的输出信息"""
        prompt_info = self.get_prompt_status(prompt_id)
        if prompt_info:
            return prompt_info.get('outputs', {})
        return {}
    
    def submit_prompt(self, workflow: Dict) -> Optional[str]:
        """提交任务到ComfyUI"""
        for attempt in range(self.max_retries):
            try:
                # 先等待队列可用
                self.wait_for_queue_available()
                
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
                    if prompt_id:
                        logger.info(f"✅ 任务提交成功，ID: {prompt_id[:8]}...")
                        return prompt_id
                    else:
                        logger.error(f"❌ 提交任务失败: 返回数据中无prompt_id")
                else:
                    logger.error(f"❌ 提交任务失败: {response.status_code} - {response.text}")
                    
            except requests.exceptions.ConnectionError as e:
                logger.error(f"❌ 连接ComfyUI服务器失败: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * 2)
                    continue
            except requests.exceptions.Timeout as e:
                logger.error(f"❌ 连接ComfyUI服务器超时: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * 2)
                    continue
            except Exception as e:
                logger.error(f"❌ 提交任务异常: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
            
            # 所有异常都走到这里
            if attempt < self.max_retries - 1:
                logger.info(f"🔄 提交任务失败，{self.retry_delay}秒后重试...")
                time.sleep(self.retry_delay)
        
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
    
    def interrupt_queue(self) -> bool:
        """中断当前任务"""
        try:
            response = self.session.post(f"{self.server_address}/interrupt")
            return response.status_code == 200
        except:
            return False

class BatchProgressTracker:
    """批处理进度跟踪器"""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or self.get_default_output_dir()
        self.batch_progress_file = "batch_progress.json"
        self.batch_time_file = "batch_times.json"
    
    def get_default_output_dir(self) -> str:
        """获取默认输出目录"""
        comfyui_output = r"F:\AI\ComfyUI_Mie_V7.0\ComfyUI\output"
        if os.path.exists(comfyui_output):
            return comfyui_output
        
        default_output = os.path.join(os.getcwd(), "output")
        os.makedirs(default_output, exist_ok=True)
        return default_output
    
    def get_existing_batches(self, video_path: str, workflow: Dict, total_batches: int) -> Tuple[List[str], Dict[str, Any]]:
        """获取已存在的批次文件和进度信息
        返回: (文件列表, 进度信息)
        """
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_files = []
        batch_status = {}
        
        # 获取所有可能的输出文件
        for ext in ['.mov', '.mp4', '.avi', '.mkv', '.webm']:
            # 搜索批次文件
            batch_files = glob.glob(os.path.join(self.output_dir, f"*{base_name}*_*%*{ext}"))
            output_files.extend(batch_files)
            
            # 搜索没有百分比的批次文件
            simple_files = glob.glob(os.path.join(self.output_dir, f"*{base_name}*{ext}"))
            for f in simple_files:
                if f not in output_files:
                    output_files.append(f)
        
        # 分析批次状态
        if output_files:
            logger.info(f"📁 已找到 {len(output_files)} 个批次文件:")
            for i, file_path in enumerate(output_files[:10]):
                file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                file_size_mb = file_size / (1024 * 1024)
                batch_num = self.extract_batch_number(file_path)
                status = f"(批号: {batch_num})" if batch_num > 0 else ""
                logger.info(f"  {i+1}. {os.path.basename(file_path)} ({file_size_mb:.1f}MB) {status}")
            
            # 分析批次进度
            completed_batches = len(output_files)
            batch_status = {
                'total_batches': total_batches,
                'completed_batches': completed_batches,
                'remaining_batches': max(0, total_batches - completed_batches),
                'percentage': (completed_batches / total_batches) * 100 if total_batches > 0 else 0,
                'files': output_files
            }
            
            logger.info(f"📊 批次进度: {completed_batches}/{total_batches} ({batch_status['percentage']:.1f}%) 完成")
        
        return output_files, batch_status
    
    def extract_batch_number(self, file_path: str) -> int:
        """从文件名中提取批次号"""
        filename = os.path.basename(file_path)
        import re
        
        # 尝试匹配批次号
        patterns = [
            r'_batch_(\d+)',
            r'_(\d+)%',
            r'_(\d+)of',
            r'_batch(\d+)',
            r'%(\d+)%'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                try:
                    return int(match.group(1))
                except:
                    continue
        
        return -1
    
    def save_progress(self, video_path: str, progress_data: Dict):
        """保存处理进度"""
        try:
            progress = {}
            if os.path.exists(self.batch_progress_file):
                with open(self.batch_progress_file, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
            
            video_key = os.path.basename(video_path)
            progress[video_key] = {
                'video_path': video_path,
                'timestamp': datetime.now().isoformat(),
                'progress': progress_data
            }
            
            with open(self.batch_progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"✅ 进度已保存: {video_key}")
        except Exception as e:
            logger.error(f"❌ 保存进度失败: {e}")
    
    def load_progress(self, video_path: str) -> Optional[Dict]:
        """加载处理进度"""
        try:
            if os.path.exists(self.batch_progress_file):
                with open(self.batch_progress_file, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
                
                video_key = os.path.basename(video_path)
                if video_key in progress:
                    return progress[video_key]
        except Exception as e:
            logger.error(f"❌ 加载进度失败: {e}")
        return None
    
    def save_batch_time(self, video_path: str, batch_num: int, process_time: float):
        """保存批次处理时间"""
        try:
            batch_times = {}
            if os.path.exists(self.batch_time_file):
                with open(self.batch_time_file, 'r', encoding='utf-8') as f:
                    batch_times = json.load(f)
            
            video_key = os.path.basename(video_path)
            if video_key not in batch_times:
                batch_times[video_key] = {}
            
            batch_times[video_key][str(batch_num)] = process_time
            
            with open(self.batch_time_file, 'w', encoding='utf-8') as f:
                json.dump(batch_times, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"⏱️  保存批次 {batch_num} 处理时间: {process_time:.1f}秒")
        except Exception as e:
            logger.error(f"❌ 保存批次时间失败: {e}")
    
    def load_batch_times(self, video_path: str) -> Dict[int, float]:
        """加载视频的批次处理时间"""
        try:
            if os.path.exists(self.batch_time_file):
                with open(self.batch_time_file, 'r', encoding='utf-8') as f:
                    batch_times = json.load(f)
                
                video_key = os.path.basename(video_path)
                if video_key in batch_times:
                    return {int(k): v for k, v in batch_times[video_key].items()}
        except Exception as e:
            logger.error(f"❌ 加载批次时间失败: {e}")
        return {}
    
    def get_average_batch_time(self, video_path: str) -> float:
        """获取平均批次处理时间"""
        batch_times = self.load_batch_times(video_path)
        if not batch_times:
            return 0.0
        
        times = list(batch_times.values())
        return sum(times) / len(times) if times else 0.0

class BatchTimeoutManager:
    """批处理超时管理器 - 简化版"""
    
    def __init__(self, timeout_per_batch: int = 300):
        """
        初始化超时管理器
        timeout_per_batch: 每个批次的最大处理时间（秒）
        """
        self.timeout_per_batch = timeout_per_batch
        self.batch_timers = {}  # 跟踪每个批次的开始时间
        self.batch_times = {}   # 记录每个批次的实际处理时间
        self.timeout_counters = {}  # 超时计数器
    
    def start_batch_timer(self, video_path: str, batch_num: int):
        """开始批次计时器"""
        video_key = os.path.basename(video_path)
        
        if video_key not in self.batch_timers:
            self.batch_timers[video_key] = {}
            self.batch_times[video_key] = {}
            self.timeout_counters[video_key] = 0
        
        self.batch_timers[video_key][batch_num] = time.time()
        logger.debug(f"⏱️  开始批次 {batch_num} 计时器: {video_key}")
    
    def end_batch_timer(self, video_path: str, batch_num: int) -> float:
        """结束批次计时器，返回处理时间"""
        video_key = os.path.basename(video_path)
        
        if (video_key in self.batch_timers and 
            batch_num in self.batch_timers[video_key]):
            
            start_time = self.batch_timers[video_key][batch_num]
            end_time = time.time()
            process_time = end_time - start_time
            
            # 保存处理时间
            self.batch_times[video_key][batch_num] = process_time
            
            # 删除计时器
            del self.batch_timers[video_key][batch_num]
            
            logger.debug(f"⏱️  批次 {batch_num} 处理完成，耗时: {process_time:.1f}秒")
            return process_time
        
        return 0.0
    
    def check_batch_timeout(self, video_path: str, batch_num: int) -> Tuple[bool, float]:
        """检查当前批次是否超时
        返回: (是否超时, 已运行时间)
        """
        video_key = os.path.basename(video_path)
        
        if (video_key not in self.batch_timers or 
            batch_num not in self.batch_timers[video_key]):
            return False, 0.0
        
        start_time = self.batch_timers[video_key][batch_num]
        elapsed = time.time() - start_time
        
        # 如果批次处理时间超过阈值
        if elapsed > self.timeout_per_batch:
            self.timeout_counters[video_key] = self.timeout_counters.get(video_key, 0) + 1
            logger.warning(f"⚠️  批次 {batch_num} 处理超时: 已运行 {elapsed:.0f} 秒，超过 {self.timeout_per_batch} 秒")
            return True, elapsed
        
        return False, elapsed
    
    def get_expected_completion_time(self, video_path: str, current_batch: int, total_batches: int) -> float:
        """根据历史数据预测完成时间"""
        video_key = os.path.basename(video_path)
        
        if video_key not in self.batch_times or not self.batch_times[video_key]:
            return 0.0
        
        # 计算已处理批次的平均时间
        completed_times = list(self.batch_times[video_key].values())
        if not completed_times:
            return 0.0
        
        avg_time = sum(completed_times) / len(completed_times)
        
        # 预测剩余时间
        remaining_batches = total_batches - current_batch
        return avg_time * remaining_batches
    
    def get_average_batch_time(self, video_path: str) -> float:
        """获取平均批次处理时间"""
        video_key = os.path.basename(video_path)
        
        if video_key in self.batch_times and self.batch_times[video_key]:
            times = list(self.batch_times[video_key].values())
            return sum(times) / len(times)
        
        return 0.0
    
    def reset_video_timer(self, video_path: str):
        """重置视频计时器"""
        video_key = os.path.basename(video_path)
        if video_key in self.timeout_counters:
            self.timeout_counters[video_key] = 0
            logger.debug(f"🔄 重置视频计时器: {video_key}")

class ComfyUI_FlashVSR_BatchProcessor:
    def __init__(self, 
                 comfyui_url: str = "http://127.0.0.1:8188", 
                 timeout_per_batch: int = 300,  # 每个批次超时时间
                 max_retries: int = 3,
                 restart_delay: int = 5,
                 startup_timeout: int = 300,
                 min_submit_interval: int = 2):  # 最小提交间隔
        """
        初始化批量处理器 - 智能简化版
        """
        # API客户端
        self.client = ComfyUI_Client(comfyui_url)
        
        # 进度跟踪器
        self.progress_tracker = BatchProgressTracker()
        
        # 超时管理器
        self.timeout_manager = BatchTimeoutManager(
            timeout_per_batch=timeout_per_batch
        )
        
        # 配置参数
        self.comfyui_url = comfyui_url
        self.timeout_per_batch = timeout_per_batch
        self.max_retries = max_retries
        self.restart_delay = restart_delay
        self.startup_timeout = startup_timeout
        self.min_submit_interval = min_submit_interval
        
        # 状态跟踪
        self.processed_files = {}
        self.failed_files = {}
        self.restart_history = []
        self.current_retry_count = 0
        self.last_submit_time = 0
        
        # 注册清理函数
        atexit.register(self.cleanup)
        
        logger.info("=" * 60)
        logger.info("ComfyUI FlashVSR 批量处理器 v22 - 智能简化版")
        logger.info(f"ComfyUI地址: {comfyui_url}")
        logger.info(f"批次超时: {timeout_per_batch}秒")
        logger.info(f"最大重试次数: {max_retries}次")
        logger.info(f"提交间隔: {min_submit_interval}秒")
        logger.info(f"输出目录: {self.progress_tracker.output_dir}")
        logger.info("=" * 60)
    
    def ensure_comfyui_running(self) -> bool:
        """确保ComfyUI在运行"""
        if self.client.is_server_running():
            logger.info("✅ ComfyUI服务器正常运行")
            return True
        
        logger.error("❌ ComfyUI服务器未运行，请先启动ComfyUI")
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
        frames_per_batch: int = 125,
        gpu_device: str = "auto"
    ) -> Dict:
        """更新工作流参数"""
        import copy
        modified_workflow = copy.deepcopy(workflow)
        
        # 设置输入视频路径
        for node_id, node_data in modified_workflow.items():
            if node_data.get("class_type") == "VHS_LoadVideo":
                node_data["inputs"]["video"] = video_path
                logger.debug(f"设置输入视频: {os.path.basename(video_path)}")
        
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
                    logger.debug(f"设置GPU设备: {device_value}")
        
        # 设置总帧数
        if total_frames is None or total_frames <= 0:
            video_info = get_video_info(video_path)
            total_frames = video_info.get('total_frames', 0)
            if total_frames <= 0:
                total_frames = 10000
                logger.warning(f"⚠️  无法获取视频总帧数，使用默认值: {total_frames}")
        
        for node_id, node_data in modified_workflow.items():
            if node_id == "50" and node_data.get("class_type") == "PrimitiveInt":
                node_data["inputs"]["value"] = total_frames
                logger.debug(f"设置总帧数: {total_frames}")
        
        # 设置每批帧数
        for node_id, node_data in modified_workflow.items():
            if node_id == "8" and node_data.get("class_type") == "PrimitiveInt":
                node_data["inputs"]["value"] = frames_per_batch
                logger.debug(f"设置每批帧数: {frames_per_batch}")
        
        # 设置输出前缀
        if output_prefix is None:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_prefix = f"flashvsr_{base_name}"
        
        for node_id, node_data in modified_workflow.items():
            if node_data.get("class_type") == "VHS_VideoCombine":
                node_data["inputs"]["filename_prefix"] = output_prefix
                logger.debug(f"设置输出前缀: {output_prefix}")
        
        return modified_workflow
    
    def wait_for_task_completion_smart(
        self, 
        prompt_id: str, 
        video_path: str, 
        workflow: Dict,
        total_frames: int,
        frames_per_batch: int,
        total_batches: int
    ) -> Tuple[bool, str, List[str], Dict[str, Any]]:
        """
        智能等待任务完成
        返回: (是否成功, 状态信息, 输出文件列表, 进度信息)
        """
        logger.info(f"⏳ 等待任务完成 (批次超时: {self.timeout_per_batch}秒)...")
        
        video_name = os.path.basename(video_path)
        start_time = time.time()
        last_status_check = 0
        status_check_interval = 5
        last_output_check = 0
        output_check_interval = 10
        queue_empty_count = 0
        max_queue_empty = 3
        output_files_found = []
        last_output_count = 0
        no_progress_count = 0
        max_no_progress = 3
        last_queue_length = 0
        consecutive_same_queue = 0
        max_consecutive_same_queue = 6  # 30秒内队列无变化
        
        # 获取历史批次处理时间
        historical_batch_time = self.progress_tracker.get_average_batch_time(video_path)
        if historical_batch_time > 0:
            logger.info(f"⏱️  历史平均批次处理时间: {historical_batch_time:.1f}秒")
        
        # 计算预期批次数
        expected_batches = total_batches
        
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # 1. 检查当前批次超时
            current_batch = last_output_count + 1
            is_timeout, batch_elapsed = self.timeout_manager.check_batch_timeout(video_path, current_batch)
            
            if is_timeout:
                logger.warning(f"⚠️  批次 {current_batch} 处理超时: {video_name} (已运行 {batch_elapsed:.0f}秒)")
                
                # 检查是否有输出进度
                output_files, progress_info = self.progress_tracker.get_existing_batches(
                    video_path, workflow, expected_batches
                )
                
                completed = len(output_files)
                if completed > last_output_count:
                    # 有新的批次完成，重置计数器
                    last_output_count = completed
                    no_progress_count = 0
                    logger.info(f"📈 检测到新批次完成: {completed}/{expected_batches}")
                    
                    # 记录批次处理时间
                    if current_batch > 1:
                        batch_time = batch_elapsed
                        self.timeout_manager.end_batch_timer(video_path, current_batch - 1)
                        self.progress_tracker.save_batch_time(video_path, current_batch - 1, batch_time)
                        logger.info(f"⏱️  批次 {current_batch-1} 处理时间: {batch_time:.1f}秒")
                else:
                    no_progress_count += 1
                    logger.warning(f"⚠️  批次 {current_batch} 无进展: {no_progress_count}/{max_no_progress}")
                    
                    if no_progress_count >= max_no_progress:
                        logger.warning(f"⚠️  连续 {max_no_progress} 个批次无进展，需要重启")
                        
                        if completed > 0:
                            return True, f"批次无进展但有部分完成({completed}/{expected_batches})", output_files, progress_info
                        else:
                            return False, f"连续批次无进展", [], {}
            
            # 2. 定期检查输出文件
            if current_time - last_output_check >= output_check_interval:
                output_files, progress_info = self.progress_tracker.get_existing_batches(
                    video_path, workflow, expected_batches
                )
                last_output_check = current_time
                
                if output_files:
                    completed = len(output_files)
                    
                    # 更新进度
                    if completed > last_output_count:
                        logger.info(f"📈 进度更新: {completed}/{expected_batches} ({completed/expected_batches*100:.1f}%)")
                        
                        # 记录完成的批次处理时间
                        for batch_num in range(last_output_count + 1, completed + 1):
                            if batch_num > 1:  # 第一个批次从任务开始计算
                                batch_time = self.timeout_manager.end_batch_timer(video_path, batch_num - 1)
                                if batch_time > 0:
                                    self.progress_tracker.save_batch_time(video_path, batch_num - 1, batch_time)
                                    logger.debug(f"⏱️  记录批次 {batch_num-1} 处理时间: {batch_time:.1f}秒")
                        
                        # 开始下一个批次的计时
                        if completed < expected_batches:
                            self.timeout_manager.start_batch_timer(video_path, completed + 1)
                        
                        last_output_count = completed
                        no_progress_count = 0
                    
                    # 检查是否完成所有批次
                    if completed >= expected_batches:
                        logger.info(f"✅ 所有批次完成: {completed}/{expected_batches}")
                        return True, f"所有批次完成", output_files, progress_info
            
            # 3. 定期检查任务状态
            if current_time - last_status_check >= status_check_interval:
                try:
                    prompt_info = self.client.get_prompt_status(prompt_id)
                    last_status_check = current_time
                    
                    if prompt_info:
                        status = prompt_info.get('status', {})
                        
                        if status.get('completed', False):
                            logger.info(f"✅ 任务 {prompt_id[:8]}... 状态: 已完成")
                            
                            # 检查输出文件
                            time.sleep(2)
                            output_files, progress_info = self.progress_tracker.get_existing_batches(
                                video_path, workflow, expected_batches
                            )
                            
                            if output_files:
                                completed = len(output_files)
                                logger.info(f"✅ 任务完成，找到 {completed}/{expected_batches} 个输出文件")
                                return True, f"任务完成", output_files, progress_info
                            else:
                                logger.warning("⚠️  任务状态为完成但未找到输出文件")
                                
                                # 给点时间让文件写入
                                time.sleep(5)
                                output_files, progress_info = self.progress_tracker.get_existing_batches(
                                    video_path, workflow, expected_batches
                                )
                                
                                if output_files:
                                    completed = len(output_files)
                                    logger.info(f"✅ 等待后找到 {completed}/{expected_batches} 个输出文件")
                                    return True, f"任务完成(延迟发现)", output_files, progress_info
                                
                                return False, "任务完成但无输出文件", [], {}
                        
                        if status.get('error', False):
                            error_msg = status.get('error_message', '未知错误')
                            logger.error(f"❌ 任务出错: {error_msg}")
                            
                            # 检查是否有部分输出
                            output_files, progress_info = self.progress_tracker.get_existing_batches(
                                video_path, workflow, expected_batches
                            )
                            
                            if output_files:
                                completed = len(output_files)
                                logger.info(f"⚠️  任务出错但有部分输出: {completed}/{expected_batches}")
                                return True, f"任务出错但有输出({completed}/{expected_batches})", output_files, progress_info
                            
                            return False, f"任务出错: {error_msg}", [], {}
                        
                        # 显示详细状态
                        if status.get('running', False):
                            logger.debug(f"⏳ 任务状态: 运行中")
                        elif status.get('pending', False):
                            logger.debug(f"⏳ 任务状态: 等待中")
                
                except Exception as e:
                    logger.debug(f"⚠️  检查任务状态时出错: {e}")
            
            # 4. 检查队列状态
            try:
                running, pending = self.client.get_queue_load()
                queue_length = running + pending
                
                if queue_length == 0:
                    queue_empty_count += 1
                    
                    if queue_empty_count >= max_queue_empty:
                        logger.info(f"🔍 队列连续 {max_queue_empty} 次为空")
                        
                        # 检查输出文件
                        output_files, progress_info = self.progress_tracker.get_existing_batches(
                            video_path, workflow, expected_batches
                        )
                        
                        if output_files:
                            completed = len(output_files)
                            logger.info(f"📊 队列为空，已有 {completed}/{expected_batches} 个批次完成")
                            
                            if completed >= expected_batches * 0.9:  # 90%完成
                                logger.info(f"✅ 队列为空且大部分批次已完成({completed}/{expected_batches})")
                                return True, f"队列为空但完成{completed}/{expected_batches}", output_files, progress_info
                        
                        logger.warning(f"⚠️  队列持续为空但无输出文件")
                        queue_empty_count = 0
                else:
                    queue_empty_count = 0
                    
                    # 检查队列是否停滞
                    if queue_length == last_queue_length:
                        consecutive_same_queue += 1
                        if consecutive_same_queue >= max_consecutive_same_queue:
                            logger.warning(f"⚠️  队列状态连续 {max_consecutive_same_queue} 次无变化，可能停滞")
                            consecutive_same_queue = 0
                    else:
                        consecutive_same_queue = 0
                        last_queue_length = queue_length
                    
                    # 每30秒显示一次队列状态
                    if elapsed % 30 == 0:
                        avg_batch_time = self.timeout_manager.get_average_batch_time(video_path)
                        if avg_batch_time > 0 and last_output_count > 0:
                            remaining_batches = expected_batches - last_output_count
                            estimated_time = avg_batch_time * remaining_batches
                            hours = int(estimated_time // 3600)
                            minutes = int((estimated_time % 3600) // 60)
                            logger.info(f"⏳ 已处理 {int(elapsed)} 秒，队列: 运行中={running}, 等待中={pending}")
                            logger.info(f"  进度: {last_output_count}/{expected_batches}，预计剩余: {hours}时{minutes}分")
                        else:
                            logger.info(f"⏳ 已处理 {int(elapsed)} 秒，队列: 运行中={running}, 等待中={pending}")
            
            except Exception as e:
                logger.debug(f"⚠️  检查队列时出错: {e}")
            
            time.sleep(2)
    
    def restart_comfyui(self, reason: str = "未知原因") -> bool:
        """重启ComfyUI"""
        logger.info(f"🔄 开始重启ComfyUI (尝试 {self.current_retry_count + 1}/{self.max_retries})")
        logger.info(f"   重启原因: {reason}")
        
        # 记录重启
        self.record_restart(
            video_path="system",
            reason=reason,
            attempt=self.current_retry_count + 1
        )
        
        # 1. 结束现有进程
        logger.info("1. 结束现有ComfyUI进程...")
        try:
            if PSUTIL_AVAILABLE:
                killed_processes = self.kill_comfyui_processes()
                logger.info(f"   已结束 {len(killed_processes)} 个进程")
            else:
                logger.warning("⚠️  psutil不可用，无法结束进程")
        except Exception as e:
            logger.error(f"❌ 结束进程失败: {e}")
        
        # 2. 等待进程结束
        logger.info("2. 等待进程结束...")
        time.sleep(3)
        
        # 3. 启动ComfyUI
        logger.info("3. 启动ComfyUI...")
        bat_path = r"F:\AI\ComfyUI_Mie_V7.0\run_nvidia_gpu_fast_fp16_accumulation_hf_mirror.bat"
        
        if not os.path.exists(bat_path):
            logger.error(f"❌ 启动脚本不存在: {bat_path}")
            return False
        
        try:
            # 在新的命令窗口中启动
            subprocess.Popen(
                f'start cmd /k "{bat_path}"',
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            logger.info(f"🚀 启动ComfyUI: {bat_path}")
        except Exception as e:
            logger.error(f"❌ 启动ComfyUI失败: {e}")
            return False
        
        # 4. 等待ComfyUI启动
        logger.info("4. 等待ComfyUI启动...")
        wait_time = 0
        while wait_time < self.startup_timeout:
            if self.client.is_server_running():
                logger.info(f"✅ ComfyUI启动成功，等待了 {wait_time} 秒")
                return True
            
            time.sleep(5)
            wait_time += 5
            logger.debug(f"   等待ComfyUI启动... {wait_time}秒")
        
        logger.error(f"❌ ComfyUI启动超时 ({self.startup_timeout}秒)")
        return False
    
    def kill_comfyui_processes(self) -> List[int]:
        """结束ComfyUI相关进程"""
        killed_pids = []
        
        if not PSUTIL_AVAILABLE:
            return killed_pids
        
        try:
            comfyui_path = r"F:\AI\ComfyUI_Mie_V7.0"
            
            for proc in psutil.process_iter(['pid', 'name', 'exe', 'cmdline']):
                try:
                    # 检查Python进程
                    if proc.info['name'] and 'python' in proc.info['name'].lower():
                        # 检查命令行参数是否包含ComfyUI路径
                        cmdline = proc.info.get('cmdline', [])
                        if cmdline and any(comfyui_path in str(arg) for arg in cmdline):
                            logger.info(f"   🔪 结束进程: {proc.info['pid']} - {proc.info['name']}")
                            proc.terminate()
                            killed_pids.append(proc.info['pid'])
                    
                    # 检查cmd进程
                    elif proc.info['name'] and 'cmd.exe' in proc.info['name'].lower():
                        # 检查命令行是否包含ComfyUI相关
                        cmdline = proc.info.get('cmdline', [])
                        if cmdline and any(comfyui_path in str(arg) for arg in cmdline):
                            logger.info(f"   🔪 结束进程: {proc.info['pid']} - {proc.info['name']}")
                            proc.terminate()
                            killed_pids.append(proc.info['pid'])
                
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            
            # 等待进程结束
            time.sleep(2)
            
            # 强制结束未退出的进程
            for pid in killed_pids[:]:
                try:
                    proc = psutil.Process(pid)
                    if proc.is_running():
                        logger.info(f"   🔫 强制结束进程 {pid}")
                        proc.kill()
                except:
                    pass
            
            # 清理GPU显存
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("✅ GPU显存已清理")
            
        except Exception as e:
            logger.error(f"❌ 结束进程时出错: {e}")
        
        return killed_pids
    
    def wait_for_submit_interval(self):
        """等待提交间隔"""
        current_time = time.time()
        time_since_last = current_time - self.last_submit_time
        
        if time_since_last < self.min_submit_interval:
            wait_time = self.min_submit_interval - time_since_last
            logger.debug(f"⏳ 等待提交间隔: {wait_time:.1f}秒")
            time.sleep(wait_time)
        
        self.last_submit_time = time.time()
    
    def process_single_video(
        self,
        workflow_template: Dict,
        video_path: str,
        output_prefix: Optional[str] = None,
        scale: float = 4.0,
        tile_size: int = 256,
        tile_overlap: int = 24,
        frames_per_batch: int = 125,
        gpu_device: str = "auto"
    ) -> Tuple[bool, str, int, List[str]]:
        """处理单个视频 - 简化版"""
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
        if frames_per_batch <= 0:
            frames_per_batch = 125
        total_batches = (total_frames + frames_per_batch - 1) // frames_per_batch
        logger.info(f"📊 视频 '{video_name}' 需要 {total_batches} 个批次 (总帧数: {total_frames}, 每批: {frames_per_batch})")
        
        # 检查是否已有输出文件
        logger.info("🔍 检查是否已有输出文件...")
        temp_workflow = self.update_workflow_parameters(
            workflow_template, video_path, output_prefix
        )
        
        # 获取现有输出文件
        existing_files, progress_info = self.progress_tracker.get_existing_batches(
            video_path, temp_workflow, total_batches
        )
        
        completed_batches = len(existing_files)
        if completed_batches >= total_batches:
            logger.info(f"✅ 视频 '{video_name}' 已有完整输出文件 ({completed_batches}/{total_batches})，跳过处理")
            return True, f"已有完整输出文件", 0, existing_files
        elif completed_batches > 0:
            logger.info(f"📊 视频 '{video_name}' 已有 {completed_batches}/{total_batches} 个批次完成")
            
            # 如果大部分已完成，从现有文件开始
            if completed_batches >= total_batches * 0.8:  # 80%完成
                logger.info(f"✅ 视频 '{video_name}' 已有 {completed_batches}/{total_batches} 完成，跳过处理")
                return True, f"大部分已处理({completed_batches}/{total_batches})", 0, existing_files
        
        # 获取历史批次处理时间
        historical_times = self.progress_tracker.load_batch_times(video_path)
        if historical_times:
            avg_time = sum(historical_times.values()) / len(historical_times)
            logger.info(f"⏱️  历史批次处理时间: 平均 {avg_time:.1f}秒/批次")
        
        while retry_count < self.max_retries and not success:
            retry_count += 1
            logger.info(f"🔄 尝试 {retry_count}/{self.max_retries}")
            
            try:
                # 等待提交间隔
                self.wait_for_submit_interval()
                
                # 清除历史记录
                logger.debug("清除ComfyUI历史记录...")
                self.client.clear_history()
                
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
                logger.info(f"📤 提交任务: {video_name}")
                prompt_id = self.client.submit_prompt(workflow)
                
                if not prompt_id:
                    status_msg = "提交任务失败"
                    logger.error(f"❌ {status_msg}")
                    time.sleep(self.restart_delay)
                    continue
                
                logger.info(f"   任务ID: {prompt_id}")
                
                # 开始第一个批次的计时
                self.timeout_manager.start_batch_timer(video_path, completed_batches + 1)
                
                # 智能等待任务完成
                task_success, task_status, output_files, progress_info = self.wait_for_task_completion_smart(
                    prompt_id=prompt_id,
                    video_path=video_path,
                    workflow=workflow,
                    total_frames=total_frames,
                    frames_per_batch=frames_per_batch,
                    total_batches=total_batches
                )
                
                if task_success:
                    success = True
                    status_msg = task_status
                    
                    # 获取最终输出文件
                    final_files, _ = self.progress_tracker.get_existing_batches(
                        video_path, workflow, total_batches
                    )
                    
                    if final_files:
                        completed = len(final_files)
                        logger.info(f"✅ 视频处理完成，生成 {completed}/{total_batches} 个输出文件")
                        
                        for i, file_path in enumerate(final_files[:3]):
                            if os.path.exists(file_path):
                                file_size = os.path.getsize(file_path)
                                file_size_mb = file_size / (1024 * 1024)
                                logger.info(f"  {i+1}. {os.path.basename(file_path)} ({file_size_mb:.1f}MB)")
                            else:
                                logger.warning(f"  {i+1}. {os.path.basename(file_path)} (文件不存在!)")
                        
                        if len(final_files) > 3:
                            logger.info(f"  ... 还有 {len(final_files)-3} 个文件")
                        
                        output_files = final_files
                    else:
                        logger.warning("⚠️  任务成功但未找到输出文件")
                        success = False
                        status_msg = "任务成功但无输出文件"
                    
                    break
                
                else:
                    status_msg = task_status
                    logger.error(f"❌ 任务失败: {status_msg}")
                    
                    # 检查是否有部分输出文件
                    partial_files, _ = self.progress_tracker.get_existing_batches(
                        video_path, workflow, total_batches
                    )
                    
                    if partial_files:
                        completed = len(partial_files)
                        logger.info(f"📊 找到 {completed}/{total_batches} 个部分输出文件")
                        
                        # 如果有部分输出，记录进度
                        if completed > 0:
                            self.progress_tracker.save_progress(video_path, {
                                'completed_batches': completed,
                                'total_batches': total_batches,
                                'output_files': [os.path.basename(f) for f in partial_files],
                                'status': 'partial_complete',
                                'last_update': datetime.now().isoformat()
                            })
                            
                            # 记录批次处理时间
                            for batch_num in range(1, completed + 1):
                                if batch_num in self.timeout_manager.batch_timers.get(video_name, {}):
                                    batch_time = self.timeout_manager.end_batch_timer(video_path, batch_num)
                                    if batch_time > 0:
                                        self.progress_tracker.save_batch_time(video_path, batch_num, batch_time)
                    
                    # 记录重启
                    self.record_restart(
                        video_path=video_path,
                        reason=status_msg,
                        attempt=retry_count
                    )
                    
                    if retry_count < self.max_retries:
                        logger.info(f"🔄 准备重试 ({retry_count}/{self.max_retries})...")
                        
                        # 重启ComfyUI
                        restart_success = self.restart_comfyui(f"批次处理超时: {status_msg}")
                        
                        if restart_success:
                            time.sleep(self.restart_delay)
                            continue
                        else:
                            logger.error("❌ 重启ComfyUI失败")
                            break
                    else:
                        logger.error(f"❌ 达到最大重试次数")
                    
            except Exception as e:
                status_msg = f"处理异常: {str(e)}"
                logger.error(f"❌ {status_msg}", exc_info=True)
                
                # 检查是否有部分输出
                partial_files, _ = self.progress_tracker.get_existing_batches(
                    video_path, temp_workflow, total_batches
                )
                
                if partial_files:
                    completed = len(partial_files)
                    logger.info(f"⚠️  异常但已有 {completed}/{total_batches} 个输出文件")
                
                # 记录重启
                self.record_restart(
                    video_path=video_path,
                    reason=status_msg,
                    attempt=retry_count
                )
                
                if retry_count < self.max_retries:
                    logger.info(f"🔄 异常后准备重试 ({retry_count}/{self.max_retries})...")
                    
                    # 重启ComfyUI
                    restart_success = self.restart_comfyui(f"处理异常: {str(e)}")
                    
                    if restart_success:
                        time.sleep(self.restart_delay)
                        continue
                else:
                    logger.error("❌ 达到最大重试次数")
        
        # 清理进度
        if success:
            self.progress_tracker.delete_progress(video_path)
        
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
        
        # 保存到文件
        try:
            with open('restart_history.json', 'w', encoding='utf-8') as f:
                json.dump(self.restart_history, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ 重启历史已保存: restart_history.json")
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
            logger.info(f"✅ 文件已移动到done目录")
            
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
        frames_per_batch: int = 125,
        gpu_device: str = "auto",
        move_to_done: bool = True,
        cleanup_after_each: bool = True
    ) -> Dict[str, Tuple[bool, str, int, List[str]]]:
        """批量处理视频"""
        logger.info(f"\n{'='*60}")
        logger.info(f"开始批量处理 {len(video_files)} 个视频")
        logger.info(f"⚙️  参数: scale={scale}, tile_size={tile_size}, tile_overlap={tile_overlap}")
        logger.info(f"🎮 GPU设备: {gpu_device}")
        logger.info(f"⏱️  批次超时: {self.timeout_per_batch}秒")
        logger.info(f"🔄 最大重试: {self.max_retries}次")
        logger.info(f"📁 输出目录: {self.progress_tracker.output_dir}")
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
            logger.info("-" * 40)
            
            # 显示视频信息
            video_info = get_video_info(video_path)
            logger.info(f"📊 视频信息: {video_name}")
            logger.info(f"   总帧数: {video_info['total_frames']}")
            logger.info(f"   帧率: {video_info['fps']:.3f} fps")
            logger.info(f"   时长: {video_info['duration']:.1f} 秒")
            logger.info(f"   分辨率: {video_info['resolution']}")
            
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
                    'output_files': [os.path.basename(f) for f in output_files] if output_files else [],
                    'timestamp': datetime.now().isoformat()
                }
                
                # 移动文件到done目录
                if move_to_done and "跳过" not in status_msg:  # 不移动跳过的文件
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
        
        if processed_count > 0:
            logger.info(f"✅ 成功文件列表:")
            for i, (video_path, (success, status, retries, files)) in enumerate(results.items()):
                if success and i < 10:  # 只显示前10个
                    logger.info(f"  {i+1}. {os.path.basename(video_path)} - {status} (重试: {retries})")
            if processed_count > 10:
                logger.info(f"  ... 还有 {processed_count-10} 个成功文件")
        
        if failed_count > 0:
            logger.info(f"❌ 失败文件列表:")
            for i, (video_path, (success, status, retries, files)) in enumerate(results.items()):
                if not success and i < 10:  # 只显示前10个
                    logger.info(f"  {i+1}. {os.path.basename(video_path)} - {status} (重试: {retries})")
            if failed_count > 10:
                logger.info(f"  ... 还有 {failed_count-10} 个失败文件")
        
        return results
    
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
            found_files = glob.glob(search_pattern)
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
        description='ComfyUI FlashVSR 批量视频处理工具 v22 - 智能简化版',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 处理目录下的所有视频
  python batch_processor_v22.py --input ./videos --batch-timeout 300
  
  # 自定义批次大小
  python batch_processor_v22.py --input ./videos --frames-per-batch 125
  
  # 指定GPU设备
  python batch_processor_v22.py --input ./videos --gpu 0

主要改进(v22):
  1. 简化超时逻辑: 只保留批次超时，去除视频总超时
  2. 批次时间追踪: 记录每个批次的实际处理时间用于智能判断
  3. 增强状态检查: 增加重试机制，避免单次检查失败
  4. 任务提交间隔: 避免短时间大量任务冲击队列
  5. 监控队列负载: 在提交前检查队列状态
  6. 详细状态追踪: 区分运行队列和等待队列
  7. 预测完成时间: 基于历史批次处理时间预测剩余时间
        """
    )
    
    # 输入参数
    parser.add_argument('--template', type=str, default='flashvsr_tile-no.json',
                       help='工作流模板 JSON 文件路径 (默认: flashvsr_tile-no.json)')
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
    
    # 超时参数
    parser.add_argument('--batch-timeout', type=int, default=300,
                       help='每个批次的最大处理时间（秒）(默认: 300)')
    parser.add_argument('--max-retries', type=int, default=3,
                       help='最大重试次数 (默认: 3)')
    parser.add_argument('--restart-delay', type=int, default=5,
                       help='重启后等待时间（秒）(默认: 5)')
    parser.add_argument('--submit-interval', type=int, default=2,
                       help='任务提交最小间隔（秒）(默认: 2)')
    
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
    logger.info(f"📁 发现 {len(video_files)} 个视频文件:")
    for i, video_path in enumerate(video_files[:10]):  # 只显示前10个
        file_size = os.path.getsize(video_path) if os.path.exists(video_path) else 0
        file_size_mb = file_size / (1024 * 1024)
        logger.info(f"  {i+1}. {os.path.basename(video_path)} ({file_size_mb:.1f}MB)")
    
    if len(video_files) > 10:
        logger.info(f"  ... 还有 {len(video_files)-10} 个文件")
    
    # 确认处理
    user_input = input(f"\n确认处理 {len(video_files)} 个视频文件? (y/n): ").strip().lower()
    if user_input not in ['y', 'yes', '是']:
        logger.info("🚫 用户取消处理")
        return
    
    # 创建批处理器实例
    processor = ComfyUI_FlashVSR_BatchProcessor(
        comfyui_url=args.server,
        timeout_per_batch=args.batch_timeout,
        max_retries=args.max_retries,
        restart_delay=args.restart_delay,
        min_submit_interval=args.submit_interval
    )
    
    try:
        # 开始批量处理
        results = processor.batch_process(
            workflow_template_path=args.template,
            video_files=video_files,
            output_prefix_base=f"flashvsr_scale{args.scale}_tile{args.tile_size}",
            scale=args.scale,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            frames_per_batch=args.frames_per_batch,
            gpu_device=args.gpu,
            move_to_done=not args.no_move,
            cleanup_after_each=not args.no_cleanup
        )
        
        # 输出详细统计
        logger.info(f"\n{'='*60}")
        logger.info("处理统计详情")
        logger.info(f"{'='*60}")
        
        success_count = 0
        fail_count = 0
        total_retries = 0
        
        for video_path, (success, status, retries, files) in results.items():
            if success:
                success_count += 1
                total_retries += retries
                logger.info(f"✅ {os.path.basename(video_path)}: 成功 (重试: {retries})")
                logger.info(f"   状态: {status}")
                if files:
                    logger.info(f"   输出文件: {len(files)} 个")
            else:
                fail_count += 1
                total_retries += retries
                logger.info(f"❌ {os.path.basename(video_path)}: 失败 (重试: {retries})")
                logger.info(f"   状态: {status}")
        
        logger.info(f"\n📊 最终统计:")
        logger.info(f"   ✅ 成功: {success_count}/{len(video_files)} ({success_count/len(video_files)*100:.1f}%)")
        logger.info(f"   ❌ 失败: {fail_count}/{len(video_files)} ({fail_count/len(video_files)*100:.1f}%)")
        
        if success_count > 0:
            avg_retries = total_retries / success_count if success_count > 0 else 0
            logger.info(f"   🔄 平均重试次数: {avg_retries:.1f}")
        
        # 重启历史
        if processor.restart_history:
            logger.info(f"\n🔄 重启历史 ({len(processor.restart_history)} 次):")
            for i, restart in enumerate(processor.restart_history[-5:]):  # 显示最近5次
                logger.info(f"   {i+1}. {restart['timestamp'][11:19]} - {restart.get('video_name', '系统')}: {restart['reason']}")
        
        # 生成处理报告
        report_path = f"batch_processing_report_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("ComfyUI FlashVSR 批量处理报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("📊 处理统计:\n")
            f.write(f"  总文件数: {len(video_files)}\n")
            f.write(f"  成功文件: {success_count}\n")
            f.write(f"  失败文件: {fail_count}\n")
            f.write(f"  成功率: {success_count/len(video_files)*100:.1f}%\n\n")
            
            f.write("📁 文件详情:\n")
            for video_path, (success, status, retries, files) in results.items():
                f.write(f"\n📂 {os.path.basename(video_path)}:\n")
                f.write(f"   状态: {'✅ 成功' if success else '❌ 失败'}\n")
                f.write(f"   重试次数: {retries}\n")
                f.write(f"   状态信息: {status}\n")
                if files:
                    f.write(f"   输出文件 ({len(files)} 个):\n")
                    for file_path in files:
                        if os.path.exists(file_path):
                            file_size = os.path.getsize(file_path) / (1024 * 1024)
                            f.write(f"     - {os.path.basename(file_path)} ({file_size:.1f}MB)\n")
                        else:
                            f.write(f"     - {os.path.basename(file_path)} (文件不存在)\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("重启历史:\n")
            for i, restart in enumerate(processor.restart_history):
                f.write(f"\n{i+1}. 时间: {restart['timestamp']}\n")
                f.write(f"   视频: {restart.get('video_name', '系统重启')}\n")
                f.write(f"   原因: {restart['reason']}\n")
                f.write(f"   尝试: {restart['attempt']}次\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("处理参数:\n")
            f.write(f"   工作流模板: {args.template}\n")
            f.write(f"   放大倍数: {args.scale}\n")
            f.write(f"   分块大小: {args.tile_size}\n")
            f.write(f"   分块重叠: {args.tile_overlap}\n")
            f.write(f"   每批帧数: {args.frames_per_batch}\n")
            f.write(f"   GPU设备: {args.gpu}\n")
            f.write(f"   批次超时: {args.batch_timeout}秒\n")
            f.write(f"   最大重试: {args.max_retries}次\n")
            f.write(f"   ComfyUI地址: {args.server}\n")
        
        logger.info(f"\n📄 详细报告已保存: {report_path}")
        
    except KeyboardInterrupt:
        logger.info("\n🛑 用户中断处理")
    except Exception as e:
        logger.error(f"\n❌ 批量处理异常: {e}", exc_info=True)
    finally:
        processor.cleanup()

if __name__ == "__main__":
    main()
