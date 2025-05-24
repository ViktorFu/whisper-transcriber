"""
Utility functions for audio processing and system management.

Contains helper functions for time parsing, GPU monitoring,
system resource calculation, and file type detection.
"""

import os
import time
import torch
import psutil
import warnings
from pathlib import Path
from typing import Tuple, Optional

# Suppress Triton kernel warnings
warnings.filterwarnings("ignore", message=".*Failed to launch Triton kernels.*")
warnings.filterwarnings("ignore", message=".*falling back to a slower.*")
warnings.filterwarnings("ignore", message=".*due to missing CUDA toolkit.*")
warnings.filterwarnings("ignore", category=UserWarning, module="whisper.timing")


def parse_time_str(t_str: str) -> float:
    """
    Convert time string to precise seconds (float).
    
    Supported formats:
    - HH:MM:SS.mmm (e.g.: 01:23:45.678)
    - HH:MM:SS     (e.g.: 01:23:45)
    - MM:SS.mmm    (e.g.: 23:45.678)
    - MM:SS        (e.g.: 23:45)
    
    Args:
        t_str: Time string to parse
        
    Returns:
        Time in seconds as float
        
    Raises:
        ValueError: If time format is invalid
    """
    t_str = t_str.strip()
    
    # Handle decimal part
    if '.' in t_str:
        time_part, decimal_part = t_str.split('.', 1)
        decimal_seconds = float('0.' + decimal_part)
    else:
        time_part = t_str
        decimal_seconds = 0.0
    
    # Parse hours:minutes:seconds part
    parts = time_part.split(':')
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h = 0
        m, s = parts
    else:
        raise ValueError(f"Invalid time format: {t_str!r}")
    
    try:
        total_seconds = float(h) * 3600 + float(m) * 60 + float(s) + decimal_seconds
        return round(total_seconds, 3)  # Preserve millisecond precision
    except ValueError:
        raise ValueError(f"Invalid time format: {t_str!r}")


def parse_time_str_ms(t_str: str) -> int:
    """
    Convert time string to precise milliseconds (int).
    
    Args:
        t_str: Time string to parse
        
    Returns:
        Time in milliseconds as integer
    """
    return int(round(parse_time_str(t_str) * 1000))


def format_seconds_to_time(seconds: float) -> str:
    """
    Convert seconds to HH:MM:SS.mmm format string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3600000
    total_ms %= 3600000
    minutes = total_ms // 60000
    total_ms %= 60000
    secs = total_ms // 1000
    ms = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format (HH:MM:SS,mmm).
    Maintains high precision to avoid cumulative errors.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        SRT-formatted timestamp string
    """
    # Use more precise calculation method
    total_ms = round(seconds * 1000)
    h = total_ms // 3600000
    total_ms %= 3600000
    m = total_ms // 60000
    total_ms %= 60000
    s = total_ms // 1000
    ms = total_ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def get_gpu_memory_info() -> Tuple[float, float, float]:
    """
    Get GPU memory information.
    
    Returns:
        Tuple of (total_memory_gb, allocated_memory_gb, free_memory_gb)
        Returns (0, 0, 0) if CUDA is not available
    """
    if torch.cuda.is_available():
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        memory_free = memory_total - memory_allocated
        return memory_total, memory_allocated, memory_free
    return 0, 0, 0


def calculate_optimal_workers() -> int:
    """
    Calculate optimal number of parallel workers based on system resources.
    
    Returns:
        Optimal number of workers for parallel processing
    """
    cpu_count = psutil.cpu_count(logical=False)  # Physical cores
    memory_gb = psutil.virtual_memory().total / 1024**3
    
    if torch.cuda.is_available():
        gpu_memory_total, _, gpu_memory_free = get_gpu_memory_info()
        
        # Limit parallel workers based on GPU memory (1.5GB per task estimated)
        gpu_workers = max(1, int(gpu_memory_free / 1.5))
        
        # Limit based on CPU cores (reserve 1-2 cores for system)
        cpu_workers = max(1, cpu_count - 1)
        
        # Limit based on system memory (2GB per task estimated)
        memory_workers = max(1, int(memory_gb / 2))
        
        # Take minimum as safe parallel count
        optimal_workers = min(gpu_workers, cpu_workers, memory_workers, 4)  # Max 4 parallel
        
        return optimal_workers
    else:
        return max(1, cpu_count // 2)


def check_gpu_environment() -> bool:
    """
    Check GPU environment configuration and display status.
    
    Returns:
        True if GPU is available and working, False otherwise
    """
    print("🔍 Checking GPU environment...")
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        memory_total, memory_allocated, memory_free = get_gpu_memory_info()
        
        print(f"✅ CUDA available - version: {torch.version.cuda}")
        print(f"🎮 GPU device: {device_name}")
        print(f"💾 Memory: {memory_allocated:.1f} / {memory_total:.1f} GB (available: {memory_free:.1f} GB)")
        print(f"📊 Device count: {device_count}")
        
        # Calculate optimal parallel workers
        optimal_workers = calculate_optimal_workers()
        print(f"🚀 Recommended parallel workers: {optimal_workers}")
        
        # Test GPU computation
        try:
            test_tensor = torch.randn(100, 100).cuda()
            _ = test_tensor @ test_tensor.T
            print(f"✅ GPU computation test passed")
            return True
        except Exception as e:
            print(f"⚠️ GPU computation test failed: {e}")
            return False
    else:
        print(f"❌ CUDA not available")
        print(f"📋 PyTorch version: {torch.__version__}")
        print(f"💡 Consider installing CUDA-enabled PyTorch version")
        return False


def is_video_file(file_path: str) -> bool:
    """
    Check if file is a video file based on extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file is a video file
    """
    video_extensions = {
        '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', 
        '.webm', '.m4v', '.3gp', '.ts', '.m2ts', '.mpg', '.mpeg'
    }
    file_ext = os.path.splitext(file_path)[1].lower()
    return file_ext in video_extensions


def is_audio_file(file_path: str) -> bool:
    """
    Check if file is an audio file based on extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file is an audio file
    """
    audio_extensions = {
        '.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma'
    }
    file_ext = os.path.splitext(file_path)[1].lower()
    return file_ext in audio_extensions


def validate_time_ranges_precision(time_ranges_sec: list) -> None:
    """
    Validate time ranges precision and provide suggestions.
    
    Args:
        time_ranges_sec: List of (start_sec, end_sec) tuples
    """
    print("\n🔍 Time range precision analysis:")
    for i, (start_sec, end_sec) in enumerate(time_ranges_sec, 1):
        duration = end_sec - start_sec
        print(f"  Range {i}: {format_seconds_to_time(start_sec)} -> {format_seconds_to_time(end_sec)} (duration: {duration:.3f}s)")
        
        # Check for millisecond precision
        start_ms = (start_sec * 1000) % 1000
        end_ms = (end_sec * 1000) % 1000
        if start_ms != 0 or end_ms != 0:
            print(f"    ✅ Millisecond precision detected - optimal subtitle sync expected")
        else:
            print(f"    ⚠️ Consider using millisecond precision (e.g.: {format_seconds_to_time(start_sec)}) for better subtitle sync")


def auto_cleanup(working_dir: Optional[str], temp_audio_file: Optional[str] = None) -> None:
    """
    Automatic cleanup function - removes working directory and temporary files.
    
    Args:
        working_dir: Working directory to clean up
        temp_audio_file: Temporary audio file to remove
    """
    import shutil
    
    cleanup_items = []
    
    # Clean up working directory
    if working_dir and os.path.exists(working_dir):
        try:
            shutil.rmtree(working_dir)
            cleanup_items.append(f"Working directory: {working_dir}")
        except Exception as e:
            print(f"⚠️ Failed to clean up working directory: {e}")
    
    # Clean up temporary audio file
    if temp_audio_file and os.path.exists(temp_audio_file):
        try:
            os.remove(temp_audio_file)
            cleanup_items.append(f"Temporary audio: {os.path.basename(temp_audio_file)}")
        except Exception as e:
            print(f"⚠️ Failed to clean up temporary audio: {e}")
    
    if cleanup_items:
        print(f"🧹 Auto-cleaned: {', '.join(cleanup_items)}") 