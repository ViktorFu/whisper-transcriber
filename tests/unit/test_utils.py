"""
Unit tests for utility functions.
"""

import pytest
import torch
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.core.utils import (
    parse_time_str,
    parse_time_str_ms,
    format_seconds_to_time,
    format_timestamp,
    is_video_file,
    is_audio_file,
)


class TestTimeUtils:
    """Test time parsing and formatting utilities."""
    
    def test_parse_time_str_hhmmss_format(self):
        """Test parsing HH:MM:SS format."""
        assert parse_time_str("01:23:45") == 5025.0
        assert parse_time_str("00:00:30") == 30.0
        assert parse_time_str("02:30:15") == 9015.0
    
    def test_parse_time_str_hhmmss_mmm_format(self):
        """Test parsing HH:MM:SS.mmm format."""
        assert parse_time_str("01:23:45.678") == 5025.678
        assert parse_time_str("00:00:30.500") == 30.5
        assert parse_time_str("02:30:15.123") == 9015.123
    
    def test_parse_time_str_mmss_format(self):
        """Test parsing MM:SS format."""
        assert parse_time_str("23:45") == 1425.0
        assert parse_time_str("05:30") == 330.0
        assert parse_time_str("00:15") == 15.0
    
    def test_parse_time_str_mmss_mmm_format(self):
        """Test parsing MM:SS.mmm format."""
        assert parse_time_str("23:45.678") == 1425.678
        assert parse_time_str("05:30.250") == 330.25
        assert parse_time_str("00:15.999") == 15.999
    
    def test_parse_time_str_invalid_format(self):
        """Test parsing invalid time formats."""
        with pytest.raises(ValueError):
            parse_time_str("invalid")
        
        with pytest.raises(ValueError):
            parse_time_str("1:2:3:4")
        
        with pytest.raises(ValueError):
            parse_time_str("")
    
    def test_parse_time_str_ms(self):
        """Test parsing time string to milliseconds."""
        assert parse_time_str_ms("01:23:45.678") == 5025678
        assert parse_time_str_ms("00:00:30") == 30000
        assert parse_time_str_ms("23:45") == 1425000
    
    def test_format_seconds_to_time(self):
        """Test formatting seconds to time string."""
        assert format_seconds_to_time(5025.678) == "01:23:45.678"
        assert format_seconds_to_time(30.0) == "00:00:30.000"
        assert format_seconds_to_time(1425.5) == "00:23:45.500"
    
    def test_format_timestamp(self):
        """Test formatting timestamp for SRT."""
        assert format_timestamp(5025.678) == "01:23:45,678"
        assert format_timestamp(30.0) == "00:00:30,000"
        assert format_timestamp(1425.5) == "00:23:45,500"
    
    def test_precision_round_trip(self):
        """Test that parsing and formatting maintain precision."""
        original = "01:23:45.678"
        parsed = parse_time_str(original)
        formatted = format_seconds_to_time(parsed)
        assert formatted == original


class TestFileUtils:
    """Test file type detection utilities."""
    
    def test_is_video_file(self):
        """Test video file detection."""
        video_files = [
            "test.mp4", "TEST.MP4", "video.avi", "movie.mkv",
            "clip.mov", "recording.wmv", "stream.flv", "web.webm",
            "mobile.m4v", "phone.3gp", "broadcast.ts", "disc.m2ts"
        ]
        
        for file_path in video_files:
            assert is_video_file(file_path), f"Failed to detect {file_path} as video"
    
    def test_is_audio_file(self):
        """Test audio file detection."""
        audio_files = [
            "song.mp3", "SONG.MP3", "audio.wav", "music.m4a",
            "track.flac", "voice.aac", "sound.ogg", "recording.wma"
        ]
        
        for file_path in audio_files:
            assert is_audio_file(file_path), f"Failed to detect {file_path} as audio"
    
    def test_not_video_file(self):
        """Test non-video files are not detected as video."""
        non_video_files = [
            "document.txt", "image.jpg", "song.mp3", "data.json",
            "script.py", "readme.md", "config.ini"
        ]
        
        for file_path in non_video_files:
            assert not is_video_file(file_path), f"Incorrectly detected {file_path} as video"
    
    def test_not_audio_file(self):
        """Test non-audio files are not detected as audio."""
        non_audio_files = [
            "document.txt", "image.jpg", "video.mp4", "data.json",
            "script.py", "readme.md", "config.ini"
        ]
        
        for file_path in non_audio_files:
            assert not is_audio_file(file_path), f"Incorrectly detected {file_path} as audio"


class TestGPUUtils:
    """Test GPU-related utilities."""
    
    @patch('torch.cuda.is_available')
    def test_gpu_not_available(self, mock_cuda_available):
        """Test GPU utilities when CUDA is not available."""
        mock_cuda_available.return_value = False
        
        from src.core.utils import get_gpu_memory_info, calculate_optimal_workers
        
        memory_info = get_gpu_memory_info()
        assert memory_info == (0, 0, 0)
        
        workers = calculate_optimal_workers()
        assert workers >= 1
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.memory_allocated')
    @patch('psutil.cpu_count')
    @patch('psutil.virtual_memory')
    def test_gpu_available(self, mock_virtual_memory, mock_cpu_count, 
                          mock_memory_allocated, mock_device_properties, 
                          mock_cuda_available):
        """Test GPU utilities when CUDA is available."""
        mock_cuda_available.return_value = True
        
        # Mock device properties
        mock_device = MagicMock()
        mock_device.total_memory = 8 * 1024**3  # 8GB
        mock_device_properties.return_value = mock_device
        
        # Mock memory usage
        mock_memory_allocated.return_value = 1 * 1024**3  # 1GB allocated
        
        # Mock system resources
        mock_cpu_count.return_value = 8  # 8 CPU cores
        mock_memory = MagicMock()
        mock_memory.total = 16 * 1024**3  # 16GB RAM
        mock_virtual_memory.return_value = mock_memory
        
        from src.core.utils import get_gpu_memory_info, calculate_optimal_workers
        
        memory_info = get_gpu_memory_info()
        assert memory_info[0] == 8.0  # Total GPU memory
        assert memory_info[1] == 1.0  # Allocated GPU memory
        assert memory_info[2] == 7.0  # Free GPU memory
        
        workers = calculate_optimal_workers()
        assert workers >= 1
        assert workers <= 4  # Should be capped at 4


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_parse_time_str_whitespace(self):
        """Test parsing time strings with whitespace."""
        assert parse_time_str("  01:23:45  ") == 5025.0
        assert parse_time_str("\t00:30:15\n") == 1815.0
    
    def test_parse_time_str_zero_values(self):
        """Test parsing time strings with zero values."""
        assert parse_time_str("00:00:00") == 0.0
        assert parse_time_str("00:00:00.000") == 0.0
        assert parse_time_str("00:00") == 0.0
    
    def test_format_seconds_edge_cases(self):
        """Test formatting edge case values."""
        assert format_seconds_to_time(0) == "00:00:00.000"
        assert format_seconds_to_time(0.001) == "00:00:00.001"
        assert format_seconds_to_time(3661.999) == "01:01:01.999"
    
    def test_file_extension_case_insensitive(self):
        """Test that file extension detection is case insensitive."""
        assert is_video_file("video.MP4")
        assert is_video_file("VIDEO.AVI")
        assert is_audio_file("song.MP3")
        assert is_audio_file("AUDIO.WAV") 