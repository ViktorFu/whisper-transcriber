"""
Pytest configuration and fixtures for Whisper Transcriber tests.
"""

import pytest
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_audio_file(temp_dir):
    """Create a sample audio file for testing."""
    # Create a minimal WAV file (just header, no actual audio data)
    wav_file = os.path.join(temp_dir, "test_audio.wav")
    
    # WAV file header (44 bytes)
    wav_header = bytearray([
        0x52, 0x49, 0x46, 0x46,  # "RIFF"
        0x24, 0x00, 0x00, 0x00,  # File size - 8
        0x57, 0x41, 0x56, 0x45,  # "WAVE"
        0x66, 0x6D, 0x74, 0x20,  # "fmt "
        0x10, 0x00, 0x00, 0x00,  # Subchunk size
        0x01, 0x00,              # Audio format (PCM)
        0x01, 0x00,              # Number of channels (mono)
        0x44, 0xAC, 0x00, 0x00,  # Sample rate (44100)
        0x88, 0x58, 0x01, 0x00,  # Byte rate
        0x02, 0x00,              # Block align
        0x10, 0x00,              # Bits per sample
        0x64, 0x61, 0x74, 0x61,  # "data"
        0x00, 0x00, 0x00, 0x00,  # Data size
    ])
    
    with open(wav_file, "wb") as f:
        f.write(wav_header)
    
    yield wav_file


@pytest.fixture
def mock_gpu_environment():
    """Mock GPU environment for testing."""
    with pytest.MonkeyPatch().context() as m:
        # Mock torch.cuda functions
        m.setattr("torch.cuda.is_available", lambda: True)
        m.setattr("torch.cuda.device_count", lambda: 1)
        m.setattr("torch.cuda.current_device", lambda: 0)
        m.setattr("torch.cuda.get_device_name", lambda x: "Mock GPU")
        
        # Mock device properties
        mock_device = MagicMock()
        mock_device.total_memory = 8 * 1024**3  # 8GB
        m.setattr("torch.cuda.get_device_properties", lambda x: mock_device)
        m.setattr("torch.cuda.memory_allocated", lambda x: 1 * 1024**3)  # 1GB
        
        yield


@pytest.fixture
def mock_no_gpu():
    """Mock environment without GPU for testing."""
    with pytest.MonkeyPatch().context() as m:
        m.setattr("torch.cuda.is_available", lambda: False)
        yield


@pytest.fixture
def sample_time_ranges():
    """Sample time ranges for testing."""
    return [
        (0.0, 30.0),
        (60.0, 90.0),
        (120.0, 180.0)
    ]


@pytest.fixture
def sample_srt_content():
    """Sample SRT content for testing."""
    return """1
00:00:01,000 --> 00:00:03,000
Hello, this is a test.

2
00:00:04,000 --> 00:00:06,000
This is the second subtitle.

3
00:00:07,000 --> 00:00:10,000
And this is the third one.
"""


@pytest.fixture
def mock_whisper_model():
    """Mock Whisper model for testing."""
    mock_model = MagicMock()
    mock_result = {
        "segments": [
            {
                "start": 1.0,
                "end": 3.0,
                "text": "Hello, this is a test."
            },
            {
                "start": 4.0,
                "end": 6.0,
                "text": "This is the second subtitle."
            }
        ]
    }
    mock_model.transcribe.return_value = mock_result
    return mock_model


def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Mark GPU tests
        if "gpu" in item.nodeid.lower():
            item.add_marker(pytest.mark.gpu)
        
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark slow tests
        if "benchmark" in item.nodeid.lower() or "slow" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow) 