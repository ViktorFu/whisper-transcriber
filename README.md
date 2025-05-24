# 🎵 Whisper Audio/Video Transcriber

A professional, high-precision audio and video transcription tool powered by OpenAI Whisper, featuring advanced audio preprocessing with Demucs vocal isolation and intelligent parallel processing.

## ✨ Features

### 🎯 Core Capabilities
- **High-Precision Transcription**: Millisecond-accurate timestamps (±1ms precision)
- **Multi-Format Support**: Audio (MP3, WAV, M4A, FLAC) and Video (MP4, AVI, MKV, MOV, etc.)
- **Intelligent Audio Preprocessing**: Automatic vocal isolation using Demucs
- **Smart Segmentation**: Silence-based splitting with configurable parameters
- **GPU Acceleration**: CUDA-accelerated processing with automatic fallback

### 🚀 Performance Optimization
- **Parallel Processing**: Multi-threaded Demucs and Whisper processing
- **Memory Management**: Dynamic GPU memory monitoring and optimization
- **Resource Allocation**: Automatic calculation of optimal worker count
- **Progressive UI**: Intelligent progress indicators with detailed status

### 🎨 User Experience
- **Intuitive GUI**: Modern tkinter interface with professional styling
- **Flexible Time Ranges**: Full audio processing or custom time intervals
- **Real-time Feedback**: Detailed processing status and performance metrics
- **Clean Workflow**: Automatic cleanup with final SRT preservation

## 🛠️ Installation

### Prerequisites
- Python 3.8+ 
- CUDA-compatible GPU (recommended)
- FFmpeg installed and accessible via PATH

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/ViktorFu/whisper-transcriber.git
cd whisper-transcriber
```

2. **Create virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install PyTorch with CUDA support** (recommended)
```bash
# For CUDA 11.8
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

## 🚀 Quick Start

### Basic Usage
```python
from src.main import main

# Run with GUI
main()
```

### Command Line Interface
```bash
python -m src.main
```

### Programmatic Usage
```python
from src.core.transcriber import WhisperTranscriber
from src.core.audio_processor import AudioProcessor

# Initialize components
processor = AudioProcessor()
transcriber = WhisperTranscriber(model="turbo", language="en")

# Process audio file
audio_path = "your_audio.mp3"
time_ranges = [(0.0, 300.0)]  # First 5 minutes
srt_path = transcriber.process_audio(audio_path, time_ranges)
```

## 📋 Configuration

### Whisper Models
- `tiny`: Fastest, least accurate
- `base`: Good balance for real-time
- `small`: Better accuracy
- `medium`: High accuracy
- `large`/`large-v2`/`large-v3`: Best accuracy
- `turbo`: Optimized for speed (recommended)

### Language Codes
- `en`: English
- `zh`: Chinese
- `ja`: Japanese
- `es`: Spanish
- `fr`: French
- `de`: German
- And [many more](https://github.com/openai/whisper#available-models-and-languages)

### Advanced Parameters
```python
# Audio processing
MIN_SILENCE_LEN_MS = 700        # Minimum silence duration for splitting
SILENCE_THRESH_DBFS = -40       # Silence threshold in dBFS
MAX_CHUNK_DURATION_MIN = 5      # Maximum chunk duration in minutes
KEEP_SILENCE_MS = 150           # Silence padding around segments

# Demucs settings
DEMUCS_MODEL = "htdemucs_ft"    # Vocal isolation model

# Parallel processing
ENABLE_PARALLEL = True          # Enable multi-threading
MAX_WORKERS = 4                 # Maximum parallel workers
```

## 🎯 Use Cases

### Content Creation
- **YouTube Videos**: Generate accurate subtitles for video content
- **Podcasts**: Create searchable transcripts with precise timestamps
- **Educational Content**: Produce accessible captions for lectures

### Professional Applications
- **Meeting Transcription**: Convert recorded meetings to text
- **Interview Processing**: Transcribe interviews with speaker separation
- **Media Production**: Generate subtitle files for films and documentaries

### Accessibility
- **Hearing Impaired**: Create accurate captions for audio content
- **Language Learning**: Generate transcripts for foreign language materials
- **Research**: Extract text from audio recordings for analysis

## 🔧 Technical Architecture

### Core Components
```
src/
├── core/
│   ├── audio_processor.py    # Audio preprocessing and segmentation
│   ├── transcriber.py        # Whisper transcription engine
│   └── utils.py              # Utility functions and helpers
├── gui/
│   └── interface.py          # GUI components and user interaction
└── main.py                   # Application entry point
```

### Processing Pipeline
1. **Input Validation**: File type detection and validation
2. **Audio Extraction**: Video-to-audio conversion (if needed)
3. **Time Range Selection**: GUI-based or programmatic time selection
4. **Initial Chunking**: High-precision audio segmentation
5. **Silence Splitting**: Intelligent silence-based subdivision
6. **Vocal Isolation**: Demucs-powered vocal enhancement
7. **Parallel Transcription**: Multi-threaded Whisper processing
8. **SRT Generation**: Standards-compliant subtitle creation
9. **Cleanup**: Automatic temporary file management

## 📊 Performance Benchmarks

### GPU Acceleration (RTX 4060 8GB)
- **Processing Speed**: 2-4x faster than CPU-only
- **Memory Efficiency**: Dynamic allocation with cleanup
- **Parallel Workers**: Up to 2 concurrent Demucs + 2 Whisper

### Accuracy Metrics
- **Timestamp Precision**: ±1ms accuracy
- **Word Error Rate**: Dependent on audio quality and model
- **Vocal Isolation**: Significant improvement in multi-speaker scenarios

## 🐛 Troubleshooting

### Common Issues

**GPU Memory Error**
```bash
# Reduce parallel workers or use CPU
export CUDA_VISIBLE_DEVICES=""
```

**FFmpeg Not Found**
```bash
# Install FFmpeg
# Windows: Download from https://ffmpeg.org/
# Ubuntu: sudo apt install ffmpeg
# macOS: brew install ffmpeg
```

**Triton Warnings**
```python
# Warnings are automatically suppressed
# No action needed - functionality is preserved
```

### Performance Optimization
- Use SSD storage for temporary files
- Close other GPU-intensive applications
- Monitor system memory usage
- Use appropriate chunk sizes for your hardware

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black src/
isort src/
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - State-of-the-art speech recognition
- [Demucs](https://github.com/facebookresearch/demucs) - Audio source separation
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [FFmpeg](https://ffmpeg.org/) - Audio/video processing

## 📞 Support

- 🐛 [Issue Tracker](https://github.com/ViktorFu/whisper-transcriber/issues)
- 💬 [Discussions](https://github.com/ViktorFu/whisper-transcriber/discussions)
- 📧 [Email Support](mailto:fuwk509@gmail.com)

---

**Made with ❤️ by ViktorFu for the audio transcription community** 