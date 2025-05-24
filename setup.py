#!/usr/bin/env python3
"""
Setup script for Whisper Audio/Video Transcriber.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    requirements = []
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)
    return requirements

setup(
    name="whisper-transcriber",
    version="1.0.0",
    author="ViktorFu",
    author_email="fuwk509@gmail.com",
    description="Professional high-precision audio and video transcription tool with Whisper and Demucs",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/ViktorFu/whisper-transcriber",
    project_urls={
        "Bug Tracker": "https://github.com/ViktorFu/whisper-transcriber/issues",
        "Documentation": "https://github.com/ViktorFu/whisper-transcriber/blob/main/README.md",
        "Source Code": "https://github.com/ViktorFu/whisper-transcriber",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Environment :: GPU :: NVIDIA CUDA",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "gpu": [
            "torch>=2.1.0+cu118",
            "torchaudio>=2.1.0+cu118",
        ],
    },
    entry_points={
        "console_scripts": [
            "whisper-transcriber=src.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "whisper",
        "transcription",
        "audio",
        "video",
        "speech-to-text",
        "subtitles",
        "srt",
        "demucs",
        "gpu",
        "parallel",
    ],
) 