#!/usr/bin/env python3
"""
Basic usage example for Whisper Audio/Video Transcriber.

This example demonstrates the simplest way to transcribe an audio file
using the default settings.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.transcriber import WhisperTranscriber


def basic_transcription_example():
    """
    Basic transcription example with default settings.
    """
    print("🎵 Basic Whisper Transcription Example")
    print("=" * 40)
    
    # Configuration
    audio_file = "sample_audio.mp3"  # Replace with your audio file
    model_name = "turbo"  # Fast and accurate model
    language = "en"  # English language
    
    # Check if audio file exists
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        print("💡 Please place an audio file in the current directory")
        print("   Supported formats: MP3, WAV, M4A, FLAC, etc.")
        return
    
    try:
        # Initialize transcriber
        transcriber = WhisperTranscriber(
            model_name=model_name,
            language=language
        )
        
        # Process entire audio file
        print(f"🔄 Processing: {audio_file}")
        print(f"🧠 Model: {model_name}")
        print(f"🌐 Language: {language}")
        
        # Get audio duration for full processing
        from pydub import AudioSegment
        audio = AudioSegment.from_file(audio_file)
        duration_seconds = len(audio) / 1000.0
        time_ranges = [(0.0, duration_seconds)]
        
        print(f"⏱️ Duration: {duration_seconds/60:.1f} minutes")
        
        # Transcribe
        srt_path = transcriber.process_audio(
            audio_file, 
            time_ranges,
            enable_parallel=True  # Use parallel processing for speed
        )
        
        print(f"\n✅ Transcription completed!")
        print(f"📄 SRT file: {srt_path}")
        
        # Validate results
        validation = transcriber.validate_srt_quality(srt_path)
        if validation["valid"]:
            print(f"🎯 Subtitle count: {validation['subtitle_count']}")
            print(f"📊 File size: {validation['file_size_bytes']} bytes")
            print(f"⏱️ Precision: {validation['timestamp_precision']}")
        
        # Cleanup temporary files
        transcriber.cleanup_files(keep_srt=True, srt_path=srt_path)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = basic_transcription_example()
    if success:
        print("\n🎉 Example completed successfully!")
    else:
        print("\n💥 Example failed!")
        sys.exit(1) 