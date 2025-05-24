#!/usr/bin/env python3
"""
Advanced usage example for Whisper Audio/Video Transcriber.

This example demonstrates advanced features including:
- Custom time ranges
- Different models and languages
- Advanced audio processing parameters
- Video file processing
- Performance optimization
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.transcriber import WhisperTranscriber
from src.core.audio_processor import AudioProcessor
from src.core.utils import check_gpu_environment


def advanced_transcription_example():
    """
    Advanced transcription example with custom settings.
    """
    print("🚀 Advanced Whisper Transcription Example")
    print("=" * 45)
    
    # Check GPU environment
    gpu_available = check_gpu_environment()
    print("=" * 45)
    
    # Configuration
    media_file = "sample_video.mp4"  # Can be video or audio
    model_name = "large"  # High accuracy model
    language = "ja"  # Japanese language
    
    # Custom time ranges (process specific segments)
    time_ranges = [
        (0.0, 30.0),      # First 30 seconds
        (60.0, 120.0),    # 1-2 minutes
        (180.0, 300.0),   # 3-5 minutes
    ]
    
    # Advanced processing parameters
    processing_params = {
        'max_subchunk_duration_ms': 3 * 60 * 1000,  # 3 minutes per chunk
        'silence_thresh_dbfs': -35,                  # Less aggressive silence detection
        'min_silence_len_ms': 500,                   # 500ms minimum silence
        'keep_silence_ms': 200,                      # Keep 200ms silence padding
        'demucs_model_name': 'htdemucs_ft'          # High-quality vocal isolation
    }
    
    # Check if media file exists
    if not os.path.exists(media_file):
        print(f"❌ Media file not found: {media_file}")
        print("💡 Please place a video/audio file in the current directory")
        print("   Supported formats: MP4, AVI, MKV, MP3, WAV, etc.")
        return False
    
    try:
        # Initialize components
        processor = AudioProcessor(base_working_dir="./advanced_processing")
        transcriber = WhisperTranscriber(
            model_name=model_name,
            language=language,
            base_working_dir="./advanced_processing"
        )
        
        print(f"🎬 Processing: {media_file}")
        print(f"🧠 Model: {model_name} (high accuracy)")
        print(f"🌐 Language: {language}")
        print(f"📊 Time ranges: {len(time_ranges)} segments")
        print(f"🚀 GPU acceleration: {'Available' if gpu_available else 'Not available'}")
        
        # Handle video files (extract audio first)
        from src.core.utils import is_video_file
        audio_file_path = media_file
        temp_audio_file = None
        
        if is_video_file(media_file):
            print("📹 Video file detected, extracting audio...")
            
            media_dir = os.path.dirname(media_file) or "."
            media_name = os.path.splitext(os.path.basename(media_file))[0]
            temp_audio_file = os.path.join(media_dir, f"{media_name}_extracted_audio.wav")
            
            if processor.extract_audio_from_video(media_file, temp_audio_file):
                audio_file_path = temp_audio_file
                print("✅ Audio extraction successful")
            else:
                print("❌ Audio extraction failed")
                return False
        
        # Display time ranges
        print("\n⏱️ Custom time ranges:")
        for i, (start, end) in enumerate(time_ranges, 1):
            duration = end - start
            print(f"   {i}. {start:>6.1f}s - {end:>6.1f}s ({duration:>4.1f}s)")
        
        print(f"\n🔧 Advanced processing parameters:")
        for key, value in processing_params.items():
            print(f"   • {key}: {value}")
        
        # Process with advanced settings
        print(f"\n🔄 Starting advanced processing pipeline...")
        
        srt_path = transcriber.process_audio(
            audio_file_path,
            time_ranges,
            enable_parallel=True,
            **processing_params
        )
        
        print(f"\n✅ Advanced transcription completed!")
        print(f"📄 SRT file: {srt_path}")
        
        # Validate and analyze results
        validation = transcriber.validate_srt_quality(srt_path)
        if validation["valid"]:
            print(f"\n📊 Results analysis:")
            print(f"   🎯 Subtitle count: {validation['subtitle_count']}")
            print(f"   📏 File size: {validation['file_size_bytes']} bytes")
            print(f"   ⏱️ Precision: {validation['timestamp_precision']}")
            
            # Calculate average subtitles per minute
            total_duration = sum(end - start for start, end in time_ranges)
            avg_subs_per_min = validation['subtitle_count'] / (total_duration / 60)
            print(f"   📈 Density: {avg_subs_per_min:.1f} subtitles/minute")
        
        # Cleanup
        transcriber.cleanup_files(
            keep_srt=True, 
            srt_path=srt_path, 
            temp_audio_file=temp_audio_file
        )
        
        print(f"\n🎉 Advanced example completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def benchmark_models():
    """
    Benchmark different Whisper models for comparison.
    """
    print("\n🏃 Model Performance Benchmark")
    print("=" * 35)
    
    # Test file (must be short for quick benchmark)
    test_file = "short_sample.wav"  # Should be ~30 seconds
    
    if not os.path.exists(test_file):
        print(f"❌ Benchmark file not found: {test_file}")
        print("💡 Please provide a short audio file for benchmarking")
        return
    
    models_to_test = ["tiny", "base", "small", "turbo"]
    language = "en"
    
    print(f"🎵 Test file: {test_file}")
    print(f"🧪 Testing models: {', '.join(models_to_test)}")
    
    results = []
    
    for model in models_to_test:
        print(f"\n🔄 Testing {model} model...")
        
        try:
            import time
            from pydub import AudioSegment
            
            # Get audio duration
            audio = AudioSegment.from_file(test_file)
            duration = len(audio) / 1000.0
            time_ranges = [(0.0, duration)]
            
            # Initialize transcriber
            transcriber = WhisperTranscriber(
                model_name=model,
                language=language,
                base_working_dir=f"./benchmark_{model}"
            )
            
            # Measure processing time
            start_time = time.time()
            srt_path = transcriber.process_audio(test_file, time_ranges, enable_parallel=False)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            # Validate results
            validation = transcriber.validate_srt_quality(srt_path)
            
            result = {
                'model': model,
                'processing_time': processing_time,
                'subtitle_count': validation.get('subtitle_count', 0),
                'file_size': validation.get('file_size_bytes', 0),
                'valid': validation.get('valid', False)
            }
            
            results.append(result)
            
            print(f"   ⏱️ Time: {processing_time:.2f}s")
            print(f"   📊 Subtitles: {result['subtitle_count']}")
            print(f"   ✅ Status: {'Success' if result['valid'] else 'Failed'}")
            
            # Cleanup
            transcriber.cleanup_files(keep_srt=False)
            
        except Exception as e:
            print(f"   ❌ Error with {model}: {e}")
            results.append({
                'model': model,
                'processing_time': float('inf'),
                'subtitle_count': 0,
                'file_size': 0,
                'valid': False
            })
    
    # Display benchmark results
    print(f"\n📊 Benchmark Results Summary:")
    print(f"{'Model':<10} {'Time (s)':<10} {'Subtitles':<12} {'Speed':<12}")
    print("-" * 50)
    
    for result in results:
        if result['valid'] and result['processing_time'] < float('inf'):
            speed_factor = duration / result['processing_time']
            print(f"{result['model']:<10} {result['processing_time']:<10.2f} "
                  f"{result['subtitle_count']:<12} {speed_factor:<12.2f}x")
        else:
            print(f"{result['model']:<10} {'FAILED':<10} {'N/A':<12} {'N/A':<12}")


if __name__ == "__main__":
    success = advanced_transcription_example()
    
    if success:
        # Optionally run benchmark
        response = input("\n🤔 Run model benchmark? (y/N): ").lower().strip()
        if response == 'y':
            benchmark_models()
    
    if success:
        print("\n🎉 Advanced example completed!")
    else:
        print("\n💥 Advanced example failed!")
        sys.exit(1) 