"""
Main application entry point for Whisper Audio/Video Transcriber.

This module provides the primary interface for the transcription application,
handling both GUI and CLI modes with comprehensive error handling and
intelligent file processing.
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Union
from pydub import AudioSegment

from .core.utils import (
    check_gpu_environment,
    is_video_file,
    is_audio_file,
    parse_time_str,
    validate_time_ranges_precision,
    auto_cleanup,
)
from .core.audio_processor import AudioProcessor
from .core.transcriber import WhisperTranscriber
from .gui.interface import prompt_media_file, prompt_string, prompt_time_ranges


def parse_command_line_args():
    """
    Parse command line arguments for the application.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Whisper Audio/Video Transcriber - Professional transcription tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main                              # Run with GUI
  python -m src.main --gui                        # Run with GUI (explicit)
  python -m src.main --cli                        # Run in CLI mode
  python -m src.main --input audio.mp3            # Process specific file
  python -m src.main --input video.mp4 --model turbo --language en
  
Supported formats:
  Audio: MP3, WAV, M4A, FLAC, AAC, OGG
  Video: MP4, AVI, MKV, MOV, WMV, FLV, WEBM, M4V, 3GP, TS, M2TS
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--gui', action='store_true', default=True,
                           help='Run with graphical user interface (default)')
    mode_group.add_argument('--cli', action='store_true',
                           help='Run in command line interface mode')
    
    # File input
    parser.add_argument('--input', '-i', type=str, 
                       help='Input audio or video file path')
    
    # Whisper parameters
    parser.add_argument('--model', '-m', type=str, default='turbo',
                       choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3', 'turbo'],
                       help='Whisper model to use (default: turbo)')
    parser.add_argument('--language', '-l', type=str,
                       help='Target language code (e.g., en, zh, ja) or auto-detect if not specified')
    
    # Time range parameters
    parser.add_argument('--time-ranges', '-t', type=str, nargs='+',
                       help='Time ranges to process (format: "start-end", e.g., "00:00:00-00:05:00")')
    parser.add_argument('--full-audio', '-f', action='store_true',
                       help='Process entire audio file (default if no time ranges specified)')
    
    # Processing parameters
    parser.add_argument('--parallel', '-p', action='store_true', default=True,
                       help='Enable parallel processing (default: enabled)')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel processing')
    
    # Advanced audio processing
    parser.add_argument('--max-chunk-duration', type=int, default=5,
                       help='Maximum chunk duration in minutes (default: 5)')
    parser.add_argument('--silence-threshold', type=int, default=-40,
                       help='Silence threshold in dBFS (default: -40)')
    parser.add_argument('--min-silence-length', type=int, default=700,
                       help='Minimum silence length in milliseconds (default: 700)')
    parser.add_argument('--demucs-model', type=str, default='htdemucs_ft',
                       help='Demucs model for vocal isolation (default: htdemucs_ft)')
    
    # Output and working directory
    parser.add_argument('--working-dir', '-w', type=str, default='./whisper_processing',
                       help='Working directory for temporary files (default: ./whisper_processing)')
    parser.add_argument('--output-dir', '-o', type=str,
                       help='Output directory for SRT file (default: same as input file)')
    
    # Debugging and verbosity
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    return parser.parse_args()


def setup_working_directory(base_path: str, media_path: str) -> str:
    """
    Set up working directory for processing.
    
    Args:
        base_path: Base working directory path
        media_path: Path to media file
        
    Returns:
        Complete working directory path
    """
    media_dir = os.path.dirname(media_path)
    media_name = os.path.splitext(os.path.basename(media_path))[0]
    
    if base_path == "./whisper_processing":
        # Use media file directory if default working dir
        working_dir = os.path.join(media_dir, f"whisper_processing_{media_name}")
    else:
        working_dir = base_path
        
    return working_dir


def parse_time_ranges_from_args(time_ranges_str: List[str]) -> List[Tuple[float, float]]:
    """
    Parse time ranges from command line arguments.
    
    Args:
        time_ranges_str: List of time range strings
        
    Returns:
        List of (start_sec, end_sec) tuples
    """
    time_ranges_sec = []
    
    for i, range_str in enumerate(time_ranges_str):
        try:
            if '-' not in range_str:
                raise ValueError("Time range must contain '-' separator")
                
            start_str, end_str = range_str.split('-', 1)
            start_sec = parse_time_str(start_str.strip())
            end_sec = parse_time_str(end_str.strip())
            
            if start_sec >= end_sec:
                print(f"⚠️ Skipping range {i+1} ('{range_str}'): Start time must be before end time")
                continue
                
            time_ranges_sec.append((start_sec, end_sec))
            print(f"  ✅ Range {i+1}: {start_str.strip()} -> {end_str.strip()}")
            
        except ValueError as e:
            print(f"❌ Skipping range {i+1} ('{range_str}'): {e}")
        except Exception as e:
            print(f"❌ Skipping range {i+1} ('{range_str}'): Unexpected error - {e}")
    
    return time_ranges_sec


def run_gui_mode():
    """
    Run the application in GUI mode.
    """
    print("🎵🎬 Whisper Audio/Video Transcriber (Professional Edition)")
    print("=" * 60)
    print("🎯 High-precision transcription with millisecond accuracy")
    print("💡 Full audio processing or custom time ranges supported")
    print("🚀 Parallel processing with GPU acceleration")
    print("=" * 60)
    
    # Check GPU environment
    gpu_available = check_gpu_environment()
    print("=" * 60)
    
    # Initialize variables
    media_path = None
    audio_file_path = None
    temp_audio_file = None
    working_dir = None
    final_srt_path = None
    
    try:
        # 1. File selection
        try:
            media_path = prompt_media_file()
            print(f"✅ Selected file: {os.path.basename(media_path)}")
        except Exception as e:
            print(f"❌ {e}")
            return
        
        # 2. Handle video files
        if is_video_file(media_path):
            print(f"📹 Video file detected, preparing to extract audio...")
            
            media_dir = os.path.dirname(media_path)
            media_name = os.path.splitext(os.path.basename(media_path))[0]
            temp_audio_file = os.path.join(media_dir, f"{media_name}_extracted_audio.wav")
            
            processor = AudioProcessor()
            if processor.extract_audio_from_video(media_path, temp_audio_file):
                audio_file_path = temp_audio_file
                print(f"✅ Audio extraction successful")
            else:
                print("❌ Audio extraction failed, exiting")
                return
                
        elif is_audio_file(media_path):
            print(f"🎵 Audio file detected, proceeding directly")
            audio_file_path = media_path
        else:
            print(f"❌ Unsupported file format: {os.path.splitext(media_path)[1]}")
            print("Supported formats: MP4, AVI, MKV, MOV, MP3, WAV, M4A, FLAC, etc.")
            return
        
        # 3. Get processing parameters
        whisper_model = prompt_string(
            title="Whisper Model",
            prompt="Select Whisper model (recommended: turbo):",
            default="turbo"
        )
        
        whisper_language = prompt_string(
            title="Recognition Language",
            prompt="Enter language code (en=English, zh=Chinese, ja=Japanese):",
            default="ja"
        )
        
        # 4. Configure time ranges and parallel processing
        time_ranges_result = prompt_time_ranges()
        
        if isinstance(time_ranges_result, tuple):
            time_ranges_data, enable_parallel = time_ranges_result
        else:
            time_ranges_data = time_ranges_result
            enable_parallel = True
        
        if time_ranges_data == "FULL_AUDIO":
            # Process entire audio
            print("🎵 User selected full audio processing, getting file duration...")
            audio = AudioSegment.from_file(audio_file_path)
            duration_seconds = len(audio) / 1000.0
            print(f"🕐 Total audio duration: {duration_seconds/60:.1f} minutes")
            
            time_ranges_sec = [(0.0, duration_seconds)]
            print("✅ Configured to process entire audio")
        else:
            # Manual time ranges
            time_ranges_sec = parse_time_ranges_from_args(time_ranges_data)
        
        if not time_ranges_sec:
            raise ValueError("❌ No valid time ranges defined. Cannot proceed.")
        
        # Validate time range precision
        validate_time_ranges_precision(time_ranges_sec)
        
        # 5. Set up working directory
        working_dir = setup_working_directory("./whisper_processing", media_path)
        print(f"📂 Working directory: {working_dir}")
        
        # 6. Initialize transcriber and process
        transcriber = WhisperTranscriber(
            model_name=whisper_model,
            language=whisper_language,
            base_working_dir=working_dir
        )
        
        print(f"\n🚀 Parallel processing: {'Enabled' if enable_parallel else 'Disabled'}")
        print("🔄 Starting processing pipeline...")
        
        srt_path = transcriber.process_audio(
            audio_file_path, 
            time_ranges_sec, 
            enable_parallel=enable_parallel
        )
        
        # 7. Validate and finalize SRT file
        final_srt_path = os.path.join(os.path.dirname(media_path), 
                                     f"{os.path.splitext(os.path.basename(media_path))[0]}.srt")
        
        validation = transcriber.validate_srt_quality(srt_path)
        
        if validation["valid"]:
            # Copy to final location
            if srt_path != final_srt_path:
                shutil.copy2(srt_path, final_srt_path)
            
            print(f"\n🎉 Transcription completed successfully!")
            print(f"📄 Subtitle file: {final_srt_path}")
            print(f"🎯 Subtitle count: {validation['subtitle_count']}")
            print(f"⏱️ Timestamp precision: {validation['timestamp_precision']}")
            
            # Cleanup
            transcriber.cleanup_files(keep_srt=True, srt_path=final_srt_path, temp_audio_file=temp_audio_file)
            
        else:
            print(f"❌ SRT validation failed: {validation['error']}")
            auto_cleanup(working_dir, temp_audio_file)
            
    except KeyboardInterrupt:
        print(f"\n\n⚠️ User interrupted operation, cleaning up...")
        auto_cleanup(working_dir, temp_audio_file)
        print(f"✋ Operation cancelled, temporary files cleaned up")
        
    except Exception as e:
        print(f"\n❌ Processing error: {e}")
        print(f"🧹 Cleaning up temporary files...")
        auto_cleanup(working_dir, temp_audio_file)
        print(f"💥 Processing failed, temporary files cleaned up")


def run_cli_mode(args):
    """
    Run the application in CLI mode.
    
    Args:
        args: Parsed command line arguments
    """
    print("🎵 Whisper Audio/Video Transcriber - CLI Mode")
    print("=" * 50)
    
    if not args.input:
        print("❌ Error: --input parameter is required in CLI mode")
        print("Usage: python -m src.main --cli --input <audio_file>")
        return
    
    media_path = args.input
    if not os.path.exists(media_path):
        print(f"❌ Error: Input file not found: {media_path}")
        return
    
    print(f"📁 Input file: {media_path}")
    
    # Initialize variables
    audio_file_path = None
    temp_audio_file = None
    working_dir = None
    
    try:
        # Handle video files
        if is_video_file(media_path):
            print("📹 Video file detected, extracting audio...")
            
            media_dir = os.path.dirname(media_path)
            media_name = os.path.splitext(os.path.basename(media_path))[0]
            temp_audio_file = os.path.join(media_dir, f"{media_name}_extracted_audio.wav")
            
            processor = AudioProcessor()
            if processor.extract_audio_from_video(media_path, temp_audio_file):
                audio_file_path = temp_audio_file
            else:
                print("❌ Audio extraction failed")
                return
                
        elif is_audio_file(media_path):
            audio_file_path = media_path
        else:
            print(f"❌ Unsupported file format: {os.path.splitext(media_path)[1]}")
            return
        
        # Set up time ranges
        if args.time_ranges:
            time_ranges_sec = parse_time_ranges_from_args(args.time_ranges)
        elif args.full_audio:
            audio = AudioSegment.from_file(audio_file_path)
            duration_seconds = len(audio) / 1000.0
            time_ranges_sec = [(0.0, duration_seconds)]
            print(f"🎵 Processing full audio ({duration_seconds/60:.1f} minutes)")
        else:
            # Default to full audio
            audio = AudioSegment.from_file(audio_file_path)
            duration_seconds = len(audio) / 1000.0
            time_ranges_sec = [(0.0, duration_seconds)]
            print(f"🎵 No time ranges specified, processing full audio ({duration_seconds/60:.1f} minutes)")
        
        if not time_ranges_sec:
            raise ValueError("No valid time ranges defined")
        
        # Set up working directory
        working_dir = setup_working_directory(args.working_dir, media_path)
        
        # Initialize transcriber
        enable_parallel = args.parallel and not args.no_parallel
        
        transcriber = WhisperTranscriber(
            model_name=args.model,
            language=args.language,
            base_working_dir=working_dir
        )
        
        print(f"🧠 Model: {args.model}")
        print(f"🌐 Language: {args.language or 'auto-detect'}")
        print(f"🚀 Parallel processing: {enable_parallel}")
        
        # Process audio
        srt_path = transcriber.process_audio(
            audio_file_path,
            time_ranges_sec,
            enable_parallel=enable_parallel,
            max_subchunk_duration_ms=args.max_chunk_duration * 60 * 1000,
            silence_thresh_dbfs=args.silence_threshold,
            min_silence_len_ms=args.min_silence_length,
            demucs_model_name=args.demucs_model
        )
        
        # Set up output path
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = os.path.dirname(media_path)
        
        final_srt_path = os.path.join(output_dir, 
                                     f"{os.path.splitext(os.path.basename(media_path))[0]}.srt")
        
        # Validate and save
        validation = transcriber.validate_srt_quality(srt_path)
        
        if validation["valid"]:
            if srt_path != final_srt_path:
                shutil.copy2(srt_path, final_srt_path)
            
            print(f"\n✅ Transcription completed successfully!")
            print(f"📄 Output: {final_srt_path}")
            print(f"📊 Subtitles: {validation['subtitle_count']}")
            
            transcriber.cleanup_files(keep_srt=True, srt_path=final_srt_path, temp_audio_file=temp_audio_file)
        else:
            print(f"❌ Validation failed: {validation['error']}")
            auto_cleanup(working_dir, temp_audio_file)
            
    except Exception as e:
        print(f"❌ Processing error: {e}")
        auto_cleanup(working_dir, temp_audio_file)


def main():
    """
    Main application entry point.
    """
    try:
        args = parse_command_line_args()
        
        if args.debug:
            import logging
            logging.basicConfig(level=logging.DEBUG)
        elif args.verbose:
            import logging
            logging.basicConfig(level=logging.INFO)
        
        # Determine mode
        if args.cli:
            run_cli_mode(args)
        else:
            # Default to GUI mode
            run_gui_mode()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        if '--debug' in sys.argv:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 