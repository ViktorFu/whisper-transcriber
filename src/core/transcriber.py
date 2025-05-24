"""
Whisper transcription engine with parallel processing capabilities.

Handles speech-to-text transcription using OpenAI Whisper with
intelligent parallel processing and SRT subtitle generation.
"""

import os
import time
import torch
import whisper
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .utils import (
    format_seconds_to_time,
    format_timestamp,
    get_gpu_memory_info,
    calculate_optimal_workers,
)


class WhisperTranscriber:
    """
    Professional Whisper transcription engine with parallel processing.
    
    Handles speech-to-text conversion with GPU acceleration,
    parallel processing, and high-precision SRT generation.
    """
    
    def __init__(self, model_name: str = "turbo", language: Optional[str] = None,
                 base_working_dir: str = "./whisper_processing"):
        """
        Initialize WhisperTranscriber.
        
        Args:
            model_name: Whisper model name (tiny, base, small, medium, large, turbo)
            language: Target language code (e.g., 'en', 'zh', 'ja') or None for auto-detect
            base_working_dir: Base directory for temporary files
        """
        self.model_name = model_name
        self.language = language
        self.base_working_dir = base_working_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Model size estimation for memory calculation (GB)
        self.model_size_map = {
            "tiny": 0.5, "base": 0.7, "small": 1.5, 
            "medium": 3.0, "large": 6.0, "large-v2": 6.0, 
            "large-v3": 6.0, "turbo": 1.0
        }
        
    def _process_single_whisper_chunk(self, args: Tuple) -> List[Tuple[float, float, str]]:
        """
        Worker function for processing a single Whisper audio chunk.
        
        Args:
            args: Tuple of (chunk_data, model_name, language, device)
            
        Returns:
            List of segment tuples (start_sec, end_sec, text)
        """
        chunk_data, model_name, language, device = args
        chunk_original_start_sec, chunk_original_end_sec, chunk_path = chunk_data
        
        if not os.path.exists(chunk_path):
            return []
        
        try:
            # Each worker loads its own model instance
            model = whisper.load_model(model_name, device=device)
            
            result = model.transcribe(
                chunk_path,
                language=language or None,
                fp16=torch.cuda.is_available(),
                word_timestamps=True,
            )
            
            chunk_segments = []
            for seg in result.get("segments", []):
                seg_start_relative = float(seg['start'])
                seg_end_relative = float(seg['end'])
                
                seg_start_orig = seg_start_relative + float(chunk_original_start_sec)
                seg_end_orig = seg_end_relative + float(chunk_original_start_sec)
                
                text = seg.get("text", "").strip()
                if text:
                    seg_start_rounded = round(seg_start_orig, 3)
                    seg_end_rounded = round(seg_end_orig, 3)
                    
                    if seg_end_rounded > seg_start_rounded:
                        chunk_segments.append((seg_start_rounded, seg_end_rounded, text))
            
            # Clean up model and GPU memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return chunk_segments
            
        except Exception as e:
            print(f"⚠️ Error processing chunk {chunk_path}: {e}")
            return []
            
    def transcribe_and_generate_srt(self, final_chunk_metadata: List[Tuple[float, float, str]], 
                                   input_audio_file: str, enable_parallel: bool = True) -> str:
        """
        Perform parallel Whisper transcription and generate SRT file.
        
        Args:
            final_chunk_metadata: List of audio chunks to transcribe
            input_audio_file: Original input audio file path
            enable_parallel: Enable parallel processing
            
        Returns:
            Path to generated SRT file
        """
        output_dir = os.path.join(self.base_working_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate Whisper's optimal parallel workers (more conservative due to multiple model instances)
        if self.device == "cuda" and enable_parallel:
            memory_total, memory_allocated, memory_free = get_gpu_memory_info()
            
            estimated_model_size = self.model_size_map.get(self.model_name, 3.0)
            
            # Calculate number of parallel models based on GPU memory
            max_parallel_models = max(1, int(memory_free / (estimated_model_size * 1.2)))  # 1.2x safety factor
            whisper_workers = min(max_parallel_models, 2)  # Max 2 parallel (conservative)
            
            print(f"💾 GPU memory status: {memory_allocated:.1f}/{memory_total:.1f} GB (available: {memory_free:.1f} GB)")
            print(f"🧠 Whisper model estimated size: {estimated_model_size:.1f} GB")
            print(f"🔄 Using {whisper_workers} parallel workers for Whisper")
            
            if whisper_workers == 1:
                print("⚠️ Limited GPU memory, using serial processing")
        else:
            whisper_workers = 1
            if not enable_parallel:
                print("🔄 User selected serial Whisper processing (more stable)")
            elif self.device == "cuda":
                print(f"🚀 Whisper will use GPU acceleration: {torch.cuda.get_device_name(0)} (serial mode)")
        
        print(f"🧠 Starting parallel Whisper transcription (Model: {self.model_name}, Language: {self.language or 'Auto-Detect'})...")
        
        start_time_transcription = time.time()
        
        # Test model loading
        print(f"🔍 Testing Whisper model loading...")
        try:
            test_model = whisper.load_model(self.model_name, device=self.device)
            print(f"✅ Whisper model '{self.model_name}' loaded successfully")
            del test_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"❌ Failed to load Whisper model: {e}")
            raise
        
        all_segments = []
        
        total_chunks = len(final_chunk_metadata)
        
        if whisper_workers == 1:
            # Serial processing
            print(f"🖋️ Starting serial Whisper transcription of {total_chunks} chunks...")
            
            model = whisper.load_model(self.model_name, device=self.device)
            
            for i, chunk_data in enumerate(final_chunk_metadata, 1):
                chunk_original_start_sec, chunk_original_end_sec, chunk_path = chunk_data
                
                if not os.path.exists(chunk_path):
                    continue
                
                # Display current processing time range
                start_time_str = format_seconds_to_time(chunk_original_start_sec)
                end_time_str = format_seconds_to_time(chunk_original_end_sec)
                print(f"   🧠 [{i}/{total_chunks}] Whisper: {start_time_str} → {end_time_str}")
                
                try:
                    result = model.transcribe(
                        chunk_path,
                        language=self.language or None,
                        fp16=torch.cuda.is_available(),
                        word_timestamps=True,
                    )
                    
                    chunk_segments_count = 0
                    for seg in result.get("segments", []):
                        seg_start_relative = float(seg['start'])
                        seg_end_relative = float(seg['end'])
                        
                        seg_start_orig = seg_start_relative + float(chunk_original_start_sec)
                        seg_end_orig = seg_end_relative + float(chunk_original_start_sec)
                        
                        text = seg.get("text", "").strip()
                        if text:
                            seg_start_rounded = round(seg_start_orig, 3)
                            seg_end_rounded = round(seg_end_orig, 3)
                            
                            if seg_end_rounded > seg_start_rounded:
                                all_segments.append((seg_start_rounded, seg_end_rounded, text))
                                chunk_segments_count += 1
                    
                    print(f"     ✅ Generated {chunk_segments_count} text segments")
                                
                except Exception as e:
                    print(f"     ⚠️ Error: {e}")
            
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        else:
            # Parallel processing
            print(f"🚀 Starting parallel Whisper transcription with {whisper_workers} workers...")
            
            worker_args = [(chunk_data, self.model_name, self.language, self.device) 
                           for chunk_data in final_chunk_metadata]
            
            with ThreadPoolExecutor(max_workers=whisper_workers) as executor:
                futures = [executor.submit(self._process_single_whisper_chunk, args) for args in worker_args]
                
                for future in tqdm(as_completed(futures), 
                                 total=len(futures), 
                                 desc=f"🧠 Whisper ({whisper_workers} workers)",
                                 unit="chunk"):
                    try:
                        chunk_segments = future.result()
                        all_segments.extend(chunk_segments)
                    except Exception as e:
                        tqdm.write(f"⚠️ Worker exception: {e}")
        
        # Sort all segments by timestamp
        all_segments.sort(key=lambda x: x[0])
        
        # Generate SRT content
        srt_lines = []
        srt_counter = 1
        
        for seg_start_rounded, seg_end_rounded, text in all_segments:
            srt_start_ts = format_timestamp(seg_start_rounded)
            srt_end_ts = format_timestamp(seg_end_rounded)
            
            srt_lines.extend([
                str(srt_counter),
                f"{srt_start_ts} --> {srt_end_ts}",
                text,
                ""
            ])
            srt_counter += 1
            
            # Show first few segments for debugging
            if srt_counter <= 4:
                print(f"  ✓ Segment {srt_counter-1}: {srt_start_ts} - {srt_end_ts}")
        
        end_time_transcription = time.time()
        
        # Show final GPU status
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_total, memory_allocated, memory_free = get_gpu_memory_info()
            print(f"💾 Post-transcription GPU memory status: {memory_allocated:.1f}/{memory_total:.1f} GB (available: {memory_free:.1f} GB)")
        
        print(f"\n✅ Finished parallel Whisper transcription. Generated {len(all_segments)} segments.")
        print(f"⏱️ Parallel transcription took {end_time_transcription - start_time_transcription:.2f} seconds.")
        print(f"🚀 Speed improvement with {whisper_workers} workers!")
        
        # Save SRT file
        srt_filename = f"output_{os.path.splitext(os.path.basename(input_audio_file))[0]}_{self.model_name}_{self.language or 'auto'}.srt"
        srt_path = os.path.join(output_dir, srt_filename)
        
        if srt_lines:
            print(f"\n💾 Saving SRT file to: {srt_path}")
            try:
                with open(srt_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(srt_lines))
                print("✅ SRT file saved successfully.")
                return srt_path
            except Exception as e:
                print(f"❌ Error saving SRT file: {e}")
                raise
        else:
            print("\n⚠️ No text was transcribed. SRT file will not be created.")
            raise RuntimeError("No text segments were generated from transcription.")
            
    def process_audio(self, audio_path: str, time_ranges: List[Tuple[float, float]], 
                     enable_parallel: bool = True, **kwargs) -> str:
        """
        Complete audio processing pipeline from raw audio to SRT.
        
        Args:
            audio_path: Path to input audio file
            time_ranges: List of (start_sec, end_sec) tuples to process
            enable_parallel: Enable parallel processing
            **kwargs: Additional parameters for audio processing
            
        Returns:
            Path to generated SRT file
        """
        from .audio_processor import AudioProcessor
        
        # Initialize audio processor
        processor = AudioProcessor(self.base_working_dir)
        
        # Set up directories
        initial_dir = os.path.join(self.base_working_dir, "01_chunks_initial_16khz")
        split_dir = os.path.join(self.base_working_dir, "02_chunks_split_by_silence")
        demucs_dir = os.path.join(self.base_working_dir, "03_chunks_vocals_demucs")
        
        # Audio processing parameters
        max_subchunk_duration_ms = kwargs.get('max_subchunk_duration_ms', 5 * 60 * 1000)  # 5 minutes
        min_silence_len_ms = kwargs.get('min_silence_len_ms', 700)
        silence_thresh_dbfs = kwargs.get('silence_thresh_dbfs', -40)
        keep_silence_ms = kwargs.get('keep_silence_ms', 150)
        demucs_model_name = kwargs.get('demucs_model_name', "htdemucs_ft")
        
        print(f"📂 Working directory: {self.base_working_dir}")
        
        # Execute processing pipeline
        print("\n🔄 Starting processing pipeline...")
        
        # 1. Initial chunking
        initial_chunk_metadata = processor.initial_chunking(audio_path, time_ranges, initial_dir)
        
        # 2. Silence-based splitting
        final_chunk_metadata = processor.silence_split(
            initial_chunk_metadata, split_dir, max_subchunk_duration_ms,
            min_silence_len_ms, silence_thresh_dbfs, keep_silence_ms
        )
        
        # 3. Vocal isolation with Demucs
        final_chunk_metadata = processor.demucs_vocal_isolation(
            final_chunk_metadata, demucs_dir, demucs_model_name, enable_parallel
        )
        
        # 4. Whisper transcription and SRT generation
        srt_path = self.transcribe_and_generate_srt(
            final_chunk_metadata, audio_path, enable_parallel
        )
        
        return srt_path
        
    def cleanup_files(self, keep_srt: bool = True, srt_path: Optional[str] = None, 
                     temp_audio_file: Optional[str] = None) -> None:
        """
        Clean up temporary files and directories.
        
        Args:
            keep_srt: Whether to keep the final SRT file
            srt_path: Path to SRT file to preserve
            temp_audio_file: Temporary audio file to clean up
        """
        print("\n--- Final Cleanup ---")
        
        cleanup_items = []
        
        # Handle SRT file
        if srt_path and os.path.exists(srt_path) and keep_srt:
            print(f"⬇️ SRT file ready: {os.path.basename(srt_path)}")
            print(f"✅ File saved: {srt_path}")
        elif srt_path:
            print("⚠️ SRT file not found or keeping disabled")
            
        # Clean up working directory completely
        if os.path.exists(self.base_working_dir):
            try:
                # Remove entire working directory and all its contents
                shutil.rmtree(self.base_working_dir)
                cleanup_items.append(f"Removed working directory: {os.path.basename(self.base_working_dir)}")
            except Exception as e:
                print(f"⚠️ Error removing working directory {self.base_working_dir}: {e}")
                # Fallback: try to remove subdirectories individually
                for subdir in ["01_chunks_initial_16khz", "02_chunks_split_by_silence", "03_chunks_vocals_demucs", "output"]:
                    subdir_path = os.path.join(self.base_working_dir, subdir)
                    if os.path.exists(subdir_path):
                        try:
                            shutil.rmtree(subdir_path)
                            cleanup_items.append(f"Removed: {os.path.basename(subdir_path)}")
                        except Exception as e2:
                            print(f"  - Error removing {os.path.basename(subdir_path)}: {e2}")
                
                # Try to remove the now-empty working directory
                try:
                    if os.path.exists(self.base_working_dir) and not os.listdir(self.base_working_dir):
                        os.rmdir(self.base_working_dir)
                        cleanup_items.append(f"Removed empty working directory: {os.path.basename(self.base_working_dir)}")
                except Exception as e3:
                    print(f"  - Could not remove working directory: {e3}")
                        
        # Clean up temporary audio file
        if temp_audio_file and os.path.exists(temp_audio_file):
            try:
                os.remove(temp_audio_file)
                cleanup_items.append(f"Removed temp audio: {os.path.basename(temp_audio_file)}")
            except Exception as e:
                print(f"⚠️ Failed to remove temporary audio: {e}")
                
        if cleanup_items:
            print(f"🧹 Cleanup completed:")
            for item in cleanup_items:
                print(f"  - {item}")
        else:
            print("🧹 No cleanup needed")
            
        print("✅ Processing pipeline completed successfully!")
        
    def validate_srt_quality(self, srt_path: str) -> Dict[str, Any]:
        """
        Validate the quality of generated SRT file.
        
        Args:
            srt_path: Path to SRT file
            
        Returns:
            Dictionary with validation results
        """
        if not os.path.exists(srt_path):
            return {"valid": False, "error": "SRT file not found"}
            
        try:
            with open(srt_path, 'r', encoding='utf-8') as f:
                srt_content = f.read().strip()
                
            if not srt_content:
                return {"valid": False, "error": "SRT file is empty"}
                
            # Count subtitle entries
            subtitle_count = srt_content.count('-->')
            
            if subtitle_count == 0:
                return {"valid": False, "error": "No valid subtitle entries found"}
                
            # Check timestamp format
            import re
            timestamp_pattern = r'\d{2}:\d{2}:\d{2},\d{3}'
            timestamps = re.findall(timestamp_pattern, srt_content)
            
            if len(timestamps) < subtitle_count * 2:
                return {"valid": False, "error": "Invalid timestamp format"}
                
            return {
                "valid": True,
                "subtitle_count": subtitle_count,
                "timestamp_precision": "millisecond",
                "file_size_bytes": os.path.getsize(srt_path)
            }
            
        except Exception as e:
            return {"valid": False, "error": f"Error reading SRT file: {e}"} 