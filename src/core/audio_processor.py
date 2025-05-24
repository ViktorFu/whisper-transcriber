"""
Audio processing pipeline for Whisper Transcriber.

Handles audio extraction from video, intelligent segmentation,
vocal isolation using Demucs, and audio preprocessing.
"""

import os
import time
import shutil
import subprocess
import threading
import torch
import soundfile as sf
from pathlib import Path
from subprocess import run, CalledProcessError
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_silence
from tqdm import tqdm
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from .utils import (
    format_seconds_to_time,
    get_gpu_memory_info,
    calculate_optimal_workers,
)


class AudioProcessor:
    """
    Professional audio processing pipeline with parallel capabilities.
    
    Handles video-to-audio extraction, intelligent segmentation,
    vocal isolation, and preparation for transcription.
    """
    
    def __init__(self, base_working_dir: str = "./whisper_processing"):
        """
        Initialize AudioProcessor.
        
        Args:
            base_working_dir: Base directory for temporary files
        """
        self.base_working_dir = base_working_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def extract_audio_from_video(self, video_path: str, output_audio_path: str) -> bool:
        """
        Extract audio from video file using ffmpeg.
        
        Args:
            video_path: Path to input video file
            output_audio_path: Path for output audio file
            
        Returns:
            True if extraction successful, False otherwise
        """
        print(f"🎵 Extracting audio from video...")
        print(f"   Video file: {video_path}")
        print(f"   Audio output: {output_audio_path}")
        
        # Use ffmpeg to extract audio (16kHz, mono, WAV format)
        ffmpeg_command = [
            'ffmpeg',
            '-i', video_path,
            '-vn',  # No video stream
            '-ac', '1',  # Mono
            '-ar', '16000',  # 16kHz sample rate
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-y',  # Overwrite output file
            '-hide_banner', '-loglevel', 'error',
            output_audio_path
        ]
        
        try:
            start_time = time.time()
            run(ffmpeg_command, check=True, capture_output=True, text=True)
            end_time = time.time()
            
            if os.path.exists(output_audio_path):
                file_size = os.path.getsize(output_audio_path) / (1024 * 1024)  # MB
                print(f"✅ Audio extraction completed! Time: {end_time - start_time:.2f} seconds")
                print(f"   Audio file size: {file_size:.2f} MB")
                return True
            else:
                print("❌ Audio file not generated")
                return False
                
        except CalledProcessError as e:
            print(f"❌ Audio extraction failed:")
            print(f"   Command: {' '.join(ffmpeg_command)}")
            print(f"   Error: {e.stderr}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error during audio extraction: {e}")
            return False
            
    def initial_chunking(self, input_audio_file: str, time_ranges_sec: List[Tuple[float, float]], 
                        initial_dir: str) -> List[Tuple[float, float, str]]:
        """
        Perform initial audio chunking using ffmpeg with high precision.
        
        Args:
            input_audio_file: Path to input audio file
            time_ranges_sec: List of (start_sec, end_sec) tuples
            initial_dir: Directory for output chunks
            
        Returns:
            List of (start_sec, end_sec, chunk_path) tuples
        """
        os.makedirs(initial_dir, exist_ok=True)
        initial_chunk_metadata = []
        
        print(f"🔪 Starting initial audio chunking using ffmpeg (Output: 16kHz Mono WAV)...")
        print(f"   Saving initial chunks to: {initial_dir}")
        
        start_time_chunking = time.time()
        total_ranges = len(time_ranges_sec)
        use_progress_bar = total_ranges > 1
        
        if use_progress_bar:
            range_iterator = tqdm(enumerate(time_ranges_sec), 
                                 total=total_ranges,
                                 desc="🔪 Extracting chunks",
                                 unit="range")
        else:
            range_iterator = enumerate(time_ranges_sec)
            print(f"🔄 Extracting from single time range...")
        
        for idx, (start_sec, end_sec) in range_iterator:
            chunk_filename = f"initial_chunk_{idx+1:03d}_{start_sec:.3f}s-{end_sec:.3f}s.wav"
            chunk_path = os.path.join(initial_dir, chunk_filename)
            duration = end_sec - start_sec
            
            # High precision ffmpeg command
            ffmpeg_command = [
                'ffmpeg',
                '-i', input_audio_file,
                '-ss', f"{start_sec:.6f}",  # Microsecond precision start time
                '-t', f"{duration:.6f}",    # Microsecond precision duration
                '-vn',                      # No video stream
                '-ac', '1',                 # Mono
                '-ar', '16000',             # 16kHz sample rate
                '-acodec', 'pcm_s16le',     # PCM 16-bit
                '-avoid_negative_ts', 'make_zero',  # Avoid negative timestamps
                '-copyts',                  # Copy timestamps
                '-start_at_zero',           # Start from zero
                '-y',                       # Overwrite output file
                '-hide_banner', '-loglevel', 'error',
                chunk_path
            ]
            
            try:
                run(ffmpeg_command, check=True, capture_output=True, text=True)
                # Verify file was created successfully and has reasonable size
                if os.path.exists(chunk_path) and os.path.getsize(chunk_path) > 44:  # WAV header size
                    initial_chunk_metadata.append((start_sec, end_sec, chunk_path))
                    if not use_progress_bar:
                        chunk_duration = end_sec - start_sec
                        print(f"   ✅ Created: {format_seconds_to_time(start_sec)} → {format_seconds_to_time(end_sec)} ({chunk_duration:.1f}s)")
                else:
                    error_msg = f"⚠️ Warning: Chunk {idx+1} was created but seems empty or invalid"
                    if use_progress_bar:
                        tqdm.write(error_msg)
                    else:
                        print(error_msg)
            except CalledProcessError as e:
                error_msg = f"❌ Error processing chunk {idx+1} ({start_sec:.3f}s-{end_sec:.3f}s): ffmpeg failed"
                if use_progress_bar:
                    tqdm.write(error_msg)
                    tqdm.write(f"   Command: {' '.join(ffmpeg_command)}")
                    tqdm.write(f"   ffmpeg stderr: {e.stderr}")
                else:
                    print(f"\n{error_msg}")
                    print(f"   Command: {' '.join(ffmpeg_command)}")
                    print(f"   ffmpeg stderr: {e.stderr}")
                    print("   Skipping this chunk and continuing...")
            except Exception as e:
                error_msg = f"❌ Unexpected error processing chunk {idx+1}: {e}"
                if use_progress_bar:
                    tqdm.write(error_msg)
                else:
                    print(f"\n{error_msg}")
                    print("   Skipping this chunk and continuing...")
        
        end_time_chunking = time.time()
        print(f"✅ Finished initial chunking. Created {len(initial_chunk_metadata)} segments.")
        print(f"⏱️ Initial chunking took {end_time_chunking - start_time_chunking:.2f} seconds.")
        
        if not initial_chunk_metadata:
            raise RuntimeError("❌ No initial chunks were successfully extracted. Check ffmpeg errors above.")
        
        return initial_chunk_metadata
        
    def silence_split(self, initial_chunk_metadata: List[Tuple[float, float, str]], 
                     split_dir: str, max_subchunk_duration_ms: int = 300000,
                     min_silence_len_ms: int = 700, silence_thresh_dbfs: int = -40,
                     keep_silence_ms: int = 150) -> List[Tuple[float, float, str]]:
        """
        Perform intelligent silence-based splitting of audio chunks.
        
        Args:
            initial_chunk_metadata: List of initial chunks
            split_dir: Directory for split chunks
            max_subchunk_duration_ms: Maximum chunk duration in milliseconds
            min_silence_len_ms: Minimum silence length for splitting
            silence_thresh_dbfs: Silence threshold in dBFS
            keep_silence_ms: Silence padding to keep around segments
            
        Returns:
            List of final chunk metadata
        """
        os.makedirs(split_dir, exist_ok=True)
        final_chunk_metadata = []
        total_chunks = len(initial_chunk_metadata)
        
        print(f"🔪 Starting intelligent silence-based splitting...")
        print(f"   📊 Input chunks: {total_chunks}")
        print(f"   ⏰ Max duration per segment: {max_subchunk_duration_ms/60000:.1f} min")
        print(f"   🔇 Silence threshold: {silence_thresh_dbfs} dBFS")
        print(f"   📁 Output directory: {split_dir}")
        
        start_time_silence_split = time.time()
        sub_chunk_counter = 0
        
        # Decide whether to show progress bar (only for multiple chunks)
        use_progress_bar = total_chunks > 1
        
        if use_progress_bar:
            chunk_iterator = tqdm(enumerate(initial_chunk_metadata), 
                                 total=total_chunks,
                                 desc="🔪 Splitting chunks",
                                 unit="chunk")
        else:
            chunk_iterator = enumerate(initial_chunk_metadata)
            print(f"🔄 Processing single audio chunk...")
        
        for initial_idx, (initial_start_sec, initial_end_sec, initial_chunk_path) in chunk_iterator:
            if not os.path.exists(initial_chunk_path):
                print(f"⚠️ Skipping chunk {initial_idx+1}: File not found")
                continue
            
            try:
                audio = AudioSegment.from_wav(initial_chunk_path)
                chunk_duration_ms = len(audio)
                chunk_duration_str = format_seconds_to_time(chunk_duration_ms / 1000.0)
                
                # Display current processing time range info
                start_time_str = format_seconds_to_time(initial_start_sec)
                end_time_str = format_seconds_to_time(initial_end_sec)
                
                if not use_progress_bar:
                    print(f"   🎵 Processing: {start_time_str} → {end_time_str} (duration: {chunk_duration_str})")
                
                current_pos_ms = 0
                chunk_segments = 0
                
                while current_pos_ms < chunk_duration_ms:
                    segment_end_ms = min(current_pos_ms + max_subchunk_duration_ms, chunk_duration_ms)
                    potential_segment = audio[current_pos_ms:segment_end_ms]
                    
                    # Find optimal silence split point
                    silences = detect_silence(
                        potential_segment,
                        min_silence_len=min_silence_len_ms,
                        silence_thresh=silence_thresh_dbfs,
                        seek_step=1
                    )
                    
                    best_split_point_in_chunk_ms = -1
                    if silences:
                        for silence_start_rel, silence_end_rel in reversed(silences):
                            silence_mid_rel = silence_start_rel + (silence_end_rel - silence_start_rel) / 2
                            split_point_candidate_ms = current_pos_ms + silence_mid_rel
                            if split_point_candidate_ms > current_pos_ms + min_silence_len_ms / 2:
                                best_split_point_in_chunk_ms = int(split_point_candidate_ms)
                                break
                    
                    if best_split_point_in_chunk_ms > current_pos_ms:
                        sub_chunk_end_ms = best_split_point_in_chunk_ms
                    else:
                        sub_chunk_end_ms = segment_end_ms
                    
                    sub_chunk_end_ms = min(sub_chunk_end_ms, chunk_duration_ms)
                    if sub_chunk_end_ms <= current_pos_ms:
                        break
                    
                    # Extract audio segment
                    extract_start = max(0, current_pos_ms - keep_silence_ms)
                    extract_end = min(chunk_duration_ms, sub_chunk_end_ms + keep_silence_ms)
                    sub_chunk_audio = audio[extract_start:extract_end]
                    
                    # Calculate timestamps
                    orig_start_sec = initial_start_sec + (current_pos_ms / 1000.0)
                    orig_end_sec = initial_start_sec + (sub_chunk_end_ms / 1000.0)
                    segment_duration = orig_end_sec - orig_start_sec
                    
                    sub_chunk_counter += 1
                    chunk_segments += 1
                    sub_chunk_filename = f"final_chunk_{initial_idx+1:03d}_{sub_chunk_counter:04d}_{orig_start_sec:.0f}s-{orig_end_sec:.0f}s.wav"
                    sub_chunk_path = os.path.join(split_dir, sub_chunk_filename)
                    sub_chunk_audio.export(sub_chunk_path, format="wav")
                    final_chunk_metadata.append((orig_start_sec, orig_end_sec, sub_chunk_path))
                    
                    # Show splitting details (only for single chunk or first few)
                    if not use_progress_bar or chunk_segments <= 3:
                        seg_start_str = format_seconds_to_time(orig_start_sec)
                        seg_end_str = format_seconds_to_time(orig_end_sec)
                        print(f"     ✂️ Segment {chunk_segments}: {seg_start_str} → {seg_end_str} ({segment_duration:.1f}s)")
                    
                    current_pos_ms = sub_chunk_end_ms
                
                if not use_progress_bar:
                    print(f"   ✅ Created {chunk_segments} segments from this chunk")
                    
            except Exception as e:
                error_msg = f"❌ Error processing chunk {initial_idx+1}: {e}"
                if use_progress_bar:
                    tqdm.write(error_msg)
                else:
                    print(error_msg)
        
        end_time_silence_split = time.time()
        processing_time = end_time_silence_split - start_time_silence_split
        
        print(f"\n🎉 Silence splitting completed!")
        print(f"   📈 Input chunks: {total_chunks}")
        print(f"   📊 Output segments: {len(final_chunk_metadata)}")
        print(f"   📈 Split ratio: {len(final_chunk_metadata)/total_chunks:.1f}x")
        print(f"   ⏱️ Processing time: {processing_time:.2f} seconds")
        print(f"   🚀 Speed: {len(final_chunk_metadata)/processing_time:.1f} segments/sec")
        
        if not final_chunk_metadata:
            raise RuntimeError("❌ No final chunks available for transcription. Check errors in initial chunking or silence splitting.")
        
        return final_chunk_metadata
        
    def _process_single_demucs_chunk(self, args: Tuple) -> Tuple[float, float, str, str]:
        """
        Worker function for processing a single Demucs audio chunk.
        
        Args:
            args: Tuple of (chunk_data, demucs_dir, base_working_dir, demucs_model_name, device)
            
        Returns:
            Tuple of (orig_start_sec, orig_end_sec, processed_path, status)
        """
        chunk_data, demucs_dir, base_working_dir, demucs_model_name, device = args
        orig_start_sec, orig_end_sec, chunk_path = chunk_data
        
        if not os.path.exists(chunk_path):
            return (orig_start_sec, orig_end_sec, chunk_path, f"File not found: {chunk_path}")
        
        try:
            from demucs.api import Separator
            # Each worker creates its own separator instance (thread-safe)
            separator = Separator(model=demucs_model_name, device=device)
            
            input_chunk_p = Path(chunk_path)
            output_vocal_filename = f"vocals_{input_chunk_p.name}"
            final_vocal_path = Path(demucs_dir) / output_vocal_filename
            
            try:
                _, separated_stems = separator.separate_audio_file(str(input_chunk_p))
                vocals_tensor = separated_stems.get("vocals")
                
                if vocals_tensor is None:
                    # Fallback: copy original file
                    shutil.copy2(str(input_chunk_p), str(final_vocal_path))
                    processed_path = str(final_vocal_path)
                else:
                    vocals_data = vocals_tensor.cpu().numpy()
                    if vocals_data.ndim > 1 and vocals_data.shape[0] > 1:
                        vocals_data = vocals_data.mean(axis=0)
                    elif vocals_data.ndim > 1 and vocals_data.shape[0] == 1:
                        vocals_data = vocals_data.squeeze(0)
                    
                    target_sr = 16000
                    if separator.samplerate != target_sr:
                        temp_demucs_native_sr_path = Path(base_working_dir) / f"temp_{threading.current_thread().ident}_{output_vocal_filename}"
                        sf.write(str(temp_demucs_native_sr_path), vocals_data, int(separator.samplerate), subtype='PCM_16')
                        
                        ffmpeg_command_resample = [
                            'ffmpeg', '-i', str(temp_demucs_native_sr_path),
                            '-ar', str(target_sr), '-ac', '1', '-y',
                            '-hide_banner', '-loglevel', 'error',
                            str(final_vocal_path)
                        ]
                        
                        result = subprocess.run(ffmpeg_command_resample, capture_output=True, text=True, check=False)
                        if result.returncode != 0:
                            shutil.copy2(str(temp_demucs_native_sr_path), str(final_vocal_path))
                        
                        if os.path.exists(temp_demucs_native_sr_path):
                            os.remove(temp_demucs_native_sr_path)
                    else:
                        sf.write(str(final_vocal_path), vocals_data, target_sr, subtype='PCM_16')
                    
                    processed_path = str(final_vocal_path)
                
                # Clean up GPU memory
                if 'vocals_tensor' in locals():
                    del vocals_tensor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return (orig_start_sec, orig_end_sec, processed_path, "success")
                
            except Exception as e:
                # Error fallback to copying original file
                try:
                    fallback_path = Path(demucs_dir) / input_chunk_p.name
                    shutil.copy2(str(input_chunk_p), str(fallback_path))
                    return (orig_start_sec, orig_end_sec, str(fallback_path), f"fallback_copy: {e}")
                except Exception as e2:
                    return (orig_start_sec, orig_end_sec, chunk_path, f"failed: {e} -> {e2}")
                    
        except Exception as e:
            return (orig_start_sec, orig_end_sec, chunk_path, f"error: {e}")
            
    def demucs_vocal_isolation(self, final_chunk_metadata: List[Tuple[float, float, str]], 
                              demucs_dir: str, demucs_model_name: str = "htdemucs_ft",
                              enable_parallel: bool = True) -> List[Tuple[float, float, str]]:
        """
        Perform parallel vocal isolation using Demucs.
        
        Args:
            final_chunk_metadata: List of audio chunks to process
            demucs_dir: Directory for Demucs output
            demucs_model_name: Demucs model name
            enable_parallel: Enable parallel processing
            
        Returns:
            Updated chunk metadata with vocal-isolated files
        """
        os.makedirs(demucs_dir, exist_ok=True)
        
        if self.device == "cuda" and enable_parallel:
            print(f"🚀 Demucs will use GPU acceleration: {torch.cuda.get_device_name(0)}")
            
            # Calculate optimal parallel workers
            optimal_workers = calculate_optimal_workers()
            # Demucs is memory-intensive, be conservative
            demucs_workers = min(optimal_workers, 2)  # Max 2 parallel
            
            memory_total, memory_allocated, memory_free = get_gpu_memory_info()
            print(f"💾 Current GPU memory status: {memory_allocated:.1f}/{memory_total:.1f} GB (available: {memory_free:.1f} GB)")
            print(f"🔄 Using {demucs_workers} parallel workers for Demucs")
        else:
            demucs_workers = 1
            if not enable_parallel:
                print("🔄 User selected serial Demucs processing (more stable)")
            elif self.device == "cuda":
                print(f"🚀 Demucs will use GPU acceleration: {torch.cuda.get_device_name(0)} (serial mode)")
        
        print(f"🎤 Starting parallel vocal isolation with Demucs (Model: {demucs_model_name}, Device: {self.device.upper()})...")
        print(f"   Saving isolated vocal chunks to: {demucs_dir}")
        
        start_time_demucs = time.time()
        
        try:
            # Test Demucs model loading
            from demucs.api import Separator
            test_separator = Separator(model=demucs_model_name, device=self.device)
            print(f"✅ Demucs model '{demucs_model_name}' loaded. Sample rate: {test_separator.samplerate}")
            del test_separator  # Release test instance
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e_load:
            print(f"❌ Failed to load Demucs model: {e_load}")
            print("   Skipping Demucs processing for all chunks.")
            return final_chunk_metadata
        
        # Prepare parallel processing arguments
        worker_args = [(chunk_data, demucs_dir, self.base_working_dir, demucs_model_name, self.device) 
                       for chunk_data in final_chunk_metadata]
        
        updated_final_chunk_metadata = []
        
        total_chunks = len(worker_args)
        processed_count = 0
        
        if demucs_workers == 1:
            # Serial processing (fallback when insufficient memory)
            print(f"🔄 Processing {total_chunks} chunks sequentially...")
            
            for i, args in enumerate(worker_args, 1):
                chunk_data = args[0]  # Get chunk info
                start_time_str = format_seconds_to_time(chunk_data[0])
                end_time_str = format_seconds_to_time(chunk_data[1])
                
                print(f"   🎤 [{i}/{total_chunks}] Demucs: {start_time_str} → {end_time_str}")
                
                result = self._process_single_demucs_chunk(args)
                orig_start_sec, orig_end_sec, processed_path, status = result
                updated_final_chunk_metadata.append((orig_start_sec, orig_end_sec, processed_path))
                processed_count += 1
                
                if "error" in status or "failed" in status:
                    print(f"     ⚠️ Issue: {status}")
                else:
                    print(f"     ✅ Success")
                    
        else:
            # Parallel processing
            print(f"🚀 Processing {total_chunks} chunks with {demucs_workers} parallel workers...")
            
            with ThreadPoolExecutor(max_workers=demucs_workers) as executor:
                futures = [executor.submit(self._process_single_demucs_chunk, args) for args in worker_args]
                
                for future in tqdm(as_completed(futures), 
                                 total=len(futures), 
                                 desc=f"🎤 Demucs ({demucs_workers} workers)",
                                 unit="chunk"):
                    try:
                        orig_start_sec, orig_end_sec, processed_path, status = future.result()
                        updated_final_chunk_metadata.append((orig_start_sec, orig_end_sec, processed_path))
                        processed_count += 1
                        
                        if "error" in status or "failed" in status:
                            tqdm.write(f"⚠️ Chunk processing issue: {status}")
                            
                    except Exception as e:
                        tqdm.write(f"❌ Worker exception: {e}")
        
        # Sort results by original order
        updated_final_chunk_metadata.sort(key=lambda x: x[0])  # Sort by start_sec
        
        end_time_demucs = time.time()
        
        # Show final GPU memory status
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Final cleanup
            memory_total, memory_allocated, memory_free = get_gpu_memory_info()
            print(f"💾 Post-processing GPU memory status: {memory_allocated:.1f}/{memory_total:.1f} GB (available: {memory_free:.1f} GB)")
        
        print(f"\n✅ Finished parallel Demucs vocal isolation. Processed {len(updated_final_chunk_metadata)} chunks.")
        print(f"⏱️ Parallel Demucs processing took {end_time_demucs - start_time_demucs:.2f} seconds.")
        print(f"🚀 Speed improvement with {demucs_workers} workers!")
        
        return updated_final_chunk_metadata 