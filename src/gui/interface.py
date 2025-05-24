"""
Graphical User Interface components for Whisper Transcriber.

Provides modern tkinter-based interfaces for file selection,
parameter configuration, and user interaction.
"""

import tkinter as tk
from tkinter import simpledialog, filedialog, messagebox, ttk
from typing import Tuple, Union, List
from ..core.utils import parse_time_str


def prompt_media_file() -> str:
    """
    Open file selection dialog for audio or video files.
    
    Returns:
        Selected file path
        
    Raises:
        RuntimeError: If no file is selected
    """
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select Audio or Video File",
        filetypes=[
            ("Media Files", "*.mp3 *.wav *.m4a *.flac *.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm *.m4v *.3gp *.ts *.m2ts"),
            ("Audio Files", "*.mp3 *.wav *.m4a *.flac *.aac *.ogg"),
            ("Video Files", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm *.m4v *.3gp *.ts *.m2ts"),
            ("All Files", "*.*")
        ]
    )
    root.destroy()
    if not path:
        raise RuntimeError("No file selected, exiting program.")
    return path


def prompt_string(title: str, prompt: str, default: str = None) -> str:
    """
    Open dialog for single-line text input.
    
    Args:
        title: Dialog window title
        prompt: Input prompt text
        default: Default value
        
    Returns:
        User input string
        
    Raises:
        RuntimeError: If no input is provided
    """
    root = tk.Tk()
    root.withdraw()
    answer = simpledialog.askstring(title=title, prompt=prompt, initialvalue=default)
    root.destroy()
    if not answer:
        raise RuntimeError(f"No input for {title}, exiting program.")
    return answer.strip()


def prompt_time_ranges() -> Tuple[Union[str, List[str]], bool]:
    """
    Modern time range input interface.
    
    Returns:
        Tuple of (time_ranges_data, enable_parallel)
        - time_ranges_data: Either "FULL_AUDIO" or list of time range strings
        - enable_parallel: Whether to enable parallel processing
    """
    root = tk.Tk()
    root.title("🎵 Whisper Time Range Configuration")
    root.geometry("580x620")
    root.resizable(False, False)
    root.configure(bg="#f8f9fa")
    
    # Set window icon (optional)
    try:
        root.iconbitmap("")  # Can set icon file
    except:
        pass
    
    # Center window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    # Store time ranges
    result = [None]
    entry_widgets = []
    
    # Create styles
    style = ttk.Style()
    style.configure("Title.TLabel", font=("Arial", 18, "bold"), background="#f8f9fa")
    style.configure("Card.TFrame", background="white", relief="solid", borderwidth=1)
    
    # Header section
    header_frame = tk.Frame(root, bg="#2196F3", height=80)
    header_frame.pack(fill=tk.X)
    header_frame.pack_propagate(False)
    
    title_label = tk.Label(header_frame, text="⏰ Time Range Configuration", 
                          font=("Arial", 18, "bold"), 
                          bg="#2196F3", fg="white")
    title_label.pack(expand=True)
    
    subtitle_label = tk.Label(header_frame, text="Configure audio transcription time ranges", 
                             font=("Arial", 10), 
                             bg="#2196F3", fg="#E3F2FD")
    subtitle_label.pack()
    
    # Main content area
    main_frame = tk.Frame(root, bg="#f8f9fa")
    main_frame.pack(fill=tk.BOTH, expand=True, padx=25, pady=(15, 5))
    
    # Full audio card
    full_audio_card = tk.Frame(main_frame, bg="white", relief="solid", borderwidth=1)
    full_audio_card.pack(fill=tk.X, pady=(0, 15))
    
    use_full_audio = tk.BooleanVar(value=True)
    
    # Full audio content
    full_audio_inner = tk.Frame(full_audio_card, bg="white")
    full_audio_inner.pack(fill=tk.X, padx=20, pady=15)
    
    # Checkbox and title
    checkbox_frame = tk.Frame(full_audio_inner, bg="white")
    checkbox_frame.pack(fill=tk.X)
    
    full_audio_cb = tk.Checkbutton(
        checkbox_frame,
        text="  📁 Process Entire Audio (Recommended)",
        variable=use_full_audio,
        font=("Arial", 12, "bold"),
        bg="white", fg="#2196F3",
        activebackground="white",
        selectcolor="#E3F2FD"
    )
    full_audio_cb.pack(side=tk.LEFT)
    
    full_audio_desc = tk.Label(full_audio_inner, 
                              text="Automatically process the entire audio file without manual time range setup",
                              font=("Arial", 9), 
                              bg="white", fg="#666")
    full_audio_desc.pack(anchor="w", pady=(5, 0))
    
    # Custom time range card
    custom_card = tk.Frame(main_frame, bg="white", relief="solid", borderwidth=1)
    custom_card.pack(fill=tk.X, pady=(0, 10))
    
    # Card title
    card_header = tk.Frame(custom_card, bg="#f5f5f5", height=40)
    card_header.pack(fill=tk.X)
    card_header.pack_propagate(False)
    
    card_title = tk.Label(card_header, text="⚙️ Custom Time Ranges", 
                         font=("Arial", 11, "bold"),
                         bg="#f5f5f5", fg="#333")
    card_title.pack(side=tk.LEFT, padx=15, pady=10)
    
    # Time range input area
    input_container = tk.Frame(custom_card, bg="white")
    input_container.pack(fill=tk.X, padx=15, pady=10)
    
    # Scrollable area
    canvas = tk.Canvas(input_container, bg="white", highlightthickness=0, height=160)
    scrollbar = ttk.Scrollbar(input_container, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas, bg="white")
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    def update_input_state():
        """Update input field states based on full audio checkbox"""
        if use_full_audio.get():
            custom_card.configure(bg="#f0f0f0")
            card_header.configure(bg="#e0e0e0")
            card_title.configure(bg="#e0e0e0", fg="#999")
            canvas.configure(state="disabled", bg="#f9f9f9")
            state = "disabled"
            bg_color = "#f9f9f9"
        else:
            custom_card.configure(bg="white")
            card_header.configure(bg="#f5f5f5")
            card_title.configure(bg="#f5f5f5", fg="#333")
            canvas.configure(state="normal", bg="white")
            state = "normal"
            bg_color = "white"
        
        for start_entry, end_entry, add_btn in entry_widgets:
            start_entry.config(state=state, bg=bg_color)
            end_entry.config(state=state, bg=bg_color)
            add_btn.config(state=state)
    
    def add_time_range_row():
        """Add a time range input row"""
        row_frame = tk.Frame(scrollable_frame, bg="white")
        row_frame.pack(fill=tk.X, pady=5, padx=5)
        
        # Create beautiful input row
        input_row = tk.Frame(row_frame, bg="#f8f9fa", relief="solid", borderwidth=1)
        input_row.pack(fill=tk.X, pady=2)
        
        # Start time
        start_frame = tk.Frame(input_row, bg="#f8f9fa")
        start_frame.pack(side=tk.LEFT, padx=10, pady=8)
        
        tk.Label(start_frame, text="Start", font=("Arial", 9, "bold"), 
                bg="#f8f9fa", fg="#555").pack()
        start_entry = tk.Entry(start_frame, width=14, font=("Consolas", 10), 
                              justify="center", relief="solid", borderwidth=1)
        start_entry.pack(pady=(2, 0))
        start_entry.insert(0, "00:00:00.000")
        
        # Arrow
        arrow_label = tk.Label(input_row, text="→", font=("Arial", 14, "bold"), 
                              bg="#f8f9fa", fg="#2196F3")
        arrow_label.pack(side=tk.LEFT, padx=5)
        
        # End time
        end_frame = tk.Frame(input_row, bg="#f8f9fa")
        end_frame.pack(side=tk.LEFT, padx=10, pady=8)
        
        tk.Label(end_frame, text="End", font=("Arial", 9, "bold"), 
                bg="#f8f9fa", fg="#555").pack()
        end_entry = tk.Entry(end_frame, width=14, font=("Consolas", 10), 
                            justify="center", relief="solid", borderwidth=1)
        end_entry.pack(pady=(2, 0))
        end_entry.insert(0, "00:05:00.000")
        
        # Add button
        add_btn = tk.Button(input_row, text="➕", width=3, height=1,
                           command=add_time_range_row, 
                           bg="#4CAF50", fg="white", 
                           font=("Arial", 11, "bold"),
                           relief="flat", cursor="hand2")
        add_btn.pack(side=tk.RIGHT, padx=10, pady=8)
        
        entry_widgets.append((start_entry, end_entry, add_btn))
        update_input_state()
        
        # Scroll to bottom
        root.after(100, lambda: canvas.yview_moveto(1.0))
    
    # Add first row
    add_time_range_row()
    
    # Format information
    info_frame = tk.Frame(input_container, bg="white")
    info_frame.pack(fill=tk.X, pady=(5, 0))
    
    info_label = tk.Label(info_frame, 
                         text="💡 Format: HH:MM:SS.mmm (e.g.: 01:23:45.678) or MM:SS (e.g.: 23:45)",
                         font=("Arial", 8), 
                         bg="white", fg="#666")
    info_label.pack()
    
    # Separator
    separator = tk.Frame(root, bg="#ddd", height=1)
    separator.pack(fill=tk.X, pady=(5, 0))
    
    # Bottom button area
    button_area = tk.Frame(root, bg="#f8f9fa", height=60)
    button_area.pack(fill=tk.X, padx=25, pady=(10, 15))
    button_area.pack_propagate(False)
    
    def confirm_input():
        """Confirm input"""
        if use_full_audio.get():
            result[0] = ("FULL_AUDIO", use_parallel.get())
            root.destroy()
            return
        
        # Collect time ranges
        ranges = []
        errors = []
        
        for i, (start_entry, end_entry, _) in enumerate(entry_widgets, 1):
            start_text = start_entry.get().strip()
            end_text = end_entry.get().strip()
            
            if not start_text and not end_text:
                continue
            
            if not start_text or not end_text:
                errors.append(f"Row {i}: Please fill in both start and end times")
                continue
            
            try:
                start_sec = parse_time_str(start_text)
                end_sec = parse_time_str(end_text)
                if start_sec >= end_sec:
                    errors.append(f"Row {i}: Start time must be less than end time")
                else:
                    ranges.append(f"{start_text}-{end_text}")
            except Exception as e:
                errors.append(f"Row {i}: Time format error - {e}")
        
        if errors:
            error_msg = "Found the following errors:\n\n" + "\n".join(errors)
            messagebox.showerror("Input Error", error_msg)
            return
        
        if not ranges:
            messagebox.showwarning("Warning", "Please enter at least one valid time range")
            return
        
        result[0] = (ranges, use_parallel.get())
        root.destroy()
    
    def cancel_input():
        """Cancel input"""
        root.destroy()
        raise RuntimeError("User cancelled time range input")
    
    # Beautiful buttons
    confirm_btn = tk.Button(button_area, text="✅ Start Processing", command=confirm_input,
                           bg="#4CAF50", fg="white", 
                           font=("Arial", 11, "bold"), 
                           width=15, height=2, relief="flat", cursor="hand2")
    confirm_btn.pack(side=tk.LEFT, pady=10)
    
    cancel_btn = tk.Button(button_area, text="❌ Cancel", command=cancel_input,
                          bg="#f44336", fg="white", 
                          font=("Arial", 11, "bold"), 
                          width=15, height=2, relief="flat", cursor="hand2")
    cancel_btn.pack(side=tk.RIGHT, pady=10)
    
    # Parallel processing option
    parallel_frame = tk.Frame(main_frame, bg="white", relief="solid", borderwidth=1)
    parallel_frame.pack(fill=tk.X, pady=(0, 15))
    
    use_parallel = tk.BooleanVar(value=True)
    
    parallel_inner = tk.Frame(parallel_frame, bg="white")
    parallel_inner.pack(fill=tk.X, padx=20, pady=15)
    
    parallel_cb_frame = tk.Frame(parallel_inner, bg="white")
    parallel_cb_frame.pack(fill=tk.X)
    
    parallel_cb = tk.Checkbutton(
        parallel_cb_frame,
        text="  🚀 Enable Parallel Processing (Recommended)",
        variable=use_parallel,
        font=("Arial", 12, "bold"),
        bg="white", fg="#FF9800",
        activebackground="white",
        selectcolor="#FFF3E0"
    )
    parallel_cb.pack(side=tk.LEFT)
    
    parallel_desc = tk.Label(parallel_inner, 
                            text="Utilize multi-core CPU and GPU for parallel processing, significantly improving speed (requires more VRAM)",
                            font=("Arial", 9), 
                            bg="white", fg="#666")
    parallel_desc.pack(anchor="w", pady=(5, 0))
    
    # Bind checkbox events
    use_full_audio.trace('w', lambda *args: update_input_state())
    
    # Initial state update
    update_input_state()
    
    # Mouse wheel support
    def on_mousewheel(event):
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    canvas.bind("<MouseWheel>", on_mousewheel)
    
    root.mainloop()
    return result[0] 