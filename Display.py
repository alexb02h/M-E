import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import queue, tkinter as tk 
from tkinterdnd2 import DND_FILES, TkinterDnD
from tkinter import ttk
import numpy as np
from IsolateVocals import create_me_stem

from ultralytics import YOLO

root = TkinterDnD.Tk()
root.title("Test")
root.geometry("650x650")
root.configure(bg="#1e1e1e")

msg_queue = queue.Queue()

def dropped_path(event_data) :
    if event_data.startswith('{') and event_data.endswith('}') : event_data = event_data[1:-1]
    return os.path.abspath(event_data)

def handle_drop(event) :
    file_path = dropped_path(event.data)
    filename = os.path.basename(file_path)
    preview_label.config(text=f"Loaded:\n{filename}", fg="#4ade80")
    log_text.insert(tk.END, f"New File Registered: {file_path}\n")
    log_text.see(tk.END)
    create_me_stem(file_path, "output_me.wav", device="mps", hf_token="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")

label = tk.Label(root, text="Hello, Loser!", font=("Arial",16))
label.pack(pady=10)

preview = tk.Frame(root, width=320, height=220, bg="#2d2d2d")
preview.pack(pady=10)
preview.pack_propagate(False)

preview_label = tk.Label(preview, text="No File Loaded", bg="#2d2d2d", fg="#aaaaaa", font=("Arial", 12))
preview_label.pack(expand=True, fill="both")

preview.drop_target_register(DND_FILES)
preview.dnd_bind('<<Drop>>', handle_drop)

log_label = tk.Label(root, text="Process Console Output : ", font=("Arial", 11, "bold"), fg="#a0a0a0", bg="#1e1e1e")
log_label.pack(anchor="w", padx=30, pady=(10, 2))

log_frame = tk.Frame(root, bg="#1e1e1e")
log_frame.pack(fill="both", expand=True, padx=30, pady=(0, 20))

log_text = tk.Text(log_frame, bg="#0d1117", fg="#4ade80", font=("Courier", 9), relief="flat", wrap="word")
log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=log_text.yview)
log_text.configure(yscrollcommand=log_scroll.set)

log_scroll.pack(side="right", fill="y")
log_text.pack(side="left", fill="both", expand=True)

root.mainloop()