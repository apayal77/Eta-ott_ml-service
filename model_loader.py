import os
import threading

_model = None
_lock = threading.Lock()

def get_whisper_model():
    global _model
    with _lock:
        if _model is None:
            print("⏳ Loading Whisper model ('base')...")
            import whisper
            # Use 'tiny' for Render Free tier compatibility (512MB RAM limit)
            model_name = os.getenv('WHISPER_MODEL', 'tiny')
            _model = whisper.load_model(model_name)
    return _model

def get_whisper_lock():
    return _lock

def clear_whisper_model():
    global _model
    with _lock:
        _model = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
