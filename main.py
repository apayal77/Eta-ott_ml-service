import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from dotenv import load_dotenv
# Extractors are now lazy-loaded inside functions

load_dotenv()

app = FastAPI(title="Eta ML Service", description="AI-powered data extraction service")

@app.on_event("startup")
async def startup_event():
    import sys
    print(f"🚀 Eta ML Service starting up on Python {sys.version}...")
    print(f"ℹ️ Working directory: {os.getcwd()}")
    print("ℹ️ Models will be lazy-loaded on first request to conserve memory.")
    print("ℹ️ Playwright browser check skipped at startup (handled in build phase).")

class ExtractionRequest(BaseModel):
    file_url: str
    content_id: str
    content_type: str  # 'pdf', 'video', 'youtube', etc.

class ExtractionResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None

class OCRRequest(BaseModel):
    image_base64: Optional[str] = None
    video_url: Optional[str] = None
    timestamp: Optional[float] = 0
    crop: Optional[dict] = None # {x, y, w, h} as 0-1 relative values
    content_type: str = 'image' # 'image' or 'video'

@app.get("/")
async def root():
    return {"status": "online", "message": "Eta ML Service is running"}

# Lightweight Embedding Model (Shared across app)
_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        print("⏳ Loading Embedding model (all-MiniLM-L6-v2)...")
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embed_model

class EmbeddingRequest(BaseModel):
    text: str

@app.post("/embeddings")
def get_embeddings(request: EmbeddingRequest):
    try:
        model = get_embed_model()
        embedding = model.encode(request.text).tolist()
        return {"success": True, "embedding": embedding}
    except Exception as e:
        return {"success": False, "error": str(e)}

# OCR Reader (Lazy-loaded)
_ocr_reader = None

def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr
            print(f"⏳ Loading EasyOCR reader (version: {getattr(easyocr, '__version__', 'unknown')})...")
            print(f"ℹ️ easyocr file: {easyocr.__file__}")
            
            # Check for Reader attribute
            if not hasattr(easyocr, 'Reader'):
                print("❌ ERROR: 'easyocr' module has no 'Reader' attribute!")
                print(f"📦 Module contents: {dir(easyocr)}")
                raise AttributeError("easyocr.Reader not found. Installation might be corrupted.")

            # Use CPU for now as it's more reliable in cloud envs without GPU
            _ocr_reader = easyocr.Reader(['en'], gpu=False)
        except Exception as e:
            print(f"❌ Failed to initialize EasyOCR: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e
    return _ocr_reader

@app.post("/ocr-frame")
def ocr_frame(request: OCRRequest):
    """
    Extract text from a video frame or image using EasyOCR
    """
    log_file = "ocr_debug.log"
    def log(msg):
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{msg}\n")
            # Safe print for Windows terminal
            print(msg.encode('ascii', 'replace').decode('ascii'))
        except:
            pass

    log(f"\n--- [OCR START] type={request.content_type} ---")
    
    try:
        import base64
        import io
        from PIL import Image
        import numpy as np

        reader = get_ocr_reader()
        clean_base64 = None
        
        if request.content_type == 'video' and request.video_url:
            from extractors.video_extractor import capture_frame_at_time
            log(f"VIDEO: {request.video_url} at {request.timestamp}s")
            
            frame_base64 = capture_frame_at_time(request.video_url, request.timestamp, request.crop)
            if not frame_base64:
                log("FAIL: Frame capture None")
                return {"success": False, "error": "Failed to capture frame from video URL", "text": ""}
            
            clean_base64 = frame_base64
            log(f"SUCCESS: Frame capture len={len(clean_base64)}")
        elif request.image_base64:
            log("IMAGE: Direct base64")
            if ',' in request.image_base64:
                clean_base64 = request.image_base64.split(',')[1]
            else:
                clean_base64 = request.image_base64
        
        if not clean_base64:
            log("ERROR: No image data")
            return {"success": False, "error": "No data source", "text": ""}

        # Convert base64 to image
        img_data = base64.b64decode(clean_base64)
        image = Image.open(io.BytesIO(img_data)).convert('RGB')
        
        # Performance Hack: Resize if image is too large (Width > 1080)
        # Deep learning OCR is exponentially slower on high-res images
        max_w = 1080
        if image.width > max_w:
            w_percent = (max_w / float(image.width))
            h_size = int((float(image.height) * float(w_percent)))
            log(f"RESIZING: {image.width}x{image.height} -> {max_w}x{h_size}")
            image = image.resize((max_w, h_size), Image.Resampling.LANCZOS)

        # Convert to numpy array for EasyOCR
        img_np = np.array(image)
        log(f"SHAPE: {img_np.shape}")
        
        # Run OCR
        log("RUNNING: EasyOCR (default mode)...")
        results = reader.readtext(img_np, detail=0)
        log(f"RESULT: Found {len(results)} items")
        
        # Combine text
        text = " ".join(results)
        log(f"TEXT: {text[:100]}...")
        
        return {
            "success": True, 
            "text": text,
            "confidence": 0.8
        }
            
    except Exception as e:
        import traceback
        err_msg = f"FATAL ERROR: {str(e)}\n{traceback.format_exc()}"
        log(err_msg)
        return {"success": False, "error": str(e), "text": ""}

# YouTube Semantic Search is lazy-loaded

class VideoSearchRequest(BaseModel):
    query: str
    selected_text: Optional[str] = ''
    transcript_segment: Optional[str] = ''
    prefer_animated: Optional[bool] = True
    prefer_coding: Optional[bool] = False
    max_duration_minutes: Optional[int] = 10
    language: Optional[str] = 'english'

@app.post("/search-videos")
def search_youtube_videos(request: VideoSearchRequest):
    """
    Advanced semantic YouTube search with intelligent ranking
    """
    try:
        from youtube_semantic_search import search_videos as semantic_search_videos
        print(f"\n{'='*60}")
        print(f"🎥 YouTube Semantic Search Request")
        print(f"   Query: {request.query}")
        print(f"   Context: {len(request.selected_text)} chars selected, {len(request.transcript_segment)} chars transcript")
        print(f"   Preferences: Animated={request.prefer_animated}, Coding={request.prefer_coding}")
        print(f"   Max Duration: {request.max_duration_minutes} min")
        print(f"{'='*60}\n")
        
        videos = semantic_search_videos(
            query=request.query,
            selected_text=request.selected_text,
            transcript_segment=request.transcript_segment,
            prefer_animated=request.prefer_animated,
            prefer_coding=request.prefer_coding,
            max_duration_minutes=request.max_duration_minutes,
            language=request.language
        )
        
        return {
            "success": True,
            "count": len(videos),
            "videos": videos
        }
    except Exception as e:
        print(f"❌ Video search error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "videos": []
        }

@app.post("/extract", response_model=ExtractionResponse)
def extract_data(request: ExtractionRequest):
    try:
        if request.content_type == 'pdf':
            from extractors.pdf_extractor import extract_pdf
            result = extract_pdf(request.file_url)
            return {"success": True, "message": "PDF extraction successful", "data": result}
        elif request.content_type == 'video':
            # Proactive check: If it's a YouTube URL but type is 'video', use youtube_extractor
            youtube_terms = ['youtube.com', 'youtu.be']
            if any(term in request.file_url for term in youtube_terms):
                print(f"🔄 Detected YouTube URL in video type, rerouting to YouTube extractor: {request.file_url}")
                from extractors.youtube_extractor import extract_youtube
                result = extract_youtube(request.file_url)
                if result.get("success"):
                    return {"success": True, "message": "YouTube extraction successful (fallback)", "data": result}
                else:
                    return {"success": False, "message": result.get("error", "YouTube fallback failed"), "data": None}
            
            from extractors.video_extractor import extract_video
            result = extract_video(request.file_url)
            return {"success": True, "message": "Video extraction successful", "data": result}
        elif request.content_type == 'youtube':
            from extractors.youtube_extractor import extract_youtube
            result = extract_youtube(request.file_url)
            if result.get("success"):
                return {"success": True, "message": "YouTube extraction successful", "data": result}
            else:
                return {"success": False, "message": result.get("error", "YouTube extraction failed"), "data": None}
        elif request.content_type == 'web':
            from extractors.web_extractor import extract_web_content
            result = extract_web_content(request.file_url)
            return {"success": True, "message": "Web content extraction successful", "data": result}
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported content type: {request.content_type}")
            
    except Exception as e:
        print(f"Extraction error: {str(e)}")
        return {"success": False, "message": str(e), "data": None}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    # Enable reload for development
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
