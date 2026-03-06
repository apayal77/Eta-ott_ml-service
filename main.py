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
    """
    Startup event for Eta ML Service
    """
    print("🚀 Eta ML Service starting up...")
    # Playwright installation is now handled in render-build.sh
    print("ℹ️ Playwright browser check skipped at startup (handled in build phase).")

class ExtractionRequest(BaseModel):
    file_url: str
    content_id: str
    content_type: str  # 'pdf', 'video', 'youtube', etc.

class ExtractionResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None

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
    uvicorn.run(app, host="0.0.0.0", port=port)
