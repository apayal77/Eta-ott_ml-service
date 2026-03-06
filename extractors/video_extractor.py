# whisper is lazy-loaded
import os
import requests
import cloudinary
import cloudinary.api
import subprocess
import shutil
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET'),
    secure=True
)


def setup_ffmpeg():
    """Ensure FFmpeg is available in the environment."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️ Global FFmpeg not found, attempting to use imageio-ffmpeg binary...")
        try:
            import imageio_ffmpeg
            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
            ffmpeg_dir = os.path.dirname(ffmpeg_bin)
            
            target_ffmpeg = os.path.join(ffmpeg_dir, "ffmpeg.exe")
            if not os.path.exists(target_ffmpeg):
                shutil.copy2(ffmpeg_bin, target_ffmpeg)
            
            if ffmpeg_dir not in os.environ["PATH"]:
                os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]
            return True
        except Exception as e:
            print(f"FFmpeg setup failed: {str(e)}")
            return False

def download_video(url, dest_path):
    """Download video with proper authentication"""
    try:
        from urllib.parse import urlparse, unquote
        
        # For Cloudinary URLs, use authenticated download
        if 'cloudinary.com' in url:
            parsed_url = urlparse(url)
            path_parts = parsed_url.path.split('/')
            
            try:
                upload_idx = path_parts.index('upload')
                res_type = path_parts[upload_idx - 1]
                after_upload = path_parts[upload_idx + 1:]
                
                if after_upload[0].startswith('v') and any(char.isdigit() for char in after_upload[0]):
                    public_id_parts = after_upload[1:]
                else:
                    public_id_parts = after_upload
                
                public_id_with_ext = unquote('/'.join(public_id_parts))
                public_id = public_id_with_ext
                
                if res_type != 'raw':
                    public_id = public_id_with_ext.rsplit('.', 1)[0]
                
                authenticated_url = cloudinary.utils.private_download_url(
                    public_id,
                    resource_type=res_type,
                    type='upload',
                    format=public_id_with_ext.split('.')[-1] if '.' in public_id_with_ext else None
                )
                
                response = requests.get(authenticated_url, stream=True, timeout=120)
                if response.status_code != 200:
                    alt_url, _ = cloudinary.utils.cloudinary_url(
                        public_id, resource_type=res_type, sign_url=True, secure=True
                    )
                    response = requests.get(alt_url, stream=True, timeout=120)
            except (ValueError, IndexError):
                response = requests.get(url, stream=True, timeout=120)
        else:
            response = requests.get(url, stream=True, timeout=120)
        
        if response.status_code != 200:
            raise Exception(f"Failed to download video: {response.status_code}")
            
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
    except Exception as e:
        print(f"Video download error: {str(e)}")
        raise e

def extract_video(file_url):
    job_id = str(uuid.uuid4())[:8]
    base_temp = os.path.abspath("temp_video_jobs")
    job_dir = os.path.join(base_temp, job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    temp_video = os.path.join(job_dir, f"source_{job_id}.mp4")
    thumb_path = os.path.join(job_dir, f"thumb_{job_id}.jpg")
    
    try:
        # 0. Set up FFmpeg
        setup_ffmpeg()

        # 1. Download video
        print(f"📥 Downloading video for job {job_id}...")
        download_video(file_url, temp_video)
        
        # 2. Extract audio for Whisper (16kHz mono is ideal)
        print(f"🔊 Extracting audio for transcription ({job_id})...")
        temp_audio = os.path.join(job_dir, f"audio_{job_id}.mp3")
        try:
            subprocess.run([
                "ffmpeg", "-i", temp_video,
                "-ar", "16000",
                "-ac", "1",
                "-ab", "64k",
                "-f", "mp3",
                temp_audio, "-y"
            ], capture_output=True, check=True)
        except Exception as audio_err:
            print(f"⚠️ Audio pre-extraction failed, using video file directly: {audio_err}")
            temp_audio = temp_video

        # 3. Transcribe with Whisper
        from model_loader import get_whisper_model, get_whisper_lock
        whisper_model = get_whisper_model()
        whisper_lock = get_whisper_lock()
        
        # Verify audio file integrity
        if not os.path.exists(temp_audio):
            raise FileNotFoundError(f"Audio file missing for transcription: {temp_audio}")
            
        file_size = os.path.getsize(temp_audio)
        print(f"🎙️ Transcribing {job_id} (Audio Size: {file_size / 1024 / 1024:.2f} MB)...")
        
        # Synchronize access to whisper model
        with whisper_lock:
            result = whisper_model.transcribe(temp_audio, fp16=False)

        # 3. Generate thumbnail from video
        thumbnail_url = None
        thumbnail_public_id = None
        try:
            print(f"🖼️ Generating thumbnail for job {job_id}...")
            subprocess.run([
                "ffmpeg", "-i", temp_video,
                "-ss", "00:00:01.000",
                "-vframes", "1",
                "-q:v", "2",
                thumb_path, "-y"
            ], capture_output=True, check=True)

            # Upload to Cloudinary
            import cloudinary.uploader
            upload_result = cloudinary.uploader.upload(
                thumb_path,
                folder="eta-thumbnails",
                resource_type="image"
            )
            thumbnail_url = upload_result.get("secure_url")
            thumbnail_public_id = upload_result.get("public_id")
            print(f"✅ Video Thumbnail uploaded: {thumbnail_url}")
        except Exception as thumb_err:
            print(f"⚠️ Video Thumbnail generation failed: {str(thumb_err)}")
        
        return {
            "text": result["text"],
            "segments": result["segments"],
            "duration": result.get("duration", 0),
            "language": result.get("language", "en"),
            "summary": result["text"][:500] + "..." if len(result["text"]) > 500 else result["text"],
            "thumbnail_url": thumbnail_url,
            "thumbnail_public_id": thumbnail_public_id
        }
        
    except Exception as e:
        print(f"❌ Video extraction error ({job_id}): {str(e)}")
        raise e
    finally:
        # Aggressive cleanup
        try:
            if os.path.exists(job_dir):
                shutil.rmtree(job_dir, ignore_errors=True)
        except Exception as cleanup_err:
            print(f"⚠️ Cleanup failed for {job_id}: {cleanup_err}")
