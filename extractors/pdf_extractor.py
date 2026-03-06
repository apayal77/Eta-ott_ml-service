import fitz  # PyMuPDF
from PIL import Image
# transformers and torch removed as they were unused or heavy
import requests
from io import BytesIO
import os
import cloudinary
import cloudinary.api
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

# Initialize model and processor once
# device = "cuda" if torch.cuda.is_available() else "cpu"
# processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
# model = AutoModelForTokenClassification.from_pretrained("microsoft/layoutlmv3-base").to(device)

def download_file(url):
    """Download file with proper authentication"""
    try:
        from urllib.parse import urlparse, unquote
        
        # For Cloudinary URLs, use authenticated download
        if 'cloudinary.com' in url:
            # We need to extract the path correctly
            parsed_url = urlparse(url)
            path_parts = parsed_url.path.split('/')
            
            try:
                # Find the index of 'upload'
                # Format: /cloud_name/raw/upload/v12345/public_id
                upload_idx = path_parts.index('upload')
                res_type = path_parts[upload_idx - 1]
                
                # Get everything after 'upload'
                after_upload = path_parts[upload_idx + 1:]
                
                # If the first part is a version code (e.g., v1), skip it
                if after_upload[0].startswith('v') and any(char.isdigit() for char in after_upload[0]):
                    public_id_parts = after_upload[1:]
                else:
                    public_id_parts = after_upload
                
                # Join parts back and UNQUOTE (decode) to get the original filename/public_id
                public_id_with_ext = unquote('/'.join(public_id_parts))
                public_id = public_id_with_ext
                
                # For images and videos, Cloudinary SDK expects public_id WITHOUT extension
                if res_type != 'raw':
                    # Only strip if it actually looks like it has an extension
                    if '.' in public_id_with_ext:
                        public_id = public_id_with_ext.rsplit('.', 1)[0]
                
                print(f"🔍 Extracted for signing: res_type={res_type}, public_id={public_id}")
                
                # Use private_download_url for robust authenticated access
                authenticated_url = cloudinary.utils.private_download_url(
                    public_id,
                    resource_type=res_type,
                    type='upload',
                    format=public_id_with_ext.split('.')[-1] if '.' in public_id_with_ext else None
                )
                
                print(f"📥 Downloading ({res_type}) from authenticated URL...")
                response = requests.get(authenticated_url, timeout=30)
                
                if response.status_code != 200:
                    print(f"⚠️ Authenticated URL failed ({response.status_code}), trying basic signed URL...")
                    # Fallback to simple signed URL
                    alt_url, _ = cloudinary.utils.cloudinary_url(
                        public_id,
                        resource_type=res_type,
                        sign_url=True,
                        secure=True
                    )
                    response = requests.get(alt_url, timeout=30)
            except (ValueError, IndexError) as e:
                print(f"⚠️ URL parsing error: {e}, using direct fallback")
                response = requests.get(url, timeout=30)
        else:
            response = requests.get(url, timeout=30)
        
        if response.status_code != 200:
            raise Exception(f"Failed to download file: {response.status_code} - {response.text[:100]}")
        return BytesIO(response.content)
    except Exception as e:
        print(f"Download error: {str(e)}")
        raise e


def extract_pdf(file_url):
    try:
        # Download PDF
        pdf_content = download_file(file_url)
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        
        full_text = ""
        page_data = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            full_text += text + "\n"
            
            page_data.append({
                "page": page_num + 1,
                "text": text
            })
            
        # Basic NLP/LLM Simulation (Ready for Gemini/OpenAI integration)
        # For now, let's extract real keywords and topics from the text
        import re
        from collections import Counter
        
        # Simple keyword extraction (words > 4 chars, ignoring common ones)
        words = re.findall(r'\w{5,}', full_text.lower())
        common_words = {'about', 'after', 'again', 'could', 'every', 'from', 'great', 'have', 'their', 'there', 'these', 'which', 'would'}
        keywords = [word for word, count in Counter(words).most_common(10) if word not in common_words]
        
        # Simple Topic detection (based on most frequent noun-like words)
        topics = [word.capitalize() for word, count in Counter(words).most_common(3) if word not in common_words]
        
        # Summary: First 1000 chars properly cleaned
        summary = full_text.strip()[:1000]
        if len(full_text) > 1000:
            summary += "..."

        # Generate thumbnail from first page
        thumbnail_url = None
        thumbnail_public_id = None
        try:
            if len(doc) > 0:
                print("🖼️ Generating thumbnail for PDF...")
                page = doc.load_page(0)
                # Use a reasonable resolution (150 DPI approx)
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                img_data = pix.tobytes("jpg")
                
                # Upload to Cloudinary
                import cloudinary.uploader
                upload_result = cloudinary.uploader.upload(
                    img_data,
                    folder="eta-thumbnails",
                    resource_type="image"
                )
                thumbnail_url = upload_result.get("secure_url")
                thumbnail_public_id = upload_result.get("public_id")
                print(f"✅ PDF Thumbnail uploaded: {thumbnail_url}")
        except Exception as thumb_err:
            print(f"⚠️ PDF Thumbnail generation failed: {str(thumb_err)}")

        return {
            "text": full_text,
            "pages": len(doc),
            "page_details": page_data,
            "summary": summary,
            "topics": topics,
            "keywords": keywords,
            "thumbnail_url": thumbnail_url,
            "thumbnail_public_id": thumbnail_public_id,
            "metadata": {
                "extracted_at": "Now",
                "method": "NLP-Advanced"
            }
        }
        
    except Exception as e:
        print(f"Error in extract_pdf: {str(e)}")
        raise e
