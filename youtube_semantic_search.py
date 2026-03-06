"""
Advanced YouTube Semantic Search with Sentence Transformers
Uses state-of-the-art NLP models for accurate video recommendations
"""

import re
from typing import List, Dict, Optional
# sentence_transformers and torch are lazy-loaded
import os
from datetime import datetime, timedelta

# Try to import YouTube API (optional)
try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    YOUTUBE_API_AVAILABLE = True
except ImportError:
    print("⚠️ google-api-python-client not installed. YouTube API features disabled.")
    print("   Install with: pip install google-api-python-client")
    YOUTUBE_API_AVAILABLE = False
    HttpError = Exception  # Fallback

# Lightweight semantic search model (all-MiniLM-L6-v2 is ~90MB vs 420MB for mpnet)
LIGHT_MODEL_NAME = 'all-MiniLM-L6-v2'
_semantic_model = None

def get_semantic_model():
    global _semantic_model
    if _semantic_model is None:
        print(f"⏳ Loading Semantic Search Model ({LIGHT_MODEL_NAME})...")
        from sentence_transformers import SentenceTransformer
        _semantic_model = SentenceTransformer(LIGHT_MODEL_NAME)
    return _semantic_model

# YouTube API configuration
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY', '')  # Add to .env file

class YouTubeSemanticSearch:
    """
    Advanced YouTube search using semantic embeddings and intelligent ranking
    """
    
    def __init__(self):
        self._model = None
        self.youtube = None
        if YOUTUBE_API_AVAILABLE and YOUTUBE_API_KEY:
            try:
                self.youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
                print("✅ YouTube Data API initialized")
            except Exception as e:
                print(f"⚠️ YouTube API initialization failed: {e}")
        elif not YOUTUBE_API_KEY:
            print("ℹ️ YouTube API key not set. Using yt-search fallback.")
        else:
            print("ℹ️ YouTube API library not available. Using yt-search fallback.")
    
    def parse_duration(self, duration_str: str) -> int:
        """
        Parse ISO 8601 duration (PT10M30S) to minutes
        """
        if not duration_str:
            return 0
        
        # Extract hours, minutes, seconds
        hours = re.search(r'(\d+)H', duration_str)
        minutes = re.search(r'(\d+)M', duration_str)
        seconds = re.search(r'(\d+)S', duration_str)
        
        total_minutes = 0
        if hours:
            total_minutes += int(hours.group(1)) * 60
        if minutes:
            total_minutes += int(minutes.group(1))
        if seconds:
            total_minutes += int(seconds.group(1)) / 60
        
        return round(total_minutes, 1)
    
    def parse_views(self, view_count: str) -> int:
        """Parse view count string to integer"""
        try:
            return int(view_count)
        except:
            return 0
    
    def calculate_engagement_score(self, views: int, likes: int, published_days_ago: int) -> float:
        """
        Calculate engagement score based on views, likes, and recency
        """
        if published_days_ago == 0:
            published_days_ago = 1
        
        # Views per day
        views_per_day = views / published_days_ago
        
        # Like ratio (likes per 1000 views)
        like_ratio = (likes / views * 1000) if views > 0 else 0
        
        # Normalize scores
        view_score = min(views_per_day / 10000, 1.0)  # 10k views/day = max
        like_score = min(like_ratio / 50, 1.0)  # 50 likes per 1k views = max
        
        return (view_score * 0.7 + like_score * 0.3)
    
    def get_days_since_published(self, published_at: str) -> int:
        """Calculate days since video was published"""
        try:
            pub_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
            now = datetime.now(pub_date.tzinfo)
            delta = now - pub_date
            return max(delta.days, 1)
        except:
            return 365  # Default to 1 year
    
    def is_animated_content(self, title: str, description: str) -> bool:
        """Detect if video is animated/visual content"""
        animated_keywords = [
            'animated', 'animation', 'visual', 'illustrated', 'explained',
            '3d', 'graphics', 'diagram', 'visualization', 'infographic',
            'whiteboard', 'drawing', 'sketch', 'motion graphics'
        ]
        
        text = f"{title} {description}".lower()
        return any(keyword in text for keyword in animated_keywords)
    
    def is_coding_content(self, title: str, description: str) -> bool:
        """Detect if video is coding/implementation content"""
        coding_keywords = [
            'code', 'coding', 'programming', 'tutorial', 'build',
            'project', 'implementation', 'hands-on', 'walkthrough',
            'step by step', 'from scratch', 'complete guide'
        ]
        
        text = f"{title} {description}".lower()
        return any(keyword in text for keyword in coding_keywords)
    
    def search_youtube_api(self, query: str, max_results: int = 30) -> List[Dict]:
        """
        Search YouTube using official API (more accurate than yt-search)
        """
        if not self.youtube:
            return []
        
        try:
            # Search for videos
            search_response = self.youtube.search().list(
                q=query,
                part='id,snippet',
                maxResults=max_results,
                type='video',
                relevanceLanguage='en',
                safeSearch='strict',
                videoEmbeddable='true',
                videoDuration='medium'  # 4-20 minutes
            ).execute()
            
            video_ids = [item['id']['videoId'] for item in search_response.get('items', [])]
            
            if not video_ids:
                return []
            
            # Get detailed video statistics
            videos_response = self.youtube.videos().list(
                part='snippet,contentDetails,statistics',
                id=','.join(video_ids)
            ).execute()
            
            videos = []
            for item in videos_response.get('items', []):
                snippet = item['snippet']
                stats = item.get('statistics', {})
                duration = item['contentDetails']['duration']
                
                videos.append({
                    'id': item['id'],
                    'url': f"https://www.youtube.com/watch?v={item['id']}",
                    'title': snippet['title'],
                    'description': snippet.get('description', ''),
                    'thumbnail': snippet['thumbnails']['high']['url'],
                    'channel': snippet['channelTitle'],
                    'published_at': snippet['publishedAt'],
                    'duration': duration,
                    'duration_minutes': self.parse_duration(duration),
                    'views': self.parse_views(stats.get('viewCount', '0')),
                    'likes': self.parse_views(stats.get('likeCount', '0')),
                    'comments': self.parse_views(stats.get('commentCount', '0'))
                })
            
            return videos
            
        except HttpError as e:
            print(f"YouTube API error: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error in YouTube API search: {e}")
            return []
    
    def search_with_fallback(self, query: str, max_results: int = 30) -> List[Dict]:
        """
        Search using YouTube API with yt-search fallback
        """
        # Try official API first
        videos = self.search_youtube_api(query, max_results)
        
        if videos:
            return videos
        
        # Fallback to youtube-search-python (more stable)
        print("⚠️ YouTube API unavailable, using youtubesearchpython fallback...")
        try:
            from youtubesearchpython import VideosSearch
            video_search = VideosSearch(query, limit=max_results)
            results = video_search.result()
            
            videos = []
            for video in results.get('result', []):
                # Extract duration
                duration_str = video.get('duration', '0:00')
                parts = duration_str.split(':')
                duration_minutes = 0
                if len(parts) == 2:
                    duration_minutes = int(parts[0]) + int(parts[1]) / 60
                elif len(parts) == 3:
                    duration_minutes = int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 60
                
                # Extract views
                view_count_text = video.get('viewCount', {}).get('text', '0 views')
                views = 0
                match = re.search(r'([\d,.]+)', view_count_text)
                if match:
                    views_str = match.group(1).replace(',', '')
                    views = int(float(views_str))
                    if 'K' in view_count_text: views *= 1000
                    elif 'M' in view_count_text: views *= 1000000
                    elif 'B' in view_count_text: views *= 1000000000

                videos.append({
                    'id': video.get('id', ''),
                    'url': video.get('link', ''),
                    'title': video.get('title', ''),
                    'description': video.get('descriptionSnippet', [{}])[0].get('text', '') if video.get('descriptionSnippet') else '',
                    'thumbnail': video.get('thumbnails', [{}])[0].get('url', ''),
                    'channel': video.get('channel', {}).get('name', ''),
                    'published_at': video.get('publishedTime', ''),
                    'duration': duration_str,
                    'duration_minutes': duration_minutes,
                    'views': views,
                    'likes': 0,
                    'comments': 0
                })
            
            return videos
            
        except Exception as e:
            print(f"Fallback search error: {e}")
            return []
    
    def semantic_search(
        self,
        query: str,
        selected_text: str = '',
        transcript_segment: str = '',
        prefer_animated: bool = True,
        prefer_coding: bool = False,
        max_duration_minutes: int = 10,
        language: str = 'english'
    ) -> List[Dict]:
        """
        Advanced semantic search with intelligent ranking
        
        Args:
            query: User's search query
            selected_text: Text selected by user (context)
            transcript_segment: Transcript from video region (context)
            prefer_animated: Boost animated/visual content
            prefer_coding: Boost coding tutorials
            max_duration_minutes: Maximum video duration (default 10 min)
            language: Preferred language
        
        Returns:
            List of ranked videos with scores
        """
        
        # 1. Build semantic context
        context_parts = [query]
        if selected_text:
            context_parts.append(selected_text[:300])
        if transcript_segment:
            context_parts.append(transcript_segment[:300])
        
        semantic_context = ' | '.join(context_parts)
        
        # 2. Generate query embedding
        print(f"🔍 Generating embedding for: '{semantic_context[:100]}...'")
        from sentence_transformers import util
        model = get_semantic_model()
        query_embedding = model.encode(semantic_context, convert_to_tensor=True)
        
        # 3. Build optimized search query
        search_parts = [query]
        
        if prefer_animated:
            search_parts.append('animated explanation visual')
        elif prefer_coding:
            search_parts.append('coding tutorial implementation')
        
        if language.lower() == 'hindi':
            search_parts.append('hindi')
        else:
            search_parts.append('english')
        
        search_query = ' '.join(search_parts)[:150]
        
        # 4. Search YouTube
        print(f"🎥 Searching YouTube: '{search_query}'")
        videos = self.search_with_fallback(search_query, max_results=30)
        
        if not videos:
            print("❌ No videos found")
            return []
        
        # 5. Filter by duration (max 10 minutes)
        videos = [v for v in videos if v['duration_minutes'] <= max_duration_minutes and v['duration_minutes'] >= 2]
        
        if not videos:
            print(f"⚠️ No videos under {max_duration_minutes} minutes, relaxing constraint...")
            videos = self.search_with_fallback(search_query, max_results=30)
            videos = [v for v in videos if v['duration_minutes'] <= 15]  # Relax to 15 min
        
        print(f"📊 Processing {len(videos)} videos for semantic ranking...")
        
        # 6. Calculate semantic similarity for each video
        scored_videos = []
        
        model = get_semantic_model()
        from sentence_transformers import util
        for video in videos:
            # Generate video embedding from title + description
            video_text = f"{video['title']} {video['description'][:500]}"
            video_embedding = model.encode(video_text, convert_to_tensor=True)
            
            # Calculate cosine similarity
            semantic_score = util.cos_sim(query_embedding, video_embedding).item()
            
            # 7. Calculate quality scores
            days_ago = self.get_days_since_published(video['published_at']) if video['published_at'] else 365
            
            # View score (normalized)
            view_score = min(video['views'] / 1000000, 1.0)  # 1M views = max
            
            # Engagement score
            engagement_score = self.calculate_engagement_score(
                video['views'],
                video['likes'],
                days_ago
            )
            
            # Recency score (prefer videos from last 2 years)
            recency_score = 1.0 if days_ago < 730 else max(0.3, 1 - (days_ago - 730) / 1825)
            
            # Duration score (prefer 5-10 min, penalize very short or long)
            duration = video['duration_minutes']
            if 5 <= duration <= 10:
                duration_score = 1.0
            elif 3 <= duration <= 12:
                duration_score = 0.9
            elif 2 <= duration <= 15:
                duration_score = 0.7
            else:
                duration_score = 0.5
            
            # Content type matching
            is_animated = self.is_animated_content(video['title'], video['description'])
            is_coding = self.is_coding_content(video['title'], video['description'])
            
            content_type_bonus = 0
            if prefer_animated and is_animated:
                content_type_bonus = 0.15
            elif prefer_coding and is_coding:
                content_type_bonus = 0.15
            
            # Quality channel bonus
            quality_channels = [
                '3blue1brown', 'khan academy', 'crash course', 'freecodecamp',
                'traversy media', 'fireship', 'academind', 'net ninja',
                'programming with mosh', 'corey schafer', 'tech with tim',
                'sentdex', 'computerphile', 'numberphile'
            ]
            
            channel_bonus = 0.1 if any(ch in video['channel'].lower() for ch in quality_channels) else 0
            
            # 8. Calculate final weighted score
            final_score = (
                semantic_score * 0.40 +           # 40% semantic relevance (increased!)
                view_score * 0.15 +               # 15% popularity
                engagement_score * 0.15 +         # 15% engagement
                recency_score * 0.10 +            # 10% recency
                duration_score * 0.10 +           # 10% duration fit
                content_type_bonus +              # 15% content type bonus
                channel_bonus                     # 10% channel quality
            )
            
            scored_videos.append({
                **video,
                'semantic_score': round(semantic_score, 4),
                'final_score': round(final_score, 4),
                'is_animated': is_animated,
                'is_coding': is_coding,
                'scores': {
                    'semantic': round(semantic_score, 3),
                    'views': round(view_score, 3),
                    'engagement': round(engagement_score, 3),
                    'recency': round(recency_score, 3),
                    'duration': round(duration_score, 3),
                    'content_type_bonus': round(content_type_bonus, 3),
                    'channel_bonus': round(channel_bonus, 3)
                }
            })
        
        # 9. Sort by final score
        scored_videos.sort(key=lambda x: x['final_score'], reverse=True)
        
        # 10. Return top results (increased to support infinite scroll/more discovery)
        top_videos = scored_videos[:30]
        
        if top_videos:
            best = top_videos[0]
            print(f"✅ Top match: '{best['title'][:60]}...'")
            print(f"   Duration: {best['duration_minutes']:.1f} min | Views: {best['views']:,}")
            print(f"   Semantic: {best['semantic_score']:.3f} | Final: {best['final_score']:.3f}")
            print(f"   Animated: {best['is_animated']} | Coding: {best['is_coding']}")
        
        return top_videos


# Global instance is now lazy-loaded
_youtube_search_instance = None

def get_youtube_search_instance():
    global _youtube_search_instance
    if _youtube_search_instance is None:
        _youtube_search_instance = YouTubeSemanticSearch()
    return _youtube_search_instance


def search_videos(
    query: str,
    selected_text: str = '',
    transcript_segment: str = '',
    prefer_animated: bool = True,
    prefer_coding: bool = False,
    max_duration_minutes: int = 10,
    language: str = 'english'
) -> List[Dict]:
    """
    Main entry point for semantic video search
    """
    return get_youtube_search_instance().semantic_search(
        query=query,
        selected_text=selected_text,
        transcript_segment=transcript_segment,
        prefer_animated=prefer_animated,
        prefer_coding=prefer_coding,
        max_duration_minutes=max_duration_minutes,
        language=language
    )
