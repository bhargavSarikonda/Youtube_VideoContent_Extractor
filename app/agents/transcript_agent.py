import asyncio
import logging
import re
import httpx
from typing import Dict, Any, Tuple, Optional, List
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import yt_dlp

from app.models.schemas import VideoMetadata

logger = logging.getLogger("agentic_pipeline.transcript_agent")


class TranscriptAgent:
    """
    Agent 2: Ultra-Fast Multilingual Video Ingestion & Transcript Extraction Agent.
    Dual-Engine High-Speed Architecture:
    1. Primary: Fast youtube-transcript-api
    2. Fallback: Ultra-fast targeted yt-dlp Subtitle Stream Parser (< 3 seconds)
    """

    @classmethod
    async def extract_transcript_and_metadata(
        cls,
        video_id: str,
        target_language: str = "en"
    ) -> Tuple[str, VideoMetadata, str]:
        """
        Concurrently extracts transcript text and video metadata for maximum throughput.
        Returns: (transcript_text, VideoMetadata, detected_language)
        """
        loop = asyncio.get_event_loop()
        
        # Run metadata fetch and transcript extraction in parallel
        metadata_task = loop.run_in_executor(None, cls._fetch_metadata, video_id)
        transcript_task = loop.run_in_executor(None, cls._fetch_transcript, video_id, target_language)

        metadata, transcript_result = await asyncio.gather(metadata_task, transcript_task)
        transcript_text, detected_lang, available_langs, is_generated = transcript_result

        metadata.detected_language = detected_lang
        metadata.available_languages = available_langs
        metadata.is_auto_generated = is_generated

        return transcript_text, metadata, detected_lang

    @classmethod
    def _fetch_metadata(cls, video_id: str) -> VideoMetadata:
        """Fetches metadata via ultra-fast oEmbed (< 200ms) with lightweight fallback."""
        title = f"YouTube Video ({video_id})"
        channel = "YouTube Creator"
        thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
        duration_seconds = 0
        duration_formatted = "00:00"

        # 1. Fast lightweight oEmbed attempt (100-200ms)
        oembed_success = False
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.get(f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json")
                if res.status_code == 200:
                    data = res.json()
                    title = data.get("title", title)
                    channel = data.get("author_name", channel)
                    thumbnail_url = data.get("thumbnail_url", thumbnail_url)
                    oembed_success = True
        except Exception:
            pass

        # 2. Fast yt-dlp metadata if oEmbed failed or for duration (timeout 3s)
        if not oembed_success:
            try:
                ydl_opts = {
                    'quiet': True,
                    'no_warnings': True,
                    'skip_download': True,
                    'extract_flat': True,
                    'socket_timeout': 3,
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
                    if info:
                        title = info.get("title", title)
                        channel = info.get("uploader", info.get("channel", channel))
                        duration_seconds = info.get("duration", 0)
                        if duration_seconds:
                            mins, secs = divmod(duration_seconds, 60)
                            hours, mins = divmod(mins, 60)
                            duration_formatted = f"{hours:02d}:{mins:02d}:{secs:02d}" if hours else f"{mins:02d}:{secs:02d}"
                        thumbnail_url = info.get("thumbnail", thumbnail_url)
            except Exception:
                pass

        return VideoMetadata(
            video_id=video_id,
            title=title,
            channel=channel,
            duration_seconds=duration_seconds,
            duration_formatted=duration_formatted,
            thumbnail_url=thumbnail_url,
            detected_language="en",
            available_languages=["en"]
        )

    @classmethod
    def _fetch_transcript(cls, video_id: str, target_language: str = "en") -> Tuple[str, str, List[str], bool]:
        """
        Attempts fast transcript extraction.
        1. Fast youtube-transcript-api
        2. Fast targeted yt-dlp JSON3 stream (< 2.5 seconds)
        """
        # 1. Primary Engine: youtube-transcript-api (short socket timeout)
        try:
            transcript_list = None
            if hasattr(YouTubeTranscriptApi, 'list'):
                try:
                    transcript_list = YouTubeTranscriptApi().list(video_id)
                except TypeError:
                    transcript_list = YouTubeTranscriptApi.list(video_id)
            elif hasattr(YouTubeTranscriptApi, 'list_transcripts'):
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            if transcript_list:
                available_codes = [t.language_code for t in transcript_list]
                selected_transcript = None
                detected_lang = "en"
                is_generated = False

                # Try finding manually created transcript
                try:
                    if target_language in available_codes:
                        selected_transcript = transcript_list.find_manually_created_transcript([target_language])
                    else:
                        selected_transcript = transcript_list.find_manually_created_transcript(available_codes)
                    detected_lang = selected_transcript.language_code
                    is_generated = False
                except Exception:
                    try:
                        selected_transcript = transcript_list.find_generated_transcript(available_codes)
                        detected_lang = selected_transcript.language_code
                        is_generated = True
                    except Exception:
                        pass

                if not selected_transcript:
                    for t in transcript_list:
                        selected_transcript = t
                        detected_lang = t.language_code
                        is_generated = getattr(t, 'is_generated', False)
                        break

                if selected_transcript:
                    raw_data = selected_transcript.fetch()
                    text_segments = []
                    for entry in raw_data:
                        if isinstance(entry, dict):
                            text = entry.get("text", "").strip()
                        else:
                            text = getattr(entry, "text", "").strip()
                        if text:
                            text_segments.append(text)

                    full_text = " ".join(text_segments)
                    if full_text.strip():
                        return full_text, detected_lang, available_codes, is_generated

        except Exception as e:
            logger.info(f"Primary transcript API timed out or blocked ({e}). Engaging high-speed yt-dlp parser...")

        # 2. Ultra-Fast Fallback Engine: Targeted yt-dlp Subtitle Stream Parser (< 2.5 seconds)
        try:
            return cls._fetch_transcript_ytdlp(video_id, target_language)
        except Exception as e:
            logger.error(f"High-speed yt-dlp transcript extraction failed: {e}")
            raise ValueError(f"Could not retrieve captions for video {video_id}. Please ensure video has captions enabled.")

    @classmethod
    def _fetch_transcript_ytdlp(cls, video_id: str, target_language: str = "en") -> Tuple[str, str, List[str], bool]:
        """
        Ultra-fast targeted yt-dlp subtitle stream parser.
        Restricts requested languages to target + en to avoid massive overhead of 'all' languages.
        """
        requested_langs = list(set([target_language, 'en', 'en-US', 'en-GB', 'en-CA', 'en-AU']))
        
        ydl_opts = {
            'skip_download': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': requested_langs,
            'quiet': True,
            'no_warnings': True,
            'socket_timeout': 5,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
            if not info:
                raise ValueError("yt-dlp could not extract video info.")

            manual_subs = info.get('subtitles', {})
            auto_subs = info.get('automatic_captions', {})

            available_codes = list(set(list(manual_subs.keys()) + list(auto_subs.keys())))
            
            # If no subtitles/captions exist on YouTube (e.g. ambient scenery, music, silent clips)
            if not available_codes:
                logger.info(f"No subtitle tracks available on YouTube for video {video_id}. Engaging Metadata & Description Fallback...")
                desc = (info.get('description') or '').strip()
                tags = info.get('tags', []) or []
                tags_str = ", ".join(tags[:10]) if tags else "General Content"
                title = info.get('title', f"Video ({video_id})")
                uploader = info.get('uploader', info.get('channel', 'Creator'))

                fallback_text = (
                    f"[Visual / Non-Speech Media Note: This video has no spoken dialogue captions on YouTube.]\n"
                    f"Video Title: {title}\n"
                    f"Creator / Channel: {uploader}\n"
                    f"Author's Summary & Description: {desc if desc else 'Pure ambient/visual video clip without extended commentary.'}\n"
                    f"Content Tags: {tags_str}"
                )
                return fallback_text, "en", ["en"], False

            is_generated = False
            tracks = []
            detected_lang = target_language if target_language in available_codes else "en"

            # 1. Prefer manual subtitles
            if target_language in manual_subs:
                tracks = manual_subs[target_language]
                detected_lang = target_language
            elif 'en' in manual_subs:
                tracks = manual_subs['en']
                detected_lang = 'en'
            elif manual_subs:
                k = next(iter(manual_subs.keys()))
                tracks = manual_subs[k]
                detected_lang = k
            # 2. Auto-generated captions
            elif target_language in auto_subs:
                tracks = auto_subs[target_language]
                detected_lang = target_language
                is_generated = True
            elif 'en' in auto_subs:
                tracks = auto_subs['en']
                detected_lang = 'en'
                is_generated = True
            elif auto_subs:
                k = next(iter(auto_subs.keys()))
                tracks = auto_subs[k]
                detected_lang = k
                is_generated = True

            if not tracks:
                logger.info(f"No matching subtitle track found for video {video_id}. Engaging Metadata Fallback...")
                desc = (info.get('description') or '').strip()
                title = info.get('title', f"Video ({video_id})")
                uploader = info.get('uploader', info.get('channel', 'Creator'))
                fallback_text = (
                    f"[Visual / Non-Speech Media Note: This video has no spoken dialogue captions on YouTube.]\n"
                    f"Video Title: {title}\n"
                    f"Creator: {uploader}\n"
                    f"Description: {desc if desc else 'Visual/ambient content.'}"
                )
                return fallback_text, "en", ["en"], False

            # Find best format: json3 or vtt
            json3_track = next((t for t in tracks if t.get('ext') == 'json3'), None)
            vtt_track = next((t for t in tracks if t.get('ext') == 'vtt'), None)
            selected_track = json3_track or vtt_track or tracks[0]

            url = selected_track.get('url')
            if not url:
                raise ValueError("No subtitle URL found in track.")

            with httpx.Client(timeout=6.0, follow_redirects=True) as client:
                res = client.get(url)
                if res.status_code != 200:
                    raise ValueError(f"Failed fetching subtitle stream: HTTP {res.status_code}")

                lines = []
                ext = selected_track.get('ext')
                if ext == 'json3':
                    data = res.json()
                    events = data.get('events', [])
                    for ev in events:
                        segs = ev.get('segs', [])
                        txt = ''.join(s.get('utf8', '') for s in segs).strip()
                        if txt and txt != '\n':
                            lines.append(txt)
                else:
                    raw_text = res.text
                    clean_lines = []
                    for line in raw_text.splitlines():
                        line = line.strip()
                        if not line or line.startswith('WEBVTT') or '-->' in line or line.isdigit():
                            continue
                        clean_lines.append(re.sub(r'<[^>]+>', '', line))
                    lines = clean_lines

                full_text = " ".join(lines).strip()
                if not full_text:
                    raise ValueError("Extracted subtitle content is empty.")

                return full_text, detected_lang, available_codes, is_generated
