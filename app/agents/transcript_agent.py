import os
import re
import tempfile
import asyncio
import logging
import httpx
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp

from app.models.schemas import VideoMetadata
from app.services.llm_service import LLMService

logger = logging.getLogger("agentic_pipeline.transcript_agent")


class TranscriptAgent:
    """
    Agent 2: Robust Multilingual Video Ingestion & Voice-to-Text Agent.
    Multi-Engine Resilient Architecture:
    1. Primary: Fast direct youtube-transcript-api (Supports all languages & auto-subs)
    2. Secondary: Fast targeted yt-dlp Subtitle Stream Parser
    3. Tertiary: Voice-to-Text Speech Transcriber (OpenAI Whisper Audio Ingestion)
    4. Fallback: Content & Metadata Context Synthesis
    """

    @classmethod
    async def extract_transcript_and_metadata(
        cls,
        video_id: str,
        target_language: str = "en"
    ) -> Tuple[str, VideoMetadata, str]:
        """
        Concurrently extracts transcript text and video metadata.
        Returns: (transcript_text, VideoMetadata, detected_language)
        """
        loop = asyncio.get_event_loop()
        
        # Run metadata fetch and transcript extraction concurrently
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
        Extracts transcript or converts voice-to-text with multi-engine fallback.
        1. youtube-transcript-api (fast closed captions)
        2. yt-dlp subtitle stream parser
        3. OpenAI Whisper Voice-to-Text (Audio speech-to-text)
        4. Metadata & Description resilient summary
        """
        # =========================================================================
        # 1. Primary Engine: youtube-transcript-api
        # =========================================================================
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

                # 1A. Try finding target language manually or generated
                if target_language in available_codes:
                    try:
                        selected_transcript = transcript_list.find_manually_created_transcript([target_language])
                        detected_lang = selected_transcript.language_code
                        is_generated = False
                    except Exception:
                        try:
                            selected_transcript = transcript_list.find_generated_transcript([target_language])
                            detected_lang = selected_transcript.language_code
                            is_generated = True
                        except Exception:
                            pass

                # 1B. Try any manually created transcript
                if not selected_transcript:
                    try:
                        selected_transcript = transcript_list.find_manually_created_transcript(available_codes)
                        detected_lang = selected_transcript.language_code
                        is_generated = False
                    except Exception:
                        pass

                # 1C. Try any generated transcript
                if not selected_transcript:
                    try:
                        selected_transcript = transcript_list.find_generated_transcript(available_codes)
                        detected_lang = selected_transcript.language_code
                        is_generated = True
                    except Exception:
                        pass

                # 1D. Fallback to iterating transcript items
                if not selected_transcript:
                    for t in transcript_list:
                        selected_transcript = t
                        detected_lang = getattr(t, 'language_code', 'en')
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
                        logger.info(f"Successfully extracted transcript via youtube-transcript-api (Language: {detected_lang})")
                        return full_text, detected_lang, available_codes, is_generated

        except Exception as e:
            logger.info(f"Primary transcript API timed out or blocked ({e}). Checking secondary engines...")

        # =========================================================================
        # 2. Secondary Engine: yt-dlp Subtitle Stream Parser
        # =========================================================================
        try:
            return cls._fetch_transcript_ytdlp(video_id, target_language)
        except Exception as e:
            logger.info(f"yt-dlp subtitle stream extraction not found or blocked ({e}). Engaging Voice-to-Text Whisper Engine...")

        # =========================================================================
        # 3. Tertiary Engine: Voice-to-Text Audio Speech Transcription (Whisper)
        # =========================================================================
        try:
            return cls._fetch_voice_to_text_whisper(video_id, target_language)
        except Exception as e:
            logger.warning(f"Voice-to-Text Whisper transcription failed: {e}. Engaging metadata synthesis fallback...")

        # =========================================================================
        # 4. Quaternary Engine: Resilient Metadata & Content Context
        # =========================================================================
        return cls._fetch_metadata_fallback(video_id)

    @classmethod
    def _fetch_transcript_ytdlp(cls, video_id: str, target_language: str = "en") -> Tuple[str, str, List[str], bool]:
        """
        Targeted yt-dlp subtitle stream parser.
        Inspects all available subtitle languages (manual & automatic).
        """
        ydl_opts = {
            'skip_download': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'allsubtitles': True,
            'quiet': True,
            'no_warnings': True,
            'socket_timeout': 5,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
            if not info:
                raise ValueError("yt-dlp could not extract video info.")

            manual_subs = info.get('subtitles', {}) or {}
            auto_subs = info.get('automatic_captions', {}) or {}

            available_codes = list(set(list(manual_subs.keys()) + list(auto_subs.keys())))
            if not available_codes:
                raise ValueError("No subtitle tracks found in yt-dlp stream.")

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
                raise ValueError("No subtitle tracks available for parsing.")

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

                logger.info(f"Successfully extracted subtitles via yt-dlp (Language: {detected_lang})")
                return full_text, detected_lang, available_codes, is_generated

    @classmethod
    def _fetch_voice_to_text_whisper(cls, video_id: str, target_language: str = "en") -> Tuple[str, str, List[str], bool]:
        """
        Voice-to-Text Engine: Downloads low-bitrate audio stream and transcribes spoken dialogue via OpenAI Whisper.
        """
        llm = LLMService()
        if not llm.is_configured():
            raise ValueError("OpenAI API Key not configured for Voice-to-Text Whisper transcription.")

        # Prepare temporary audio destination (compatible with Vercel /tmp)
        tmp_dir = Path("/tmp") if os.path.exists("/tmp") else Path(tempfile.gettempdir())
        output_template = str(tmp_dir / f"yt_audio_{video_id}.%(ext)s")
        expected_audio_file = None

        ydl_opts = {
            'format': 'ba[ext=m4a]/ba/b',
            'outtmpl': output_template,
            'quiet': True,
            'no_warnings': True,
            'max_filesize': 25 * 1024 * 1024,  # 25 MB max limit for Whisper API
            'socket_timeout': 10,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
                if not info:
                    raise ValueError("Failed downloading audio stream from YouTube.")
                
                # Identify downloaded audio path
                ext = info.get('ext', 'm4a')
                candidate_path = str(tmp_dir / f"yt_audio_{video_id}.{ext}")
                if os.path.exists(candidate_path):
                    expected_audio_file = candidate_path
                else:
                    # Search tmp_dir for matching file
                    matches = list(tmp_dir.glob(f"yt_audio_{video_id}.*"))
                    if matches:
                        expected_audio_file = str(matches[0])

            if not expected_audio_file or not os.path.exists(expected_audio_file):
                raise FileNotFoundError(f"Downloaded audio file for video {video_id} could not be located.")

            # Perform Whisper speech-to-text transcription
            whisper_text, detected_lang, _ = llm.transcribe_audio_sync(
                audio_file_path=expected_audio_file,
                language=target_language if target_language in ["en", "hi", "es", "fr", "de", "it", "ja", "ko", "pt", "ru", "zh"] else None
            )

            if not whisper_text.strip():
                raise ValueError("Whisper transcription yielded empty voice output.")

            logger.info(f"Successfully transcribed spoken voice via Whisper (Language: {detected_lang}, Words: {len(whisper_text.split())})")
            return whisper_text, detected_lang, [detected_lang], True

        finally:
            # Clean up temporary audio files
            if expected_audio_file and os.path.exists(expected_audio_file):
                try:
                    os.remove(expected_audio_file)
                except Exception:
                    pass
            for f in tmp_dir.glob(f"yt_audio_{video_id}.*"):
                try:
                    f.unlink()
                except Exception:
                    pass

    @classmethod
    def _fetch_metadata_fallback(cls, video_id: str) -> Tuple[str, str, List[str], bool]:
        """
        Resilient Metadata & Content Fallback:
        Synthesizes video title, channel, description, and tags when no spoken dialogue or captions exist.
        """
        title = f"YouTube Video ({video_id})"
        channel = "YouTube Creator"
        desc = ""
        tags_str = "General Content"

        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'skip_download': True,
                'extract_flat': True,
                'socket_timeout': 4,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
                if info:
                    title = info.get("title", title)
                    channel = info.get("uploader", info.get("channel", channel))
                    desc = (info.get("description") or "").strip()
                    tags = info.get("tags", []) or []
                    if tags:
                        tags_str = ", ".join(tags[:10])
        except Exception:
            pass

        fallback_text = (
            f"[Visual / Non-Speech Media Overview]\n"
            f"Video Title: {title}\n"
            f"Creator / Channel: {channel}\n"
            f"Content Description: {desc if desc else 'Video presentation without closed captions.'}\n"
            f"Keywords & Topics: {tags_str}"
        )
        return fallback_text, "en", ["en"], False

