import re
from urllib.parse import urlparse, parse_qs
from typing import Tuple, Optional


class YouTubeURLAgent:
    """
    Agent 1: YouTube URL Validation & Extraction Agent.
    Strictly permits ONLY valid YouTube links (youtube.com, youtu.be, shorts).
    Rejects any third-party or arbitrary URLs.
    """

    ALLOWED_DOMAINS = {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
        "youtu.be",
        "www.youtu.be",
    }

    # Standard 11 character YouTube video ID regex pattern
    VIDEO_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{11}$")

    @classmethod
    def validate_and_extract(cls, url: str) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
        """
        Validates URL and extracts the canonical video ID and canonical URL.
        Returns: (is_valid, video_id, canonical_url, error_message)
        """
        if not url or not isinstance(url, str):
            return False, None, None, "URL cannot be empty."

        clean_url = url.strip()

        # Add https scheme if omitted
        if not clean_url.startswith(("http://", "https://")):
            clean_url = "https://" + clean_url

        try:
            parsed = urlparse(clean_url)
        except Exception as e:
            return False, None, None, f"Malformed URL format: {str(e)}"

        hostname = (parsed.hostname or "").lower()

        # 1. Strict Domain Verification
        if hostname not in cls.ALLOWED_DOMAINS:
            return False, None, None, (
                f"Invalid domain '{hostname}'. Only official YouTube URLs "
                "(youtube.com, youtu.be, m.youtube.com) are permitted."
            )

        video_id = None

        # 2. Path & Query String Extraction
        if "youtu.be" in hostname:
            # Format: https://youtu.be/VIDEO_ID
            path_parts = parsed.path.strip("/").split("/")
            if path_parts and path_parts[0]:
                video_id = path_parts[0]
        else:
            # Format: https://www.youtube.com/watch?v=VIDEO_ID
            if parsed.path == "/watch":
                query_params = parse_qs(parsed.query)
                v_param = query_params.get("v")
                if v_param:
                    video_id = v_param[0]
            elif parsed.path.startswith("/shorts/"):
                # Format: https://www.youtube.com/shorts/VIDEO_ID
                parts = parsed.path.split("/shorts/")
                if len(parts) > 1 and parts[1]:
                    video_id = parts[1].split("/")[0].split("?")[0]
            elif parsed.path.startswith("/embed/"):
                # Format: https://www.youtube.com/embed/VIDEO_ID
                parts = parsed.path.split("/embed/")
                if len(parts) > 1 and parts[1]:
                    video_id = parts[1].split("/")[0].split("?")[0]
            elif parsed.path.startswith("/v/"):
                # Format: https://www.youtube.com/v/VIDEO_ID
                parts = parsed.path.split("/v/")
                if len(parts) > 1 and parts[1]:
                    video_id = parts[1].split("/")[0].split("?")[0]

        # 3. Video ID Integrity Verification
        if not video_id or not cls.VIDEO_ID_REGEX.match(video_id):
            return False, None, None, "Could not extract a valid 11-character YouTube video ID from the provided URL."

        canonical_url = f"https://www.youtube.com/watch?v={video_id}"
        return True, video_id, canonical_url, None
