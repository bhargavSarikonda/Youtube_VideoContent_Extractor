import pytest
from app.agents.url_agent import YouTubeURLAgent


def test_valid_youtube_urls():
    valid_cases = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "http://youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s&feature=shared",
        "youtube.com/watch?v=dQw4w9WgXcQ",
        "youtu.be/dQw4w9WgXcQ",
    ]
    for url in valid_cases:
        is_valid, video_id, canonical_url, err = YouTubeURLAgent.validate_and_extract(url)
        assert is_valid is True, f"Failed on valid URL: {url}"
        assert video_id == "dQw4w9WgXcQ"
        assert canonical_url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert err is None


def test_invalid_and_malicious_urls():
    invalid_cases = [
        "https://vimeo.com/12345678",
        "https://dailymotion.com/video/x7tgad0",
        "https://google.com",
        "https://fake-youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.malicious.org/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=short",  # Invalid ID length
        "not_a_url",
        "",
        None,
    ]
    for url in invalid_cases:
        is_valid, video_id, canonical_url, err = YouTubeURLAgent.validate_and_extract(url)
        assert is_valid is False, f"Should have failed on invalid URL: {url}"
        assert video_id is None
        assert err is not None
