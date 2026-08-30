import sys
import uvicorn

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

if __name__ == "__main__":
    print("=" * 70)
    print(">> Starting Agentic YouTube Video Content Extractor & Intelligence Suite")
    print(">> Server URL: http://127.0.0.1:8000")
    print(">> Real-time SSE Streams: http://127.0.0.1:8000/api/stream/{job_id}")
    print(">> Guardrails: Harm, Violence & Abuse Detection Active")
    print(">> Grounding: Strict 6-10 Lines Summarizer Active")
    print(">> HITL: Human-in-the-Loop Gateway Active")
    print("=" * 70)
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
