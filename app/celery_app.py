from celery import Celery
from app.config import settings

celery_app = Celery(
    "agentic_youtube_extractor",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.tasks.pipeline_tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 min hard limit
    task_soft_time_limit=240,
    worker_prefetch_multiplier=4,  # High throughput tuning
    worker_max_tasks_per_child=1000,
)
