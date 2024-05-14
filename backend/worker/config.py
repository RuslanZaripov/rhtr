"""Module with Celery configurations to Audio Length worker."""
from kombu import Queue

# Set worker to ack only when return or failing (unhandled expection)
TASK_ACKS_LATE = True

# Worker only gets one task at a time
WORKER_PREFETCH_MULTIPLIER = 1

# Create queue for worker
TASK_QUEUES = [Queue(name="rhtr")]

# Set Redis key TTL (Time to live)
RESULT_EXPIRES = 60 * 60 * 48  # 48 hours in seconds
