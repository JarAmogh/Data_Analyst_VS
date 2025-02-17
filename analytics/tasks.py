from celery import shared_task
import logging

from data_platform.celery import app

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,  # Set the default logging level
#     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
# )

# Create a logger instance
logger = logging.getLogger("celery")

@app.task(bind=True)
def example_task(self):
    from funcs.check import Check
    Check.check()
