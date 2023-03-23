from pydantic import BaseSettings
from multiprocessing import cpu_count


class Settings(BaseSettings):
    websocket_connection_limit: int = 1000
    worker_limit: int = cpu_count() - 1
    task_timeout: int = 5


settings = Settings()
