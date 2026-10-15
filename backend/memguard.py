import gc

import psutil

from backend import config


class MemoryLimitError(Exception):
    pass


def check_memory() -> None:
    used = psutil.Process().memory_info().rss / 1048576
    if used > config.MEM_SOFT_LIMIT_MB:
        gc.collect()
        used = psutil.Process().memory_info().rss / 1048576
        if used > config.MEM_SOFT_LIMIT_MB:
            raise MemoryLimitError(
                "Memory limit reached on the free server. "
                "Please try again in a minute, or upload a smaller file."
            )