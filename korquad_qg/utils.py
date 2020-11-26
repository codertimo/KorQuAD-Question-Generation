import logging
import sys
from typing import Optional, TextIO

import tqdm


class TQDMHandler(logging.Handler):
    def __init__(self, stream: Optional[TextIO] = None):
        super().__init__()
        if stream is None:
            stream = sys.stdout

        self.stream = stream

    def flush(self):
        self.acquire()
        try:
            if self.stream and hasattr(self.stream, "flush"):
                self.stream.flush()
        finally:
            self.release()

    def emit(self, record: logging.LogRecord):
        try:
            message = self.format(record)
            tqdm.tqdm.write(message, self.stream)
            self.flush()
        except RecursionError:  # see https://github.com/python/cpython/blob/a62ad4730c9b575f140f24074656c0257c86a09a/Lib/logging/__init__.py#L1086
            raise
        except Exception:
            self.handleError(record)
