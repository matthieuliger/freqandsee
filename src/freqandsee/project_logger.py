import logging
from pretty_logger import get_logger  # type: ignore

logger = get_logger(
    name="freqandsee",
    path="logs/freqandsee.log",
    level=logging.DEBUG,
    add_console_hander=True,
)
