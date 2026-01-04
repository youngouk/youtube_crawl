import logging
import sys


def setup_logger(name: str = "MedicalRAG") -> logging.Logger:
    """로거를 설정하고 반환합니다."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add handler if not exists
    if not logger.handlers:
        logger.addHandler(handler)

    return logger


logger = setup_logger()
