# logging_config.py
import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    
    logger = logging.getLogger('api_server')
    if logger.hasHandlers():
        logger.info('Logger already configured')
        return logger
    
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = RotatingFileHandler('api_server.log', maxBytes=1024*1024*5, backupCount=5)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger