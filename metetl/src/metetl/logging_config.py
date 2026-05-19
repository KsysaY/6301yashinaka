import logging
import os

def setup_logging():
    if not os.path.exists("logs"):
        os.makedirs("logs")

    logger = logging.getLogger("metetl")
    logger.setLevel(logging.DEBUG)

    file = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

    console = logging.Formatter('%(levelname)s: %(message)s')

    fh = logging.FileHandler("logs/metetl.log", encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(file)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(console)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger