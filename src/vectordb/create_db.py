import sys
sys.path.append("../")
from src.vectordb.search import *
import logging

from src.vectordb.ingest import create_knowledge_vectordb

logger = logging.getLogger(__name__)
logging.basicConfig(encoding='utf-8', level=logging.DEBUG)


def run():
    logging.info("Creating database for Knowledge Base")
    try:
        create_knowledge_vectordb()
    except Exception as e:
        logger.error(f"Error for Wikivoyage Documents: {e}")


if __name__ == "__main__":
    run()
