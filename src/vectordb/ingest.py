import logging
import sys
from typing import Optional, Callable

sys.path.append("../")
logger = logging.getLogger(__name__)
logging.basicConfig(encoding='utf-8', level=logging.DEBUG)
from src.vectordb.helpers import preprocess_kb
from src.vectordb.schema import KnowledgeBase
from src.vectordb.helpers import set_uri
import lancedb
import pandas as pd
from src.data_directories import *


def _create_table_and_ingest_data(table_name: str, schema: object,
                                  preprocessor: Optional[Callable] = None):
    """
    Generalized function to create a table and ingest data into the database.
    Args:
        - table_name: str, name of the table to create.
        - schema: object, schema of the table.
        - data_fetcher: Callable, function to fetch the data.
        - preprocessor: Optional[Callable], function to preprocess the data (default is None).
    """
    uri = set_uri(run_local=True)

    db = lancedb.connect(uri)
    logger.info(f"Connected to DB. Reading data for table {table_name} now...")

    df = pd.read_csv(kbase_dir + "merged_listing.csv")

    if preprocessor:
        df = preprocessor(df)

    logger.info(f"Finished reading data for {table_name}, attempting to create table and ingest the data...")

    db.drop_table(table_name, ignore_missing=True)
    table = db.create_table(table_name, schema=schema)

    table.add(df.to_dict('records'))
    logger.info(f"Completed ingestion for {table_name}.")


def create_knowledge_vectordb():
    """
    Creates the table for the knowledge base and ingests data
    """

    _create_table_and_ingest_data(
        table_name="conv_trs_kb",
        schema=KnowledgeBase,
        preprocessor=preprocess_kb,
    )


if __name__ == "__main__":
    create_knowledge_vectordb()
