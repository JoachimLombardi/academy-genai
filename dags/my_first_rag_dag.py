"""
## Simple RAG DAG to ingest new knowledge data into a vector database

This DAG ingests text data from markdown files, chunks the text, and then ingests 
the chunks into a Weaviate vector database.
"""

from airflow.decorators import dag, task
from airflow.models.baseoperator import chain
from airflow.operators.empty import EmptyOperator
from airflow.providers.weaviate.hooks.weaviate import WeaviateHook
from airflow.providers.weaviate.operators.weaviate import WeaviateIngestOperator
from pendulum import datetime, duration
import os
import logging
import pandas as pd
from weaviate.classes.config import Configure

t_log = logging.getLogger("airflow.task")

# Variables used in the DAG
_INGESTION_FOLDERS_LOCAL_PATHS = os.getenv("INGESTION_FOLDERS_LOCAL_PATHS")

_WEAVIATE_CONN_ID = os.getenv("WEAVIATE_CONN_ID")
_WEAVIATE_COLLECTION_NAME = os.getenv("WEAVIATE_CLASS_NAME")
_WEAVIATE_VECTORIZER = os.getenv("WEAVIATE_VECTORIZER")
_WEAVIATE_SCHEMA_PATH = os.getenv("WEAVIATE_SCHEMA_PATH")

_CREATE_COLLECTION_TASK_ID = "create_collection"
_COLLECTION_ALREADY_EXISTS_TASK_ID = "collection_already_exists"

@dag(
    dag_display_name="📚 Ingest Knowledge Base",
    start_date=datetime(2024, 5, 1),
    schedule="@daily",
    catchup=False,
    max_consecutive_failed_dag_runs=5,
    tags=["RAG"],
    default_args={
        "retries": 3,
        "retry_delay": duration(minutes=5),
        "owner": "jojo",
    },
    doc_md=__doc__,
    description="Ingest knowledge into the vector database for RAG.",
)
def my_first_rag_dag():

    @task.branch(retries=4)
    def check_collection(
        conn_id: str,
        collection_name: str,
        create_collection_task_id: str,
        collection_already_exists_task_id: str,
    ) -> str:
        """
        Check if the target collection exists in the Weaviate schema.
        Args:
            conn_id: The connection ID to use.
            collection_name: The name of the collection to check.
            create_collection_task_id: The task ID to execute if the collection does not exist.
            collection_already_exists_task_id: The task ID to execute if the collection already exists.
        Returns:
            str: Task ID of the next task to execute.
        """

        # connect to Weaviate using the Airflow connection `conn_id`
        hook = WeaviateHook(conn_id)

        client = hook.get_conn()

        # check if the collection exists in the Weaviate database
        collection = client.collections.exists(collection_name)

        print(client.is_ready())
        
        if collection:
            t_log.info(f"Collection {collection_name} already exists.")
            return collection_already_exists_task_id
        else:
            t_log.info(f"Class {collection_name} does not exist yet.")
            return create_collection_task_id

    check_collection_obj = check_collection(
        conn_id=_WEAVIATE_CONN_ID,
        collection_name=_WEAVIATE_COLLECTION_NAME,
        create_collection_task_id=_CREATE_COLLECTION_TASK_ID,
        collection_already_exists_task_id=_COLLECTION_ALREADY_EXISTS_TASK_ID,
    )

    @task
    def create_collection(conn_id: str, collection_name: str):
        """
        Create a collection in the Weaviate schema.
        Args:
            conn_id: The connection ID to use.
            collection_name: The name of the collection to create.
        """
        hook = WeaviateHook(conn_id)
        hook.create_collection(name=collection_name, vectorizer_config=Configure.Vectorizer.text2vec_transformers(model="sentence-transformers/all-MiniLM-L6-v2")
        )
        
    create_collection_obj = create_collection(conn_id=_WEAVIATE_CONN_ID,collection_name=_WEAVIATE_COLLECTION_NAME)

    collection_already_exists = EmptyOperator(task_id=_COLLECTION_ALREADY_EXISTS_TASK_ID)

    weaviate_ready = EmptyOperator(task_id="weaviate_ready", trigger_rule="none_failed")

    chain(check_collection_obj, [create_collection_obj, collection_already_exists], weaviate_ready)

my_first_rag_dag()