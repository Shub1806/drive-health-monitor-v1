"""Snowflake access for the RAG layer.

Snowflake is used for exact lookups of stored anomaly results and drive telemetry, and later for
storing knowledge metadata and explanations. It is not used for embeddings or retrieval.
"""


def connect():
    """Open a Snowflake connection with the credentials from .env, and nothing else."""
    import snowflake.connector

    from src import config

    return snowflake.connector.connect(**config.snowflake_connection_params(config.load_settings()))
