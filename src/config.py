"""Connection settings and session builders.

Credentials come only from the environment (.env locally). Nothing here prints
or logs a value; errors name the missing keys, never their contents.
"""
import os
from typing import Dict, Mapping, Optional

from dotenv import load_dotenv

REQUIRED_KEYS = (
    "SNOWFLAKE_ACCOUNT",
    "SNOWFLAKE_USER",
    "SNOWFLAKE_PASSWORD",
    "SNOWFLAKE_ROLE",
    "SNOWFLAKE_WAREHOUSE",
    "SNOWFLAKE_DATABASE",
)
DEFAULT_SCHEMA = "BRONZE"


def load_settings(env: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """Read Snowflake settings, raising if any required key is missing or blank.

    `env` exists so tests can pass a dict instead of touching the real .env.
    """
    if env is None:
        load_dotenv()
        env = os.environ

    missing = [key for key in REQUIRED_KEYS if not env.get(key, "").strip()]
    if missing:
        raise ValueError("Missing required settings in .env: " + ", ".join(missing))

    settings = {key: env[key].strip() for key in REQUIRED_KEYS}
    settings["SNOWFLAKE_SCHEMA"] = env.get("SNOWFLAKE_SCHEMA", "").strip() or DEFAULT_SCHEMA
    return settings


def snowflake_connection_params(settings: Mapping[str, str]) -> Dict[str, str]:
    """Map settings to the connection dict the Snowflake clients expect."""
    return {
        "account": settings["SNOWFLAKE_ACCOUNT"],
        "user": settings["SNOWFLAKE_USER"],
        "password": settings["SNOWFLAKE_PASSWORD"],
        "role": settings["SNOWFLAKE_ROLE"],
        "warehouse": settings["SNOWFLAKE_WAREHOUSE"],
        "database": settings["SNOWFLAKE_DATABASE"],
        "schema": settings["SNOWFLAKE_SCHEMA"],
    }


def get_spark_session(app_name: str = "drive_health_monitor"):
    """Start a Spark session backed by Snowflake, using only .env credentials."""
    # Imported here because the package is heavy and tests of the helpers above don't need it.
    from snowflake import snowpark_connect

    params = snowflake_connection_params(load_settings())
    return snowpark_connect.init_spark_session(connection_parameters=params, app_name=app_name)


def smoke_test() -> None:
    """Connect, run a trivial query, and show which objects the session was configured for."""
    # The CURRENT_* context functions are not supported through Snowpark Connect, so the
    # object names below come from the settings, not from the live session.
    settings = load_settings()
    spark = get_spark_session()
    # Stopping explicitly avoids a gRPC thread-pool error when the interpreter exits with the session open.
    try:
        print("select 1 ->", spark.sql("SELECT 1 AS ok").collect()[0]["ok"])
        print("configured role:", settings["SNOWFLAKE_ROLE"])
        print("configured warehouse:", settings["SNOWFLAKE_WAREHOUSE"])
        print("configured database:", settings["SNOWFLAKE_DATABASE"])
        print("configured schema:", settings["SNOWFLAKE_SCHEMA"])
    finally:
        spark.stop()


if __name__ == "__main__":
    smoke_test()
