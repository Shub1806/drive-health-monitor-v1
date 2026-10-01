import os

import pytest


@pytest.fixture(scope="session")
def spark():
    """Plain local Spark for transform tests; no Snowflake connection involved."""
    # Spark 3.5 needs Java 17; the Java that ships with the Snowflake package is one.
    import jdk4py

    os.environ["JAVA_HOME"] = str(jdk4py.JAVA_HOME)
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[1]")
        .appName("drive-health-tests")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()
