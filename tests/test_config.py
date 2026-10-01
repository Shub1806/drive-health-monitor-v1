import pytest

from src.config import load_settings, snowflake_connection_params

FULL_ENV = {
    "SNOWFLAKE_ACCOUNT": "org-acct",
    "SNOWFLAKE_USER": "someone",
    "SNOWFLAKE_PASSWORD": "not-a-real-password",
    "SNOWFLAKE_ROLE": "ACCOUNTADMIN",
    "SNOWFLAKE_WAREHOUSE": "DRIVE_WH",
    "SNOWFLAKE_DATABASE": "DRIVE_HEALTH",
}


def test_missing_keys_are_named_but_values_are_not_leaked():
    env = dict(FULL_ENV)
    del env["SNOWFLAKE_USER"]
    env["SNOWFLAKE_PASSWORD"] = "   "

    with pytest.raises(ValueError) as err:
        load_settings(env)

    message = str(err.value)
    assert "SNOWFLAKE_USER" in message
    assert "SNOWFLAKE_PASSWORD" in message
    assert "org-acct" not in message


def test_schema_defaults_to_bronze():
    assert load_settings(FULL_ENV)["SNOWFLAKE_SCHEMA"] == "BRONZE"


def test_connection_params_mapping():
    params = snowflake_connection_params(load_settings({**FULL_ENV, "SNOWFLAKE_SCHEMA": "SILVER"}))

    assert params == {
        "account": "org-acct",
        "user": "someone",
        "password": "not-a-real-password",
        "role": "ACCOUNTADMIN",
        "warehouse": "DRIVE_WH",
        "database": "DRIVE_HEALTH",
        "schema": "SILVER",
    }
