from typing import Any

import requests
from sqlalchemy import Engine, create_engine

from openchatbi.catalog.token_service import apply_token_for_user
from openchatbi.utils import log


def get_requests_session(token: str, header_extra_params: dict) -> requests.Session:
    """Create HTTP session with bearer token authentication."""
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}"})
    if header_extra_params:
        session.headers.update(header_extra_params)
    return session


def create_sqlalchemy_engine_instance(data_warehouse_config: dict[str, Any]) -> Engine:
    """
    Create SQLAlchemy engine instance from data warehouse config

    Args:
        data_warehouse_config: Config dict with 'uri' and optional 'token_service'

    Returns:
        Configured SQLAlchemy engine
    """
    database_uri = data_warehouse_config.get("uri")

    engine_args = {"echo": True}

    # Handle Presto authentication
    if "presto" in database_uri and "token_service" in data_warehouse_config:
        token_service = data_warehouse_config.get("token_service")
        user_name = data_warehouse_config.get("user_name")
        password = data_warehouse_config.get("password")
        header_extra_params = data_warehouse_config.get("header_extra_params", {})
        token = apply_token_for_user(token_service, user_name, password)
        log(f"Applied presto token: {token} for user: {user_name}")
        engine_args["connect_args"] = {
            "protocol": "https",
            "requests_session": get_requests_session(token, header_extra_params),
        }
        database_uri = database_uri.format(user_name=user_name)

    engine = create_engine(database_uri, **engine_args)

    return engine
