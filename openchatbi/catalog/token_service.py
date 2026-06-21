"""Token service for authentication with external services."""

import json

import requests

REQUEST_TIMEOUT_SECONDS = 10


class TokenService:
    """Service for managing authentication tokens.

    Handles token application, validation, and authentication
    with external services.
    """

    base_url: str | None = None
    token: str | None = None
    user_name: str | None = None
    password: str | None = None

    def __init__(self, user_name: str, password: str):
        """Initialize token service."""
        self.user_name = user_name
        self.password = password

    def apply_token(self):
        """Apply for authentication token using credentials."""
        assert self.base_url is not None, "base_url must be set before calling apply_token"
        response = requests.post(
            self.base_url + "/apply_token",
            data=json.dumps({"user_name": self.user_name, "password": self.password}),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        resp_json = response.json()
        self.token = resp_json.get("token")


def apply_token_for_user(token_url: str, user_name: str, password: str):
    """Apply for token and return token with username.

    Args:
        token_url (str): Base URL for token service.
        user_name (str): The user name.
        password (str): The password.

    Returns:
        token
    """
    token_service = TokenService(user_name, password)
    token_service.base_url = token_url
    token_service.apply_token()
    return token_service.token
