"""Token service for authentication with external services."""

import json

import requests


class TokenService:
    """Service for managing authentication tokens.

    Handles token application, validation, and authentication
    with external services.
    """

    base_url = None
    token = None
    user_name = None
    password = None

    def __init__(self, user_name: str, password: str):
        """Initialize token service."""
        self.user_name = user_name
        self.password = password

    def apply_token(self):
        """Apply for authentication token using credentials."""
        response = requests.post(
            self.base_url + "/apply_token", data=json.dumps({"user_name": self.user_name, "password": self.password})
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
