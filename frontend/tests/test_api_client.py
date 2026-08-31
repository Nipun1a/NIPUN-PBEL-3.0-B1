"""
Unit tests for frontend/utils/api_client.py (Task 13.5).

Tests:
  - Correct URL construction via _url()
  - GET/POST/PUT/DELETE delegate to session methods with correct args
  - Content-Type header is set to application/json on init
  - ConnectionError propagates from GET
  - download() returns response.content bytes
"""
from __future__ import annotations

import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import requests

from frontend.utils.api_client import APIClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client_with_mock_session(base_url="http://localhost:8000"):
    """Create an APIClient and replace its session with a MagicMock."""
    client = APIClient(base_url)
    mock_session = MagicMock(spec=requests.Session)
    # Propagate the real headers update behaviour — just record that it was called
    mock_session.headers = MagicMock()
    client.session = mock_session
    return client, mock_session


# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------

class TestURLConstruction:
    def test_url_joins_base_and_path(self):
        client = APIClient("http://localhost:8000")
        assert client._url("/api/test") == "http://localhost:8000/api/test"

    def test_url_strips_trailing_slash_from_base(self):
        client = APIClient("http://localhost:8000/")
        assert client._url("/api/test") == "http://localhost:8000/api/test"

    def test_url_handles_path_without_leading_slash(self):
        client = APIClient("http://localhost:8000")
        assert client._url("api/test") == "http://localhost:8000/api/test"

    def test_url_with_custom_base(self):
        client = APIClient("http://192.168.1.10:9000")
        assert client._url("/api/attendance") == "http://192.168.1.10:9000/api/attendance"

    def test_url_nested_path(self):
        client = APIClient("http://localhost:8000")
        assert client._url("/api/students/101") == "http://localhost:8000/api/students/101"


# ---------------------------------------------------------------------------
# Content-Type header
# ---------------------------------------------------------------------------

class TestContentTypeHeader:
    def test_content_type_set_to_application_json(self):
        """The session must be initialised with Content-Type: application/json."""
        with patch("frontend.utils.api_client.requests.Session") as MockSession:
            mock_session_instance = MagicMock()
            MockSession.return_value = mock_session_instance

            APIClient("http://localhost:8000")

            mock_session_instance.headers.update.assert_called_once_with(
                {"Content-Type": "application/json"}
            )


# ---------------------------------------------------------------------------
# HTTP method delegation
# ---------------------------------------------------------------------------

class TestHTTPMethods:
    def test_get_calls_session_get_with_correct_url(self):
        client, mock_session = _make_client_with_mock_session()
        client.get("/api/students")
        mock_session.get.assert_called_once_with("http://localhost:8000/api/students")

    def test_get_forwards_kwargs(self):
        client, mock_session = _make_client_with_mock_session()
        client.get("/api/students", params={"page": 1})
        mock_session.get.assert_called_once_with(
            "http://localhost:8000/api/students", params={"page": 1}
        )

    def test_post_calls_session_post_with_correct_url(self):
        client, mock_session = _make_client_with_mock_session()
        client.post("/api/students", json={"name": "Alice"})
        mock_session.post.assert_called_once_with(
            "http://localhost:8000/api/students", json={"name": "Alice"}
        )

    def test_put_calls_session_put_with_correct_url(self):
        client, mock_session = _make_client_with_mock_session()
        client.put("/api/settings/", json={"recognition_threshold": 0.7})
        mock_session.put.assert_called_once_with(
            "http://localhost:8000/api/settings/",
            json={"recognition_threshold": 0.7},
        )

    def test_delete_calls_session_delete_with_correct_url(self):
        client, mock_session = _make_client_with_mock_session()
        client.delete("/api/attendance/5")
        mock_session.delete.assert_called_once_with(
            "http://localhost:8000/api/attendance/5"
        )

    def test_get_returns_session_response(self):
        client, mock_session = _make_client_with_mock_session()
        mock_response = MagicMock()
        mock_session.get.return_value = mock_response

        result = client.get("/api/students")
        assert result is mock_response

    def test_post_returns_session_response(self):
        client, mock_session = _make_client_with_mock_session()
        mock_response = MagicMock()
        mock_session.post.return_value = mock_response

        result = client.post("/api/students", json={})
        assert result is mock_response

    def test_put_returns_session_response(self):
        client, mock_session = _make_client_with_mock_session()
        mock_response = MagicMock()
        mock_session.put.return_value = mock_response

        result = client.put("/api/settings/", json={})
        assert result is mock_response

    def test_delete_returns_session_response(self):
        client, mock_session = _make_client_with_mock_session()
        mock_response = MagicMock()
        mock_session.delete.return_value = mock_response

        result = client.delete("/api/attendance/5")
        assert result is mock_response


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------

class TestErrorPropagation:
    def test_connection_error_propagates_from_get(self):
        client, mock_session = _make_client_with_mock_session()
        mock_session.get.side_effect = requests.ConnectionError("Connection refused")

        with pytest.raises(requests.ConnectionError):
            client.get("/api/students")

    def test_connection_error_propagates_from_post(self):
        client, mock_session = _make_client_with_mock_session()
        mock_session.post.side_effect = requests.ConnectionError("Connection refused")

        with pytest.raises(requests.ConnectionError):
            client.post("/api/students", json={})

    def test_connection_error_propagates_from_put(self):
        client, mock_session = _make_client_with_mock_session()
        mock_session.put.side_effect = requests.ConnectionError("Connection refused")

        with pytest.raises(requests.ConnectionError):
            client.put("/api/settings/", json={})

    def test_connection_error_propagates_from_delete(self):
        client, mock_session = _make_client_with_mock_session()
        mock_session.delete.side_effect = requests.ConnectionError("Connection refused")

        with pytest.raises(requests.ConnectionError):
            client.delete("/api/attendance/5")

    def test_connection_error_propagates_from_download(self):
        client, mock_session = _make_client_with_mock_session()
        mock_session.get.side_effect = requests.ConnectionError("Connection refused")

        with pytest.raises(requests.ConnectionError):
            client.download("/api/export/attendance")


# ---------------------------------------------------------------------------
# download()
# ---------------------------------------------------------------------------

class TestDownload:
    def test_download_returns_response_content_bytes(self):
        client, mock_session = _make_client_with_mock_session()
        fake_bytes = b"PK\x03\x04" + b"\x00" * 100  # fake xlsx header
        mock_response = MagicMock()
        mock_response.content = fake_bytes
        mock_session.get.return_value = mock_response

        result = client.download("/api/export/attendance")
        assert result == fake_bytes
        assert isinstance(result, bytes)

    def test_download_calls_session_get_with_correct_url(self):
        client, mock_session = _make_client_with_mock_session()
        mock_response = MagicMock()
        mock_response.content = b"data"
        mock_session.get.return_value = mock_response

        client.download("/api/export/attendance")
        mock_session.get.assert_called_once_with(
            "http://localhost:8000/api/export/attendance", params=None
        )

    def test_download_passes_params_to_session_get(self):
        client, mock_session = _make_client_with_mock_session()
        mock_response = MagicMock()
        mock_response.content = b"data"
        mock_session.get.return_value = mock_response

        client.download("/api/export/attendance", params={"date": "2024-01-15"})
        mock_session.get.assert_called_once_with(
            "http://localhost:8000/api/export/attendance",
            params={"date": "2024-01-15"},
        )

    def test_download_calls_raise_for_status(self):
        """download() must call raise_for_status to propagate HTTP errors."""
        client, mock_session = _make_client_with_mock_session()
        mock_response = MagicMock()
        mock_response.content = b"data"
        mock_session.get.return_value = mock_response

        client.download("/api/export/attendance")
        mock_response.raise_for_status.assert_called_once()

    def test_download_empty_response_returns_empty_bytes(self):
        client, mock_session = _make_client_with_mock_session()
        mock_response = MagicMock()
        mock_response.content = b""
        mock_session.get.return_value = mock_response

        result = client.download("/api/export/students")
        assert result == b""
