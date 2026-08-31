import requests
from typing import Any, Optional


class APIClient:
    """Thin wrapper around requests.Session for all Backend API calls."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def get(self, path: str, **kwargs) -> requests.Response:
        return self.session.get(self._url(path), **kwargs)

    def post(self, path: str, **kwargs) -> requests.Response:
        return self.session.post(self._url(path), **kwargs)

    def put(self, path: str, **kwargs) -> requests.Response:
        return self.session.put(self._url(path), **kwargs)

    def delete(self, path: str, **kwargs) -> requests.Response:
        return self.session.delete(self._url(path), **kwargs)

    def download(self, path: str, params: Optional[dict] = None) -> bytes:
        """Download a file (e.g., xlsx export) and return raw bytes."""
        resp = self.session.get(self._url(path), params=params)
        resp.raise_for_status()
        return resp.content
