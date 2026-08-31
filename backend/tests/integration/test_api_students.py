"""
Integration tests for the /api/students endpoints.

Uses FastAPI's TestClient via the shared ``integration_client`` fixture
defined in conftest.py.  The fixture patches recognition_service and lifespan
via unittest.mock.patch() — NOT via sys.modules — so unit tests running in the
same process are not affected.

Test coverage:
  - POST /api/students        → 201 Created
  - POST same roll_number     → 409 Conflict
  - GET  /api/students/{rn}   → 200 with student data
  - GET  /api/students/none   → 404 Not Found
  - PUT  /api/students/{rn}   → 200 Updated
  - DELETE /api/students/{rn} → 200 Deleted

Requirements: 3.1, 3.2, 3.4, 3.5, 3.6
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Tests — all use `integration_client` fixture from conftest.py
# ---------------------------------------------------------------------------

STUDENT_PAYLOAD = {
    "roll_number": "TEST001",
    "name": "Alice Test",
    "department": "CS",
    "email": "alice@example.com",
    "phone": "1234567890",
}


def test_create_student_returns_201(integration_client):
    """POST /api/students with valid data → 201 and student data in response."""
    response = integration_client.post("/api/students/", json=STUDENT_PAYLOAD)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["roll_number"] == STUDENT_PAYLOAD["roll_number"]
    assert body["name"] == STUDENT_PAYLOAD["name"]
    assert body["department"] == STUDENT_PAYLOAD["department"]


def test_create_duplicate_student_returns_409(integration_client):
    """POST same roll_number twice → second request returns 409 Conflict."""
    integration_client.post("/api/students/", json=STUDENT_PAYLOAD)
    response = integration_client.post("/api/students/", json=STUDENT_PAYLOAD)
    assert response.status_code == 409, response.text


def test_get_existing_student_returns_200(integration_client):
    """GET /api/students/{rn} for existing student → 200 with student data."""
    integration_client.post("/api/students/", json=STUDENT_PAYLOAD)
    response = integration_client.get(f"/api/students/{STUDENT_PAYLOAD['roll_number']}")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["roll_number"] == STUDENT_PAYLOAD["roll_number"]
    assert body["name"] == STUDENT_PAYLOAD["name"]


def test_get_nonexistent_student_returns_404(integration_client):
    """GET /api/students/nonexistent → 404 Not Found."""
    response = integration_client.get("/api/students/DOES_NOT_EXIST_9999")
    assert response.status_code == 404, response.text


def test_update_student_returns_200(integration_client):
    """PUT /api/students/{rn} with valid update → 200 and updated data."""
    integration_client.post("/api/students/", json=STUDENT_PAYLOAD)
    update_payload = {"name": "Alice Updated", "department": "Math"}
    response = integration_client.put(
        f"/api/students/{STUDENT_PAYLOAD['roll_number']}",
        json=update_payload,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["name"] == "Alice Updated"
    assert body["department"] == "Math"
    assert body["roll_number"] == STUDENT_PAYLOAD["roll_number"]


def test_update_nonexistent_student_returns_404(integration_client):
    """PUT /api/students/nonexistent → 404 Not Found."""
    response = integration_client.put("/api/students/DOES_NOT_EXIST_9999", json={"name": "Ghost"})
    assert response.status_code == 404, response.text


def test_delete_student_returns_200(integration_client):
    """DELETE /api/students/{rn} → 200 with success message."""
    integration_client.post("/api/students/", json=STUDENT_PAYLOAD)
    rn = STUDENT_PAYLOAD["roll_number"]

    with patch("backend.services.student_service.shutil.rmtree"), \
         patch("backend.services.student_service.EmbeddingStore"), \
         patch("backend.services.student_service.os.path.isdir", return_value=False):
        response = integration_client.delete(f"/api/students/{rn}")

    assert response.status_code == 200, response.text
    body = response.json()
    assert "deleted" in body.get("message", "").lower()


def test_delete_nonexistent_student_returns_404(integration_client):
    """DELETE /api/students/nonexistent → 404 Not Found."""
    with patch("backend.services.student_service.shutil.rmtree"), \
         patch("backend.services.student_service.EmbeddingStore"):
        response = integration_client.delete("/api/students/DOES_NOT_EXIST_9999")
    assert response.status_code == 404, response.text


def test_student_not_found_after_delete(integration_client):
    """After deleting a student, GET should return 404."""
    integration_client.post("/api/students/", json=STUDENT_PAYLOAD)
    rn = STUDENT_PAYLOAD["roll_number"]

    with patch("backend.services.student_service.shutil.rmtree"), \
         patch("backend.services.student_service.EmbeddingStore"), \
         patch("backend.services.student_service.os.path.isdir", return_value=False):
        integration_client.delete(f"/api/students/{rn}")

    response = integration_client.get(f"/api/students/{rn}")
    assert response.status_code == 404, response.text
