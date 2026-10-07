import pytest

from app import app


@pytest.fixture
def client():
    app.config.update(TESTING=True, SECRET_KEY="test-secret")
    with app.test_client() as test_client:
        yield test_client


def test_health_endpoint_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_homepage_renders(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b"Keyword Spotter" in response.data


def test_transcribe_without_file_shows_validation_message(client):
    response = client.post("/transcribe", data={})

    assert response.status_code == 200
    assert b"File not found" in response.data
