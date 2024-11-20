import pytest
from app import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_predict_success(client):
    with open("test_image.jpg", "rb") as img:
        response = client.post("/predict", data={"image": img})
    assert response.status_code == 200
    assert response.json["status"] == "success"

def test_predict_no_file(client):
    response = client.post("/predict", data={})
    assert response.status_code == 400
    assert response.json["status"] == "fail"
    assert "No file part in the request" in response.json["message"]

def test_large_file(client):
    # Simulate a large file
    large_content = b"x" * (1_000_001)
    response = client.post("/predict", data={"image": (large_content, "large_file.jpg")})
    assert response.status_code == 413
    assert "Payload content length greater than maximum allowed" in response.json["message"]
