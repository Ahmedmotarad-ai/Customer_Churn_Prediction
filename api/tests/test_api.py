"""API tests: health, valid/invalid predictions, robustness, failures."""
import app as app_module


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_shape(client, flagged_customer):
    r = client.post("/predict", json=flagged_customer)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"churn", "churn_label", "churn_probability"}
    assert body["churn"] in (0, 1)
    assert body["churn_label"] == ("Yes" if body["churn"] else "No")
    assert 0.0 <= body["churn_probability"] <= 1.0


def test_predict_flagged_customer(client, flagged_customer):
    # first dataset row is actually labeled No, but the model flags it
    # as a churner (proba 0.8193) — a stable false positive used here
    # to pin the decision boundary, not as ground truth
    body = client.post("/predict", json=flagged_customer).json()
    assert body["churn"] == 1
    assert body["churn_probability"] > 0.5


def test_predict_low_risk_stays(client, low_risk):
    body = client.post("/predict", json=low_risk).json()
    assert body["churn"] == 0
    assert body["churn_probability"] < 0.5


def test_predict_negative_tenure_rejected(client, flagged_customer):
    bad = dict(flagged_customer, tenure=-5)
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_missing_field_rejected(client, flagged_customer):
    bad = dict(flagged_customer)
    bad.pop("Contract")
    assert client.post("/predict", json=bad).status_code == 422


def test_health_reports_model_details(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "LogisticRegression"
    assert body["n_features"] == 18


def test_model_loaded_once_and_reused(client, flagged_customer):
    from app import load_bundle
    first = load_bundle()
    client.post("/predict", json=flagged_customer)
    assert load_bundle() is first


def test_predict_unknown_category_tolerated(client, flagged_customer):
    # encoder uses handle_unknown="ignore": unseen values must not crash
    weird = dict(flagged_customer, Contract="Weekly")
    r = client.post("/predict", json=weird)
    assert r.status_code == 200
    assert 0.0 <= r.json()["churn_probability"] <= 1.0


def test_missing_model_predict_returns_503(client, flagged_customer, tmp_path,
                                           monkeypatch):
    monkeypatch.setattr(app_module, "MODEL_PATH", tmp_path / "nope.pkl")
    monkeypatch.setattr(app_module, "_bundle", None)
    r = client.post("/predict", json=flagged_customer)
    assert r.status_code == 503
    assert "train.py" in r.json()["detail"]


def test_missing_model_health_reports_degraded(client, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "MODEL_PATH", tmp_path / "nope.pkl")
    monkeypatch.setattr(app_module, "_bundle", None)
    r = client.get("/health")
    assert r.status_code == 503
    body = r.json()
    assert body["model_loaded"] is False
    assert body["status"] == "degraded"


def test_corrupt_model_returns_clean_500(client, flagged_customer, tmp_path,
                                         monkeypatch):
    bad = tmp_path / "bad.pkl"
    bad.write_bytes(b"not a pickle")
    monkeypatch.setattr(app_module, "MODEL_PATH", bad)
    monkeypatch.setattr(app_module, "_bundle", None)
    r = client.post("/predict", json=flagged_customer)
    assert r.status_code == 500
    assert r.json() == {"detail": "model failed to load."}


def test_threshold_env_changes_label(client, flagged_customer, monkeypatch):
    # known churner scores 0.8193: movable threshold flips the label
    monkeypatch.setenv("CHURN_THRESHOLD", "0.99")
    assert client.post("/predict", json=flagged_customer).json()["churn"] == 0
    monkeypatch.setenv("CHURN_THRESHOLD", "0.01")
    assert client.post("/predict", json=flagged_customer).json()["churn"] == 1
    monkeypatch.setenv("CHURN_THRESHOLD", "not-a-number")
    assert client.post("/predict", json=flagged_customer).json()["churn"] == 1
