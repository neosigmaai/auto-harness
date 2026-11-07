import json
from uuid import uuid4

import httpx
import pytest

from scripts import demo_recorded_review as demo


def fixture_report(tmp_path, monkeypatch, status="SUCCEEDED"):
    monkeypatch.setattr(demo, "ROOT", tmp_path)
    directory = tmp_path / "runs" / "recorded"
    directory.mkdir(parents=True)
    job_id = str(uuid4())
    control = {"job_id": job_id, "database_url": "postgresql://fixture:private@localhost/fixture",
               "api_key": "aos_fixture_private_value"}
    (directory / "control.json").write_text(json.dumps(control))
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"run_directory": "runs/recorded", "job": {"job_id": job_id, "status": status}}))
    return report, directory, control


def test_recorded_context_requires_matching_terminal_job(tmp_path, monkeypatch):
    report, directory, control = fixture_report(tmp_path, monkeypatch)
    assert demo.load_recorded_context(report)["job_id"] == control["job_id"]
    control["job_id"] = str(uuid4())
    (directory / "control.json").write_text(json.dumps(control))
    with pytest.raises(demo.InspectionError, match="different jobs"):
        demo.load_recorded_context(report)


def test_active_report_and_external_control_directory_are_rejected(tmp_path, monkeypatch):
    report, _, _ = fixture_report(tmp_path, monkeypatch, "RUNNING")
    with pytest.raises(demo.InspectionError, match="terminal"):
        demo.load_recorded_context(report)
    value = json.loads(report.read_text())
    value["job"]["status"] = "SUCCEEDED"
    value["run_directory"] = "../outside-runs"
    report.write_text(json.dumps(value))
    with pytest.raises(demo.InspectionError, match="missing or invalid"):
        demo.load_recorded_context(report)


def test_child_environments_do_not_forward_provider_credentials(monkeypatch):
    for key in demo.PRIVATE_ENVIRONMENT:
        monkeypatch.setenv(key, "DO_NOT_FORWARD")
    api, client = demo.child_environments({"database_url": "postgresql://fixture/db", "api_key": "aos_service_fixture"})
    assert "DO_NOT_FORWARD" not in str((api, client))
    assert api["NEOSIGMA_OPEN_API_KEY"] == api["BOOTSTRAP_ADMIN_TOKEN"] == api["AOS_API_KEY"] == ""
    assert client["AOS_API_KEY"] == "aos_service_fixture"
    assert "DATABASE_URL" not in client and "NEOSIGMA_OPEN_API_KEY" not in client


@pytest.mark.asyncio
async def test_inspection_api_rejects_all_mutations_before_database_access():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=demo.create_inspection_app()), base_url="http://test") as client:
        for method, path in (("POST", "/v1/jobs"), ("POST", "/v1/jobs/anything/cancel"),
                             ("POST", "/v1/orgs"), ("DELETE", "/v1/jobs/anything")):
            response = await client.request(method, path, json={})
            assert response.status_code == 405
        assert (await client.get("/v1/healthz")).status_code == 200
