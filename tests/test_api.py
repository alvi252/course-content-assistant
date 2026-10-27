from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient

from conftest import CountingEmbeddings, FakeLLM, new_idea, with_settings
from course_assistant.api import create_app
from course_assistant.config import Settings
from course_assistant.embeddings import HashingEmbeddings
from course_assistant.errors import ProviderError, ProviderNotConfiguredError
from course_assistant.services import Services, build_services

IDEA_KEYS = {
    "id", "status", "format", "format_label", "title", "hook", "outline", "quote", "source",
    "created_at", "used_at", "script", "caption", "warnings",
}  # fmt: skip
SOURCE_KEYS = {"chunk_id", "doc_path", "course", "lesson", "section", "text"}
USAGE_KEYS = {"input_tokens", "output_tokens", "embedding_tokens", "cost_usd"}


@pytest.fixture
def client(services: Services) -> Iterator[TestClient]:
    with TestClient(create_app(services.settings, services=services)) as c:
        yield c


def _make(client: TestClient, **body: Any) -> dict[str, Any]:
    response = client.post("/ideas", json=body)
    assert response.status_code == 200, response.text
    return response.json()  # type: ignore[no-any-return]


def test_post_ideas_returns_the_contract(client: TestClient) -> None:
    data = _make(client, count=4)
    assert set(data) == {
        "ideas", "dropped", "requested", "candidates", "fresh_passages_left", "empty_reason",
        "model", "latency_ms", "usage",
    }  # fmt: skip
    assert (data["requested"], data["candidates"], data["fresh_passages_left"]) == (4, 4, 8)
    assert data["empty_reason"] is None
    assert data["model"] == "fake-llm"
    assert set(data["usage"]) == USAGE_KEYS
    assert set(data["latency_ms"]) == {"selection", "llm", "total"}
    idea = data["ideas"][0]
    assert set(idea) == IDEA_KEYS
    assert set(idea["source"]) == SOURCE_KEYS
    assert idea["status"] == "suggested"
    assert idea["format_label"] == "Educational post"
    assert idea["source"]["course"] in {"Sunday Batch Cooking", "Weeknight Rescue Meals"}
    assert idea["source"]["lesson"].startswith("Lesson ")
    assert idea["quote"] in idea["source"]["text"]
    assert idea["used_at"] is None and idea["script"] is None and idea["caption"] is None


def test_post_ideas_defaults_come_from_settings(client: TestClient, fake_llm: FakeLLM) -> None:
    data = _make(client)
    assert data["requested"] == 15
    assert data["candidates"] == 8  # only eight eligible passages in the fixture library
    assert len(data["ideas"]) == 8
    assert "Use the format" not in fake_llm.calls[0]["system"]  # null format = any


def test_post_ideas_passes_format_and_topic(client: TestClient, fake_llm: FakeLLM) -> None:
    client.app.state.services.retriever.min_similarity = 0.0  # type: ignore[attr-defined]
    _make(client, count=3, format="post", topic="  breakfast  ")
    call = fake_llm.calls[0]
    assert 'Use the format "post"' in call["system"]
    assert '"breakfast"' in call["user"]


def test_skip_used_days_reaches_the_selection(
    client: TestClient, services: Services, fake_llm: FakeLLM
) -> None:
    chunk = next(c for c in services.retriever.all_chunks() if c.heading == "Tomato sauce")
    [idea] = services.store.add_ideas([new_idea(chunk)])
    ten_days_ago = datetime.now(UTC) - timedelta(days=10)
    services.store.set_idea_status(idea.id, "used", now=ten_days_ago)

    def sent_tomato(**body: Any) -> bool:
        fake_llm.calls.clear()
        _make(client, count=30, **body)
        return "tomato sauce becomes pasta night" in fake_llm.calls[0]["user"].lower()

    assert sent_tomato(skip_used_days=30) is False  # used 10 days ago: inside a 30-day window
    assert sent_tomato(skip_used_days=5) is True  # outside a 5-day window
    assert sent_tomato(skip_used_days=0) is True
    assert sent_tomato() is False  # the default (30) applies when omitted


def test_idea_warns_about_numbers_not_in_the_passage(client: TestClient, fake_llm: FakeLLM) -> None:
    fake_llm.idea_edit = lambda ideas: [
        dict(i, title="Seventy dinners from one tray", hook="Same tray", outline=["Roast"])
        for i in ideas
    ]
    idea = _make(client, count=1)["ideas"][0]
    assert idea["warnings"] == ["Number seventy is in the title but not in the source passage."]


def test_post_ideas_reports_dropped(client: TestClient, fake_llm: FakeLLM) -> None:
    fake_llm.fmt = "carousel"
    data = _make(client, count=2, format="post")
    assert data["ideas"] == []
    assert data["dropped"][0] == {"reason": "wrong_format", "title": "Idea 1"}


def test_post_ideas_with_nothing_eligible_skips_the_llm(
    client: TestClient, fake_llm: FakeLLM
) -> None:
    for idea in _make(client, count=30)["ideas"]:
        assert client.post(f"/ideas/{idea['id']}/use").status_code == 200
    fake_llm.calls.clear()

    data = _make(client, count=5)

    assert fake_llm.calls == []
    assert data["ideas"] == [] and data["dropped"] == []
    assert (data["candidates"], data["fresh_passages_left"], data["model"]) == (0, 0, None)
    assert set(data["usage"]) == USAGE_KEYS


@pytest.mark.parametrize(
    "body",
    [
        {"count": 0},
        {"count": 31},
        {"count": "many"},
        {"format": "reel"},
        {"topic": "x" * 201},
        {"skip_used_days": -1},
        {"skip_used_days": 366},
    ],
)
def test_post_ideas_validates_input(client: TestClient, body: dict[str, Any]) -> None:
    assert client.post("/ideas", json=body).status_code == 422


def test_topic_is_trimmed_and_blank_becomes_none(client: TestClient, fake_llm: FakeLLM) -> None:
    _make(client, count=2, topic="   ")
    assert "topic" not in fake_llm.calls[0]["user"]
    _make(client, count=2, topic="x" * 200, skip_used_days=0)  # exactly the limit is fine


def test_max_count_follows_the_settings(services: Services) -> None:
    small = with_settings(services.settings, ideas_max_count=5)
    with TestClient(create_app(small, services=services)) as c:
        assert c.post("/ideas", json={"count": 6}).status_code == 422
        assert c.post("/ideas", json={"count": 5}).status_code == 200


def test_list_ideas_filters_and_orders(client: TestClient) -> None:
    ideas = _make(client, count=3)["ideas"]
    client.post(f"/ideas/{ideas[0]['id']}/dismiss")
    everything = client.get("/ideas").json()["ideas"]
    assert [i["id"] for i in everything] == sorted((i["id"] for i in ideas), reverse=True)
    dismissed = client.get("/ideas", params={"status": "dismissed"}).json()["ideas"]
    assert [i["id"] for i in dismissed] == [ideas[0]["id"]]
    assert len(client.get("/ideas", params={"limit": 2}).json()["ideas"]) == 2
    assert client.get("/ideas", params={"status": "bogus"}).status_code == 422
    assert client.get("/ideas", params={"limit": 0}).status_code == 422


def test_use_dismiss_reset_cycle(client: TestClient) -> None:
    idea = _make(client, count=1)["ideas"][0]
    used = client.post(f"/ideas/{idea['id']}/use").json()
    assert used["status"] == "used" and used["used_at"]
    assert set(used) == IDEA_KEYS
    dismissed = client.post(f"/ideas/{idea['id']}/dismiss").json()
    assert (dismissed["status"], dismissed["used_at"]) == ("dismissed", None)
    reset = client.post(f"/ideas/{idea['id']}/reset").json()
    assert (reset["status"], reset["used_at"]) == ("suggested", None)
    assert reset["source"]["text"]


@pytest.mark.parametrize("action", ["use", "dismiss", "reset", "script"])
def test_unknown_idea_is_404(client: TestClient, action: str) -> None:
    assert client.post(f"/ideas/9999/{action}").status_code == 404


def test_script_endpoint(client: TestClient, fake_llm: FakeLLM) -> None:
    fake_llm.script_reply = {"script": "Keep it for 12 days. Game changer.", "caption": "Cap"}
    idea = _make(client, count=1)["ideas"][0]

    response = client.post(f"/ideas/{idea['id']}/script")

    assert response.status_code == 200
    data = response.json()
    assert set(data) == {"idea", "script", "caption", "warnings", "model", "latency_ms", "usage"}
    assert data["script"].startswith("Keep it")
    assert len(data["warnings"]) == 2
    assert data["idea"]["script"] == data["script"] and data["idea"]["caption"] == "Cap"
    assert client.get("/ideas").json()["ideas"][0]["script"] == data["script"]


def test_script_for_removed_lesson_is_409(client: TestClient, settings: Settings) -> None:
    idea = _make(client, count=1)["ideas"][0]
    (settings.docs_dir / idea["source"]["doc_path"]).unlink()
    assert client.post("/ingest").status_code == 200
    response = client.post(f"/ideas/{idea['id']}/script")
    assert response.status_code == 409
    listed = client.get("/ideas").json()["ideas"][0]
    assert listed["source"]["text"] is None  # the chunk is gone, the idea stays


def test_provider_failure_is_502(client: TestClient, fake_llm: FakeLLM) -> None:
    def boom(system: str, user: str) -> str:
        raise ProviderError("OpenAI", RuntimeError("secret detail"))

    idea = _make(client, count=1)["ideas"][0]
    fake_llm.responder = boom
    for response in (client.post("/ideas", json={}), client.post(f"/ideas/{idea['id']}/script")):
        assert response.status_code == 502
        assert "secret detail" not in response.text


def test_unusable_model_reply_is_502(client: TestClient, fake_llm: FakeLLM) -> None:
    fake_llm.responder = lambda system, user: "not json"
    assert client.post("/ideas", json={"count": 2}).status_code == 502


def test_coverage(client: TestClient) -> None:
    ideas = _make(client, count=30)["ideas"]
    first = ideas[0]
    client.post(f"/ideas/{first['id']}/use")

    data = client.get("/coverage").json()

    assert set(data) == {"lessons", "totals"}
    assert data["totals"] == {"lessons": 4, "never_used": 3, "passages": 9, "used_passages": 1}
    paths = [row["doc_path"] for row in data["lessons"]]
    assert paths == sorted(paths)
    row = next(r for r in data["lessons"] if r["doc_path"] == first["source"]["doc_path"])
    assert set(row) == {
        "doc_path", "course", "lesson", "passages", "suggested", "used", "last_used_at",
    }  # fmt: skip
    assert (row["used"], row["suggested"]) == (1, 2)
    assert row["last_used_at"]
    assert row["course"] == first["source"]["course"]


def test_voice(client: TestClient) -> None:
    data = client.get("/voice").json()
    assert data["author"] == "Mira Calder"
    assert "Cook once, eat twice" in data["phrases"]
    assert "game changer" in data["never_say"]
    assert "Warm, practical" in data["style"]
    assert [s["format_label"] for s in data["samples"]] == [
        "Educational post",
        "Carousel",
        "Talking-head video",
    ]
    assert set(data["samples"][0]) == {"path", "title", "format", "format_label", "text"}


def test_formats(client: TestClient) -> None:
    assert client.get("/formats").json() == [
        {"key": "talking_head", "label": "Talking-head video"},
        {"key": "carousel", "label": "Carousel"},
        {"key": "text_graphic", "label": "Text graphic"},
        {"key": "post", "label": "Educational post"},
    ]


def test_health_and_documents(client: TestClient) -> None:
    idea = _make(client, count=1)["ideas"][0]
    client.post(f"/ideas/{idea['id']}/use")
    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert (health["documents"], health["chunks"], health["ideas_used"]) == (4, 9, 1)
    assert health["llm_model"] == "fake-llm"
    assert health["embedding_model"].startswith("hashing")
    assert len(client.get("/documents").json()) == 4


def test_ingest_requires_the_admin_token_when_set(services: Services) -> None:
    secured = with_settings(services.settings, admin_token="s3cret-test")
    with TestClient(create_app(secured, services=services)) as c:
        assert c.post("/ingest").status_code == 401
        assert c.post("/ingest", headers={"X-Admin-Token": "wrong"}).status_code == 401
        ok = c.post("/ingest", headers={"X-Admin-Token": "s3cret-test"}, json={"rebuild": False})
        assert ok.status_code == 200 and ok.json()["changed"] is False


def test_security_headers_and_removed_ask_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert "default-src 'self'" in response.headers["Content-Security-Policy"]
    assert client.post("/ask", json={"question": "hi"}).status_code in {404, 405}


def test_app_title(client: TestClient) -> None:
    assert client.get("/openapi.json").json()["info"]["title"] == "Course Content Assistant"


def test_empty_reason_in_the_response(
    client: TestClient, settings: Settings, fake_llm: FakeLLM
) -> None:
    data = _make(client, count=3, topic="quantum chromodynamics lattice gluon")
    assert (data["ideas"], data["empty_reason"]) == ([], "no_topic_match")

    fake_llm.idea_edit = lambda ideas: []
    assert _make(client, count=3)["empty_reason"] == "model_returned_none"

    fake_llm.idea_edit = None
    for idea in _make(client, count=30)["ideas"]:
        client.post(f"/ideas/{idea['id']}/use")
    assert _make(client, count=3)["empty_reason"] == "all_recently_used"


def test_empty_index_reason(settings: Settings, embeddings: CountingEmbeddings) -> None:
    empty = build_services(settings, embedding_provider=embeddings, llm=FakeLLM())
    with TestClient(create_app(settings, services=empty)) as c:
        assert _make(c)["empty_reason"] == "empty_index"


def test_health_reports_the_limits_for_the_page(client: TestClient) -> None:
    health = client.get("/health").json()
    assert (health["ideas_default_count"], health["ideas_max_count"]) == (15, 30)
    assert (health["skip_used_days_default"], health["skip_used_days_max"]) == (30, 365)


def test_non_ascii_admin_token_is_401_not_500(services: Services) -> None:
    secured = with_settings(services.settings, admin_token="s3cret-test")
    with TestClient(create_app(secured, services=services)) as c:
        response = c.post("/ingest", headers={"X-Admin-Token": "t\u00f6ken".encode("latin-1")})
        assert response.status_code == 401


def test_missing_api_key_is_503_with_a_clear_message(
    settings: Settings, embeddings: CountingEmbeddings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    svc = build_services(settings, embedding_provider=embeddings)  # the real, lazy OpenAI chat
    svc.ingestor.run()
    with TestClient(create_app(settings, services=svc)) as c:
        response = c.post("/ideas", json={"count": 2})
        assert response.status_code == 503
        assert response.json()["detail"] == "No API key configured for OpenAI"
        listing = c.get("/ideas")  # the rest of the app keeps working
        assert listing.status_code == 200


def test_missing_embedding_key_is_503_too(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from course_assistant.embeddings import OpenAIEmbeddings

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    svc = build_services(settings, embedding_provider=OpenAIEmbeddings(), llm=FakeLLM())
    with TestClient(create_app(settings, services=svc)) as c:
        response = c.post("/ingest")
        assert response.status_code == 503
        assert "No API key configured for OpenAI" in response.json()["detail"]


def test_other_provider_failures_stay_502(client: TestClient, fake_llm: FakeLLM) -> None:
    def fail(system: str, user: str) -> str:
        raise ProviderError("OpenAI", RuntimeError("down"))

    fake_llm.responder = fail
    assert client.post("/ideas", json={}).status_code == 502
    assert issubclass(ProviderNotConfiguredError, ProviderError)


def test_embedding_model_mismatch_on_a_topic_request_is_409(
    client: TestClient, services: Services
) -> None:
    services.embedder.provider = HashingEmbeddings(dim=64)
    response = client.post("/ideas", json={"count": 3, "topic": "breakfast"})
    assert response.status_code == 409
    assert response.json()["detail"] == (
        "The index was built with a different embedding model; run ingest --rebuild."
    )


def test_idea_source_follows_an_edited_lesson(client: TestClient, settings: Settings) -> None:
    ideas = _make(client, count=30)["ideas"]
    idea = next(i for i in ideas if i["source"]["section"] == "Herb oil")
    client.post(f"/ideas/{idea['id']}/use")
    path = settings.docs_dir / idea["source"]["doc_path"]
    path.write_text(
        path.read_text(encoding="utf-8").replace("## Herb oil\n\n", "## Herb oil\n\nAn easy one. "),
        encoding="utf-8",
    )
    assert client.post("/ingest").status_code == 200

    listed = next(i for i in client.get("/ideas").json()["ideas"] if i["id"] == idea["id"])
    assert listed["source"]["text"].startswith("An easy one.")
    assert listed["source"]["chunk_id"] != idea["source"]["chunk_id"]
    coverage = client.get("/coverage").json()
    assert coverage["totals"]["used_passages"] == 1
    assert client.post(f"/ideas/{idea['id']}/script").status_code == 200


def test_a_half_finished_reingest_does_not_break_the_log_endpoints(
    client: TestClient, services: Services
) -> None:
    idea = _make(client, count=2)["ideas"][0]
    with services.store.connect() as conn:  # one chunk already re-embedded with another model
        conn.execute("UPDATE chunks SET model = 'text-embedding-3-large' WHERE rowid = 1")

    assert client.get("/ideas").status_code == 200
    assert client.get("/coverage").status_code == 200
    assert client.post(f"/ideas/{idea['id']}/use").status_code == 200
    assert client.post(f"/ideas/{idea['id']}/reset").status_code == 200
    assert client.post("/ideas", json={"count": 2}).status_code == 200  # no search needed
    topical = client.post("/ideas", json={"count": 2, "topic": "breakfast"})
    assert topical.status_code == 409  # only search needs the vectors
    assert "ingest --rebuild" in topical.json()["detail"]
