from __future__ import annotations

from pathlib import Path

import pytest

from conftest import CountingEmbeddings, with_settings
from course_assistant.config import Settings
from course_assistant.embeddings import HashingEmbeddings
from course_assistant.retrieval import reciprocal_rank_fusion, tokenize
from course_assistant.services import Services, build_services


def test_relevant_section_ranks_first(services: Services) -> None:
    result = services.retriever.search("breakfast for dinner when the fridge looks empty")
    top = result.chunks[0]
    assert (top.chunk.doc_path, top.chunk.heading) == (
        "weeknight-rescue-meals/01-fridge-raid.md",
        "Breakfast for dinner",
    )
    similarities = [c.similarity for c in result.chunks]
    assert all(s >= services.retriever.min_similarity for s in similarities)


def test_delivery_question_finds_the_delivery_section(services: Services) -> None:
    result = services.retriever.search("tomato sauce simmer olive oil")
    assert result.chunks[0].chunk.heading == "Tomato sauce"
    assert result.chunks[0].vector_rank == 1


def test_unrelated_question_returns_nothing(services: Services) -> None:
    result = services.retriever.search("quantum chromodynamics lattice gluon")
    assert result.chunks == []
    assert result.best_similarity < services.retriever.min_similarity


def test_top_k_limits_results(services: Services) -> None:
    services.retriever.min_similarity = -1.0  # let everything through
    assert len(services.retriever.search("dinner", top_k=2).chunks) == 2


def test_vector_only_mode_scores_by_similarity(
    settings: Settings, embeddings: CountingEmbeddings
) -> None:
    from conftest import FakeLLM

    svc = build_services(
        with_settings(settings, hybrid=False), embedding_provider=embeddings, llm=FakeLLM()
    )
    svc.ingestor.run()
    result = svc.retriever.search("sausages on one tray, turn halfway so the edges brown")
    assert result.chunks[0].chunk.doc_path == "weeknight-rescue-meals/02-fifteen-minutes.md"
    assert result.chunks[0].score == pytest.approx(result.chunks[0].similarity)
    assert result.chunks[0].keyword_rank is None


def test_new_documents_are_picked_up_without_restart(services: Services, docs_dir: Path) -> None:
    query = "kombucha fermentation scoby"
    assert services.retriever.search(query).chunks == []
    (docs_dir / "kombucha.md").write_text(
        "# Kombucha\n\nKombucha fermentation needs a scoby and a warm corner of the kitchen."
    )
    services.ingestor.run()
    assert services.retriever.search(query).chunks[0].chunk.doc_path == "kombucha.md"


def test_index_built_with_another_model_is_rejected(services: Services) -> None:
    services.embedder.provider = HashingEmbeddings(dim=64)
    with pytest.raises(RuntimeError, match="ingest --rebuild"):
        services.retriever.search("batch cooking")


def test_query_embeddings_are_cached(services: Services, embeddings: CountingEmbeddings) -> None:
    services.retriever.search("herb oil jar")
    calls = len(embeddings.calls)
    services.retriever.search("herb oil jar")
    assert len(embeddings.calls) == calls


def test_reciprocal_rank_fusion_rewards_agreement() -> None:
    scores = reciprocal_rank_fusion([[1, 2, 3], [3, 1]], k=60)
    assert scores[1] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[3] == pytest.approx(1 / 63 + 1 / 61)
    assert scores[2] == pytest.approx(1 / 62)
    assert sorted(scores, key=scores.__getitem__, reverse=True) == [1, 3, 2]


def test_tokenize_drops_stopwords_and_folds_plurals() -> None:
    assert tokenize("The deliveries of parcels to France") == ["delivery", "parcel", "france"]
