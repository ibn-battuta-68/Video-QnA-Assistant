import json
from dataclasses import replace

import numpy as np
import pytest
from benchmark import parse_choice, read_questions, run_benchmark, score_predictions
from PIL import Image
from retrieval import BM25, Hit, HybridRetriever, reciprocal_rank_fusion
from video_processing import (
    Chunk,
    Config,
    Speech,
    align_speech,
    chunk_shots,
    detect_shots,
    make_grid,
)
from video_rag import VideoRAG, select_frames


@pytest.fixture
def tiny_video(tmp_path):
    import cv2

    path = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (96, 64))
    if not writer.isOpened():
        pytest.skip("MJPG video encoder unavailable")
    for color in [(0, 0, 230), (20, 20, 20), (230, 0, 0)]:
        for _ in range(20):
            frame = np.full((64, 96, 3), color, dtype=np.uint8)
            writer.write(frame)
    writer.release()
    return path


class FakeModels:
    def __init__(self):
        self.calls = []

    def encode_text(self, texts):
        return np.array([[1.0, 0.0] if "red" in text.casefold() else [0.0, 1.0] for text in texts])

    def encode_visual_query(self, query):
        return np.array([1.0, 0.0])

    def encode_images(self, paths):
        from PIL import Image

        vectors = []
        for path in paths:
            with Image.open(path) as image:
                red, green, blue = image.resize((1, 1)).getpixel((0, 0))
            vectors.append([1.0, 0.0] if red > max(green, blue) else [0.0, 1.0])
        return np.array(vectors)

    def generate(self, prompt, grid, *, choice=False):
        self.calls.append((prompt, grid, choice))
        return '{"answer": 2}' if choice else "A red object is visible."


@pytest.fixture
def fake_models():
    return FakeModels()


def test_long_shots_are_bounded_and_cuts_preserved():
    shots = [(0, 2), (2, 95), (95, 100)]
    chunks = chunk_shots(shots, 30)
    assert chunks[0].start == 0 and chunks[-1].end == 100
    assert all(c.end - c.start <= 30 for c in chunks)
    for chunk in chunks:
        start, end = shots[chunk.shot_id]
        assert start <= chunk.start < chunk.end <= end
    assert all(a.end == b.start for a, b in zip(chunks, chunks[1:], strict=False))


def test_word_alignment_assigns_boundary_words_once():
    chunks = chunk_shots([(0, 2), (2, 4)], 30)
    align_speech(chunks, [Speech(2.0, 2.5, "next"), Speech(0, 1, "first"), Speech(1.8, 2.2, "cut")])
    assert chunks[0].transcript == "first"
    assert chunks[1].transcript == "cut next"


@pytest.mark.parametrize(
    "shots", [[(1, 2)], [(0, 2), (3, 4)], [(0, 2), (1, 3)], [(0, 0)], [(0, float("inf"))]]
)
def test_invalid_shots_rejected(shots):
    with pytest.raises(ValueError):
        chunk_shots(shots, 30)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_durations_rejected(value):
    with pytest.raises(ValueError):
        Config(max_chunk_seconds=value)


def test_bm25_matches_exact_terms_even_in_single_document():
    assert BM25(["a red hammer"]).scores("HAMMER!")[0] > 0
    scores = BM25(["red hammer hammer", "blue kettle", ""]).scores("hammer")
    assert scores[0] > 0 and list(scores[1:]) == [0, 0]
    assert not BM25(["", ""]).scores("anything").any()


def test_rrf_weights_and_duplicate_ids():
    result = reciprocal_rank_fusion({"dense": [0, 0, 1], "bm25": [1, 2]}, {"dense": 1, "bm25": 2})
    assert result[0][0] == 1
    assert result[0][1] == pytest.approx(1 / 62 + 2 / 61)
    assert result[0][2] == {"dense": 2, "bm25": 1}


def make_retriever(models):
    chunks = [
        Chunk("a", 0, 0, 1, caption="red cup", frames=["a.jpg"], frame_times=[0.5]),
        Chunk("b", 1, 1, 2, caption="blue wrench", frames=["b.jpg"], frame_times=[1.5]),
    ]
    return HybridRetriever(chunks, np.eye(2), np.eye(2), models)


def test_hybrid_combines_semantic_and_exact_matches(fake_models):
    retriever = make_retriever(fake_models)
    dense = retriever.retrieve("red wrench", mode="dense")
    hybrid = retriever.retrieve("red wrench", mode="hybrid")
    assert dense[0].chunk.id == "a"
    assert any("bm25" in hit.ranks and "dense_image" in hit.ranks for hit in hybrid)
    assert retriever.retrieve("wrench", mode="bm25")[0].chunk.id == "b"


def test_sparse_retrieval_does_not_call_embedding_models():
    retriever = make_retriever(object())
    assert retriever.retrieve("wrench", mode="bm25")[0].chunk.id == "b"
    assert retriever.retrieve("nonexistent", mode="bm25") == []


def test_visual_only_chunks_still_retrievable(fake_models):
    chunk = Chunk("a", 0, 0, 1, frames=["a.jpg"], frame_times=[0.5])
    retriever = HybridRetriever([chunk], [[0, 0]], [[1, 0]], fake_models)
    hit = retriever.retrieve("red")[0]
    assert hit.chunk.id == "a" and hit.ranks == {"dense_image": 1}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_k": 0},
        {"mode": "wrong"},
        {"dense_weight": -1},
        {"dense_weight": 0, "sparse_weight": 0},
    ],
)
def test_invalid_retrieval_options(fake_models, kwargs):
    with pytest.raises(ValueError):
        make_retriever(fake_models).retrieve("red", **kwargs)


def test_actual_shot_detection_finds_synthetic_hard_cuts(tiny_video):
    shots = detect_shots(tiny_video, Config())
    assert len(shots) == 3
    assert shots[0][0] == 0
    assert [end for _, end in shots] == pytest.approx([2, 4, 6], abs=0.11)


def test_no_cut_video_covers_full_duration(tiny_video):
    shots = detect_shots(tiny_video, Config(scene_threshold=10000))
    assert shots == pytest.approx([(0, 6)])


def test_grid_keeps_two_columns_and_three_rows(tmp_path):
    paths = []
    colors = [(240, 0, 0), (0, 240, 0), (0, 0, 240)]
    for i, color in enumerate(colors):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (12, 8), color).save(path)
        paths.append(path)
    output = make_grid(paths, tmp_path / "grid.png")
    with Image.open(output) as grid:
        assert grid.size == (24, 24)
        assert grid.getpixel((6, 4)) == colors[0]
        assert grid.getpixel((18, 4)) == colors[1]
        assert grid.getpixel((6, 12)) == colors[2]
        assert grid.getpixel((18, 20)) == (0, 0, 0)


def test_frame_selection_diversifies_chunks_then_sorts_time():
    late = Chunk(
        "late", 1, 10, 20, frames=[f"l{i}" for i in range(6)], frame_times=list(range(10, 16))
    )
    early = Chunk("early", 0, 0, 10, frames=[f"e{i}" for i in range(6)], frame_times=list(range(6)))
    frames = select_frames([Hit(late, 1, {}), Hit(early, 0.9, {})])
    assert len(frames) == 6
    assert sum(path.startswith("e") for path, _ in frames) == 3
    assert [t for _, t in frames] == sorted(t for _, t in frames)


def test_ingestion_persistence_retrieval_and_qa(tiny_video, tmp_path, fake_models):
    config = Config(transcribe=False)
    index_dir = tmp_path / "index"
    index = VideoRAG.ingest(tiny_video, index_dir, config, fake_models)
    assert len(index.chunks) == 3
    assert all(c.caption and c.frames for c in index.chunks)
    assert not list(tmp_path.glob(".index-*"))
    loaded = VideoRAG.load(index_dir, fake_models)
    result = loaded.answer("What is red?", grid_path=tmp_path / "retrieved.jpg")
    assert result["sources"] and (tmp_path / "retrieved.jpg").is_file()
    assert all("bm25" in source["ranks"] for source in result["sources"])
    assert len(fake_models.calls) == 4
    assert "Frame times:" in fake_models.calls[-1][0]
    cached = VideoRAG.ingest(tiny_video, index_dir, config, fake_models)
    assert len(cached.chunks) == 3 and len(fake_models.calls) == 4
    with pytest.raises(ValueError, match="different video or configuration"):
        VideoRAG.ingest(tiny_video, index_dir, replace(config, captions=False), fake_models)


def test_failed_ingestion_is_not_cached(tiny_video, tmp_path, fake_models):
    def fail(*args, **kwargs):
        raise RuntimeError("Model unavailable")

    fake_models.generate = fail
    with pytest.raises(RuntimeError, match="unavailable"):
        VideoRAG.ingest(tiny_video, tmp_path / "index", Config(transcribe=False), fake_models)
    assert not (tmp_path / "index").exists()
    assert not list(tmp_path.glob(".index-*"))


@pytest.mark.parametrize(
    ("text", "expected"), [('{"answer": 4}', 4), ("2", 2), ("C", 2), ("Final Answer: B", 1)]
)
def test_choice_parsing(text, expected):
    assert parse_choice(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        '{"answer": true}',
        '{"answer": 5}',
        '{"answer": "2"}',
        "2 or 3",
        "There are 4 people, so maybe 1",
        "-1",
        "",
    ],
)
def test_invalid_choices_are_not_guessed(text):
    with pytest.raises(ValueError):
        parse_choice(text)


def test_failures_count_in_accuracy_denominator():
    result = score_predictions(
        {"a": 2, "b": True, "c": 9}, {"a": 2, "b": 1, "c": 0}, ["a", "b", "c"]
    )
    assert result["accuracy"] == pytest.approx(1 / 3)
    assert result["failed_or_missing"] == 2
    assert score_predictions({"a": 2}, {}, ["a"])["accuracy"] is None


def fixtures(tmp_path):
    questions = [
        {
            "q_uid": uid,
            "question": "What is happening?",
            **{f"option {i}": f"action {i}" for i in range(5)},
        }
        for uid in ("a", "b", "c")
    ]
    qpath, apath = tmp_path / "questions.json", tmp_path / "answers.json"
    qpath.write_text(json.dumps(questions), encoding="utf-8")
    apath.write_text(json.dumps({"a": 2, "b": 1}), encoding="utf-8")
    videos = tmp_path / "videos"
    videos.mkdir()
    return qpath, apath, videos


def test_benchmark_subset_failure_resume_and_config_guard(tmp_path, monkeypatch, fake_models):
    qpath, apath, videos = fixtures(tmp_path)
    calls = []

    class FakeIndex:
        def answer(self, question, **kwargs):
            assert "action 2" in kwargs["options"]
            assert "answer" not in question
            return {"answer": '{"answer": 2}', "sources": [], "frames": []}

    def ingest(video, *args):
        calls.append(video.stem)
        if video.stem == "b" and len(calls) == 2:
            raise FileNotFoundError("missing video")
        return FakeIndex()

    monkeypatch.setattr("benchmark.VideoRAG.ingest", ingest)
    kwargs = {"answers_path": apath, "models": fake_models}
    config = Config(transcribe=False)
    args = (qpath, videos, tmp_path / "out", tmp_path / "cache", config)
    metrics = run_benchmark(*args, **kwargs)
    assert metrics["questions"] == 2 and metrics["failed_or_missing"] == 1
    assert metrics["accuracy"] == 0.5
    log = tmp_path / "out" / "results.jsonl"
    with log.open("a", encoding="utf-8") as stream:
        stream.write('{"q_uid":')
    metrics = run_benchmark(*args, **kwargs)
    assert calls == ["a", "b", "b"]
    assert metrics["valid_predictions"] == 2
    predictions = json.loads((tmp_path / "out" / "predictions.json").read_text())
    assert predictions == {"a": 2, "b": 2}
    assert len([json.loads(line) for line in log.read_text().splitlines()]) == 3
    with pytest.raises(ValueError, match="different benchmark run"):
        run_benchmark(*args, **kwargs, mode="dense")


def test_full_benchmark_includes_unlabeled_examples(tmp_path, monkeypatch, fake_models):
    qpath, apath, videos = fixtures(tmp_path)
    monkeypatch.setattr(
        "benchmark.VideoRAG.ingest",
        lambda *args: (_ for _ in ()).throw(FileNotFoundError("missing")),
    )
    metrics = run_benchmark(
        qpath,
        videos,
        tmp_path / "out",
        tmp_path / "cache",
        Config(transcribe=False),
        answers_path=apath,
        full=True,
        models=fake_models,
    )
    assert metrics["questions"] == 3
    assert metrics["labeled_questions"] == 2
    assert metrics["accuracy"] == 0


def test_duplicate_question_ids_rejected(tmp_path):
    qpath, _, _ = fixtures(tmp_path)
    questions = json.loads(qpath.read_text())
    questions.append(questions[0])
    qpath.write_text(json.dumps(questions))
    with pytest.raises(ValueError, match="duplicate"):
        read_questions(qpath)


def test_invalid_benchmark_settings_fail_before_indexing(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        run_benchmark(
            tmp_path / "missing.json",
            tmp_path,
            tmp_path / "out",
            tmp_path / "cache",
            Config(transcribe=False),
            top_k=0,
        )
    assert not (tmp_path / "out").exists()
