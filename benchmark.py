import argparse
import csv
import hashlib
import json
import re
import time
from pathlib import Path

from retrieval import validate_retrieval
from video_processing import Config
from video_rag import (
    Models,
    VideoRAG,
    add_ingestion_options,
    add_retrieval_options,
    make_config,
)


def read_questions(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("questions.json must be a nonempty list of EgoSchema questions")
    seen = set()
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Each question must be an object")
        uid = item.get("q_uid")
        if not isinstance(uid, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", uid) or uid in seen:
            raise ValueError(f"Missing, unsafe, or duplicate q_uid: {uid}")
        seen.add(uid)
        if not isinstance(item.get("question"), str) or not item["question"].strip():
            raise ValueError(f"Missing question text for {uid}")
        if any(
            not isinstance(item.get(f"option {i}"), str) or not item[f"option {i}"].strip()
            for i in range(5)
        ):
            raise ValueError(f"Missing answer options for {uid}")
    return data


def read_answers(path: Path) -> dict[str, int]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError("Answer labels must be a nonempty q_uid-to-integer object")
    if any(
        not isinstance(uid, str) or type(answer) is not int or not 0 <= answer <= 4
        for uid, answer in data.items()
    ):
        raise ValueError("Answer labels must be integers from 0 through 4")
    return data


def parse_choice(response: str) -> int:
    response = response.strip()
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        answer = parsed.get("answer")
        if type(answer) is int and 0 <= answer <= 4:
            return answer
        raise ValueError("Model returned an invalid JSON answer")
    match = re.fullmatch(
        r"(?:final\s+answer\s*:\s*|answer\s*:\s*)?([0-4A-E])\.?", response, re.IGNORECASE
    )
    if match:
        value = match.group(1).upper()
        return int(value) if value.isdigit() else ord(value) - ord("A")
    raise ValueError("Model response must unambiguously select one option from 0 through 4")


def score_predictions(predictions: dict, answers: dict, selected_ids: list[str]) -> dict:
    selected = set(selected_ids)
    valid = {
        uid: answer
        for uid, answer in predictions.items()
        if uid in selected and type(answer) is int and 0 <= answer <= 4
    }
    labeled = {uid: answers[uid] for uid in selected_ids if uid in answers}
    correct = sum(valid.get(uid) == answer for uid, answer in labeled.items())
    return {
        "questions": len(selected_ids),
        "valid_predictions": len(valid),
        "failed_or_missing": len(selected_ids) - len(valid),
        "coverage": len(valid) / len(selected_ids) if selected_ids else 0.0,
        "labeled_questions": len(labeled),
        "correct": correct,
        "accuracy": correct / len(labeled) if labeled else None,
    }


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def run_benchmark(
    questions_path: Path,
    videos_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    config: Config,
    *,
    answers_path: Path | None = None,
    full: bool = False,
    limit: int | None = None,
    mode: str = "hybrid",
    top_k: int = 5,
    candidate_k: int = 50,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    models=None,
) -> dict:
    validate_retrieval(mode, top_k, candidate_k, dense_weight, sparse_weight)
    questions = read_questions(questions_path)
    answers = read_answers(answers_path) if answers_path else {}
    if answers and not full:
        questions = [q for q in questions if q["q_uid"] in answers]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        questions = questions[:limit]
    if not questions:
        raise ValueError("No questions match the requested benchmark split")
    output_dir, cache_dir, videos_dir = Path(output_dir), Path(cache_dir), Path(videos_dir)
    if not videos_dir.is_dir():
        raise ValueError(f"Video directory does not exist: {videos_dir}")
    config_hash = hashlib.sha256(json.dumps(config.to_dict(), sort_keys=True).encode()).hexdigest()[
        :16
    ]
    signature = {
        "version": 1,
        "config": config.to_dict(),
        "questions": questions,
        "videos_dir": str(videos_dir.resolve()),
        "mode": mode,
        "top_k": top_k,
        "candidate_k": candidate_k,
        "dense_weight": dense_weight,
        "sparse_weight": sparse_weight,
        "labels": {q["q_uid"]: answers[q["q_uid"]] for q in questions if q["q_uid"] in answers},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    run_path = output_dir / "run.json"
    if run_path.exists():
        if json.loads(run_path.read_text(encoding="utf-8")) != signature:
            raise ValueError(
                "Output directory contains a different benchmark run; choose another directory"
            )
    elif any(output_dir.iterdir()):
        raise ValueError(
            "Output directory contains files without run metadata; choose an empty directory"
        )
    else:
        write_json(run_path, signature)
    log_path = output_dir / "results.jsonl"
    records = {}
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if i != len(lines) - 1:
                    raise ValueError("Corrupt benchmark log") from None
                # Discard an interrupted final append before writing new records.
                log_path.write_text("\n".join(lines[:i]) + ("\n" if i else ""), encoding="utf-8")
                break
            if type(record.get("prediction")) is int and 0 <= record["prediction"] <= 4:
                records[record["q_uid"]] = record
        if log_path.stat().st_size and not log_path.read_bytes().endswith(b"\n"):
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write("\n")
    models = models or Models(config)
    started = time.perf_counter()
    attempts = 0
    try:
        with log_path.open("a", encoding="utf-8") as log:
            for i, question in enumerate(questions):
                uid = question["q_uid"]
                if uid in records:
                    continue
                attempts += 1
                attempt_start = time.perf_counter()
                record = {"q_uid": uid, "prediction": None}
                print(f"[{i + 1}/{len(questions)}] {uid}", flush=True)
                try:
                    video = videos_dir / f"{uid}.mp4"
                    index = VideoRAG.ingest(video, cache_dir / config_hash / uid, config, models)
                    ingest_elapsed = time.perf_counter() - attempt_start
                    result = index.answer(
                        question["question"],
                        options=[question[f"option {j}"] for j in range(5)],
                        top_k=top_k,
                        mode=mode,
                        candidate_k=candidate_k,
                        dense_weight=dense_weight,
                        sparse_weight=sparse_weight,
                    )
                    record.update(
                        raw_response=result["answer"],
                        sources=result["sources"],
                        frames=result["frames"],
                    )
                    record["prediction"] = parse_choice(result["answer"])
                    record["index_seconds"] = ingest_elapsed
                    record["qa_seconds"] = time.perf_counter() - attempt_start - ingest_elapsed
                    records[uid] = record
                except Exception as error:
                    record["error"] = f"{type(error).__name__}: {error}"
                record["elapsed_seconds"] = time.perf_counter() - attempt_start
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
    finally:
        predictions = {
            q["q_uid"]: records[q["q_uid"]]["prediction"]
            for q in questions
            if q["q_uid"] in records
        }
        metrics = score_predictions(predictions, answers, [q["q_uid"] for q in questions])
        metrics.update(
            retrieval_mode=mode,
            audio=config.transcribe,
            visual_captions=config.captions,
            attempts_this_session=attempts,
            session_seconds=time.perf_counter() - started,
            mean_successful_qa_seconds=(
                sum(r.get("qa_seconds", 0) for r in records.values()) / len(records)
            )
            if records
            else None,
        )
        write_json(output_dir / "predictions.json", predictions)
        write_json(output_dir / "metrics.json", metrics)
        with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["q_uid", "answer"])
            writer.writerows(predictions.items())
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run EgoSchema evaluation or score predictions")
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--videos", type=Path)
    parser.add_argument("--answers", type=Path)
    parser.add_argument("--predictions", type=Path, help="Score an existing prediction JSON")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache", type=Path, default=Path("data/indexes/egoschema"))
    parser.add_argument(
        "--full", action="store_true", help="Include questions without public labels"
    )
    parser.add_argument("--limit", type=int, help="Run only the first N selected questions")
    add_ingestion_options(parser, benchmark=True)
    add_retrieval_options(parser)
    args = parser.parse_args(argv)
    if args.predictions:
        if not args.answers:
            parser.error("--answers is required when scoring predictions")
    elif not all((args.questions, args.videos, args.output)):
        parser.error("--questions, --videos, and --output are required to run evaluation")
    try:
        if args.predictions:
            answers = read_answers(args.answers)
            predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
            if not isinstance(predictions, dict):
                raise ValueError("Predictions must be a q_uid-to-integer object")
            result = score_predictions(predictions, answers, list(answers))
        else:
            result = run_benchmark(
                args.questions,
                args.videos,
                args.output,
                args.cache,
                make_config(args),
                answers_path=args.answers,
                full=args.full,
                limit=args.limit,
                mode=args.mode,
                top_k=args.top_k,
                candidate_k=args.candidate_k,
                dense_weight=args.dense_weight,
                sparse_weight=args.sparse_weight,
            )
    except (ValueError, OSError, ImportError) as error:
        parser.exit(1, f"benchmark.py: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not args.predictions and result["failed_or_missing"]:
        parser.exit(1, "Some questions failed. Inspect results.jsonl and rerun to resume.\n")


if __name__ == "__main__":
    main()
