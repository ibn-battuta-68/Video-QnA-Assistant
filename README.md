# Video RAG

Video question answering with scene-aware chunks, hybrid retrieval, and local multimodal models. The pipeline combines video frames, visual captions, and speech transcripts to retrieve evidence for a question. LLaVA receives timestamped text and a **two-column, three-row grid of up to six frames** to generate an answer.

## Files

```text
./
  video_processing.py   # Settings, shot detection, chunking, transcripts, and frame grids
  retrieval.py          # Dense text/image retrieval, BM25, and rank fusion
  video_rag.py          # Model loading, indexing, answering, and command-line entry point
  benchmark.py          # EgoSchema evaluation, scoring, and resumable predictions
  test_video_rag.py     # Pipeline and evaluation checks
  video_rag_demo.ipynb   # Interactive walkthrough
  requirements.txt
  README.md
  .gitignore
```

`video_rag.py` is the entry point for video indexing and question answering. `benchmark.py` runs EgoSchema evaluation. Both use the same video processing and retrieval code.

## Setup

Prerequisites:

- Python 3.10-3.12.
- [Ollama](https://ollama.com/download) for LLaVA inference.
- [FFmpeg](https://ffmpeg.org/download.html), including `ffmpeg` and `ffprobe` on `PATH`, for audio transcription and merging downloaded video/audio streams. Conda users can install it with `conda install -c conda-forge ffmpeg`.

Run all commands from the repository root. Create a virtual environment:

```sh
python -m venv .venv
```

Activate it in PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```sh
source .venv/bin/activate
```

Install the Python dependencies:

```sh
python -m pip install -r requirements.txt
```

Start the Ollama app, or run `ollama serve` in another terminal if its server is not already running. Download the vision model:

```sh
ollama pull llava
```

MiniLM, CLIP, and Whisper weights download when first used. GPU acceleration is optional. The default Ollama endpoint is `http://localhost:11434`; set `OLLAMA_HOST` to use another server. No API key is needed for the default local setup.

## Use

Place a video at `data/videos/example.mp4`, or replace that path with your own video file:

```sh
python video_rag.py ingest data/videos/example.mp4 --index data/indexes/example
python video_rag.py ask "What happens after the person opens the drawer?" --index data/indexes/example --grid outputs/evidence.jpg
```

The `ask` command prints JSON containing the answer, retrieved sources, branch ranks, scores, and frame timestamps. `--grid` also saves the evidence image.

To skip speech transcription, add `--no-transcribe` to `ingest`. Visual captions remain enabled, so silent footage can still be searched with BM25. Download a video from a supported URL with:

```sh
python video_rag.py download "https://www.youtube.com/watch?v=VIDEO_ID" --output data/videos/example.mp4
```

The pipeline can also be called from Python:

```python
from pathlib import Path
from video_processing import Config
from video_rag import VideoRAG

rag = VideoRAG.ingest(Path("data/videos/example.mp4"), Path("data/indexes/example"), Config())
result = rag.answer("What is happening?", grid_path=Path("outputs/evidence.jpg"))
print(result["answer"])
```

For an interactive walkthrough, open [video_rag_demo.ipynb](video_rag_demo.ipynb). It covers indexing, question answering, grid display, retrieval comparisons, and loading a saved index. EgoSchema evaluation runs separately through `benchmark.py`.

To use JupyterLab:

```sh
python -m pip install jupyterlab
jupyter lab video_rag_demo.ipynb
```

Select the project's Python environment and set the video path and question in the notebook before running its cells.

## How retrieval works

1. **Detect shots:** PySceneDetect finds visual cuts. Each shot becomes a chunk; shots longer than 30 seconds are subdivided. Chunks stay within detected shot boundaries. Continuous videos with no cuts still get full coverage through the duration cap.
2. **Prepare evidence:** Frames are sampled within chunks at roughly three-second intervals. Whisper words are assigned once by timestamp midpoint. LLaVA captions each chunk from a grid, giving BM25 useful text even without speech.
3. **Retrieve:** MiniLM searches caption/transcript embeddings. CLIP searches frame embeddings using the question, with each chunk represented by its best frame match. BM25 searches the same caption/transcript text for exact terms. Text and image embeddings use separate spaces.
4. **Combine:** Weighted reciprocal rank fusion merges the three rankings with constant 60. Dense text and images each get half the dense weight; BM25 gets the sparse weight. Both branch weights default to 1. Use `--mode dense`, `--mode bm25`, or `--mode hybrid` to compare them.
5. **Answer:** Up to six frames are spread across retrieved chunks, sorted by time, and arranged left to right, top to bottom in the grid. Timestamped captions and transcripts accompany the image.

## Configuration

Defaults are defined by `Config` in `video_processing.py` and can be overridden with command-line flags.

| Setting | Default | Flag |
| --- | --- | --- |
| Shot detection threshold | 27 | `--scene-threshold` |
| Minimum shot length | 0.5 seconds | `--min-scene-seconds` |
| Maximum chunk duration | 30 seconds | `--max-chunk-seconds` |
| Frame sampling interval | Approximately 3 seconds | `--frame-interval` |
| Extracted frame width | 640 pixels | `--frame-width` |
| Text embeddings | `sentence-transformers/all-MiniLM-L6-v2` | `--text-model` |
| Image/text embeddings | `clip-ViT-B-32` | `--image-model` |
| Vision-language model | `llava` | `--vlm-model` |
| Speech recognition | Whisper `base` | `--whisper-model` |
| Retrieval mode | `hybrid` | `--mode` |
| Retrieved chunks | 5 | `--top-k` |
| Candidates per retrieval ranking | 50 | `--candidate-k` |

`--no-captions` skips visual caption generation during indexing. Without captions, BM25 relies on the transcript; if both captions and transcription are disabled, use dense or hybrid retrieval for frame-based evidence. `--dense-weight` and `--sparse-weight` control fusion weights.

Custom image embedding models must support both image and text inputs in the same embedding space. The Ollama model must support image input and, for benchmark predictions, structured JSON output.

Run `python video_rag.py ingest --help`, `python video_rag.py ask --help`, or `python benchmark.py --help` for all options.

Each index stores JSON metadata, JPEG frames, caption grids, and NumPy vectors for one video. Repeating ingestion with identical video content and settings reuses it. Changed content or settings require another index directory. Answering uses the model configuration stored in the index. Failed index builds are cleaned up.

## EgoSchema

Obtain the videos using the [official EgoSchema instructions](https://github.com/egoschema/EgoSchema). The benchmark provides five choices per question and public answers for a 500-question subset; full-set answers are hidden. Keep dataset files outside Git:

```text
data/egoschema/
  questions.json
  subset_answers.json
  videos/
    <q_uid>.mp4
```

Download annotations from the official repository (use `curl.exe` in PowerShell):

```sh
curl -L --create-dirs -o data/egoschema/questions.json https://raw.githubusercontent.com/egoschema/EgoSchema/main/questions.json
curl -L --create-dirs -o data/egoschema/subset_answers.json https://raw.githubusercontent.com/egoschema/EgoSchema/main/subset_answers.json
```

Evaluate the public subset:

```sh
python benchmark.py --questions data/egoschema/questions.json --answers data/egoschema/subset_answers.json --videos data/egoschema/videos --output outputs/egoschema-hybrid
```

Add `--limit 5` with a separate output directory for a smoke test. With `--answers`, only questions with public labels are selected by default. Labels are used only for selection/scoring; models see the question, five choices, and video evidence. Nothing is trained on EgoSchema.

Benchmarking defaults to **vision-only**, including visual captions and CLIP frames. `--with-audio` adds speech and should be reported as a separate experiment. Normal video indexing includes audio by default.

Compare retrieval by rerunning with `--mode dense` or `--mode bm25` and a separate `--output` directory. The ingestion cache is shared for identical settings. Evaluation writes:

| File | Contents |
| --- | --- |
| `run.json` | Question selection and configuration |
| `results.jsonl` | Predictions or errors, raw model responses, evidence, and timing |
| `metrics.json` | Accuracy, coverage, failures, and timing |
| `predictions.json` | Official `{q_uid: 0..4}` prediction format |
| `predictions.csv` | Convenience CSV; adapt columns to the current Kaggle sample if needed |

Rerun the same command to resume. Successful predictions are skipped and failures retried. Failed/malformed answers count as incorrect; coverage is reported separately. Incomplete runs exit with a nonzero status. Use a new output directory when changing settings or question selection, and keep videos/model weights unchanged during a resumed run.

Predict the full question set:

```sh
python benchmark.py --questions data/egoschema/questions.json --videos data/egoschema/videos --full --output outputs/egoschema-full
```

Without labels, local accuracy is `null`. Use the official evaluation process for full-set scoring; the script exports predictions and does not submit them. `--full --answers ...` predicts every question while scoring the labeled intersection locally.

Score saved predictions against all public labels:

```sh
python benchmark.py --predictions outputs/egoschema-hybrid/predictions.json --answers data/egoschema/subset_answers.json
```

No measured EgoSchema results are published with this repository. Report the evaluated count, coverage, models, retrieval mode, settings, and audio use with your results. Generation uses a fixed seed and zero temperature, but changes in hardware or model weights may still affect answers.

## Limitations

- Shot detection identifies visual cuts rather than semantic scene changes. Continuous footage relies on the chunk duration cap.
- Answers are based on retrieved text and at most six sampled frames; brief actions and events outside the retrieved evidence may be missed.
- Visual captions and speech transcripts are model-generated and can contain errors.
- Retrieval loads one video's vectors into memory. The project supports per-video question answering rather than searching a collection of videos.

## References

- [EgoSchema: dataset and evaluation](https://github.com/egoschema/EgoSchema)
- [PySceneDetect documentation](https://www.scenedetect.com/docs/latest/)
- [Sentence Transformers pretrained models](https://sbert.net/docs/sentence_transformer/pretrained_models.html)
- [Ollama Python client](https://github.com/ollama/ollama-python)
