import math
import subprocess
from bisect import bisect_right
from dataclasses import asdict, dataclass, field
from math import ceil, isfinite
from pathlib import Path


@dataclass(frozen=True)
class Config:
    scene_threshold: float = 27.0
    min_scene_seconds: float = 0.5
    max_chunk_seconds: float = 30.0
    frame_interval: float = 3.0
    frame_width: int = 640
    text_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    image_model: str = "clip-ViT-B-32"
    vlm_model: str = "llava"
    whisper_model: str = "base"
    transcribe: bool = True
    captions: bool = True
    seed: int = 42

    def __post_init__(self):
        for name in ("scene_threshold", "min_scene_seconds", "max_chunk_seconds", "frame_interval"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.frame_width < 32:
            raise ValueError("frame_width must be at least 32")
        for name in ("text_model", "image_model", "vlm_model", "whisper_model"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} cannot be empty")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Speech:
    start: float
    end: float
    text: str


@dataclass
class Chunk:
    id: str
    shot_id: int
    start: float
    end: float
    transcript: str = ""
    caption: str = ""
    frames: list[str] = field(default_factory=list)
    frame_times: list[float] = field(default_factory=list)

    @property
    def text(self):
        return "\n".join(part for part in (self.caption, self.transcript) if part.strip())


def chunk_shots(shots: list[tuple[float, float]], max_seconds: float) -> list[Chunk]:
    if not isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be finite and positive")
    chunks = []
    previous_end = 0.0
    for shot_id, (start, end) in enumerate(shots):
        if not all(isfinite(t) for t in (start, end)) or start < 0 or end <= start:
            raise ValueError("Shots must have finite, increasing nonnegative timestamps")
        if abs(start - previous_end) > 1e-5:
            raise ValueError("Shots must cover the video contiguously from zero")
        count = max(1, ceil((end - start) / max_seconds))
        step = (end - start) / count
        for part in range(count):
            chunks.append(
                Chunk(
                    id=f"shot_{shot_id:05d}_{part:03d}",
                    shot_id=shot_id,
                    start=start + part * step,
                    end=end if part == count - 1 else start + (part + 1) * step,
                )
            )
        previous_end = end
    return chunks


def align_speech(chunks: list[Chunk], speech: list[Speech]) -> None:
    if not chunks:
        return
    starts = [chunk.start for chunk in chunks]
    texts = [[] for _ in chunks]
    for segment in sorted(speech, key=lambda s: (s.start, s.end)):
        if (
            not all(isfinite(t) for t in (segment.start, segment.end))
            or segment.end < segment.start
        ):
            raise ValueError("Invalid speech timestamps")
        midpoint = (segment.start + segment.end) / 2
        i = bisect_right(starts, midpoint) - 1
        if 0 <= i < len(chunks) and midpoint < chunks[i].end and segment.text.strip():
            texts[i].append(segment.text.strip())
    for chunk, words in zip(chunks, texts, strict=True):
        chunk.transcript = " ".join(words)


def detect_shots(video_path: Path, config: Config) -> list[tuple[float, float]]:
    import cv2
    from scenedetect import ContentDetector, SceneManager, open_video

    probe = cv2.VideoCapture(str(video_path))
    try:
        fps = probe.get(cv2.CAP_PROP_FPS)
        count = probe.get(cv2.CAP_PROP_FRAME_COUNT)
        if not probe.isOpened() or not math.isfinite(fps) or fps <= 0 or count <= 0:
            raise ValueError(f"Cannot read video metadata: {video_path}")
    finally:
        probe.release()
    video = open_video(str(video_path))
    manager = SceneManager()
    manager.add_detector(
        ContentDetector(
            threshold=config.scene_threshold,
            min_scene_len=max(1, round(config.min_scene_seconds * fps)),
        )
    )
    manager.detect_scenes(video=video)
    scenes = manager.get_scene_list(start_in_scene=True)

    def seconds(timecode):
        return timecode.seconds if hasattr(timecode, "seconds") else timecode.get_seconds()

    return [(seconds(start), seconds(end)) for start, end in scenes] or [(0.0, count / fps)]


def extract_frames(video_path: Path, chunks: list[Chunk], root: Path, config: Config) -> None:
    import cv2

    (root / "frames").mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Cannot decode video: {video_path}")
        for chunk in chunks:
            count = max(1, math.ceil((chunk.end - chunk.start) / config.frame_interval))
            for i in range(count):
                target = chunk.start + (i + 0.5) * (chunk.end - chunk.start) / count
                frame_number = math.floor(target * fps)
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"Failed to decode frame at {target:.3f}s in {video_path}")
                height, width = frame.shape[:2]
                frame = cv2.resize(
                    frame, (config.frame_width, max(1, round(height * config.frame_width / width)))
                )
                relative = f"frames/{chunk.id}_{i:04d}.jpg"
                if not cv2.imwrite(str(root / relative), frame):
                    raise OSError(f"Could not save frame: {root / relative}")
                chunk.frames.append(relative)
                chunk.frame_times.append(frame_number / fps)
    finally:
        capture.release()


def make_grid(paths: list[Path], destination: Path) -> Path:
    """Arrange up to six frames in a two-column, three-row grid."""
    from PIL import Image, ImageOps

    if not paths:
        raise ValueError("Cannot build an image grid without frames")
    with Image.open(paths[0]) as first:
        size = first.size
    grid = Image.new("RGB", (size[0] * 2, size[1] * 3))
    try:
        for i, path in enumerate(paths[:6]):
            with Image.open(path) as image:
                tile = ImageOps.pad(image.convert("RGB"), size)
                grid.paste(tile, (size[0] * (i % 2), size[1] * (i // 2)))
                tile.close()
        destination.parent.mkdir(parents=True, exist_ok=True)
        grid.save(destination)
    finally:
        grid.close()
    return destination


def transcribe_video(video_path: Path, root: Path, config: Config) -> list[Speech]:
    import json

    import whisper

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    if not probe.stdout.strip():
        return []
    audio_path = root / "audio.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(audio_path),
        ],
        check=True,
    )
    try:
        result = whisper.load_model(config.whisper_model).transcribe(
            str(audio_path),
            word_timestamps=True,
            fp16=False,
        )
        (root / "transcript.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        speech = []
        for segment in result["segments"]:
            words = segment.get("words")
            if words:
                speech.extend(Speech(float(w["start"]), float(w["end"]), w["word"]) for w in words)
            else:
                speech.append(
                    Speech(float(segment["start"]), float(segment["end"]), segment["text"])
                )
        return speech
    finally:
        audio_path.unlink(missing_ok=True)


def download_video(url: str, destination: Path) -> Path:
    import yt_dlp

    destination.parent.mkdir(parents=True, exist_ok=True)
    options = {
        "outtmpl": str(destination.with_suffix(".%(ext)s")),
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(url, download=True)
        path = Path(downloader.prepare_filename(info))
    merged = path.with_suffix(".mp4")
    return merged if merged.is_file() else path
