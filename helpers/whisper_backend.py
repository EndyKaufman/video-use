"""Local ASR backend: faster-whisper → Scribe-shaped transcript JSON.

Drop-in replacement for the ElevenLabs Scribe call in `transcribe.py`. It writes the
same cache path and the same schema, so `pack_transcripts.py`, `render.py` and
`timeline_view.py` work unchanged:

    {"words": [{"type": "word", "text": "Hello", "start": 0.52, "end": 0.81,
                "speaker_id": "speaker_0"},
               {"type": "spacing", "text": " ", "start": 0.81, "end": 1.34}, ...],
     "language": "en", "backend": "faster-whisper:small"}

Honest differences from hosted Scribe — read these before trusting a cut list:

  * **No diarization.** Every word gets `speaker_id: "speaker_0"`. Phrases still break
    on silence, so packed transcripts stay readable, but you cannot tell speakers
    apart from text alone. `--num-speakers` is accepted and ignored.
  * **No audio events.** No `(laughter)` / `(applause)` / `(sigh)` tags. Those come
    from Scribe's tagger; whisper does not have one.
  * **Fillers are unreliable, not absent.** In the verified sample `small` emitted
    `UMM.` as its own phrase, so filler-cutting can use tokens — but whisper
    orthography varies (`uhm`/`um`/dropped), so cross-check the silence gap too
    (`find_silences` in timeline_view sees them fine) and cut on the pause either way.
  * **Word timestamps drift more** — expect 50–200ms. Hard Rule 7 (pad every cut
    edge, 30–200ms) matters more here than with Scribe; lean toward the wider end.

Tuning via env, all optional:
  VIDEO_USE_WHISPER_MODEL        tiny|base|small|medium|large-v3|distil-large-v3   (default small)
  VIDEO_USE_WHISPER_DEVICE       cuda|cpu|auto                                     (default auto)
  VIDEO_USE_WHISPER_COMPUTE      float16|int8|int8_float16|auto                    (default auto)
  VIDEO_USE_WHISPER_MODEL_DIR    where CT2 models are cached                       (default <repo>/.model_cache)
  VIDEO_USE_WHISPER_VAD          0 to disable voice-activity filtering             (default 1)

Usage:
    python helpers/transcribe.py <video>                 # routes here by default
    python helpers/whisper_backend.py <video>            # directly, same CLI
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_MODEL = None
_MODEL_LOCK = threading.Lock()
_MODEL_KEY: tuple[str, str, str] | None = None


def env(name: str, default: str) -> str:
    return (os.environ.get(name) or default).strip() or default


def model_name() -> str:
    return env("VIDEO_USE_WHISPER_MODEL", "small")


def model_dir() -> Path:
    return Path(env("VIDEO_USE_WHISPER_MODEL_DIR", str(REPO_ROOT / ".model_cache")))


def cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def resolve_device_compute(device: str | None = None, compute: str | None = None) -> tuple[str, str]:
    """Pick (device, compute_type). 'auto'/'cpu' never raise; 'cuda' may fail upstream."""
    device = (device or env("VIDEO_USE_WHISPER_DEVICE", "auto")).lower()
    compute = (compute or env("VIDEO_USE_WHISPER_COMPUTE", "auto")).lower()

    if device == "auto":
        device = "cuda" if cuda_available() else "cpu"
    if compute in ("", "auto"):
        compute = "float16" if device == "cuda" else "int8"
    return device, compute


def load_model(name: str | None = None, device: str | None = None, compute: str | None = None):
    """Process-wide model singleton, guarded by a lock.

    Batch mode calls this from several threads at once; loading twice would double the
    RAM for the same weights and CTranslate2 would then fight itself for the GPU.
    """
    global _MODEL, _MODEL_KEY
    from faster_whisper import WhisperModel

    name = name or model_name()
    device, compute = resolve_device_compute(device, compute)
    key = (name, device, compute)

    with _MODEL_LOCK:
        if _MODEL is None or _MODEL_KEY != key:
            t0 = time.time()
            _MODEL = WhisperModel(
                name,
                device=device,
                compute_type=compute,
                download_root=str(model_dir()),
            )
            _MODEL_KEY = key
            print(f"  whisper: loaded {name} on {device}/{compute} in {time.time() - t0:.1f}s", flush=True)
    return _MODEL, key


def run_whisper(
    wav_path: Path,
    language: str | None = None,
    model: str | None = None,
    device: str | None = None,
    compute: str | None = None,
) -> dict:
    """Transcribe a 16 kHz mono wav into the Scribe-shaped dict."""
    mgr, (m_name, m_device, m_compute) = load_model(model, device, compute)

    seg_iter, info = mgr.transcribe(
        str(wav_path),
        language=language or None,
        word_timestamps=True,
        vad_filter=env("VIDEO_USE_WHISPER_VAD", "1") not in ("0", "false", "no"),
        vad_parameters={"min_speech_duration_ms": 200},
        condition_on_previous_text=False,
        beam_size=5,
    )

    words: list[dict] = []
    segments: list[dict] = []
    prev_end: float | None = None

    for seg in seg_iter:
        for w in seg.words or []:
            text = (w.word or "").strip()
            if not text:
                continue
            start = float(w.start)
            end = float(w.end)
            if prev_end is not None and start - prev_end >= 0.05:
                words.append(
                    {"type": "spacing", "text": " ", "start": round(prev_end, 3), "end": round(start, 3)}
                )
            words.append(
                {
                    "type": "word",
                    "text": text,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "speaker_id": "speaker_0",
                    "logid": None,
                    "words_confidence": round(float(getattr(w, "probability", 0.0) or 0.0), 4),
                    "temperature": 0.0,
                }
            )
            prev_end = end
        segments.append(
            {
                "start": round(float(seg.start), 3),
                "end": round(float(seg.end), 3),
                "text": (seg.text or "").strip(),
                "speaker_id": "speaker_0",
            }
        )

    return {
        "words": words,
        "segments": segments,
        "language": getattr(info, "language", language),
        "language_probability": round(float(getattr(info, "language_probability", 0.0) or 0.0), 4),
        "duration": round(float(getattr(info, "duration", 0.0) or 0.0), 3),
        "text": " ".join(s["text"] for s in segments).strip(),
        "backend": f"faster-whisper:{m_name}",
        "device": m_device,
        "compute_type": m_compute,
        "diarized": False,
        "audio_events": False,
    }


def main() -> None:
    """Same CLI as transcribe.py — it delegates there for cache/extract plumbing."""
    import transcribe

    transcribe.main()


if __name__ == "__main__":
    main()
