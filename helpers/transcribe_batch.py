"""Batch-transcribe every video in a directory with 4 parallel workers.

Walks <videos_dir> for common video extensions, runs ElevenLabs Scribe on
each, writes transcripts to <videos_dir>/edit/transcripts/<name>.json.

Cached per-file: any source that already has a transcript is skipped.

Usage:
    python helpers/transcribe_batch.py <videos_dir>
    python helpers/transcribe_batch.py <videos_dir> --workers 4
    python helpers/transcribe_batch.py <videos_dir> --num-speakers 2
    python helpers/transcribe_batch.py <videos_dir> --edit-dir /custom/edit
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pyutf8  # noqa: F401  # forces UTF-8 stdio; see pyutf8.py

from transcribe import load_api_key, resolve_backend, transcribe_one, transcript_path


VIDEO_EXTS = {".mp4", ".MP4", ".mov", ".MOV", ".mkv", ".MKV", ".avi", ".AVI", ".m4v"}


def find_videos(videos_dir: Path) -> list[Path]:
    videos = sorted(
        p for p in videos_dir.iterdir()
        if p.is_file() and p.suffix in VIDEO_EXTS
    )
    return videos


def main() -> None:
    ap = argparse.ArgumentParser(description="Parallel batch transcription of a videos directory")
    ap.add_argument("videos_dir", type=Path, help="Directory containing source videos")
    ap.add_argument(
        "--edit-dir",
        type=Path,
        default=None,
        help="Edit output directory (default: <videos_dir>/edit)",
    )
    ap.add_argument("--workers", type=int, default=None,
                    help="Parallel workers. Default: 4 for hosted Scribe (network bound), "
                         "1 for local whisper (one model, compute bound — extra workers "
                         "only oversubscribe cores and slow every file down).")
    ap.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional ISO language code. Omit to auto-detect per file.",
    )
    ap.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Optional number of speakers. Improves diarization when known (Scribe only).",
    )
    ap.add_argument(
        "--audio-track",
        type=int,
        default=0,
        help="Zero-based audio track to transcribe (OBS: 0 = game, 1 = mic).",
    )
    ap.add_argument(
        "--asr",
        choices=("whisper", "scribe"),
        default=None,
        help="ASR backend (default: local whisper, or $VIDEO_USE_ASR).",
    )
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help="Whisper model name, e.g. small | large-v3 | distil-large-v3.",
    )
    ap.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default=None,
        help="Whisper device (default: auto).",
    )
    ap.add_argument(
        "--compute-type",
        dest="compute",
        type=str,
        default=None,
        help="Whisper compute type: float16 | int8 | auto.",
    )
    args = ap.parse_args()

    backend = resolve_backend(args.asr)
    workers = args.workers if args.workers is not None else (1 if backend == "whisper" else 4)
    if backend == "whisper" and workers > 1:
        print(f"note: whisper is compute-bound; {workers} workers will share one model "
              f"and may be slower than 1. Set --workers 1 unless you have GPU headroom.")

    videos_dir = args.videos_dir.resolve()
    if not videos_dir.is_dir():
        sys.exit(f"not a directory: {videos_dir}")

    edit_dir = (args.edit_dir or (videos_dir / "edit")).resolve()
    (edit_dir / "transcripts").mkdir(parents=True, exist_ok=True)

    videos = find_videos(videos_dir)
    if not videos:
        sys.exit(f"no videos found in {videos_dir}")

    already_cached = [v for v in videos
                      if transcript_path(edit_dir, v, args.audio_track).exists()]
    pending = [v for v in videos if v not in already_cached]

    print(f"found {len(videos)} videos ({len(already_cached)} cached, {len(pending)} to transcribe)")
    if not pending:
        print("nothing to do")
        return

    api_key = load_api_key(required=backend == "scribe")

    print(f"transcribing {len(pending)} files with {workers} parallel workers ({backend} backend)")
    t0 = time.time()

    errors: list[tuple[Path, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                transcribe_one,
                video=v,
                edit_dir=edit_dir,
                api_key=api_key,
                language=args.language,
                num_speakers=args.num_speakers,
                verbose=False,
                audio_track=args.audio_track,
                asr=backend,
                model=args.model,
                device=args.device,
                compute=args.compute,
            ): v
            for v in pending
        }
        for fut in as_completed(futures):
            v = futures[fut]
            try:
                out = fut.result()
                print(f"  + {v.stem}  →  {out.name}")
            except Exception as e:
                errors.append((v, str(e)))
                print(f"  x {v.stem}  FAILED: {e}")

    dt = time.time() - t0
    print(f"\ndone in {dt:.1f}s")
    if errors:
        print(f"{len(errors)} failures:")
        for v, msg in errors:
            print(f"  {v.name}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
