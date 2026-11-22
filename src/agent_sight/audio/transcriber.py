"""Optional local speech-to-text fallback (faster-whisper preferred, openai-whisper accepted)."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from agent_sight.config import Settings, get_settings
from agent_sight.errors import AgentError, ErrorCode
from agent_sight.models import Transcript, TranscriptSegment, Word

log = logging.getLogger("agent_sight.whisper")
_models: dict[tuple[str, str, str], Any] = {}
_lock = threading.Lock()


def available_backend() -> str | None:
    try:
        import faster_whisper  # type: ignore[import-not-found]  # noqa: F401

        return "faster-whisper"
    except ImportError:
        pass
    try:
        import whisper  # type: ignore[import-not-found]  # noqa: F401

        return "openai-whisper"
    except ImportError:
        return None


def _faster_model(s: Settings) -> Any:
    from faster_whisper import WhisperModel  # type: ignore[import-not-found]

    key = ("fw", s.whisper_model, s.whisper_device)
    with _lock:
        if key not in _models:
            compute = s.whisper_compute_type
            if compute == "auto":
                compute = "int8" if s.whisper_device in ("auto", "cpu") else "float16"
            _models[key] = WhisperModel(s.whisper_model, device=s.whisper_device, compute_type=compute)
        return _models[key]


def transcribe(
    audio_path: Path,
    *,
    language: str | None = None,
    word_timestamps: bool = False,
    settings: Settings | None = None,
) -> Transcript:
    s = settings or get_settings()
    backend = available_backend()
    if backend is None:
        raise AgentError(
            ErrorCode.DEPENDENCY_MISSING,
            "No local Whisper backend is installed.",
            "pip install 'agent-sight[whisper]' (installs faster-whisper), then retry.",
        )
    lang = language.split("-")[0] if language else None
    try:
        if backend == "faster-whisper":
            model = _faster_model(s)
            seg_iter, info = model.transcribe(
                str(audio_path), language=lang, vad_filter=True, word_timestamps=word_timestamps,
                beam_size=1,
            )
            segments = [
                TranscriptSegment(
                    start=float(seg.start), end=float(seg.end), text=seg.text.strip(),
                    words=[
                        Word(start=float(w.start), end=float(w.end), text=w.word.strip(),
                             probability=float(w.probability))
                        for w in (seg.words or [])
                    ] if word_timestamps else None,
                )
                for seg in seg_iter if seg.text.strip()
            ]
            detected = getattr(info, "language", None)
        else:
            import whisper  # type: ignore[import-not-found]

            key = ("ow", s.whisper_model, s.whisper_device)
            with _lock:
                if key not in _models:
                    _models[key] = whisper.load_model(s.whisper_model)
            result = _models[key].transcribe(str(audio_path), language=lang, word_timestamps=word_timestamps)
            segments = [
                TranscriptSegment(
                    start=float(seg["start"]), end=float(seg["end"]), text=seg["text"].strip(),
                    words=[
                        Word(start=float(w["start"]), end=float(w["end"]), text=w["word"].strip(),
                             probability=w.get("probability"))
                        for w in seg.get("words", [])
                    ] if word_timestamps else None,
                )
                for seg in result.get("segments", []) if seg["text"].strip()
            ]
            detected = result.get("language")
    except AgentError:
        raise
    except Exception as exc:
        raise AgentError(ErrorCode.TRANSCRIPTION_FAILED, f"{backend} failed: {exc}") from exc
    if not segments:
        raise AgentError(ErrorCode.NO_TRANSCRIPT, "No speech was detected in the audio track.",
                         "The video may be silent or music-only; use visual tools instead.")
    return Transcript(language=detected or lang, source="whisper", segments=segments)
