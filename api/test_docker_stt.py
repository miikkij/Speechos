"""Unit tests for Docker STT engine management: no Docker required."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from src.docker_stt import (
    DOCKER_STT_ENGINES,
    is_docker_stt_engine,
    get_docker_stt_config,
    _parse_text_response,
    _parse_linto_response,
    _parse_whisperx_response,
)


class TestEngineRegistry:
    """Test the DOCKER_STT_ENGINES registry."""

    def test_expected_engines_present(self):
        expected = {
            "speaches", "whisper-asr",
            "linto-nemo", "linto-whisper", "linto-nemo-1.1b",
        }
        assert set(DOCKER_STT_ENGINES.keys()) == expected

    def test_excluded_engines(self):
        # vosk: WebSocket-only, not REST
        assert "vosk" not in DOCKER_STT_ENGINES
        # whisper-cpp: requires manual model download
        assert "whisper-cpp" not in DOCKER_STT_ENGINES
        # whisperx-api: upstream bug, VAD model download fails
        assert "whisperx-api" not in DOCKER_STT_ENGINES

    def test_all_engines_have_required_keys(self):
        required = {"service", "port", "health_url", "transcribe_url", "file_field", "parser"}
        for name, cfg in DOCKER_STT_ENGINES.items():
            missing = required - set(cfg.keys())
            assert not missing, f"Engine {name} missing keys: {missing}"

    def test_port_allocations(self):
        expected_ports = {
            "speaches": 36320,
            "whisper-asr": 36321,
            "linto-nemo": 36323,
            "linto-whisper": 36324,
            "linto-nemo-1.1b": 36327,
        }
        for engine, port in expected_ports.items():
            assert DOCKER_STT_ENGINES[engine]["port"] == port, f"{engine} port mismatch"

    def test_no_duplicate_ports(self):
        ports = [cfg["port"] for cfg in DOCKER_STT_ENGINES.values()]
        assert len(ports) == len(set(ports)), "Duplicate ports found"

    def test_linto_engines_require_accept_header(self):
        for name in ("linto-nemo", "linto-whisper", "linto-nemo-1.1b"):
            headers = DOCKER_STT_ENGINES[name].get("headers", {})
            assert headers.get("Accept") == "application/json", f"{name} missing Accept header"

    def test_speaches_has_model_field(self):
        extra = DOCKER_STT_ENGINES["speaches"].get("extra_fields", {})
        assert "model" in extra


class TestIsDockerSttEngine:
    def test_known_engines(self):
        for engine in DOCKER_STT_ENGINES:
            assert is_docker_stt_engine(engine)

    def test_native_engines_are_not_docker(self):
        for engine in ("faster-whisper", "vosk", "nemo", "whisperx", "moonshine"):
            assert not is_docker_stt_engine(engine)

    def test_get_config_returns_dict(self):
        cfg = get_docker_stt_config("speaches")
        assert isinstance(cfg, dict)
        assert cfg["port"] == 36320

    def test_get_config_returns_none_for_unknown(self):
        assert get_docker_stt_config("nonexistent") is None


class TestResponseParsers:
    """Test response parsers with mock httpx responses."""

    def _mock_response(self, json_data=None, text_data=None, content_type="application/json"):
        resp = MagicMock()
        resp.headers = {"content-type": content_type}
        if json_data is not None:
            resp.json.return_value = json_data
        resp.text = text_data or ""
        return resp

    def test_text_response_json(self):
        resp = self._mock_response(json_data={"text": "Hello world"})
        result = _parse_text_response(resp)
        assert result["text"] == "Hello world"
        assert result["segments"] == []

    def test_text_response_plain(self):
        resp = self._mock_response(text_data="Hello world", content_type="text/plain")
        result = _parse_text_response(resp)
        assert result["text"] == "Hello world"

    def test_linto_response(self):
        resp = self._mock_response(json_data={
            "text": "Hello world",
            "segments": [
                {"start": 0.0, "end": 1.5, "text": "Hello"},
                {"start": 1.5, "end": 2.5, "text": "world"},
            ],
            "language": "en",
        })
        result = _parse_linto_response(resp)
        assert result["text"] == "Hello world"
        assert len(result["segments"]) == 2
        assert result["segments"][0]["text"] == "Hello"
        assert result["language"] == "en"

    def test_linto_response_no_segments(self):
        resp = self._mock_response(json_data={"text": "Hello"})
        result = _parse_linto_response(resp)
        assert result["text"] == "Hello"
        assert result["segments"] == []

    def test_whisperx_response(self):
        resp = self._mock_response(json_data={
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "Hello"},
                {"start": 1.0, "end": 2.0, "text": "world"},
            ],
        })
        result = _parse_whisperx_response(resp)
        assert result["text"] == "Hello world"
        assert len(result["segments"]) == 2

    def test_whisperx_response_with_text_field(self):
        resp = self._mock_response(json_data={
            "text": "Override text",
            "segments": [{"start": 0, "end": 1, "text": "seg"}],
        })
        result = _parse_whisperx_response(resp)
        assert result["text"] == "Override text"


class TestWebMDecoding:
    """Test that WebM/Opus audio decoding preserves amplitude."""

    def test_wav_to_webm_roundtrip_preserves_amplitude(self):
        """Regression test: PyAV float audio must not be divided by 32768."""
        import subprocess
        import tempfile
        import os
        from src.audio import read_audio, float32_to_wav_bytes

        # Create a 1-second test tone WAV
        sr = 16000
        t = np.linspace(0, 1, sr, dtype=np.float32)
        tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        wav_bytes = float32_to_wav_bytes(tone, sr)

        wav_path = tempfile.mktemp(suffix=".wav")
        webm_path = tempfile.mktemp(suffix=".webm")
        try:
            with open(wav_path, "wb") as f:
                f.write(wav_bytes)
            subprocess.run(
                ["ffmpeg", "-y", "-i", wav_path, "-c:a", "libopus", "-b:a", "64k", webm_path],
                capture_output=True, check=True,
            )
            with open(webm_path, "rb") as f:
                webm_bytes = f.read()

            audio_decoded, sr_decoded = read_audio(webm_bytes)
            rms = float(np.sqrt(np.mean(audio_decoded**2)))

            # Original tone has RMS ~0.354; after lossy Opus encoding it should
            # still be in the same ballpark, NOT crushed to near-zero.
            assert rms > 0.1, f"WebM decode RMS too low ({rms:.6f}), float audio may have been incorrectly normalized"
            assert sr_decoded == 48000  # Opus always decodes at 48kHz
        finally:
            os.unlink(wav_path)
            if os.path.exists(webm_path):
                os.unlink(webm_path)
