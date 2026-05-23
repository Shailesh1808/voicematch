import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import base64
import tempfile

import numpy as np
import pytest
import soundfile as sf

from backend.audio_processor import (
    AudioTooShortError,
    InvalidAudioError,
    SAMPLE_RATE,
    convert_to_wav,
    extract_embedding,
    generate_spectrogram,
)


@pytest.fixture
def real_wav():
    """Path to a real reference WAV file."""
    return "reference_data/sample_1.wav"


@pytest.fixture
def short_wav():
    """A 1-second silent WAV — below the 2-second minimum duration."""
    audio = np.zeros(SAMPLE_RATE * 1, dtype=np.float32)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    sf.write(tmp_path, audio, SAMPLE_RATE)
    yield tmp_path
    Path(tmp_path).unlink(missing_ok=True)


@pytest.fixture
def silent_wav():
    """A 3-second fully silent WAV — long enough in duration but no voiced frames."""
    audio = np.zeros(SAMPLE_RATE * 3, dtype=np.float32)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    sf.write(tmp_path, audio, SAMPLE_RATE)
    yield tmp_path
    Path(tmp_path).unlink(missing_ok=True)


class TestExtractEmbedding:

    def test_returns_correct_shape(self, real_wav):
        embedding = extract_embedding(real_wav)
        assert embedding.shape == (123,)

    def test_returns_unit_vector(self, real_wav):
        embedding = extract_embedding(real_wav)
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-5

    def test_raises_for_short_recording(self, short_wav):
        with pytest.raises(AudioTooShortError):
            extract_embedding(short_wav)

    def test_raises_for_silent_recording(self, silent_wav):
        with pytest.raises(AudioTooShortError):
            extract_embedding(silent_wav)

    def test_raises_for_invalid_file(self, tmp_path):
        bad_file = tmp_path / "bad.wav"
        bad_file.write_bytes(b"this is not audio")
        with pytest.raises(InvalidAudioError):
            extract_embedding(str(bad_file))

    def test_same_file_returns_same_embedding(self, real_wav):
        emb1 = extract_embedding(real_wav)
        emb2 = extract_embedding(real_wav)
        np.testing.assert_array_almost_equal(emb1, emb2)

    def test_different_files_return_different_embeddings(self):
        emb1 = extract_embedding("reference_data/sample_1.wav")
        emb2 = extract_embedding("reference_data/sample_2.wav")
        assert not np.allclose(emb1, emb2)


class TestGenerateSpectrogram:

    def test_returns_string(self, real_wav):
        result = generate_spectrogram(real_wav)
        assert isinstance(result, str)

    def test_returns_valid_base64(self, real_wav):
        result = generate_spectrogram(real_wav)
        try:
            decoded = base64.b64decode(result)
            assert len(decoded) > 0
        except Exception:
            pytest.fail("generate_spectrogram did not return valid base64")

    def test_returns_png(self, real_wav):
        result = generate_spectrogram(real_wav)
        decoded = base64.b64decode(result)
        assert decoded[:8] == b'\x89PNG\r\n\x1a\n'


class TestConvertToWav:

    def test_converts_valid_wav(self, real_wav, tmp_path):
        output = str(tmp_path / "output.wav")
        convert_to_wav(real_wav, output)
        assert Path(output).exists()
        assert Path(output).stat().st_size > 0

    def test_raises_for_invalid_input(self, tmp_path):
        bad_file = tmp_path / "bad.webm"
        bad_file.write_bytes(b"not real audio")
        with pytest.raises(InvalidAudioError):
            convert_to_wav(str(bad_file), str(tmp_path / "out.wav"))
