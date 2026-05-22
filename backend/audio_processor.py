"""
Audio feature extraction for VoiceMatch.

Implements the full MFCC pipeline: format conversion, loading, normalization,
silence trimming, MFCC extraction with delta features, cepstral mean normalization,
energy-based frame filtering, mean pooling, and L2 normalization.
Also generates mel spectrogram PNGs for display.
All librosa calls are confined to this module.
"""

import base64
import io
import logging
import subprocess
import tempfile
from pathlib import Path

import librosa
import librosa.display 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512
MIN_DURATION_SECONDS = 2.0
MAX_DURATION_SECONDS = 15.0
MIN_VOICED_FRAMES = 20
SILENCE_TRIM_DB = 20
N_MELS = 128
MINIMUM_RMS_THRESHOLD = 0.01


class AudioTooShortError(Exception):
    pass


class InvalidAudioError(Exception):
    pass


def convert_to_wav(input_path: str, output_path: str) -> None:
    """Convert any audio file to a 16 kHz mono WAV using ffmpeg.

    Parameters
    ----------
    input_path : str
        Path to the source audio file in any browser-supported format.
    output_path : str
        Destination path for the converted WAV file.

    Raises
    ------
    InvalidAudioError
        If ffmpeg exits with a non-zero return code, indicating the input
        could not be decoded or written.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        str(output_path),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        logger.error(
            "ffmpeg failed for %s: %s",
            input_path,
            result.stderr.decode(errors="replace"),
        )
        raise InvalidAudioError(
            f"Could not convert audio file: {input_path.name}"
        )


def extract_embedding(audio_path: str) -> np.ndarray:
    """Extract a 120-dimensional L2-normalised MFCC embedding from an audio file.

    Runs the full VoiceMatch pipeline:
      1. Convert to 16 kHz mono WAV via ffmpeg.
      2. Load with librosa.
      3. Validate minimum duration (>= 2 s).
      4. Peak-amplitude normalise.
      5. Trim leading/trailing silence.
      6. Extract 40 MFCCs.
      7. Compute delta and delta-delta features; stack to (120, T).
      8. Cepstral mean normalisation.
      9. Energy-based voiced-frame filtering.
      10. Mean-pool over voiced frames → (120,).
      11. L2 normalise.

    Parameters
    ----------
    audio_path : str
        Path to the source audio file (any format supported by ffmpeg).

    Returns
    -------
    np.ndarray
        Unit vector of shape (120,) representing the vocal character.

    Raises
    ------
    AudioTooShortError
        If the recording is shorter than MIN_DURATION_SECONDS, or if
        fewer than MIN_VOICED_FRAMES remain after energy filtering.
    InvalidAudioError
        If ffmpeg cannot decode the input file.
    """
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = Path(tmp_wav.name)
    tmp_wav.close()

    try:
        # Step 1: convert to 16 kHz mono WAV
        convert_to_wav(audio_path, str(tmp_path))

        # Step 2: load audio
        audio, _ = librosa.load(str(tmp_path), sr=SAMPLE_RATE, mono=True)

        # Step 3: validate duration bounds
        duration = len(audio) / SAMPLE_RATE
        if duration < MIN_DURATION_SECONDS:
            raise AudioTooShortError("Recording is too short. Please record for at least 2 seconds.")
        if duration > MAX_DURATION_SECONDS:
            raise InvalidAudioError("Recording is too long. Please record no more than 15 seconds.")

        # Step 3b: reject recordings with insufficient raw energy before normalisation
        # amplifies them — after normalisation even sub-threshold noise reaches peak=1.
        raw_rms = float(np.sqrt(np.mean(audio ** 2)))
        if raw_rms < MINIMUM_RMS_THRESHOLD:
            raise AudioTooShortError(
                "No audio detected. Please make sure your microphone is working and try again."
            )

        # Step 4: peak amplitude normalisation
        audio = audio / (np.max(np.abs(audio)) + 1e-9)

        # Step 5: trim leading/trailing silence
        audio, _ = librosa.effects.trim(audio, top_db=SILENCE_TRIM_DB)

        # Step 5b: validate duration after trimming
        trimmed_duration = len(audio) / SAMPLE_RATE
        if trimmed_duration < MIN_DURATION_SECONDS:
            raise AudioTooShortError("Recording contains too much silence. Please record at least 2 seconds of clear audio.")

        # Step 6: extract MFCCs
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )

        # Step 7: delta and delta-delta features → (120, T)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        features = np.vstack([mfcc, delta, delta2])

        # Step 8: cepstral mean normalisation
        features = features - np.mean(features, axis=1, keepdims=True)

        # Step 9: energy-based frame filtering
        rms = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)[0]
        threshold = max(
            float(np.mean(rms) - np.std(rms)),
            MINIMUM_RMS_THRESHOLD
        )
        voiced_mask = rms > threshold

        if np.sum(voiced_mask) < MIN_VOICED_FRAMES:
            raise AudioTooShortError(
                "Recording contains too much background noise. Please try again in a quieter environment."
            )

        features = features[:, voiced_mask]

        # Step 10: mean pool over voiced frames
        embedding = np.mean(features, axis=1)

        # Step 11: L2 normalise
        embedding = embedding / (np.linalg.norm(embedding) + 1e-9)

        return embedding

    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def generate_spectrogram(audio_path: str) -> str:
    """Generate a mel spectrogram PNG from an audio file and return it as base64.

    Converts the input to a 16 kHz mono WAV, computes a mel spectrogram,
    renders it with matplotlib (magma colourmap, transparent background),
    and encodes the resulting PNG as a base64 string suitable for embedding
    directly in a JSON response.

    Parameters
    ----------
    audio_path : str
        Path to the source audio file (any format supported by ffmpeg).

    Returns
    -------
    str
        Base64-encoded PNG of the mel spectrogram.

    Raises
    ------
    InvalidAudioError
        If ffmpeg cannot decode the input file.
    """
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = Path(tmp_wav.name)
    tmp_wav.close()

    try:
        convert_to_wav(audio_path, str(tmp_path))

        audio, _ = librosa.load(str(tmp_path), sr=SAMPLE_RATE, mono=True)

        S = librosa.feature.melspectrogram(
            y=audio, sr=SAMPLE_RATE, n_mels=N_MELS
        )
        S_db = librosa.power_to_db(S, ref=np.max)

        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(
            S_db,
            sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            x_axis="time",
            y_axis="mel",
            cmap="magma",
            ax=ax,
        )
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        plt.tight_layout()
        fig.patch.set_alpha(0)

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)

        return base64.b64encode(buf.read()).decode("utf-8")

    finally:
        if tmp_path.exists():
            tmp_path.unlink()
