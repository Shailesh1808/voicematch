"""
Flask application entry point for VoiceMatch.

Handles HTTP routing for /api/compare, /api/health, and /api/spectrogram/<vocalist_id>.
Precomputes reference embeddings and spectrograms at startup.
All audio processing is delegated to audio_processor.py and similarity.py.
"""

import logging
import os
import tempfile
from pathlib import Path

import soundfile as sf

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

import audio_processor
import similarity
from audio_processor import AudioTooShortError, InvalidAudioError

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

VOCALISTS = [
    {"id": "sample_1",  "name": "Luna",   "style": "Alto, Soul"},
    {"id": "sample_2",  "name": "Atlas",  "style": "Baritone, Folk"},
    {"id": "sample_3",  "name": "Echo",   "style": "Soprano, Pop"},
    {"id": "sample_4",  "name": "Riven",  "style": "Tenor, Rock"},
    {"id": "sample_5",  "name": "Sol",    "style": "Mezzo, Jazz"},
    {"id": "sample_6",  "name": "Vesper", "style": "Bass, Classical"},
    {"id": "sample_7",  "name": "Coda",   "style": "Alto, R&B"},
    {"id": "sample_8",  "name": "Fern",   "style": "Soprano, Indie"},
    {"id": "sample_9",  "name": "Birch",  "style": "Tenor, Acoustic"},
    {"id": "sample_10", "name": "Wren",   "style": "Mezzo, Country"},
]

reference_embeddings = {}
reference_spectrograms = {}


def precompute_references() -> None:
    """Load all reference vocalist .wav files and precompute their embeddings and spectrograms.

    Reads REFERENCE_DATA_DIR from the environment (defaults to ../reference_data relative
    to this file). For each vocalist in VOCALISTS, extracts an MFCC embedding and generates
    a mel spectrogram PNG. Both are stored in module-level dicts keyed by vocalist ID.

    Raises
    ------
    RuntimeError
        If no reference files could be loaded at all.
    """
    raw_dir = os.environ.get("REFERENCE_DATA_DIR")
    if raw_dir:
        reference_data_dir = Path(raw_dir)
    else:
        reference_data_dir = Path(__file__).parent.parent / "reference_data"

    logger.info("Loading reference data from: %s", reference_data_dir)

    for vocalist in VOCALISTS:
        vid = vocalist["id"]
        name = vocalist["name"]
        wav_path = reference_data_dir / f"{vid}.wav"

        if not wav_path.exists():
            logger.warning("Reference file not found, skipping: %s", wav_path)
            continue

        try:
            embedding = audio_processor.extract_embedding(str(wav_path))
            spectrogram = audio_processor.generate_spectrogram(str(wav_path))
            reference_embeddings[vid] = embedding
            reference_spectrograms[vid] = spectrogram
            logger.info("Loaded reference: %s", name)
        except Exception:
            logger.exception("Failed to process reference file for %s (%s)", name, wav_path)

    if not reference_embeddings:
        logger.critical("No reference embeddings loaded. Check reference_data/ folder.")
        raise RuntimeError("No reference embeddings loaded. Check reference_data/ folder.")

    logger.info("Precomputed %d reference profiles", len(reference_embeddings))


precompute_references()

app = Flask(__name__)
# CORS(app, origins=["https://voicematch-mu.vercel.app/"])
CORS(app, origins="*")


@app.route("/api/health", methods=["GET"])
def health():
    """Return server status and number of loaded reference profiles."""
    return jsonify({"status": "ok", "references_loaded": len(reference_embeddings)}), 200


@app.route("/api/compare", methods=["POST"])
def compare():
    """Accept a user audio upload and return ranked vocalist similarity results.

    Expects multipart/form-data with a field named "audio". Extracts an MFCC
    embedding, ranks it against all loaded reference profiles, and returns the
    sorted results alongside the user's mel spectrogram as a base64 PNG.

    Returns
    -------
    JSON 200
        {"results": [...], "top_match": str, "user_spectrogram": str}
    JSON 400
        If the audio field is missing, the recording is too short, or the file
        cannot be decoded.
    JSON 500
        For any other unexpected error.
    """
    if "audio" not in request.files:
        return jsonify({"error": "no_audio", "message": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    logger.info(
        "POST /api/compare — received audio file: %s, size: %s bytes",
        audio_file.filename or "unnamed",
        request.content_length or "unknown",
    )

    original_filename = audio_file.filename or ""
    suffix = Path(original_filename).suffix if original_filename else ".webm"
    if not suffix:
        suffix = ".webm"

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = Path(tmp_file.name)

    try:
        audio_file.save(tmp_file)
        tmp_file.close()

        info = sf.info(str(tmp_path))
        if info.duration > 15:
            return jsonify({
                "error": "recording_too_long",
                "message": "Recording is too long. Please record no more than 15 seconds.",
            }), 400

        user_embedding = audio_processor.extract_embedding(str(tmp_path))
        user_spectrogram = audio_processor.generate_spectrogram(str(tmp_path))

        results = similarity.rank_results(user_embedding, reference_embeddings, VOCALISTS)
        top_match = similarity.get_top_match(results)

        logger.info("POST /api/compare → 200, top match: %s", top_match)
        return jsonify({
            "results": results,
            "top_match": top_match,
            "user_spectrogram": user_spectrogram,
        }), 200

    except AudioTooShortError as e:
        logger.warning("AudioTooShortError: %s", str(e))
        return jsonify({
            "error": "recording_too_short",
            "message": str(e),
        }), 400

    except InvalidAudioError as e:
        logger.warning("InvalidAudioError: %s", str(e))
        return jsonify({
            "error": "invalid_audio",
            "message": str(e),
        }), 400

    except Exception:
        logger.exception("Unexpected error during /api/compare")
        return jsonify({
            "error": "server_error",
            "message": "Something went wrong. Please try again.",
        }), 500

    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@app.route("/api/spectrogram/<vocalist_id>", methods=["GET"])
def spectrogram(vocalist_id: str):
    """Return the precomputed mel spectrogram for a reference vocalist.

    Parameters
    ----------
    vocalist_id : str
        The vocalist's ID (e.g. "sample_1").

    Returns
    -------
    JSON 200
        {"image": "<base64 PNG>"}
    JSON 404
        If the vocalist_id is not found in the loaded references.
    """
    if vocalist_id not in reference_spectrograms:
        return jsonify({"error": "not_found", "message": "Vocalist not found"}), 404

    return jsonify({"image": reference_spectrograms[vocalist_id]}), 200


if __name__ == "__main__":
    port = int(os.environ.get("FLASK_PORT", 5000))
    debug = os.environ.get("FLASK_ENV") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
