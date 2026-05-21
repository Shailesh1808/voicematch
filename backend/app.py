"""
Flask application entry point for VoiceMatch.

Handles HTTP routing for /api/compare, /api/health, and /api/spectrogram/<vocalist_id>.
Precomputes reference embeddings and spectrograms at startup.
All audio processing is delegated to audio_processor.py and similarity.py.
"""
