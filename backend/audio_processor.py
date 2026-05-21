"""
Audio feature extraction for VoiceMatch.

Implements the full MFCC pipeline: format conversion, loading, normalization,
silence trimming, MFCC extraction with delta features, cepstral mean normalization,
energy-based frame filtering, mean pooling, and L2 normalization.
Also generates mel spectrogram PNGs for display.
All librosa calls are confined to this module.
"""
