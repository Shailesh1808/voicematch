# VoiceMatch — Audio Pipeline Design Document

## 1. The Core Question This Solves
When two people sing, their voices sound different even if they sing the same
note at the same pitch. One voice might be warmer, breathier, more nasal,
richer in harmonics. These differences are what make a voice recognizable.

The pipeline converts a raw audio recording into a compact set of numbers
that captures these differences — called an embedding or feature vector.
Once we have embeddings for the user and each reference vocalist, we
mathematically measure how similar they are.

## 2. Key Concepts

### Sound wave
Audio is stored as numbers representing air pressure over time. A 44.1kHz
WAV file has 44,100 numbers per second. Too much raw data to compare
directly — we need to summarize it.

### Frequency
Low frequency = deep/bass sound. High frequency = bright/treble sound.
A human voice contains many frequencies at once. The mix of those
frequencies is what makes your voice sound like you.

### Spectrogram
A 2D picture of sound:
- X axis = time
- Y axis = frequency
- Color/brightness = how loud that frequency is at that moment

### Mel Spectrogram
Human ears don't hear all frequencies equally. The mel scale remaps
frequencies to match human pitch perception. A mel spectrogram uses
this scale. Important because we want to measure what a human ear
would notice — not just what the microphone physically captures.

### MFCCs (Mel-Frequency Cepstral Coefficients)
Take a mel spectrogram → apply one more compression step (DCT) →
get a small set of numbers (we use 40) summarizing the shape of the spectrum.

If a mel spectrogram is a detailed photograph of a voice,
MFCCs are a 40-number sketch of that photograph.
Captures essential vocal character. Standard feature for voice similarity.

## 3. Handling Different Lengths — Mean Pooling

User records 2–15 seconds. References are 6–18 seconds.
We cannot compare vectors of different lengths directly.

Solution: mean pooling over voiced frames only.
Average each coefficient across all voiced time frames.
Collapses any recording into a fixed-length vector.

Recording A (5s):  (120, 215 frames) → mean → (120,)
Recording B (12s): (120, 516 frames) → mean → (120,)
Both become the same shape. Directly comparable.

Alternatives considered:
- DTW (Dynamic Time Warping): more accurate but complex and slow
- Fixed-length windowing: wastes data, introduces crop artifacts
Mean pooling is simple, fast, explainable, and captures average vocal
character — which is what we want.

DECISIONS.md entry: Mean pooling vs DTW

## 4. Handling Noise and Silence

### Problem
Real browser recordings contain:
- Leading silence (before the user starts singing)
- Trailing silence (after they stop)
- Mid-recording silence (pauses between phrases)
- Background noise (room, fan, street)
- Breath sounds between phrases

If not handled, the MFCC embedding represents silence and noise as much
as the voice. Two people in quiet rooms appear more similar than they are.

### Solutions (simple, no models)

1. Trim leading/trailing silence
   librosa.effects.trim(top_db=20)
   Removes everything below 1% of peak loudness from both ends.

2. Energy-based frame filtering (handles mid-recording silence + breaths)
   - Compute RMS energy per time frame
   - voiced_mask = energy > (mean_energy - 1 standard deviation)
   - Mean pool only over voiced frames
   - Adaptive threshold — adjusts to each recording's loudness level
   - If fewer than 10 voiced frames remain → raise error

3. Cepstral mean normalization
   Subtract the mean of each MFCC coefficient across all frames.
   Removes constant contribution of background noise and mic coloration.
   Standard technique in speaker recognition.

### What this covers
- Silence before singing:         ✅ trim
- Silence after singing:          ✅ trim
- Pauses between phrases:         ✅ energy frame filtering
- Breath sounds:                  ✅ energy frame filtering (low energy)
- Constant background noise:      ✅ cepstral mean normalization
- Loud background noise/music:    ❌ out of scope — noted in README

DECISIONS.md entry: Cepstral mean normalization for noise robustness

## 5. Full Pipeline

Step 1: FORMAT CONVERSION
  ffmpeg converts WebM/MP4/etc → 16kHz mono WAV
  Why 16kHz: voice lives below 8kHz, 16kHz captures everything needed,
  half the data of 44.1kHz

Step 2: LOAD & VALIDATE
  librosa.load(path, sr=16000, mono=True)
  Check duration ≥ 2 seconds. If not → raise AudioTooShortError

Step 3: PRE-PROCESS
  Peak amplitude normalization — quiet and loud recordings compare fairly
  librosa.effects.trim(top_db=20) — remove leading/trailing silence

Step 4: EXTRACT MFCCs
  librosa.feature.mfcc(y, sr=16000, n_mfcc=40, n_fft=2048, hop_length=512)
  Result shape: (40, T) where T = number of time frames

Step 5: DELTA FEATURES
  delta  = librosa.feature.delta(mfcc)          shape: (40, T)
  delta2 = librosa.feature.delta(mfcc, order=2) shape: (40, T)
  Concatenate: np.vstack([mfcc, delta, delta2])  shape: (120, T)
  Why: static features capture what the voice is,
       delta captures how it moves,
       delta-delta captures acceleration of that movement

Step 6: CEPSTRAL MEAN NORMALIZATION
  features -= np.mean(features, axis=1, keepdims=True)
  Suppresses constant background noise and microphone coloration

Step 7: ENERGY-BASED FRAME FILTERING
  rms = librosa.feature.rms(y=audio, hop_length=512)
  threshold = mean(rms) - std(rms)
  voiced_mask = rms > threshold
  If sum(voiced_mask) < 10 → raise AudioTooShortError
  Apply mask to features: features[:, voiced_mask]

Step 8: MEAN POOL
  embedding = np.mean(features[:, voiced_mask], axis=1)
  Shape: (120,)

Step 9: L2 NORMALIZE
  embedding = embedding / np.linalg.norm(embedding)
  Required for cosine similarity to be meaningful
  Removes magnitude differences (volume) from the comparison

Final output: a 120-dimensional unit vector representing vocal character

## 6. Comparison — Cosine Similarity

cosine_similarity(A, B) = A · B  (since both are L2 normalized)

Why cosine and not Euclidean distance:
- Euclidean measures how far apart two points are in space
- Cosine measures the direction they point, ignoring magnitude
- For voice embeddings, direction = vocal character
- Magnitude = how energetic the recording is
- We want direction, not magnitude

DECISIONS.md entry: Cosine similarity vs Euclidean distance

## 7. Score Normalization for Display

Raw cosine scores → softmax → percentages

exp_scores  = np.exp(raw_scores)
percentages = (exp_scores / exp_scores.sum()) * 100

Why softmax and not simple division:
Simple division on scores like [0.84, 0.81, 0.79] produces
[34%, 33%, 32%] — meaningless, no clear winner.
Softmax amplifies differences so the top match visually stands out
while still summing to 100%.

DECISIONS.md entry: Softmax normalization for display scores

## 8. Mel Spectrogram Generation (Bonus)

Separate from MFCC pipeline. Same input audio, different output.

S    = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=128)
S_db = librosa.power_to_db(S, ref=np.max)
Render to PNG with matplotlib, encode as base64 string

Generated at startup for all 10 reference vocalists (held in memory).
Generated on-demand for user recordings, returned in compare response.

## 9. Sanity Check

Feed sample_1.wav through the pipeline as a user recording.
Expected: Luna scores ~1.0, all others much lower.
Repeat with sample_6.wav → Vesper scores ~1.0.

If all vocalists score ~10% each:
- The embedding is not discriminative
- Likely cause: energy/loudness dominating the similarity
- Fix: verify L2 normalization and cepstral mean normalization are applied

## 10. Parameters

| Parameter      | Value | Reason                                        |
|----------------|-------|-----------------------------------------------|
| Sample rate    | 16000 | Voice < 8kHz; Nyquist means 16kHz is enough   |
| n_mfcc         | 40    | More than traditional 13; finer vocal detail  |
| n_fft          | 2048  | ~128ms window; good time/frequency balance    |
| hop_length     | 512   | ~32ms step; standard for speech/singing       |
| Delta features | Yes×3 | Static + velocity + acceleration              |
| Vector size    | 120   | 40 MFCCs × 3 (static + delta + delta-delta)   |
| Similarity     | Cosine| Direction not magnitude                       |
| Display scores | Softmax %| Amplifies differences, sums to 100%        |

## 11. What We Would Improve With More Time
- VGGish or CLAP pretrained embeddings — far better voice identity capture
- Multiple samples per vocalist — average embeddings for robust profiles
- DTW for comparison — more accurate than mean pooling, more complex