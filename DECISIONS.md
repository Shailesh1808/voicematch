# VoiceMatch — Design Decisions

This document records the key technical decisions made during the build of VoiceMatch, the alternatives that were considered, and the rationale for each choice.

---

## 1. MFCCs as the Core Voice Representation

**Decision:** Use 40 Mel-Frequency Cepstral Coefficients (MFCCs) as the primary voice feature, augmented with delta and delta-delta features to produce a 120-dimensional embedding.

**Alternatives considered:**
- **Raw waveform** — passing the raw audio signal directly into a similarity function. Rejected because raw waveforms are extremely sensitive to timing, pitch, and recording conditions. Two people singing the same phrase will produce completely different waveforms even if their voices sound similar.
- **Mel spectrogram** — using the full 2D time-frequency representation. Rejected as a direct comparison target because it's high-dimensional and not aggregated into a fixed-size vector, making similarity computation expensive and noisy.
- **Pretrained embeddings (CLAP, VGGish, openl3)** — using a neural network trained on large audio datasets to produce embeddings. A stronger approach but adds hundreds of megabytes to the Docker image and significantly increases cold start time on Railway. Identified as the primary future improvement.

**Why MFCCs:**
MFCCs capture the shape of the vocal tract — the resonance characteristics that make each voice unique — independently of what pitch or note is being sung. They are the industry standard for voice characterisation in speech recognition and speaker verification. The delta and delta-delta features add temporal dynamics (how the voice changes over time), making the embedding significantly more discriminative than static MFCCs alone.

**Impact on VoiceMatch:** The 120-dimensional MFCC embedding is the single number that represents a user's voice. Every comparison, ranking, and similarity score flows from this vector.

---

## 2. Cosine Similarity over Euclidean Distance

**Decision:** Compare voice embeddings using cosine similarity (dot product of two L2-normalised vectors).

**Alternatives considered:**
- **Euclidean distance** — measures the straight-line distance between two points in 120-dimensional space. Rejected because it is sensitive to the magnitude of vectors, not just their direction. Two identical voices recorded at different volumes would appear dissimilar under Euclidean distance.
- **Manhattan distance** — sum of absolute differences. Similar magnitude sensitivity problem as Euclidean.
- **Pearson correlation** — statistically equivalent to cosine similarity on zero-mean vectors, but more complex to compute with no practical benefit here.

**Why cosine similarity:**
After L2 normalisation, all embeddings lie on the surface of a unit hypersphere. Cosine similarity measures the angle between two vectors, which captures how similar their vocal *character* is regardless of recording loudness or microphone sensitivity. Two users singing at different volumes will still produce embeddings pointing in the same direction, giving a high similarity score.

**Impact on VoiceMatch:** Users get consistent similarity scores regardless of how loudly or quietly they sing.

---

## 3. Softmax Normalisation for Display Scores

**Decision:** Convert raw cosine similarity scores to display percentages using softmax with temperature scaling.

**Alternatives considered:**
- **Raw cosine similarity** — displaying the raw value between -1 and 1. Rejected because users find raw similarity scores unintuitive, and the differences between vocalists are often small (e.g. 0.72 vs 0.68), making rankings feel arbitrary.
- **Min-max normalisation** — rescaling so the lowest score is 0% and the highest is 100%. Rejected because it would always make the best match 100% and the worst 0%, regardless of how close or far apart the actual similarities are.
- **Linear percentage** — multiplying cosine similarity by 100. Rejected because negative cosine similarities (very dissimilar voices) would produce negative percentages, which are confusing to display.

**Why softmax:**
Softmax converts a set of real numbers into a probability distribution that sums to 100%. It naturally amplifies differences between scores (the highest score gets a disproportionately large share) while keeping all values positive and summing to a meaningful total. The temperature parameter controls how spread out the distribution is.

**Impact on VoiceMatch:** The percentage scores shown to users are intuitive and always sum to 100%, making the ranking feel like a genuine breakdown of vocal similarity.

---

## 4. Cepstral Mean Normalisation (CMN)

**Decision:** Subtract the per-feature mean across time from all MFCC frames before pooling.

**Alternatives considered:**
- **No normalisation** — using raw MFCCs directly. Rejected because microphone quality, room acoustics, and background noise all introduce a consistent offset across all frames. Two people with identical voices but different microphones would get different embeddings.
- **Global standardisation** — subtracting a fixed mean computed from a training corpus. Rejected because we have no training corpus and this would require recomputing the normalisation statistics when new reference voices are added.
- **Per-utterance variance normalisation** — also dividing by the standard deviation. Considered but not implemented because it can over-normalise quiet, expressive passages where variance is naturally low.

**Why CMN:**
CMN removes the channel effect — the systematic bias introduced by recording conditions — by centring each feature dimension around zero. Because the offset is consistent across time within a single recording, subtracting the mean removes it exactly. This makes the embedding represent vocal character rather than microphone character.

**Impact on VoiceMatch:** Users can record on any device — phone, laptop, headset — and get fair comparisons against the reference vocalists, who were also recorded on specific equipment.

---

## 5. Three-Layer Silence Detection

**Decision:** Implement three separate checks to reject silent or near-silent recordings, at different stages of the pipeline.

The three layers are:
1. **Raw RMS gate** — reject if RMS energy before normalisation is below 0.01
2. **Post-trim duration check** — reject if less than 2 seconds of audio remains after silence trimming
3. **Voiced frame filtering** — discard frames with energy below (mean RMS − 1 standard deviation), reject if fewer than 10 voiced frames remain

**Alternatives considered:**
- **Single duration check only** — rejecting recordings under 2 seconds. Rejected because a 5-second recording of someone breathing quietly would pass, but produce a meaningless embedding.
- **Single energy check only** — rejecting low-RMS recordings. Rejected because a recording could have high peak energy (a cough at the start) but contain no singing.
- **VAD (Voice Activity Detection)** — using a dedicated algorithm like WebRTC VAD to identify voiced segments. A stronger approach but adds a dependency and complexity for marginal gain at this scale.

**Why three layers:**
Each layer catches a different failure mode:
- Layer 1 catches completely silent recordings before normalisation amplifies them
- Layer 2 catches recordings that are mostly silence with a brief sound
- Layer 3 catches recordings with background noise but no actual singing, and also removes unvoiced frames from the embedding computation to improve accuracy

**Impact on VoiceMatch:** Users get specific, actionable error messages instead of a confusing result from a meaningless embedding. The pipeline only produces embeddings from recordings that contain real singing.

---

## 6. Precomputing Reference Embeddings at Startup

**Decision:** Extract embeddings and spectrograms from all 10 reference WAV files once when the Flask app starts, and store them in memory for the lifetime of the process.

**Alternatives considered:**
- **Compute on every request** — running the full MFCC pipeline on all 10 reference files for every user recording. Rejected because it would add ~2–3 seconds of latency to every comparison and waste CPU on identical computations.
- **Cache to disk** — saving embeddings to `.npy` files and loading them on startup. A valid approach but adds file management complexity. Chosen as a future optimisation for cold start performance.
- **Lazy loading** — computing each reference embedding the first time it is needed. Rejected because the first user after a cold start would experience the worst possible latency.

**Why precompute at startup:**
The reference embeddings never change between deployments. Computing them once at startup costs ~20 seconds on cold start but reduces every subsequent `/api/compare` request to milliseconds of similarity computation. The tradeoff is clearly worth it at any scale beyond a single user.

**Impact on VoiceMatch:** Response times for `/api/compare` are fast and consistent for all users.

---

## 7. Mean Pooling over Voiced Frames

**Decision:** Aggregate the per-frame feature matrix into a single embedding vector by taking the mean across all voiced frames.

**Alternatives considered:**
- **Max pooling** — taking the maximum value in each feature dimension across frames. Rejected because it captures the most extreme moment in the recording rather than the overall character. A single loud or expressive moment would dominate the embedding.
- **First frame only** — using the MFCC vector from the first voiced frame. Rejected because a single 32ms window is far too little data to characterise a voice.
- **Concatenation with fixed length** — padding or truncating to a fixed number of frames and flattening. Rejected because it is sensitive to recording length and requires choosing an arbitrary fixed length.
- **Learned aggregation (attention pooling)** — using a neural network to weight frames by importance. A stronger approach but requires training data and a model, which is out of scope.

**Why mean pooling:**
Mean pooling produces a single vector that represents the average vocal character across the entire recording. It is robust to the recording length (a 2-second and a 10-second recording of the same voice will produce similar embeddings), computationally trivial, and well-understood. By restricting pooling to voiced frames only, we ensure the mean reflects actual singing rather than silence.

**Impact on VoiceMatch:** Users can record for anywhere between 2 and 15 seconds and get consistent, comparable results.