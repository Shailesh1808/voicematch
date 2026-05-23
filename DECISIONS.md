# VoiceMatch - Design Decisions

This document records the key technical decisions made during the build of VoiceMatch, including what alternatives were considered and why each choice was made.

---

## 1. MFCCs as the Core Voice Representation

**Decision:** Use 40 Mel-Frequency Cepstral Coefficients (MFCCs) as the primary voice feature, augmented with delta and delta-delta features to produce a 120-dimensional embedding.

**Alternatives considered:**

Raw waveform comparison was ruled out immediately. Raw waveforms are extremely sensitive to timing, pitch, and recording conditions -- two people singing the same phrase will produce completely different waveforms even if their voices sound nearly identical.

Using the full mel spectrogram as a comparison target was also considered. The problem is that it produces a high-dimensional 2D matrix that varies in size with recording length, making direct comparison expensive and unreliable without further aggregation.

Pretrained audio embeddings like CLAP, VGGish, or openl3 would produce better results and were identified as the primary future improvement. They were not used in this version because they add hundreds of megabytes to the Docker image and significantly increase cold start time on Railway.

**Why MFCCs:**

MFCCs capture the shape of the vocal tract - the resonance characteristics that make a voice sound like a particular person - independently of what pitch or note is being sung. They are the standard feature for voice characterisation in both speech recognition and speaker verification. The delta and delta-delta features add temporal dynamics, capturing how the voice changes over time rather than just what it sounds like at a single moment.

---

## 2. Cosine Similarity over Euclidean Distance

**Decision:** Compare voice embeddings using cosine similarity, which is the dot product of two L2-normalised vectors.

**Alternatives considered:**

Euclidean distance measures the straight-line distance between two points in the embedding space. It was ruled out because it is sensitive to vector magnitude, not just direction. Two identical voices recorded at different volumes would appear dissimilar.

Manhattan distance has the same magnitude sensitivity problem.

Pearson correlation is mathematically equivalent to cosine similarity on zero-mean vectors but more complex to compute with no practical benefit here.

**Why cosine similarity:**

After L2 normalisation, every embedding sits on the surface of a unit hypersphere. Cosine similarity measures the angle between two vectors, which captures how similar the vocal characters are regardless of recording loudness or microphone sensitivity. Two people singing at very different volumes will still get embeddings pointing in similar directions, giving a high similarity score.

---

## 3. Softmax Normalisation for Display Scores

**Decision:** Convert raw cosine similarity scores to display percentages using softmax.

**Alternatives considered:**

Displaying raw cosine similarity values (between -1 and 1) was considered. The problem is that differences between vocalists are often small (e.g. 0.72 vs 0.68), making the rankings feel arbitrary and hard to interpret.

Min-max normalisation would rescale scores so the lowest is 0% and the highest is 100%. This was rejected because it would always show one vocalist at 100% and another at 0% regardless of how close or far apart the actual similarities are.

Multiplying cosine similarity by 100 to get a percentage was also considered. This fails because negative cosine similarities would produce negative percentages.

**Why softmax:**

Softmax converts a set of real numbers into a probability distribution that sums to 100%. It amplifies the differences between scores naturally - the highest score gets a larger share - while keeping all values positive. The temperature parameter controls how spread out the distribution is.

---

## 4. Cepstral Mean Normalisation (CMN)

**Decision:** Subtract the per-feature mean across time from all MFCC frames before pooling.

**Alternatives considered:**

Using raw MFCCs without any normalisation would mean that microphone quality, room acoustics, and background noise all introduce a consistent offset across frames. Two people with nearly identical voices but different microphones would get different embeddings.

Global standardisation (subtracting a fixed mean from a training corpus) was not possible here because there is no training corpus, and it would require recomputing statistics every time new reference vocalists are added.

Per-utterance variance normalisation (also dividing by the standard deviation) was considered but not implemented. It can over-normalise quiet or expressive passages where variance is naturally low.

**Why CMN:**

The bias introduced by recording conditions is consistent across all frames within a single recording. Subtracting the mean removes it exactly. The result is an embedding that represents vocal character rather than microphone or room character.

---

## 5. Three-Layer Silence Detection

**Decision:** Implement three separate silence checks at different stages of the pipeline rather than a single check.

The three layers are:
1. Raw RMS gate - reject if RMS energy before normalisation is below 0.01
2. Post-trim duration check - reject if less than 2 seconds remains after silence trimming
3. Voiced frame filtering - discard frames below (mean RMS minus one standard deviation), reject if fewer than 10 voiced frames remain

**Alternatives considered:**

A single duration check (rejecting recordings under 2 seconds) would miss a 5-second recording of someone breathing quietly. It would pass the check but produce a meaningless embedding.

A single energy check would miss recordings with high peak energy from a single cough or noise at the start, but no actual singing.

WebRTC VAD (Voice Activity Detection) is a dedicated algorithm for identifying voiced segments. It would be more accurate but adds a dependency and complexity for marginal gain at this scale.

**Why three layers:**

Each layer catches a different failure mode. Layer 1 catches completely silent recordings before normalisation amplifies them. Layer 2 catches recordings that are mostly silence with a brief sound. Layer 3 catches recordings with background noise but no singing, and also filters out unvoiced frames from the embedding computation to improve accuracy.

Each failure mode also gets its own specific error message, which is better for users than a generic failure.

---

## 6. Precomputing Reference Embeddings at Startup

**Decision:** Extract embeddings and spectrograms from all 10 reference WAV files once when Flask starts, and keep them in memory for the lifetime of the process.

**Alternatives considered:**

Computing reference embeddings on every request would add several seconds of latency to each comparison and waste CPU on identical work.

Caching embeddings to disk as `.npy` files and loading them at startup is a valid approach and was identified as a future improvement for reducing cold start time.

Lazy loading (computing each reference embedding the first time it is needed) would mean the first user after a cold start gets the worst possible experience.

**Why precompute at startup:**

The reference embeddings never change between deployments. Computing them once at startup costs around 20 seconds on cold start but makes every subsequent request take milliseconds. The trade-off is clearly worth it.

---

## 7. Mean Pooling over Voiced Frames

**Decision:** Aggregate the per-frame feature matrix into a single embedding vector by averaging across all voiced frames.

**Alternatives considered:**

Max pooling takes the maximum value in each feature dimension across frames. This was rejected because it captures the most extreme moment in the recording rather than the overall character, which is what we want.

Using only the first voiced frame was rejected because a single 32ms window is far too little data to characterise a voice.

Padding or truncating to a fixed number of frames and flattening was rejected because it is sensitive to recording length and requires choosing an arbitrary fixed length.

Attention pooling (using a neural network to weight frames by importance) would be a stronger approach but requires training data and a model.

**Why mean pooling:**

Mean pooling is robust to recording length - a 2-second and a 10-second recording of the same voice produce similar embeddings. It is computationally trivial and well-understood. By restricting pooling to voiced frames only, the mean reflects actual singing rather than silence.

---

## 8. Adding F0 Pitch Features Alongside MFCCs

**Decision:** Extend the embedding from 120 to 123 dimensions by adding log-scaled fundamental frequency (F0) and its delta and delta-delta features, extracted using the YIN algorithm.

**Alternatives considered:**

The initial version used 120-dimensional MFCC embeddings only. The 10/10 self-match sanity check passed, confirming the pipeline was technically correct. However, real user testing showed most recordings clustering around the same 2 or 3 vocalists regardless of who was singing.

Chroma features capture the musical notes being sung regardless of octave. They are more useful for melody matching than voice type matching.

Spectral contrast captures the difference between spectral peaks and valleys, which is useful for distinguishing breathy from powerful voices. It adds less discriminative power than pitch for distinguishing vocal types.

Pretrained embeddings would capture pitch implicitly along with many other features, but were not used for reasons described in Decision 1.

**Why F0:**

MFCCs are pitch-independent by design. This is a strength for speaker recognition (you want to match a voice regardless of what note is being sung) but a weakness for vocalist type matching. A bass and a soprano singing the same phrase will have similar MFCCs if their vocal tract shapes are similar. F0 directly captures where in the frequency range a vocalist naturally sings, which is the primary distinguishing characteristic between voice types like Bass, Tenor, Mezzo, and Soprano.

Converting to log scale with `log1p` means the difference between 100 Hz and 200 Hz (one octave) is treated the same as 200 Hz to 400 Hz (also one octave), which matches how humans perceive pitch intervals.

A frame-count alignment step was also required because YIN and the STFT use slightly different padding, producing arrays that differ by one frame in some cases.

**Trade-offs:**

YIN pitch extraction adds roughly 20% to processing time per request. The embedding dimension increase from 120 to 123 has negligible impact on similarity computation. With more time, the relative weight of pitch features versus MFCC features could be tuned by scaling the F0 rows before stacking - currently all 123 dimensions are weighted equally in the cosine similarity.