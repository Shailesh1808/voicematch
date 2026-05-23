# VoiceMatch

VoiceMatch is a web app where you record yourself singing and find out which of 10 reference vocalists your voice is most similar to. It uses MFCC features combined with pitch (F0) analysis and cosine similarity to rank how closely your vocal character matches each reference.

**Live demo:** https://voicematch-mu.vercel.app

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, Vite, Tailwind CSS |
| Backend | Python 3.11, Flask, flask-cors |
| Audio processing | librosa, soundfile, ffmpeg |
| Numerical computation | numpy, scipy |
| Deployment (Frontend) | Vercel |
| Deployment (Backend) | Railway (Dockerfile) |

---

## Project Structure

```
voicematch/
├── backend/
│   ├── app.py                  # Flask application and API endpoints
│   ├── audio_processor.py      # MFCC + F0 pipeline and spectrogram generation
│   ├── similarity.py           # Cosine similarity, softmax, ranking
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── reference_data/         # 10 reference vocalist WAV files
│   └── DECISIONS.md            # Design decision log
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── api/voiceApi.js     # API calls to Flask backend
│   │   └── components/
│   │       ├── Recorder.jsx
│   │       ├── ResultsView.jsx
│   │       ├── WaveformDisplay.jsx
│   │       ├── SpectrogramView.jsx
│   │       └── ErrorMessage.jsx
│   ├── package.json
│   └── vite.config.js
└── tests/
    ├── test_audio_processor.py
    └── test_similarity.py
```

---

## Local Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- ffmpeg installed and available in PATH

**Install ffmpeg:**
- macOS: `brew install ffmpeg`
- Ubuntu: `sudo apt install ffmpeg`
- Windows: download from https://ffmpeg.org and add to PATH

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:

```
FLASK_PORT=5000
FLASK_ENV=development
REFERENCE_DATA_DIR=./reference_data
```

Run the server:

```bash
python app.py
```

The API will be available at `http://localhost:5000`.

### Frontend

```bash
cd frontend
npm install
```

Create a `.env.local` file:

```
VITE_API_URL=http://localhost:5000
```

Run the dev server:

```bash
npm run dev
```

The app will be available at `http://localhost:5173`.

### Running Tests

```bash
cd backend
pytest tests/ -v
```

31 tests covering the audio pipeline and similarity logic.

---

## API Reference

### GET /api/health

Returns server status and number of loaded reference profiles.

**Response:**
```json
{
  "status": "ok",
  "references_loaded": 10
}
```

### POST /api/compare

Accepts a voice recording and returns ranked similarity results.

**Request:** `multipart/form-data` with a field named `audio` containing the recording file.

**Response (200):**
```json
{
  "results": [
    {
      "rank": 1,
      "id": "sample_3",
      "name": "Echo",
      "style": "Soprano, Pop",
      "similarity": 0.8724,
      "percentage": 18.3
    }
  ],
  "top_match": "Echo",
  "user_spectrogram": "<base64 PNG>"
}
```

**Error responses:**
- `400 recording_too_short` - recording under 2 seconds or too much silence
- `400 recording_too_long` - recording exceeds 15 seconds
- `400 invalid_audio` - file could not be decoded
- `500 server_error` - unexpected error

### GET /api/spectrogram/<vocalist_id>

Returns the precomputed mel spectrogram for a reference vocalist.

**Response (200):**
```json
{
  "image": "<base64 PNG>"
}
```

---

## How the Audio Pipeline Works

When you submit a recording, VoiceMatch runs the following steps:

1. **Format conversion.** ffmpeg converts the browser recording (WebM or OGG) to a 16 kHz mono WAV. This is necessary because browsers produce compressed formats that librosa cannot reliably decode.

2. **Validation.** The recording is rejected if it is shorter than 2 seconds, longer than 15 seconds, or if the RMS energy is below a minimum threshold. The energy check catches silent recordings before normalisation can amplify them into something that looks like real audio.

3. **Normalisation.** Peak amplitude normalisation scales the audio so the loudest sample is 1.0. This ensures quiet and loud singers are treated equally - the comparison is about vocal character, not volume.

4. **Silence trimming.** Leading and trailing silence is removed. A second duration check then confirms at least 2 seconds of actual audio remains after trimming.

5. **MFCC extraction.** 40 Mel-Frequency Cepstral Coefficients are extracted per frame using a 2048-sample FFT window and 512-sample hop length. MFCCs capture the shape of the vocal tract - what makes your voice sound like you - independently of the note you are singing.

6. **Delta features.** First-order (delta) and second-order (delta-delta) derivatives are computed and stacked with the MFCCs, giving a (120, T) feature matrix. These add temporal dynamics to the representation - not just what the voice sounds like at a moment, but how it moves over time.

7. **Pitch (F0) features.** The fundamental frequency is extracted per frame using the YIN algorithm with a range of C2 to C7, covering the full singing voice range. F0 values are converted to a log scale and stacked with their own delta and delta-delta, adding 3 more rows to produce a (123, T) matrix. This step was added after initial testing showed most users clustering around the same top results - MFCCs alone do not distinguish vocalists whose primary difference is pitch range.

8. **Cepstral mean normalisation.** The per-feature mean is subtracted across time from the full (123, T) matrix. This removes the systematic bias introduced by different microphones and room acoustics.

9. **Voiced frame filtering.** Frames with energy below one standard deviation of the mean are discarded, keeping only frames where you are actively singing.

10. **Mean pooling.** The feature matrix is averaged across voiced frames, producing a single 123-dimensional vector.

11. **L2 normalisation.** The vector is scaled to unit length, which is required for cosine similarity to work correctly.

The resulting embedding is compared against 10 precomputed reference embeddings using cosine similarity. Results are ranked and converted to display percentages using softmax normalisation.

### Sanity Check

Each reference WAV file was fed through the pipeline as a simulated user recording. All 10 files ranked their own vocalist at position 1, confirming the embeddings are discriminative. This is the self-match test described in the challenge brief.

---

## What I Tried and What I Would Improve

**What worked well**

MFCCs with delta and delta-delta features produced clearly discriminative embeddings. The 10/10 self-match sanity check passed cleanly. The three-layer silence detection caught every failure mode reliably in testing, and precomputing reference embeddings at startup kept response times fast.

**What I added during development**

F0 pitch features were added after observing that users were clustering around the same top results. MFCCs capture timbre but not pitch, so vocalists whose main difference is their pitch range were underrepresented in the rankings.

**What I would improve with more time**

The biggest improvement would be replacing MFCCs with a pretrained audio embedding model like CLAP or openl3. These models are trained on large singing datasets and capture musical characteristics that MFCCs miss entirely. The trade-off is a much larger Docker image and slower cold starts on Railway.

I would also replace the Flask development server with Gunicorn for production use, and cache reference embeddings to disk so cold starts do not require reprocessing all 10 WAV files every time the container restarts.

---

## Trade-offs and Known Limitations

**Accuracy ceiling**

Even with pitch features added, the embedding misses vibrato, breath control, dynamics, and higher-level musical style. A pretrained model would improve results significantly.

**Reference sample quality**

The pipeline is correct - the 10/10 self-match test confirms this - but the reference samples vary in duration (5.9s to 18.3s) and recording conditions, which affects how representative each vocalist's embedding is.

**Development server**

The backend runs Flask's built-in development server. For real production traffic this should be replaced with Gunicorn:

```bash
gunicorn -w 2 -b 0.0.0.0:8080 app:app
```

**No persistent storage**

Results are not stored. Each comparison is stateless - no history, no user accounts.

---

## Scaling to 1000 Vocalists

At 10 references, sequential cosine similarity search is instant. At 1000, the search itself is still fast (123-dimensional dot products are cheap), but other things break down.

**Startup time.** Precomputing 1000 embeddings at startup would take several minutes. The fix is to precompute them offline, save to a `.npy` file, and load that file at startup instead of reprocessing audio.

**Storage.** 1000 embeddings x 123 dimensions x 8 bytes is about 1MB - negligible. The bigger problem is the WAV files. At roughly 500KB each, 1000 files is 500MB in the Docker image. These should live in object storage (S3 or similar) and be loaded on demand.

**Search at scale.** At 100,000+ references, brute-force search would be too slow. Approximate nearest-neighbour libraries like FAISS or Annoy would be needed.

**Data management.** At 1000 vocalists, managing files and embeddings by hand is not practical. A proper setup would use a database for metadata, a vector store for embeddings, and an admin interface for adding new vocalists without redeploying.

---

## Deployment

### Backend (Railway)

Deployed as a Docker container. The Dockerfile installs ffmpeg via apt, installs Python dependencies, and runs Flask.

Key environment variables:
- `FLASK_PORT` - port Flask listens on (set to 8080 to match Railway networking)
- `REFERENCE_DATA_DIR` - path to reference WAV files inside the container
- `FLASK_ENV` - set to `production` in deployment

### Frontend (Vercel)

Deployed on Vercel. Set the environment variable `VITE_API_URL` to the Railway backend URL in the Vercel project settings.