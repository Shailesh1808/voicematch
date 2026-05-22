# VoiceMatch

VoiceMatch is a web application that records a user's singing voice and ranks it against 10 reference vocalist profiles by similarity. It uses MFCC-based audio embeddings and cosine similarity to produce a ranked breakdown of how closely the user's vocal character matches each reference.

**Live demo:** https://voicematch-mu.vercel.app

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, Vite, Tailwind CSS |
| Backend | Python 3.11, Flask, flask-cors |
| Audio processing | librosa, soundfile, ffmpeg |
| Numerical computation | numpy, scipy |
| Deployment — Frontend | Vercel |
| Deployment — Backend | Railway (Dockerfile) |

---

## Project Structure

```
voicematch/
├── backend/
│   ├── app.py                  # Flask application, API endpoints
│   ├── audio_processor.py      # Full MFCC pipeline and spectrogram generation
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

Set environment variables (or create a `.env` file):

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

### `GET /api/health`

Returns server status and number of loaded reference profiles.

**Response:**
```json
{
  "status": "ok",
  "references_loaded": 10
}
```

### `POST /api/compare`

Accepts a voice recording and returns ranked similarity results.

**Request:** `multipart/form-data` with field `audio` containing the recording file.

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
- `400 recording_too_short` — recording under 2 seconds or too much silence
- `400 invalid_audio` — file could not be decoded
- `500 server_error` — unexpected error

### `GET /api/spectrogram/<vocalist_id>`

Returns the precomputed mel spectrogram for a reference vocalist.

**Response (200):**
```json
{
  "image": "<base64 PNG>"
}
```

---

## How the Audio Pipeline Works

When a user submits a recording, VoiceMatch runs the following steps:

1. **Format conversion** — ffmpeg converts the browser recording (WebM/OGG) to a 16 kHz mono WAV. This is necessary because browsers produce compressed formats that librosa cannot reliably decode.

2. **Validation** — the recording is rejected if it is shorter than 2 seconds, or if the RMS energy is below a minimum threshold (catching silent recordings before normalisation amplifies them).

3. **Normalisation** — peak amplitude normalisation scales the audio so the loudest sample is 1.0. This ensures quiet and loud singers are compared on equal terms.

4. **Silence trimming** — leading and trailing silence is removed. A second duration check ensures at least 2 seconds of actual audio remains.

5. **MFCC extraction** — 40 Mel-Frequency Cepstral Coefficients are extracted per frame using a 2048-sample FFT window and 512-sample hop length. MFCCs capture the shape of the vocal tract — the resonance characteristics that make each voice unique — independently of pitch.

6. **Delta features** — first-order (delta) and second-order (delta-delta) derivatives are computed and stacked with the MFCCs, producing a (120, T) feature matrix. This adds temporal dynamics to the representation.

7. **Cepstral mean normalisation** — the per-feature mean is subtracted across time, removing the systematic bias introduced by microphone quality and room acoustics.

8. **Voiced frame filtering** — frames with energy below one standard deviation of the mean are discarded, keeping only frames where the user is actively singing.

9. **Mean pooling** — the feature matrix is averaged across voiced frames, producing a single 120-dimensional vector.

10. **L2 normalisation** — the vector is scaled to unit length, enabling cosine similarity comparison.

The resulting embedding is compared against 10 precomputed reference embeddings using cosine similarity. Results are ranked and converted to display percentages using softmax normalisation.

---

## Trade-offs and Known Limitations

**MFCC accuracy ceiling**
MFCCs are a strong baseline but do not capture pitch, vibrato, or higher-level musical characteristics. Two singers with similar vocal tract shapes but very different styles may score highly similar. A pretrained audio embedding model (CLAP, VGGish, openl3) would produce significantly better results at the cost of a larger Docker image and slower cold starts.

**Reference sample quality**
The accuracy of the rankings is bounded by the quality and representativeness of the 10 reference WAV files. Short or poorly recorded reference samples will produce unreliable results for all users.

**Development server in production**
The backend currently runs Flask's built-in development server. For production traffic this should be replaced with Gunicorn:
```bash
gunicorn -w 2 -b 0.0.0.0:8080 app:app
```

**No persistent storage**
Results are not stored. Each comparison is stateless — there is no history, user accounts, or analytics.

---

## Scaling to 1000 Vocalists

The current architecture compares a user embedding against all reference embeddings sequentially. At 10 references this is instant. At 1000 references, a naive sequential search would still be fast (~1ms), but there are several other bottlenecks that would need addressing:

**Startup time**
Precomputing 1000 embeddings at startup would take several minutes and exhaust memory on small servers. The fix is to precompute all embeddings offline, serialise them to a `.npy` file, and load the file at startup instead of reprocessing WAV files.

**Memory**
1000 embeddings × 120 dimensions × 8 bytes = ~1MB. This is negligible. The bigger concern is storing 1000 reference WAV files in the Docker image — at ~500KB each, that is ~500MB. The fix is to store WAV files in object storage (S3, R2) and load only on first access.

**Similarity search**
At 1000+ references, brute-force cosine similarity search remains fast because 120-dimensional dot products are cheap. At 100,000+ references, approximate nearest-neighbour search (FAISS, Annoy) would be needed.

**Reference data management**
At 1000 vocalists, managing reference WAV files and embeddings manually is impractical. A proper pipeline would store metadata in a database, embeddings in a vector store, and audio files in object storage, with an admin interface for adding new vocalists without redeploying.

---

## Deployment

### Backend (Railway)

The backend is deployed as a Docker container on Railway. The Dockerfile installs ffmpeg via apt, installs Python dependencies, and runs Flask.

Key environment variables:
- `FLASK_PORT` — port Flask listens on (set to 8080 to match Railway's networking)
- `REFERENCE_DATA_DIR` — path to reference WAV files inside the container
- `FLASK_ENV` — set to `production` in deployment

### Frontend (Vercel)

The frontend is deployed on Vercel. Set the environment variable `VITE_API_URL` to the Railway backend URL in the Vercel project settings.