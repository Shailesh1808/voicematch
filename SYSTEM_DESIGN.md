# VoiceMatch — System Design Document

## 1. What the system does
A user visits the web app, records themselves singing for 2–15 seconds,
and receives a ranked list showing how similar their voice is to each of
10 reference vocalists. The comparison is based on audio feature analysis —
we are comparing the character of the voice, not what notes were sung.

## 2. Architecture Overview

Browser (React)
  [Record] → [Waveform] → [Upload] → [Results View]
      │
      │  POST /api/compare (multipart/form-data)
      ▼
Flask Backend
  ├── /api/compare
  ├── /api/health
  └── /api/spectrogram/<vocalist_id>
      │
      ▼
  Audio Processor
    1. Convert to WAV (ffmpeg)
    2. Load & normalize audio (librosa)
    3. Extract MFCC embedding
    4. Cosine similarity vs reference profiles
    5. Rank & normalize scores
      │
      ▼
  Reference Profiles (in-memory dict)
  Precomputed at startup from reference_data/*.wav

## 3. Data Flow

At server startup (once):
1. Flask loads all 10 .wav files from reference_data/
2. Extracts an MFCC embedding for each
3. Generates a mel spectrogram PNG for each
4. Stores embeddings and spectrograms in memory — keyed by vocalist ID
5. Server is ready to accept requests

When a user records and submits:
1. Browser records audio using MediaRecorder API
2. On stop, sends audio blob to POST /api/compare
3. Flask receives raw audio bytes
4. ffmpeg converts to standard 16kHz mono WAV
5. librosa extracts MFCC embedding using the same pipeline as references
6. Cosine similarity computed against all 10 reference embeddings
7. Scores normalized with softmax → percentages
8. Mel spectrogram generated for user recording
9. Flask returns ranked JSON + user spectrogram as base64 PNG
10. React renders results

## 4. API Contract

### POST /api/compare
Request:
  Content-Type: multipart/form-data
  Body: audio file (any browser-native format)

Success 200:
{
  "results": [
    {
      "rank": 1,
      "name": "Luna",
      "style": "Alto, Soul",
      "similarity": 0.847,
      "percentage": 24.3
    },
    ...all 10 vocalists ranked
  ],
  "top_match": "Luna",
  "user_spectrogram": "<base64-encoded PNG>"
}

Error 400 — too short:
{ "error": "recording_too_short", "message": "Recording must be at least 2 seconds" }

Error 400 — bad audio:
{ "error": "invalid_audio", "message": "Could not process audio file" }

Error 500:
{ "error": "server_error", "message": "Something went wrong. Please try again." }

### GET /api/health
{ "status": "ok", "references_loaded": 10 }

### GET /api/spectrogram/<vocalist_id>
{ "image": "<base64-encoded PNG>" }

## 5. Folder Structure

voicematch/
├── CLAUDE.md
├── DECISIONS.md
├── README.md
├── SYSTEM_DESIGN.md
├── AUDIO_PIPELINE.md
├── .gitignore
├── reference_data/            # NOT committed to git
│   ├── sample_1.wav
│   └── ...
├── backend/
│   ├── app.py
│   ├── audio_processor.py
│   ├── similarity.py
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── components/
│       │   ├── Recorder.jsx
│       │   ├── WaveformDisplay.jsx
│       │   ├── ResultsView.jsx
│       │   ├── SpectrogramView.jsx
│       │   └── ErrorMessage.jsx
│       └── api/
│           └── voiceApi.js
└── tests/
    ├── test_audio_processor.py
    └── test_similarity.py

## 6. Key Technical Boundaries

Backend is responsible for:
- All audio processing
- Format conversion (any browser format → WAV)
- Keeping reference profiles in memory
- Returning consistent ranked normalized scores

Frontend is responsible for:
- Recording and enforcing duration limits (2s min, 15s max)
- Mic permission denial error — never reaches backend
- Sending audio as a blob, displaying what it receives
- No audio math in the browser

Neither does:
- No database
- No user accounts
- No persistence

## 7. Error Handling Strategy

| Scenario                        | Where caught | What happens                              |
|---------------------------------|--------------|-------------------------------------------|
| User denies mic access          | Frontend     | Show mic required message                 |
| Recording < 2 seconds           | Frontend     | Show error before upload                  |
| Browser sends unsupported format| Backend      | ffmpeg converts silently or returns error |
| Reference file missing          | Backend      | Log warning, skip that vocalist           |
| All reference files missing     | Backend      | Exit with clear error on startup          |
| Network failure during upload   | Frontend     | Show connection error message             |
| Corrupt audio bytes             | Backend      | Return invalid_audio 400                  |
| Too much silence in recording   | Backend      | Return recording_too_short 400            |

## 8. What is deliberately out of scope
- User authentication
- Saving results
- Comparing against more than 10 vocalists
- Scientific accuracy claims — scores are normalized similarity values for UI only