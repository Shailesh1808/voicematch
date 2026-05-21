# CLAUDE.md — VoiceMatch Project Instructions

## What This Project Is
VoiceMatch is a web app where a user records themselves singing (2–15 seconds)
and receives a ranked similarity comparison against 10 reference vocalists.
It uses MFCC-based audio feature extraction and cosine similarity.
No database. No model training. No authentication.

## Tech Stack
- Backend: Python 3.11, Flask
- Audio processing: librosa, soundfile, numpy
- Audio conversion: ffmpeg (system dependency)
- Frontend: React 18, Vite, Tailwind CSS
- Testing: pytest (backend)

## Folder Structure
voicematch/
├── CLAUDE.md
├── DECISIONS.md
├── README.md
├── .gitignore
├── reference_data/        # .wav files — NOT committed to git
├── backend/
│   ├── app.py             # Flask app, routes, startup precomputation
│   ├── audio_processor.py # MFCC extraction, mel spectrogram generation
│   ├── similarity.py      # Cosine similarity, ranking, normalization
│   └── requirements.txt
└── frontend/
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

## Reference Vocalist Data
The 10 reference vocalists are:
| ID         | Name   | Style             |
|------------|--------|-------------------|
| sample_1   | Luna   | Alto, Soul        |
| sample_2   | Atlas  | Baritone, Folk    |
| sample_3   | Echo   | Soprano, Pop      |
| sample_4   | Riven  | Tenor, Rock       |
| sample_5   | Sol    | Mezzo, Jazz       |
| sample_6   | Vesper | Bass, Classical   |
| sample_7   | Coda   | Alto, R&B         |
| sample_8   | Fern   | Soprano, Indie    |
| sample_9   | Birch  | Tenor, Acoustic   |
| sample_10  | Wren   | Mezzo, Country    |

## Audio Pipeline (DO NOT deviate from this)
1. Convert incoming audio to 16kHz mono WAV using ffmpeg
2. Load with librosa.load(sr=16000, mono=True)
3. Peak amplitude normalization
4. Trim leading/trailing silence: librosa.effects.trim(top_db=20)
5. Extract MFCCs: n_mfcc=40, n_fft=2048, hop_length=512
6. Compute delta and delta-delta MFCCs
7. Cepstral mean normalization (subtract per-coefficient mean)
8. Energy-based frame filtering:
   - Compute RMS energy per frame
   - voiced_mask = energy > (mean_energy - 1 std dev)
   - If fewer than 10 voiced frames → raise AudioTooShortError
9. Mean pool over voiced frames only → shape (120,)
10. L2 normalize the final vector

Same pipeline runs for reference files at startup AND user uploads.
Reference embeddings are stored in a module-level dict at startup.
No file reads happen during request handling.

## API Endpoints
POST /api/compare
  - Receives: multipart/form-data with audio file
  - Returns: ranked results + user spectrogram (base64 PNG)

GET /api/health
  - Returns: status + number of references loaded

GET /api/spectrogram/<vocalist_id>
  - Returns: base64 PNG of reference vocalist mel spectrogram

## Error Handling Rules
- Never let an unhandled exception reach the client
- Always return JSON error responses, never HTML error pages
- Error response shape: { "error": "<code>", "message": "<human readable>" }
- Error codes: recording_too_short, invalid_audio, server_error
- Log all errors server-side with full tracebacks
- If a reference file is missing at startup: log a warning and skip it
- If ALL reference files are missing at startup: exit with a clear error message

## Coding Rules
- Every function must have a docstring explaining what it does,
  its parameters, and what it returns
- No magic numbers — use named constants at the top of each file
- Keep functions small and single-purpose
- audio_processor.py handles all librosa calls — nowhere else
- similarity.py handles all numpy/math — nowhere else
- app.py only handles HTTP concerns — no audio math in routes

## Git Commit Rules
Write commits in this format:
  <type>: <short description>

Types:
  feat     — new feature
  fix      — bug fix
  test     — adding or fixing tests
  docs     — documentation only
  refactor — code restructure, no behavior change
  chore    — setup, config, dependencies

Examples:
  feat: add MFCC extraction with delta features
  test: add unit tests for cosine similarity ranking
  docs: add audio pipeline section to README
  chore: add requirements.txt and .gitignore

DO NOT make commits that say "fix", "wip", "update", or "changes".
Each commit should be self-contained and tell the story of what was built.

## What NOT To Do
- Do not use a database of any kind
- Do not install packages not listed in requirements.txt without asking
- Do not put audio math in app.py
- Do not put HTTP logic in audio_processor.py or similarity.py
- Do not commit reference_data/*.wav files
- Do not commit __pycache__, .env, node_modules, or dist/
- Do not use print() for logging — use Python's logging module
- Do not hardcode file paths — use pathlib.Path relative to the file

## Environment Variables
REFERENCE_DATA_DIR  — path to reference_data/ folder
                      default: ../reference_data relative to backend/
FLASK_ENV           — development or production
FLASK_PORT          — default 5000

## Testing Requirements
- Tests live in tests/ at the project root
- Run with: pytest tests/ from the project root
- test_audio_processor.py must cover:
    - MFCC extraction returns correct shape (120,)
    - Silence trimming works
    - Energy frame filtering excludes silent frames
    - Duration validation raises error for clips under 2 seconds
- test_similarity.py must cover:
    - Cosine similarity of identical vectors = 1.0
    - Cosine similarity ranks correctly
    - Softmax normalization sums to 100%
    - Top match is correctly identified

## Frontend Dev Server
cd frontend && npm run dev
Runs on http://localhost:5173
Proxies /api/* to http://localhost:5000

## Backend Dev Server
cd backend && flask run
Runs on http://localhost:5000

## First Thing Every Session
Before writing any code, read this file completely.
Then read the relevant section of the design docs:
  - System Design: SYSTEM_DESIGN.md
  - Audio Pipeline: AUDIO_PIPELINE.md
Then confirm the task before starting.