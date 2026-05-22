import { useRef, useState } from 'react'
import { compareVoice } from '../api/voiceApi'
import ErrorMessage from './ErrorMessage'
import ResultsView from './ResultsView'
import WaveformDisplay from './WaveformDisplay'

const MIN_DURATION = 2
const MAX_DURATION = 15

function formatDuration(seconds) {
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return `${m}:${String(s).padStart(2, '0')}`
}

export default function Recorder() {
  const [recordingState, setRecordingState] = useState('idle')
  const [duration, setDuration] = useState(0)
  const [error, setError] = useState(null)
  const [results, setResults] = useState(null)
  const [analyserNode, setAnalyserNode] = useState(null)

  const mediaRecorderRef = useRef(null)
  const audioChunksRef = useRef([])
  const streamRef = useRef(null)
  const timerRef = useRef(null)
  const audioContextRef = useRef(null)
  const maxDurationTimerRef = useRef(null)
  // Track duration in a ref so handleRecordingStop can read the current value
  const durationRef = useRef(0)

  async function startRecording() {
    setError(null)
    setResults(null)
    setDuration(0)
    durationRef.current = 0

    let stream
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    } catch {
      setError('Microphone access is required. Please allow microphone access and try again.')
      return
    }

    streamRef.current = stream

    const audioCtx = new AudioContext()
    audioContextRef.current = audioCtx
    const source = audioCtx.createMediaStreamSource(stream)
    const analyser = audioCtx.createAnalyser()
    analyser.fftSize = 2048
    source.connect(analyser)
    setAnalyserNode(analyser)

    const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? 'audio/webm;codecs=opus'
      : MediaRecorder.isTypeSupported('audio/webm')
      ? 'audio/webm'
      : ''

    const recorder = mimeType
      ? new MediaRecorder(stream, { mimeType })
      : new MediaRecorder(stream)

    mediaRecorderRef.current = recorder
    audioChunksRef.current = []

    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunksRef.current.push(e.data)
    }

    recorder.onstop = handleRecordingStop

    recorder.start(100)

    timerRef.current = setInterval(() => {
      durationRef.current += 1
      setDuration((d) => d + 1)
    }, 1000)

    maxDurationTimerRef.current = setTimeout(stopRecording, MAX_DURATION * 1000)

    setRecordingState('recording')
  }

  function stopRecording() {
    if (recordingState !== 'recording') return
    clearInterval(timerRef.current)
    clearTimeout(maxDurationTimerRef.current)
    streamRef.current?.getTracks().forEach((t) => t.stop())
    mediaRecorderRef.current?.stop()
  }

  async function handleRecordingStop() {
    if (durationRef.current < MIN_DURATION) {
      setError('Recording too short. Please record for at least 2 seconds.')
      setRecordingState('idle')
      setDuration(0)
      durationRef.current = 0
      return
    }

    setRecordingState('processing')

    const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' })

    try {
      const data = await compareVoice(blob)
      setResults(data)
      setRecordingState('done')
    } catch (err) {
      if (err.message) {
        setError(err.message)
      } else if (err.error === 'recording_too_short') {
        setError('Recording is too short. Please record for at least 2 seconds.')
      } else if (err.error === 'invalid_audio') {
        setError('Could not process your recording. Please try again.')
      } else {
        setError('Something went wrong. Please try again.')
      }
      setRecordingState('idle')
    }
  }

  function resetRecording() {
    setRecordingState('idle')
    setResults(null)
    setError(null)
    setDuration(0)
    durationRef.current = 0
    setAnalyserNode(null)
  }

  if (recordingState === 'idle') {
    return (
      <div className="flex flex-col items-center gap-6">
        {error && (
          <div className="w-full">
            <ErrorMessage message={error} />
          </div>
        )}
        <button
          onClick={startRecording}
          className="flex h-32 w-32 flex-col items-center justify-center rounded-full bg-black text-white hover:bg-gray-800 transition-colors"
        >
          <svg className="h-10 w-10" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z" />
            <path d="M19 10v2a7 7 0 01-14 0v-2H3v2a9 9 0 008 8.94V23h2v-2.06A9 9 0 0021 12v-2h-2z" />
          </svg>
          <span className="mt-1 text-sm font-semibold">Record</span>
        </button>
        <p className="text-sm text-gray-500">Record 2–15 seconds of your voice singing</p>
      </div>
    )
  }

  if (recordingState === 'recording') {
    return (
      <div className="flex flex-col items-center gap-4">
        <WaveformDisplay isRecording={true} analyserNode={analyserNode} />

        <div className="text-2xl font-mono text-gray-900">
          {formatDuration(duration)}
        </div>

        <button
          onClick={stopRecording}
          className="rounded-full border-2 border-red-500 text-red-500 hover:bg-red-50 px-6 py-2 text-sm font-medium transition-colors"
        >
          Stop
        </button>

        <p className="text-sm text-gray-500">Recording… (max {MAX_DURATION}s)</p>
      </div>
    )
  }

  if (recordingState === 'processing') {
    return (
      <div className="flex flex-col items-center gap-4 py-12">
        <div className="h-12 w-12 animate-spin rounded-full border-4 border-black border-t-transparent" />
        <p className="text-gray-600">Analysing your voice…</p>
      </div>
    )
  }

  // done
  return (
    <div className="flex flex-col items-center gap-6 w-full">
      <ResultsView results={results} />
      <button
        onClick={resetRecording}
        className="border border-gray-300 text-gray-600 hover:bg-gray-50 rounded-lg px-4 py-2 text-sm transition-colors"
      >
        Record Again
      </button>
    </div>
  )
}
