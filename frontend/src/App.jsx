import Recorder from './components/Recorder'

export default function App() {
  return (
    <div className="w-full min-h-screen bg-white">
      <div className="mx-auto max-w-2xl px-6 py-12">
        <header className="mb-10">
          <h1 className="text-3xl font-bold text-gray-900 tracking-tight">VoiceMatch</h1>
          <p className="text-base text-gray-500 mt-1">Discover which vocalist your voice resembles</p>
        </header>

        <main>
          <Recorder />
        </main>

        <footer className="mt-12 text-center">
          <p className="text-xs text-gray-400">Powered by MFCC audio analysis</p>
        </footer>
      </div>
    </div>
  )
}
