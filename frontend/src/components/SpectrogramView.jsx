import { useEffect, useState } from 'react'
import { getSpectrogram } from '../api/voiceApi'

export default function SpectrogramView({ userSpectrogram, topMatchId }) {
  const [referenceImage, setReferenceImage] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    getSpectrogram(topMatchId)
      .then((data) => setReferenceImage(data.image))
      .catch(() => setReferenceImage(null))
      .finally(() => setLoading(false))
  }, [topMatchId])

  return (
    <div className="mt-8 pt-8 border-t border-gray-100">
      <div className="flex flex-col gap-6">
        <div>
          <p className="text-xs uppercase tracking-widest text-gray-400 mb-2">Your Voice</p>
          <div className="rounded-lg overflow-hidden border border-gray-100">
            <img
              src={`data:image/png;base64,${userSpectrogram}`}
              alt="Your voice spectrogram"
              className="w-full"
            />
          </div>
        </div>

        <div>
          <p className="text-xs uppercase tracking-widest text-gray-400 mb-2">Closest Match</p>
          <div className="rounded-lg overflow-hidden border border-gray-100">
            {loading ? (
              <div className="flex h-32 items-center justify-center bg-gray-50">
                <div className="h-6 w-6 animate-spin rounded-full border-2 border-black border-t-transparent" />
              </div>
            ) : referenceImage ? (
              <img
                src={`data:image/png;base64,${referenceImage}`}
                alt="Reference vocalist spectrogram"
                className="w-full"
              />
            ) : (
              <p className="text-sm text-gray-400 p-4">Could not load spectrogram.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
