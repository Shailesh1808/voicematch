import { useState } from 'react'
import SpectrogramView from './SpectrogramView'
import { getSpectrogram } from '../api/voiceApi'

export default function ResultsView({ results }) {
  const { results: ranked, top_match, user_spectrogram } = results
  const topResult = ranked[0]
  const maxPercentage = ranked[0].percentage

  const [openIds, setOpenIds] = useState(new Set())
  const [spectrogramCache, setSpectrogramCache] = useState({})

  async function toggleSpectrogram(id) {
    setOpenIds(prev => {
      const next = new Set(prev)
      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }
      return next
    })

    if (!spectrogramCache[id]) {
      try {
        const data = await getSpectrogram(id)
        setSpectrogramCache(prev => ({ ...prev, [id]: data.image }))
      } catch {
        setSpectrogramCache(prev => ({ ...prev, [id]: null }))
      }
    }
  }

  return (
    <div className="w-full space-y-6">
      {/* Top match header */}
      <div className="rounded-xl bg-gray-50 border border-gray-100 p-6 text-center">
        <p className="text-xs uppercase tracking-widest text-gray-400">Your closest match</p>
        <p className="mt-2 text-4xl font-bold text-gray-900">{top_match}</p>
        <p className="mt-1 text-base text-gray-500">{topResult.style}</p>
      </div>

      {/* Ranked list */}
      <div>
        {ranked.map((r) => (
          <div key={r.id} className="border-b border-gray-100 last:border-0">
            {/* The row itself */}
            <div
              className="flex items-center gap-4 py-3 cursor-pointer"
              onClick={() => toggleSpectrogram(r.id)}
            >
              {/* Rank */}
              <span
                className={`text-sm w-6 shrink-0 ${
                  r.rank === 1 ? 'text-black font-medium' : 'text-gray-400'
                }`}
              >
                {r.rank}
              </span>

              {/* Name + style */}
              <div className="min-w-0 flex-1">
                <div className="flex items-baseline gap-2">
                  <span
                    className={`text-sm ${
                      r.rank === 1 ? 'font-semibold text-gray-900' : 'font-medium text-gray-900'
                    }`}
                  >
                    {r.name}
                  </span>
                  <span className="truncate text-xs text-gray-400">{r.style}</span>
                </div>

                {/* Progress bar */}
                <div className="mt-1.5 w-full bg-gray-100 rounded-full overflow-hidden" style={{ height: r.rank === 1 ? '8px' : '6px' }}>
                  <div
                    className="bg-black rounded-full h-full transition-all"
                    style={{ width: `${(r.percentage / maxPercentage) * 100}%` }}
                  />
                </div>
              </div>

              {/* Scores */}
              <div className="shrink-0 text-right">
                <p className="text-sm text-gray-600 w-12">{r.percentage.toFixed(1)}%</p>
                <p className="text-xs text-gray-400 w-16">{Math.max(0, r.similarity).toFixed(4)}</p>
              </div>

              {/* Chevron */}
              <button
                className="ml-2 shrink-0 text-gray-400 transition-transform duration-200"
                style={{ transform: openIds.has(r.id) ? 'rotate(180deg)' : 'rotate(0deg)' }}
                aria-label="Toggle spectrogram"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
                  stroke="currentColor" strokeWidth="2">
                  <polyline points="6 9 12 15 18 9" />
                </svg>
              </button>
            </div>

            {/* Expanded spectrogram */}
            {openIds.has(r.id) && (
              <div className="pb-4 px-2">
                {!spectrogramCache[r.id] && spectrogramCache[r.id] !== null ? (
                  <div className="flex justify-center py-6">
                    <div className="h-5 w-5 animate-spin rounded-full border-2
                      border-black border-t-transparent" />
                  </div>
                ) : spectrogramCache[r.id] ? (
                  <img
                    src={`data:image/png;base64,${spectrogramCache[r.id]}`}
                    alt={`${r.name} spectrogram`}
                    className="w-full rounded-lg border border-gray-100"
                  />
                ) : (
                  <p className="text-xs text-gray-400 py-2">
                    Could not load spectrogram.
                  </p>
                )}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Spectrograms */}
      {user_spectrogram && (
        <SpectrogramView
          userSpectrogram={user_spectrogram}
          topMatchId={topResult.id}
        />
      )}
    </div>
  )
}
