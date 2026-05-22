import SpectrogramView from './SpectrogramView'

export default function ResultsView({ results }) {
  const { results: ranked, top_match, user_spectrogram } = results
  const topResult = ranked[0]

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
          <div
            key={r.id}
            className="flex items-center gap-4 py-3 border-b border-gray-100 last:border-0"
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
                  style={{ width: `${r.percentage}%` }}
                />
              </div>
            </div>

            {/* Scores */}
            <div className="shrink-0 text-right">
              <p className="text-sm text-gray-600 w-12">{r.percentage.toFixed(1)}%</p>
              <p className="text-xs text-gray-400 w-16">{r.similarity.toFixed(4)}</p>
            </div>
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
