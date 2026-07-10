import { useState, useEffect } from 'react'
import { Database, Download, RefreshCw, CheckCircle } from 'lucide-react'

export default function TrainingDataPanel() {
  const [stats,    setStats]    = useState(null)
  const [recent,   setRecent]   = useState([])
  const [loading,  setLoading]  = useState(false)
  const [dlStatus, setDlStatus] = useState(null)   // null | 'loading' | 'done' | 'error'

  async function refresh() {
    try {
      const [s, r] = await Promise.all([
        fetch('/api/training-data/stats').then(x => x.json()),
        fetch('/api/training-data?limit=5').then(x => x.json()),
      ])
      setStats(s)
      setRecent(r.rows ?? [])
    } catch {}
  }

  useEffect(() => { refresh() }, [])

  async function downloadCSV() {
    setDlStatus('loading')
    try {
      const res  = await fetch('/api/training-data?limit=50000')
      const data = await res.json()
      const rows = data.rows

      if (!rows || rows.length === 0) {
        setDlStatus('error')
        setTimeout(() => setDlStatus(null), 3000)
        return
      }

      // Build CSV
      const exclude = ['features_json']
      const headers = Object.keys(rows[0]).filter(k => !exclude.includes(k))
      const csvRows = [
        headers.join(','),
        ...rows.map(r =>
          headers.map(h => {
            const v = r[h]
            // Quote strings that contain commas or quotes
            if (typeof v === 'string' && (v.includes(',') || v.includes('"'))) {
              return `"${v.replace(/"/g, '""')}"`
            }
            return v ?? ''
          }).join(',')
        ),
      ]

      const blob = new Blob([csvRows.join('\n')], { type: 'text/csv;charset=utf-8;' })
      const url  = URL.createObjectURL(blob)
      const a    = document.createElement('a')
      a.href     = url
      a.download = `nyc_predictions_${new Date().toISOString().slice(0,10)}.csv`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)

      setDlStatus('done')
      setTimeout(() => setDlStatus(null), 3000)
    } catch {
      setDlStatus('error')
      setTimeout(() => setDlStatus(null), 3000)
    }
  }

  const total = stats?.total ?? 0

  return (
    <div className="rounded-2xl border border-white/8 bg-white/[0.02] overflow-hidden">
      <div className="flex items-center justify-between px-5 py-3.5 border-b border-white/6">
        <div className="flex items-center gap-2">
          <Database size={13} className="text-white/40" />
          <span className="text-white/70 font-semibold text-sm">Training Data</span>
          {total > 0 && (
            <span className="text-xs bg-white/8 border border-white/10 text-white/50
                            rounded-full px-2 py-0.5 font-medium">
              {total.toLocaleString()} rows
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={refresh}
            className="p-1.5 rounded-lg hover:bg-white/8 text-white/25 hover:text-white/50 transition-colors"
          >
            <RefreshCw size={12} />
          </button>
          <button
            onClick={downloadCSV}
            disabled={total === 0 || dlStatus === 'loading'}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold
                       border transition-all ${
              dlStatus === 'done'
                ? 'bg-emerald-500/15 border-emerald-500/30 text-emerald-300'
                : dlStatus === 'error'
                ? 'bg-red-500/15 border-red-500/30 text-red-300'
                : total === 0
                ? 'bg-white/4 border-white/8 text-white/20 cursor-not-allowed'
                : 'bg-white/6 border-white/12 text-white/60 hover:bg-white/10 hover:text-white/80'
            }`}
          >
            {dlStatus === 'loading' ? (
              <RefreshCw size={11} className="animate-spin" />
            ) : dlStatus === 'done' ? (
              <CheckCircle size={11} />
            ) : (
              <Download size={11} />
            )}
            {dlStatus === 'loading' ? 'Preparing…' : dlStatus === 'done' ? 'Saved!' : 'Download CSV'}
          </button>
        </div>
      </div>

      <div className="p-4 space-y-4">

        {/* stat row */}
        {stats && (
          <div className="grid grid-cols-3 gap-2">
            <StoreNum label="Avg price" value={stats.avg_price != null ? `$${Math.round(stats.avg_price)}` : '—'} />
            <StoreNum label="Min price" value={stats.min_price != null ? `$${Math.round(stats.min_price)}` : '—'} />
            <StoreNum label="Max price" value={stats.max_price != null ? `$${Math.round(stats.max_price)}` : '—'} />
          </div>
        )}

        {/* recent rows preview */}
        {recent.length > 0 ? (
          <div className="rounded-xl border border-white/6 overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-white/6 bg-white/[0.02]">
                    {['Borough', 'Room type', 'Guests', 'Price', 'Cache', 'Time'].map(h => (
                      <th key={h} className="px-3 py-2 text-left text-white/30 font-semibold">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {recent.map((row, i) => (
                    <tr key={i} className="border-b border-white/[0.04] last:border-0 hover:bg-white/[0.02]">
                      <td className="px-3 py-2 text-white/70">{row.borough}</td>
                      <td className="px-3 py-2 text-white/50">{row.room_type?.replace('/apt', '')}</td>
                      <td className="px-3 py-2 text-white/50">{row.accommodates}</td>
                      <td className="px-3 py-2 text-white font-bold">${Math.round(row.predicted_price)}</td>
                      <td className="px-3 py-2">
                        <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
                          row.cache_hit
                            ? 'bg-emerald-500/15 text-emerald-400'
                            : 'bg-white/6 text-white/30'
                        }`}>
                          {row.cache_hit ? 'hit' : 'miss'}
                        </span>
                      </td>
                      <td className="px-3 py-2 text-white/30 font-mono">
                        {row.timestamp?.slice(11, 19)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          <div className="text-center py-6 text-white/20 text-sm">
            Make a prediction — it will appear here
          </div>
        )}

        <p className="text-white/20 text-[11px] text-center">
          All predictions stored → Download CSV → use for nightly retraining
        </p>
      </div>
    </div>
  )
}

function StoreNum({ label, value }) {
  return (
    <div className="rounded-lg bg-white/[0.025] border border-white/6 p-2.5 text-center">
      <p className="text-white text-base font-bold leading-none mb-0.5">{value}</p>
      <p className="text-white/30 text-[11px]">{label}</p>
    </div>
  )
}
