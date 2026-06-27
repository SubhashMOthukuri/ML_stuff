import { useState } from 'react'
import { FlaskConical, RefreshCw, TrendingUp, TrendingDown, Minus, CheckCircle2, XCircle, AlertTriangle, ChevronUp, ChevronDown } from 'lucide-react'
import { usePolling } from '../hooks/usePolling'

const REC_CONFIG = {
  promote:            { label: 'Promote',          color: '#10b981', bg: 'rgba(16,185,129,0.1)',  border: 'rgba(16,185,129,0.2)',  icon: CheckCircle2 },
  monitor:            { label: 'Monitor',          color: '#f59e0b', bg: 'rgba(245,158,11,0.1)',  border: 'rgba(245,158,11,0.2)',  icon: AlertTriangle },
  reject:             { label: 'Reject',           color: '#e11d48', bg: 'rgba(225,29,72,0.1)',   border: 'rgba(225,29,72,0.2)',   icon: XCircle },
  insufficient_data:  { label: 'Need More Data',   color: '#6b7280', bg: 'rgba(107,114,128,0.1)', border: 'rgba(107,114,128,0.2)', icon: AlertTriangle },
}

const DIST_LABELS = {
  '<-15%':       { label: '< −15%', color: '#e11d48' },
  '-15%_to_-5%': { label: '−15% to −5%', color: '#f97316' },
  '-5%_to_5%':   { label: '−5% to +5%  ✓', color: '#10b981' },
  '5%_to_15%':   { label: '+5% to +15%', color: '#f97316' },
  '>15%':        { label: '> +15%', color: '#e11d48' },
}

export default function ShadowPage() {
  const { data, loading, error, refresh, lastFetch } = usePolling('/api/shadow/stats', 10000)
  const [confirm, setConfirm] = useState(null)   // 'promote' | 'reject' | null
  const [actioning, setActioning] = useState(false)
  const [actionMsg, setActionMsg] = useState(null)

  async function doAction(type) {
    setActioning(true)
    setActionMsg(null)
    try {
      const method = type === 'clear' ? 'DELETE' : 'POST'
      const url    = type === 'clear' ? '/api/shadow/comparisons' : `/api/shadow/${type}`
      const r = await fetch(url, { method })
      const body = await r.json()
      setActionMsg({ ok: r.ok, text: r.ok ? (body.message ?? 'Done') : (body.detail ?? 'Failed') })
      refresh()
    } catch (e) {
      setActionMsg({ ok: false, text: e.message })
    } finally {
      setActioning(false)
      setConfirm(null)
    }
  }

  return (
    <div className="space-y-6">
      {/* header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-white text-xl font-bold tracking-tight flex items-center gap-2">
            <FlaskConical size={20} className="text-violet-400" />
            Shadow Lab
          </h1>
          <p className="text-white/35 text-sm mt-1">
            Challenger runs silently — users always see champion results
          </p>
        </div>
        <button onClick={refresh}
          className="flex items-center gap-2 px-3.5 py-2 rounded-xl border border-white/10
                     bg-white/[0.03] text-white/60 text-sm hover:text-white hover:border-white/20 transition-colors">
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          Refresh
        </button>
      </div>

      {/* action feedback */}
      {actionMsg && (
        <div className={`rounded-xl border px-4 py-3 text-sm ${
          actionMsg.ok
            ? 'border-emerald-500/25 bg-emerald-500/8 text-emerald-300'
            : 'border-red-500/25 bg-red-500/8 text-red-300'
        }`}>
          {actionMsg.text}
        </div>
      )}

      {error && (
        <div className="rounded-xl border border-red-500/25 bg-red-500/8 px-4 py-3 text-red-300 text-sm">
          {error}
        </div>
      )}

      {loading && !data && <Skeleton />}

      {data && (
        <>
          {/* status banner */}
          <div className={`rounded-2xl border p-5 flex items-center justify-between ${
            data.active
              ? 'border-violet-500/20 bg-violet-500/6'
              : 'border-white/8 bg-white/[0.02]'
          }`}>
            <div className="flex items-center gap-3">
              <span className={`w-2.5 h-2.5 rounded-full ${data.active ? 'bg-violet-400 animate-pulse' : 'bg-zinc-600'}`} />
              <span className="text-white font-semibold">
                {data.active ? 'Shadow Active' : 'Shadow Disabled'}
              </span>
              {data.active && (
                <span className="text-white/40 text-sm">
                  · challenger.onnx loaded
                </span>
              )}
            </div>
            {lastFetch && (
              <span className="text-white/30 text-xs">
                Updated {lastFetch.toLocaleTimeString()}
              </span>
            )}
          </div>

          {/* no data message */}
          {!data.active && (
            <div className="rounded-2xl border border-white/8 bg-white/[0.02] p-10 text-center">
              <p className="text-white/50 text-sm">No challenger model loaded.</p>
              <p className="text-white/30 text-xs mt-2 max-w-sm mx-auto leading-relaxed">
                Run <code className="text-violet-400 font-mono">python scripts/retrain.py</code> and
                let the validation gate fail — challenger.onnx will be saved automatically.
                Restart the API to activate shadow mode.
              </p>
            </div>
          )}

          {data.n_comparisons > 0 && (
            <>
              {/* KPI row */}
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                <KPI label="Comparisons" value={data.n_comparisons?.toLocaleString()}
                     sub={`${data.total_all_time?.toLocaleString()} all time`} />
                <KPI label="Champion Mean" value={`$${data.champion_mean_usd?.toFixed(0)}`}
                     sub="current model" />
                <KPI label="Challenger Mean" value={`$${data.challenger_mean_usd?.toFixed(0)}`}
                     sub="new model" />
                <KPI
                  label="Mean Bias"
                  value={`${data.mean_diff_pct > 0 ? '+' : ''}${data.mean_diff_pct?.toFixed(2)}%`}
                  sub={`±${data.std_diff_pct?.toFixed(1)}% std`}
                  accent={Math.abs(data.mean_diff_pct) <= 5 ? '#10b981' : Math.abs(data.mean_diff_pct) <= 15 ? '#f59e0b' : '#e11d48'}
                  icon={data.mean_diff_pct > 0.5 ? TrendingUp : data.mean_diff_pct < -0.5 ? TrendingDown : Minus}
                />
              </div>

              {/* agreement bars */}
              <div className="rounded-2xl border border-white/8 bg-[#0f0f14] p-5 space-y-4">
                <h3 className="text-white text-sm font-semibold">Agreement Rate</h3>
                <Bar label="Within ±5%" value={data.agreement_within_5pct}
                     color="#10b981" threshold={0.8} />
                <Bar label="Within ±10%" value={data.agreement_within_10pct}
                     color="#10b981" threshold={0.9} />
              </div>

              {/* diff distribution */}
              {data.diff_distribution && (
                <div className="rounded-2xl border border-white/8 bg-[#0f0f14] p-5">
                  <h3 className="text-white text-sm font-semibold mb-4">Price Difference Distribution</h3>
                  <div className="space-y-2.5">
                    {Object.entries(DIST_LABELS).map(([key, { label, color }]) => {
                      const pct = (data.diff_distribution[key] ?? 0) * 100
                      return (
                        <div key={key} className="flex items-center gap-3">
                          <span className="text-white/50 text-xs w-[120px] shrink-0">{label}</span>
                          <div className="flex-1 h-5 rounded bg-white/5 overflow-hidden relative">
                            <div
                              className="h-full rounded transition-all duration-500"
                              style={{ width: `${pct}%`, background: color, opacity: 0.7 }}
                            />
                          </div>
                          <span className="text-white/60 text-xs w-10 text-right shrink-0">
                            {pct.toFixed(1)}%
                          </span>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )}

              {/* bottom grid: by-borough + recommendation */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {/* borough breakdown */}
                {data.by_borough && Object.keys(data.by_borough).length > 0 && (
                  <div className="rounded-2xl border border-white/8 bg-[#0f0f14] p-5">
                    <h3 className="text-white text-sm font-semibold mb-4">By Borough</h3>
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="text-white/30 text-xs">
                          <th className="text-left pb-2 font-medium">Borough</th>
                          <th className="text-right pb-2 font-medium">n</th>
                          <th className="text-right pb-2 font-medium">Mean Δ</th>
                          <th className="text-right pb-2 font-medium">Std</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-white/[0.04]">
                        {Object.entries(data.by_borough).map(([b, s]) => (
                          <tr key={b}>
                            <td className="py-2 text-white/70">{b}</td>
                            <td className="py-2 text-right text-white/40">{s.n}</td>
                            <td className="py-2 text-right">
                              <DiffBadge v={s.mean_diff_pct} />
                            </td>
                            <td className="py-2 text-right text-white/40">
                              ±{s.std_diff_pct?.toFixed(1)}%
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}

                {/* recommendation */}
                <div className="rounded-2xl border border-white/8 bg-[#0f0f14] p-5 flex flex-col">
                  <h3 className="text-white text-sm font-semibold mb-4">Recommendation</h3>
                  <RecCard rec={data.recommendation} reason={data.recommendation_reason} />

                  <div className="mt-auto pt-5 space-y-2">
                    {/* confirm dialog */}
                    {confirm && (
                      <div className="rounded-xl border border-white/15 bg-white/5 p-3 text-sm mb-2">
                        <p className="text-white/80 mb-2">
                          {confirm === 'promote'
                            ? 'Promote challenger to champion? This swaps the ONNX files. Restart the API to apply.'
                            : 'Discard challenger? This cannot be undone.'}
                        </p>
                        <div className="flex gap-2">
                          <button
                            onClick={() => doAction(confirm)}
                            disabled={actioning}
                            className={`px-3 py-1.5 rounded-lg text-xs font-semibold text-white transition-colors ${
                              confirm === 'promote' ? 'bg-emerald-600 hover:bg-emerald-500' : 'bg-rose-600 hover:bg-rose-500'
                            }`}>
                            {actioning ? 'Working…' : 'Confirm'}
                          </button>
                          <button onClick={() => setConfirm(null)}
                            className="px-3 py-1.5 rounded-lg text-xs text-white/50 hover:text-white/80">
                            Cancel
                          </button>
                        </div>
                      </div>
                    )}

                    <div className="flex gap-2 flex-wrap">
                      <button
                        onClick={() => setConfirm('promote')}
                        className="flex-1 px-3 py-2 rounded-xl text-xs font-semibold text-white
                                   bg-emerald-600/80 hover:bg-emerald-500 border border-emerald-500/30
                                   transition-colors">
                        Promote Challenger
                      </button>
                      <button
                        onClick={() => setConfirm('reject')}
                        className="flex-1 px-3 py-2 rounded-xl text-xs font-semibold text-white/60
                                   bg-white/5 hover:bg-rose-500/20 hover:text-white border border-white/10
                                   transition-colors">
                        Reject
                      </button>
                      <button
                        onClick={() => doAction('clear')}
                        className="px-3 py-2 rounded-xl text-xs font-semibold text-white/40
                                   bg-white/5 hover:bg-white/10 border border-white/10 transition-colors">
                        Clear Data
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </>
          )}

          {/* disabled + no comparisons */}
          {data.active && data.n_comparisons === 0 && (
            <div className="rounded-2xl border border-white/8 bg-white/[0.02] p-8 text-center">
              <p className="text-white/50 text-sm">Shadow is active but no comparisons yet.</p>
              <p className="text-white/30 text-xs mt-1">
                Send predictions to <code className="text-violet-400">/api/predict</code> to start collecting data.
              </p>
            </div>
          )}
        </>
      )}
    </div>
  )
}

function KPI({ label, value, sub, accent, icon: Icon }) {
  return (
    <div className="rounded-2xl border border-white/8 bg-[#0f0f14] p-5">
      <p className="text-white/40 text-xs font-medium uppercase tracking-wide mb-2">{label}</p>
      <p className="font-bold text-2xl leading-none" style={{ color: accent ?? '#fff' }}>
        {Icon && <Icon size={16} className="inline mr-1 mb-0.5 opacity-80" />}
        {value ?? '—'}
      </p>
      {sub && <p className="text-white/30 text-xs mt-1.5">{sub}</p>}
    </div>
  )
}

function Bar({ label, value, color, threshold }) {
  const pct    = (value ?? 0) * 100
  const passes = value >= threshold
  return (
    <div className="flex items-center gap-3">
      <span className="text-white/50 text-xs w-24 shrink-0">{label}</span>
      <div className="flex-1 h-2 rounded-full bg-white/8 overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${pct}%`, background: passes ? color : '#f59e0b' }}
        />
      </div>
      <span className="text-xs font-semibold w-12 text-right shrink-0"
            style={{ color: passes ? color : '#f59e0b' }}>
        {pct.toFixed(1)}%
      </span>
    </div>
  )
}

function DiffBadge({ v }) {
  const color = Math.abs(v) <= 5 ? '#10b981' : Math.abs(v) <= 15 ? '#f59e0b' : '#e11d48'
  return (
    <span className="text-xs font-mono font-semibold" style={{ color }}>
      {v > 0 ? '+' : ''}{v?.toFixed(2)}%
    </span>
  )
}

function RecCard({ rec, reason }) {
  const cfg = REC_CONFIG[rec] ?? REC_CONFIG.monitor
  const Icon = cfg.icon
  return (
    <div className="rounded-xl border p-4 flex gap-3"
         style={{ background: cfg.bg, borderColor: cfg.border }}>
      <Icon size={18} style={{ color: cfg.color }} className="shrink-0 mt-0.5" />
      <div>
        <p className="text-sm font-bold" style={{ color: cfg.color }}>{cfg.label}</p>
        <p className="text-white/50 text-xs mt-1 leading-relaxed">{reason}</p>
      </div>
    </div>
  )
}

function Skeleton() {
  return (
    <div className="space-y-4 animate-pulse">
      <div className="h-16 rounded-2xl bg-white/5" />
      <div className="grid grid-cols-4 gap-4">
        {[...Array(4)].map((_, i) => <div key={i} className="h-24 rounded-2xl bg-white/5" />)}
      </div>
      <div className="h-32 rounded-2xl bg-white/5" />
    </div>
  )
}
