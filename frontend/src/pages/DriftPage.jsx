import { Activity, RefreshCw, Clock } from 'lucide-react'
import { usePolling } from '../hooks/usePolling'

const STATUS_CFG = {
  ok:               { label: 'No Drift',         color: '#10b981', bg: 'rgba(16,185,129,0.1)',  border: 'rgba(16,185,129,0.2)'  },
  warning:          { label: 'Drift Warning',    color: '#f59e0b', bg: 'rgba(245,158,11,0.1)',  border: 'rgba(245,158,11,0.2)'  },
  critical:         { label: 'Critical Drift',   color: '#e11d48', bg: 'rgba(225,29,72,0.1)',   border: 'rgba(225,29,72,0.2)'   },
  insufficient_data:{ label: 'Insufficient Data',color: '#6b7280', bg: 'rgba(107,114,128,0.1)', border: 'rgba(107,114,128,0.2)' },
  no_baseline:      { label: 'No Baseline',      color: '#6b7280', bg: 'rgba(107,114,128,0.1)', border: 'rgba(107,114,128,0.2)' },
}

const FEATURE_LABELS = {
  accommodates:          'Accommodates',
  bedrooms:              'Bedrooms',
  bathrooms:             'Bathrooms',
  minimum_nights:        'Min. Nights',
  number_of_reviews:     'Review Count',
  reviews_per_month:     'Reviews / Month',
  review_scores_rating:  'Review Score',
  amenity_count:         'Amenity Count',
  host_listings_count:   'Host Listings',
}

export default function DriftPage() {
  // Read-only poll on mount — no push_alert so page loads don't create alert noise.
  // The "Run Check" button uses the push_alert=true URL to explicitly fire an alert if warranted.
  const { data, loading, error, refresh, lastFetch } = usePolling('/api/drift?window_hours=168', null)

  async function runCheck() {
    await fetch('/api/drift?window_hours=168&push_alert=true')
    refresh()
  }

  return (
    <div className="space-y-6">
      {/* header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-white text-xl font-bold tracking-tight flex items-center gap-2">
            <Activity size={20} className="text-amber-400" />
            Drift Monitor
          </h1>
          <p className="text-white/35 text-sm mt-1">
            PSI-based feature drift · 7-day rolling window
          </p>
        </div>
        <button onClick={runCheck}
          className="flex items-center gap-2 px-3.5 py-2 rounded-xl border border-white/10
                     bg-white/[0.03] text-white/60 text-sm hover:text-white hover:border-white/20 transition-colors">
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          Run Check
        </button>
      </div>

      {error && (
        <div className="rounded-xl border border-red-500/25 bg-red-500/8 px-4 py-3 text-red-300 text-sm">
          {error}
        </div>
      )}

      {loading && !data && <Skeleton />}

      {data && (
        <>
          {/* status banner */}
          {(() => {
            const cfg = STATUS_CFG[data.status] ?? STATUS_CFG.no_baseline
            return (
              <div className="rounded-2xl border p-5" style={{ background: cfg.bg, borderColor: cfg.border }}>
                <div className="flex items-center justify-between flex-wrap gap-3">
                  <div className="flex items-center gap-3">
                    <span className="w-3 h-3 rounded-full animate-pulse"
                          style={{ background: cfg.color }} />
                    <span className="text-white font-bold text-lg">{cfg.label}</span>
                    {data.alerts?.length > 0 && (
                      <span className="text-xs px-2 py-0.5 rounded-full font-semibold"
                            style={{ background: 'rgba(255,255,255,0.1)', color: cfg.color }}>
                        {data.alerts.length} alert{data.alerts.length > 1 ? 's' : ''}
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-4 text-sm" style={{ color: cfg.color + 'bb' }}>
                    <span>{data.n_samples?.toLocaleString()} samples</span>
                    <span>·</span>
                    <span>{data.window_hours}h window</span>
                    {lastFetch && (
                      <>
                        <span>·</span>
                        <span className="flex items-center gap-1">
                          <Clock size={12} />
                          {lastFetch.toLocaleTimeString()}
                        </span>
                      </>
                    )}
                  </div>
                </div>
                {data.alerts?.length > 0 && (
                  <ul className="mt-3 space-y-1">
                    {data.alerts.map((a, i) => (
                      <li key={i} className="text-sm" style={{ color: cfg.color + 'cc' }}>
                        · {a}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )
          })()}

          {/* prediction drift */}
          {data.prediction_drift && Object.keys(data.prediction_drift).length > 0 && (
            <Section title="Prediction Price Drift">
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                <MiniKPI label="Current Mean" value={`$${data.prediction_drift.current_mean?.toFixed(0)}`} />
                <MiniKPI label="Baseline Mean" value={`$${data.prediction_drift.baseline_mean?.toFixed(0)}`} />
                <MiniKPI
                  label="Shift"
                  value={`${data.prediction_drift.shift_pct > 0 ? '+' : ''}${data.prediction_drift.shift_pct?.toFixed(1)}%`}
                  color={Math.abs(data.prediction_drift.shift_pct) > 15 ? '#e11d48' : '#10b981'}
                />
                <MiniKPI
                  label="Threshold"
                  value={`±${data.prediction_drift.threshold_pct?.toFixed(0)}%`}
                  color="#6b7280"
                />
              </div>
            </Section>
          )}

          {/* feature drift PSI */}
          {data.feature_drift && Object.keys(data.feature_drift).length > 0 && (
            <Section title="Feature Drift (PSI)">
              <div className="space-y-3">
                <div className="grid grid-cols-[1fr_160px_80px_80px] text-xs text-white/30 font-medium pb-1 border-b border-white/[0.06]">
                  <span>Feature</span>
                  <span className="text-right">PSI Score</span>
                  <span className="text-right">Value</span>
                  <span className="text-right">Status</span>
                </div>
                {(() => {
                  // scale bars to the actual max PSI so they're always readable
                  const psiValues = Object.values(data.feature_drift).map(f => f.psi ?? 0)
                  const maxPsi    = Math.max(...psiValues, 0.30)   // floor at 0.30 for normal ranges
                  return Object.entries(data.feature_drift).map(([feat, info]) => {
                  const psi     = info.psi ?? 0
                  const status  = info.status ?? 'ok'
                  const barPct  = Math.min((psi / maxPsi) * 100, 100)
                  const color   = status === 'critical' ? '#e11d48' : status === 'warning' ? '#f59e0b' : '#10b981'
                  return (
                    <div key={feat}
                         className="grid grid-cols-[1fr_160px_80px_80px] items-center gap-2 py-1">
                      <span className="text-white/70 text-sm">
                        {FEATURE_LABELS[feat] ?? feat}
                      </span>
                      <div className="h-2 rounded-full bg-white/8 overflow-hidden">
                        <div className="h-full rounded-full transition-all duration-700"
                             style={{ width: `${barPct}%`, background: color }} />
                      </div>
                      <span className="text-right text-sm font-mono font-semibold"
                            style={{ color }}>
                        {psi.toFixed(3)}
                      </span>
                      <div className="text-right">
                        <StatusPill status={status} />
                      </div>
                    </div>
                  )
                })})()}
                <div className="pt-2 flex gap-4 text-[11px] text-white/30 flex-wrap">
                  <span className="flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-emerald-500" /> {'< 0.10 OK'}
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-amber-500" /> {'0.10–0.20 Warning'}
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-rose-500" /> {'> 0.20 Critical'}
                  </span>
                  <span className="text-white/20 ml-auto">bars scale to max PSI in window</span>
                </div>
              </div>
            </Section>
          )}

          {/* categorical drift */}
          {data.categorical_drift && Object.keys(data.categorical_drift).length > 0 && (
            <Section title="Categorical Drift">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {Object.entries(data.categorical_drift).map(([col, info]) => {
                  const psi    = info.psi ?? 0
                  const status = info.status ?? 'ok'
                  const color  = status === 'critical' ? '#e11d48' : status === 'warning' ? '#f59e0b' : '#10b981'
                  return (
                    <div key={col} className="rounded-xl border border-white/8 bg-[#0f0f14] p-4 flex items-center gap-4">
                      <div className="flex-1">
                        <p className="text-white/70 text-sm font-medium capitalize">
                          {col === 'room_type' ? 'Room Type' : col.charAt(0).toUpperCase() + col.slice(1)}
                        </p>
                        <p className="text-white/30 text-xs mt-0.5">PSI score</p>
                      </div>
                      <div className="text-right">
                        <p className="text-xl font-bold font-mono" style={{ color }}>
                          {psi.toFixed(3)}
                        </p>
                        <StatusPill status={status} />
                      </div>
                    </div>
                  )
                })}
              </div>
            </Section>
          )}
        </>
      )}
    </div>
  )
}

function Section({ title, children }) {
  return (
    <div className="rounded-2xl border border-white/8 bg-[#0f0f14] p-5 space-y-4">
      <h3 className="text-white text-sm font-semibold">{title}</h3>
      {children}
    </div>
  )
}

function MiniKPI({ label, value, color }) {
  return (
    <div className="rounded-xl border border-white/8 bg-white/[0.03] p-4">
      <p className="text-white/40 text-xs mb-1">{label}</p>
      <p className="text-lg font-bold font-mono" style={{ color: color ?? '#fff' }}>{value}</p>
    </div>
  )
}

function StatusPill({ status }) {
  const cfg = STATUS_CFG[status] ?? STATUS_CFG.ok
  return (
    <span className="inline-block text-[10px] font-bold px-2 py-0.5 rounded-full"
          style={{ color: cfg.color, background: cfg.bg, border: `1px solid ${cfg.border}` }}>
      {status?.toUpperCase()}
    </span>
  )
}

function Skeleton() {
  return (
    <div className="space-y-4 animate-pulse">
      <div className="h-20 rounded-2xl bg-white/5" />
      <div className="h-32 rounded-2xl bg-white/5" />
      <div className="h-64 rounded-2xl bg-white/5" />
    </div>
  )
}
