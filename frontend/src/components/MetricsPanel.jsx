import { Activity, Zap, TrendingUp, Clock, RefreshCw } from 'lucide-react'

export default function MetricsPanel({ metrics }) {
  if (!metrics) {
    return (
      <div className="rounded-2xl border border-white/8 bg-white/[0.02] p-6 text-center">
        <div className="w-10 h-10 rounded-xl bg-white/5 flex items-center justify-center mx-auto mb-3">
          <Activity size={16} className="text-white/20" />
        </div>
        <p className="text-white/30 text-sm">Start the API to see<br />live metrics</p>
      </div>
    )
  }

  const t   = metrics.totals ?? {}
  const p   = metrics.predictions ?? {}
  const lat = metrics.endpoints?.['/predict']?.latency_ms ?? {}
  const maxLat = lat.max ?? 0

  return (
    <div className="rounded-2xl border border-white/8 bg-white/[0.02] overflow-hidden">
      <div className="flex items-center justify-between px-5 py-3.5 border-b border-white/6">
        <div className="flex items-center gap-2">
          <Activity size={13} className="text-white/40" />
          <span className="text-white/70 font-semibold text-sm">Live Metrics</span>
        </div>
        <div className="flex items-center gap-1.5 text-white/20 text-xs">
          <RefreshCw size={10} className="animate-spin" style={{ animationDuration: '4s' }} />
          5s
        </div>
      </div>

      <div className="p-4 space-y-4">

        {/* 4-stat row */}
        <div className="grid grid-cols-2 gap-2.5">
          <MiniStat icon={Clock}      color="blue"    label="Uptime"      value={metrics.uptime_human} />
          <MiniStat icon={Zap}        color="violet"  label="Requests"    value={t.requests} />
          <MiniStat icon={TrendingUp} color="emerald" label="Predictions" value={p.total} />
          <MiniStat icon={Activity}   color="rose"    label="Error rate"  value={t.error_rate} />
        </div>

        {/* latency */}
        <div className="rounded-xl border border-white/6 bg-black/20 p-3.5 space-y-3">
          <p className="text-white/30 text-[11px] font-semibold uppercase tracking-widest">
            Latency · /predict
          </p>
          <div className="grid grid-cols-3 gap-2 text-center">
            <LatNum label="avg" value={lat.avg != null ? `${lat.avg}ms` : null} />
            <LatNum label="p95" value={lat.p95 != null ? `${lat.p95}ms` : null} highlight />
            <LatNum label="max" value={lat.max != null ? `${lat.max}ms` : null} />
          </div>
          {maxLat > 0 && lat.avg != null && (
            <div className="relative h-1 bg-white/8 rounded-full overflow-hidden">
              <div className="absolute h-full bg-emerald-500/50 rounded-full transition-all duration-700"
                   style={{ width: `${Math.min((lat.avg / maxLat) * 100, 100)}%` }} />
              {lat.p95 != null && (
                <div className="absolute top-0 h-full w-px bg-amber-400/60"
                     style={{ left: `${Math.min((lat.p95 / maxLat) * 100, 100)}%` }} />
              )}
            </div>
          )}
        </div>

        {/* price distribution */}
        {p.total > 0 && (
          <div className="rounded-xl border border-white/6 bg-black/20 p-3.5 space-y-2">
            <p className="text-white/30 text-[11px] font-semibold uppercase tracking-widest">
              Price distribution
            </p>
            <div className="grid grid-cols-3 gap-2 text-center">
              <LatNum label="min"  value={p.min  != null ? `$${Math.round(p.min)}`  : null} />
              <LatNum label="avg"  value={p.mean != null ? `$${Math.round(p.mean)}` : null} highlight />
              <LatNum label="max"  value={p.max  != null ? `$${Math.round(p.max)}`  : null} />
            </div>
          </div>
        )}

      </div>
    </div>
  )
}

const COLORS = {
  blue:    'bg-blue-500/10 border-blue-500/15 text-blue-400',
  violet:  'bg-violet-500/10 border-violet-500/15 text-violet-400',
  emerald: 'bg-emerald-500/10 border-emerald-500/15 text-emerald-400',
  rose:    'bg-rose-500/10 border-rose-500/15 text-rose-400',
}

function MiniStat({ icon: Icon, color, label, value }) {
  return (
    <div className={`rounded-xl border p-3 ${COLORS[color]}`}>
      <Icon size={13} className="mb-2 opacity-70" />
      <p className="text-white text-lg font-bold leading-none mb-0.5">{value ?? '—'}</p>
      <p className="text-white/35 text-[11px]">{label}</p>
    </div>
  )
}

function LatNum({ label, value, highlight }) {
  return (
    <div>
      <p className={`text-base font-bold leading-none mb-0.5 ${highlight ? 'text-white' : 'text-white/60'}`}>
        {value ?? '—'}
      </p>
      <p className="text-white/25 text-[11px]">{label}</p>
    </div>
  )
}
