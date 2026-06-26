import { Clock, Zap, TrendingUp, AlertCircle, RefreshCw } from 'lucide-react'

function StatCard({ icon: Icon, label, value, color }) {
  const palette = {
    blue:    { bg: 'bg-blue-500/10',    border: 'border-blue-500/15',    icon: 'text-blue-400' },
    violet:  { bg: 'bg-violet-500/10',  border: 'border-violet-500/15',  icon: 'text-violet-400' },
    emerald: { bg: 'bg-emerald-500/10', border: 'border-emerald-500/15', icon: 'text-emerald-400' },
    amber:   { bg: 'bg-amber-500/10',   border: 'border-amber-500/15',   icon: 'text-amber-400' },
  }[color] ?? {}

  return (
    <div className={`rounded-2xl border p-4 ${palette.bg} ${palette.border}`}>
      <Icon size={15} className={`mb-3 ${palette.icon}`} />
      <p className="text-white text-2xl font-bold leading-none mb-1">{value ?? '—'}</p>
      <p className="text-white/40 text-xs">{label}</p>
    </div>
  )
}

export default function MetricsPanel({ metrics }) {
  if (!metrics) {
    return (
      <div className="rounded-2xl border border-white/8 bg-white/[0.03] p-8 text-center">
        <div className="w-10 h-10 rounded-xl bg-white/5 flex items-center justify-center mx-auto mb-3">
          <Zap size={16} className="text-white/20" />
        </div>
        <p className="text-white/30 text-sm">Start the API to see<br/>live metrics here</p>
      </div>
    )
  }

  const t   = metrics.totals ?? {}
  const p   = metrics.predictions ?? {}
  const lat = metrics.endpoints?.['/predict']?.latency_ms ?? {}

  return (
    <div className="rounded-2xl border border-white/8 bg-white/[0.03] overflow-hidden">
      {/* header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-white/6">
        <span className="text-white font-semibold text-sm">Live Metrics</span>
        <div className="flex items-center gap-1.5 text-white/25 text-xs">
          <RefreshCw size={10} className="animate-spin" style={{ animationDuration: '4s' }} />
          auto-refresh 5s
        </div>
      </div>

      <div className="p-5 space-y-5">

        {/* 4 stat cards */}
        <div className="grid grid-cols-2 gap-3">
          <StatCard icon={Clock}       label="Uptime"       value={metrics.uptime_human} color="blue" />
          <StatCard icon={Zap}         label="Requests"     value={t.requests}            color="violet" />
          <StatCard icon={TrendingUp}  label="Predictions"  value={p.total}               color="emerald" />
          <StatCard icon={AlertCircle} label="Error rate"   value={t.error_rate}          color="amber" />
        </div>

        {/* latency block */}
        <div className="rounded-xl border border-white/6 bg-white/[0.025] p-4 space-y-3">
          <p className="text-white/30 text-xs font-semibold uppercase tracking-widest">
            Latency · /predict
          </p>
          <div className="grid grid-cols-3 gap-3 text-center">
            <LatStat label="avg" value={lat.avg != null ? `${lat.avg} ms` : null} />
            <LatStat label="p95" value={lat.p95 != null ? `${lat.p95} ms` : null} highlight />
            <LatStat label="max" value={lat.max != null ? `${lat.max} ms` : null} />
          </div>
          {lat.max != null && lat.avg != null && (
            <div className="relative h-1.5 bg-white/8 rounded-full overflow-hidden">
              <div className="absolute h-full bg-emerald-500/60 rounded-full transition-all duration-700"
                   style={{ width: `${(lat.avg / lat.max) * 100}%` }} />
              {lat.p95 != null && (
                <div className="absolute top-0 h-full w-px bg-amber-400/70"
                     style={{ left: `${(lat.p95 / lat.max) * 100}%` }} />
              )}
            </div>
          )}
        </div>

        {/* prediction distribution */}
        {p.total > 0 && (
          <div className="rounded-xl border border-white/6 bg-white/[0.025] p-4 space-y-3">
            <p className="text-white/30 text-xs font-semibold uppercase tracking-widest">
              Prediction distribution
            </p>
            <div className="grid grid-cols-3 gap-3 text-center">
              <LatStat label="min"  value={p.min  != null ? `$${Math.round(p.min)}`  : null} />
              <LatStat label="mean" value={p.mean != null ? `$${Math.round(p.mean)}` : null} highlight />
              <LatStat label="max"  value={p.max  != null ? `$${Math.round(p.max)}`  : null} />
            </div>
          </div>
        )}

      </div>
    </div>
  )
}

function LatStat({ label, value, highlight }) {
  return (
    <div>
      <p className={`text-lg font-bold leading-none mb-1 ${highlight ? 'text-white' : 'text-white/70'}`}>
        {value ?? '—'}
      </p>
      <p className="text-white/30 text-xs">{label}</p>
    </div>
  )
}
