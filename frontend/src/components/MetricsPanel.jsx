import { useEffect, useState } from 'react'
import { Zap, TrendingUp, AlertCircle, Clock, BarChart2, RefreshCw } from 'lucide-react'

function MetricCard({ icon: Icon, label, value, sub, color = 'rose' }) {
  const colors = {
    rose:    'from-rose-600/20 to-rose-800/10 border-rose-500/20',
    emerald: 'from-emerald-600/20 to-emerald-800/10 border-emerald-500/20',
    blue:    'from-blue-600/20 to-blue-800/10 border-blue-500/20',
    amber:   'from-amber-600/20 to-amber-800/10 border-amber-500/20',
    violet:  'from-violet-600/20 to-violet-800/10 border-violet-500/20',
  }
  const iconColors = {
    rose: 'text-rose-400', emerald: 'text-emerald-400',
    blue: 'text-blue-400', amber: 'text-amber-400', violet: 'text-violet-400',
  }

  return (
    <div className={`rounded-2xl border bg-gradient-to-br p-4 ${colors[color]}`}>
      <div className="flex items-start justify-between mb-3">
        <Icon size={16} className={iconColors[color]} />
        {sub && <span className="text-white/30 text-xs">{sub}</span>}
      </div>
      <p className="text-white text-2xl font-bold tracking-tight">{value ?? '—'}</p>
      <p className="text-white/50 text-xs mt-1">{label}</p>
    </div>
  )
}

export default function MetricsPanel({ metrics }) {
  const [tick, setTick] = useState(0)

  useEffect(() => {
    const id = setInterval(() => setTick(t => t + 1), 1000)
    return () => clearInterval(id)
  }, [])

  if (!metrics) {
    return (
      <div className="rounded-2xl border border-white/10 bg-white/5 p-6 text-center text-white/40 text-sm">
        Start the API to see live metrics
      </div>
    )
  }

  const t   = metrics.totals ?? {}
  const p   = metrics.predictions ?? {}
  const lat = metrics.endpoints?.['/predict']?.latency_ms ?? {}

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-white font-semibold">Live Metrics</h2>
        <div className="flex items-center gap-1.5 text-white/40 text-xs">
          <RefreshCw size={11} className="animate-spin" style={{ animationDuration: '3s' }} />
          auto-refresh 5s
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <MetricCard icon={Clock}     label="Uptime"       value={metrics.uptime_human}    color="blue" />
        <MetricCard icon={Zap}       label="Requests"     value={t.requests}               color="violet" />
        <MetricCard icon={TrendingUp} label="Predictions"  value={p.total}                  color="emerald" />
        <MetricCard icon={AlertCircle} label="Error rate"  value={t.error_rate}             color="amber" />
      </div>

      <div className="rounded-2xl border border-white/10 bg-white/5 p-4 space-y-3">
        <p className="text-white/60 text-xs font-medium uppercase tracking-wider">Latency · /predict</p>
        <div className="grid grid-cols-3 gap-2 text-center">
          <LatVal label="avg" value={lat.avg} unit="ms" />
          <LatVal label="p95" value={lat.p95} unit="ms" />
          <LatVal label="max" value={lat.max} unit="ms" />
        </div>
        {lat.avg != null && (
          <LatencyBar avg={lat.avg} p95={lat.p95} max={lat.max} />
        )}
      </div>

      {p.total > 0 && (
        <div className="rounded-2xl border border-white/10 bg-white/5 p-4 space-y-3">
          <p className="text-white/60 text-xs font-medium uppercase tracking-wider">Prediction distribution</p>
          <div className="grid grid-cols-3 gap-2 text-center">
            <LatVal label="min"  value={p.min  ? `$${Math.round(p.min)}`  : null} />
            <LatVal label="mean" value={p.mean ? `$${Math.round(p.mean)}` : null} />
            <LatVal label="max"  value={p.max  ? `$${Math.round(p.max)}`  : null} />
          </div>
        </div>
      )}
    </div>
  )
}

function LatVal({ label, value, unit }) {
  return (
    <div>
      <p className="text-white font-semibold text-lg">
        {value != null ? `${value}${unit ? ` ${unit}` : ''}` : '—'}
      </p>
      <p className="text-white/40 text-xs mt-0.5">{label}</p>
    </div>
  )
}

function LatencyBar({ avg, p95, max }) {
  if (!max) return null
  const pct = v => `${Math.min(100, (v / max) * 100).toFixed(1)}%`
  return (
    <div className="relative h-2 bg-white/10 rounded-full overflow-hidden">
      <div className="absolute left-0 top-0 h-full bg-emerald-500/70 rounded-full transition-all duration-500"
           style={{ width: pct(avg) }} />
      {p95 && (
        <div className="absolute top-0 h-full w-0.5 bg-amber-400"
             style={{ left: pct(p95) }} />
      )}
    </div>
  )
}
