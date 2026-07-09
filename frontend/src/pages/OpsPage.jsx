import { useState } from 'react'
import { Server, Database, Inbox, Download, RefreshCw, Trash2, Cpu, Clock, Zap, AlertCircle, HardDrive, GitBranch, ExternalLink } from 'lucide-react'
import { usePolling } from '../hooks/usePolling'

export default function OpsPage() {
  const health   = usePolling('/api/health', 5000)
  const metrics  = usePolling('/api/metrics/summary', 5000)
  const dlq      = usePolling('/api/dlq?n=20', 10000)
  const tdStats  = usePolling('/api/training-data/stats', 10000)
  const modelReg = usePolling('/api/v1/model-info', null)   // fetch once; user refreshes manually

  const h = health.data ?? {}
  const m = metrics.data ?? {}
  const lat = m.endpoints?.['/v1/predict']?.latency_ms ?? {}
  const cache = h.cache ?? {}

  const [downloading, setDownloading]         = useState(false)
  const [clearingDlq, setClearingDlq]         = useState(false)
  const [clearingPreds, setClearingPreds]     = useState(false)
  const [confirmClear, setConfirmClear]       = useState(false)

  async function downloadCSV() {
    setDownloading(true)
    try {
      const r = await fetch('/api/training-data?limit=50000')
      const body = await r.json()
      const rows = body.rows ?? []
      if (rows.length === 0) return
      const cols = Object.keys(rows[0])
      const csv = [
        cols.join(','),
        ...rows.map(row => cols.map(c => {
          const v = row[c]
          const s = v === null || v === undefined ? '' : String(v)
          return s.includes(',') || s.includes('"') ? `"${s.replace(/"/g, '""')}"` : s
        }).join(',')),
      ].join('\n')
      const blob = new Blob([csv], { type: 'text/csv' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `nyc_predictions_${new Date().toISOString().slice(0, 10)}.csv`
      a.click()
      URL.revokeObjectURL(url)
    } finally {
      setDownloading(false)
    }
  }

  async function clearDlq() {
    setClearingDlq(true)
    try {
      await fetch('/api/dlq', { method: 'DELETE' })
      dlq.refresh()
      health.refresh()
    } finally {
      setClearingDlq(false)
    }
  }

  async function clearPredictions() {
    setClearingPreds(true)
    setConfirmClear(false)
    try {
      await fetch('/api/predictions', { method: 'DELETE' })
      tdStats.refresh()
      health.refresh()
    } finally {
      setClearingPreds(false)
    }
  }

  const apiAlive = health.data !== null && !health.error

  return (
    <div className="space-y-6">
      {/* header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-white text-xl font-bold tracking-tight flex items-center gap-2">
            <Server size={20} className="text-sky-400" />
            Operations
          </h1>
          <p className="text-white/35 text-sm mt-1">
            System health · cache · dead-letter queue · training data export
          </p>
        </div>
        <div className={`flex items-center gap-2 px-3.5 py-1.5 rounded-full text-xs font-semibold border ${
          apiAlive
            ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'
            : 'bg-red-500/10 border-red-500/20 text-red-400'
        }`}>
          <span className={`w-1.5 h-1.5 rounded-full animate-pulse ${apiAlive ? 'bg-emerald-400' : 'bg-red-400'}`} />
          {apiAlive ? 'API healthy' : 'API offline'}
        </div>
      </div>

      {/* system KPIs */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        <KPI icon={Clock}      label="Uptime"      value={h.uptime ?? '—'}                       color="#60a5fa" />
        <KPI icon={Zap}        label="Requests"    value={h.total_requests?.toLocaleString() ?? '—'} color="#a78bfa" />
        <KPI icon={Cpu}        label="Predictions" value={h.total_predictions?.toLocaleString() ?? '—'} color="#34d399" />
        <KPI icon={AlertCircle}label="Error Rate"  value={h.error_rate ?? '—'}                   color="#f87171" />
        <KPI icon={HardDrive}  label="Store Rows"  value={h.store_rows?.toLocaleString() ?? '—'} color="#fbbf24" />
        <KPI icon={Inbox}      label="DLQ Size"    value={h.dlq_size ?? '—'}                     color={h.dlq_size > 0 ? '#f87171' : '#34d399'} />
      </div>

      {/* model + backend */}
      <div className="rounded-2xl border border-white/8 bg-[#0f0f14] p-5 flex flex-wrap items-center gap-x-8 gap-y-3">
        <Inline label="Model" value="XGBoost (ONNX Runtime)" />
        <Inline label="Backend" value={h.inference_backend?.replace('ExecutionProvider', '') ?? '—'} />
        <Inline label="R² (test)" value={h.model_r2 ?? '—'} />
        <Inline label="Latency p95" value={lat.p95 != null ? `${lat.p95}ms` : '—'} />
        <Inline label="Latency avg" value={lat.avg != null ? `${lat.avg}ms` : '—'} />
      </div>

      {/* MLflow Model Registry */}
      <ModelRegistryCard data={modelReg.data} loading={modelReg.loading} onRefresh={modelReg.refresh} />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* cache panel */}
        <Card title="Redis Cache" icon={Database} accent="#34d399"
              action={<RefreshBtn onClick={() => health.refresh()} spinning={health.loading} />}>
          <div className="flex items-center gap-2 mb-4">
            <span className={`w-2 h-2 rounded-full ${cache.connected ? 'bg-emerald-400' : 'bg-red-400'}`} />
            <span className="text-sm text-white/60">
              {cache.connected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <Metric label="Session Hits"   value={cache.hits ?? 0} />
            <Metric label="Session Misses" value={cache.misses ?? 0} />
            <Metric label="Hit Rate"       value={`${cache.hit_rate_pct?.toFixed(1) ?? 0}%`} accent="#34d399" />
            <Metric label="Redis Lifetime" value={`${(cache.redis_hits ?? 0).toLocaleString()} hits`} />
          </div>
          {/* hit rate bar */}
          <div className="mt-4">
            <div className="h-2 rounded-full bg-white/8 overflow-hidden">
              <div className="h-full bg-emerald-500/70 rounded-full transition-all duration-700"
                   style={{ width: `${Math.min(cache.hit_rate_pct ?? 0, 100)}%` }} />
            </div>
          </div>
        </Card>

        {/* training data / predictions store */}
        <Card title="Prediction Store" icon={Download} accent="#fbbf24"
              action={<RefreshBtn onClick={() => tdStats.refresh()} spinning={tdStats.loading} />}>
          <div className="grid grid-cols-2 gap-3 mb-4">
            <Metric label="Total Logged" value={tdStats.data?.total?.toLocaleString() ?? '—'} />
            <Metric label="Cache Hits"   value={tdStats.data?.cache_hits?.toLocaleString() ?? '—'} />
            <Metric label="Avg Price"    value={tdStats.data?.avg_price != null ? `$${Math.round(tdStats.data.avg_price)}` : '—'} accent="#fbbf24" />
            <Metric label="Price Range"  value={tdStats.data?.min_price != null ? `$${Math.round(tdStats.data.min_price)}–${Math.round(tdStats.data.max_price)}` : '—'} />
          </div>
          <div className="space-y-2">
            <button onClick={downloadCSV} disabled={downloading || !tdStats.data?.total}
              className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl
                         bg-amber-500/15 border border-amber-500/25 text-amber-300 text-sm font-semibold
                         hover:bg-amber-500/25 transition-colors disabled:opacity-40">
              <Download size={15} className={downloading ? 'animate-bounce' : ''} />
              {downloading ? 'Preparing CSV…' : 'Download CSV (up to 50k rows)'}
            </button>

            {/* clear predictions — two-step confirm so it can't be fat-fingered */}
            {!confirmClear ? (
              <button
                onClick={() => setConfirmClear(true)}
                disabled={!tdStats.data?.total || clearingPreds}
                className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl
                           bg-red-500/8 border border-red-500/20 text-red-400/70 text-sm font-medium
                           hover:bg-red-500/15 hover:text-red-300 transition-colors disabled:opacity-30">
                <Trash2 size={14} />
                Clear all predictions (resets drift window)
              </button>
            ) : (
              <div className="flex gap-2">
                <button
                  onClick={clearPredictions}
                  disabled={clearingPreds}
                  className="flex-1 flex items-center justify-center gap-1.5 px-4 py-2.5 rounded-xl
                             bg-red-600/20 border border-red-500/40 text-red-300 text-sm font-semibold
                             hover:bg-red-600/30 transition-colors">
                  <Trash2 size={14} />
                  {clearingPreds ? 'Clearing…' : 'Yes, delete all'}
                </button>
                <button
                  onClick={() => setConfirmClear(false)}
                  className="flex-1 px-4 py-2.5 rounded-xl border border-white/10
                             bg-white/[0.03] text-white/50 text-sm hover:text-white/70 transition-colors">
                  Cancel
                </button>
              </div>
            )}
          </div>
        </Card>
      </div>

      {/* DLQ table */}
      <Card title="Dead Letter Queue" icon={Inbox} accent="#f87171"
            action={
              <div className="flex items-center gap-2">
                <RefreshBtn onClick={() => dlq.refresh()} spinning={dlq.loading} />
                {(dlq.data?.size ?? 0) > 0 && (
                  <button onClick={clearDlq} disabled={clearingDlq}
                    className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-red-500/25
                               bg-red-500/10 text-red-400 text-xs font-medium hover:bg-red-500/20 transition-colors">
                    <Trash2 size={12} />
                    {clearingDlq ? 'Clearing…' : 'Clear'}
                  </button>
                )}
              </div>
            }>
        <p className="text-white/40 text-sm mb-4">
          {dlq.data?.size ?? 0} failed request{(dlq.data?.size ?? 0) !== 1 ? 's' : ''} ·
          showing latest {dlq.data?.entries?.length ?? 0}
        </p>
        {(dlq.data?.entries?.length ?? 0) === 0 ? (
          <div className="text-center py-8 text-white/30 text-sm">
            <Inbox size={28} className="mx-auto mb-2 opacity-30" />
            No failed requests — queue is empty
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-white/30 text-xs border-b border-white/[0.06]">
                  <th className="text-left pb-2 font-medium">Time</th>
                  <th className="text-left pb-2 font-medium">Endpoint</th>
                  <th className="text-left pb-2 font-medium">Error</th>
                  <th className="text-right pb-2 font-medium">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/[0.04]">
                {dlq.data.entries.map((e, i) => (
                  <tr key={i} className="text-white/60">
                    <td className="py-2.5 text-white/40 text-xs whitespace-nowrap">
                      {e.timestamp ? new Date(e.timestamp).toLocaleTimeString() : '—'}
                    </td>
                    <td className="py-2.5 font-mono text-xs">{e.endpoint ?? '—'}</td>
                    <td className="py-2.5 text-red-300/80 text-xs max-w-xs truncate">{e.error ?? '—'}</td>
                    <td className="py-2.5 text-right">
                      <span className="text-xs font-mono px-2 py-0.5 rounded bg-red-500/10 text-red-400">
                        {e.status_code ?? '5xx'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  )
}

function KPI({ icon: Icon, label, value, color }) {
  return (
    <div className="rounded-2xl border border-white/8 bg-[#0f0f14] p-4">
      <Icon size={15} style={{ color }} className="mb-2 opacity-80" />
      <p className="text-white font-bold text-lg leading-none">{value}</p>
      <p className="text-white/35 text-[11px] mt-1.5">{label}</p>
    </div>
  )
}

function Inline({ label, value }) {
  return (
    <div>
      <p className="text-white/35 text-xs">{label}</p>
      <p className="text-white font-semibold text-sm mt-0.5">{value}</p>
    </div>
  )
}

function Card({ title, icon: Icon, accent, action, children }) {
  return (
    <div className="rounded-2xl border border-white/8 bg-[#0f0f14] overflow-hidden">
      <div className="flex items-center justify-between px-5 py-3.5 border-b border-white/[0.06]">
        <div className="flex items-center gap-2">
          <Icon size={15} style={{ color: accent }} />
          <span className="text-white font-semibold text-sm">{title}</span>
        </div>
        {action}
      </div>
      <div className="p-5">{children}</div>
    </div>
  )
}

function Metric({ label, value, accent }) {
  return (
    <div className="rounded-xl border border-white/6 bg-white/[0.02] p-3">
      <p className="text-white/35 text-xs mb-1">{label}</p>
      <p className="font-bold text-base" style={{ color: accent ?? '#fff' }}>{value}</p>
    </div>
  )
}

function RefreshBtn({ onClick, spinning }) {
  return (
    <button onClick={onClick}
      className="p-1.5 rounded-lg text-white/40 hover:text-white hover:bg-white/5 transition-colors">
      <RefreshCw size={13} className={spinning ? 'animate-spin' : ''} />
    </button>
  )
}

function ModelRegistryCard({ data, loading, onRefresh }) {
  const reg = data?.registry ?? null
  const fromMlflow = reg?.source === 'mlflow'

  const kv = fromMlflow
    ? [
        { label: 'Model Name',   value: reg.model_name },
        { label: 'Version',      value: `v${reg.version}`, accent: '#a78bfa' },
        { label: 'Run ID',       value: reg.run_id?.slice(0, 8) + '…' },
        { label: 'Run Name',     value: reg.run_name ?? '—' },
        { label: 'Git SHA',      value: reg.git_sha ?? '—' },
        { label: 'Data Date',    value: reg.data_date ?? '—' },
        { label: 'R² test',      value: reg.metrics?.r2_test?.toFixed(4) ?? '—', accent: '#34d399' },
        { label: 'RMSE (log)',   value: reg.metrics?.rmse_log?.toFixed(4) ?? '—' },
        { label: 'MAE ($)',      value: reg.metrics?.mae_dollar != null ? `$${Math.round(reg.metrics.mae_dollar)}` : '—' },
      ]
    : reg
    ? [
        { label: 'Source',       value: 'local report (MLflow offline)' },
        { label: 'Best Model',   value: reg.best_model ?? '—' },
        { label: 'Trained',      value: reg.timestamp ? new Date(reg.timestamp).toLocaleDateString() : '—' },
        { label: 'R² test',      value: reg.metrics?.r2?.toFixed(4) ?? '—', accent: '#34d399' },
        { label: 'MAE ($)',      value: reg.metrics?.mae_dollar != null ? `$${Math.round(reg.metrics.mae_dollar)}` : '—' },
      ]
    : []

  return (
    <Card title="Model Registry" icon={GitBranch} accent="#a78bfa"
          action={
            <div className="flex items-center gap-2">
              <RefreshBtn onClick={onRefresh} spinning={loading} />
              {fromMlflow && reg.mlflow_ui && (
                <a href={reg.mlflow_ui} target="_blank" rel="noreferrer"
                   className="flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-medium
                              text-violet-400 border border-violet-500/25 bg-violet-500/10
                              hover:bg-violet-500/20 transition-colors">
                  <ExternalLink size={11} />
                  MLflow UI
                </a>
              )}
            </div>
          }>
      {!reg ? (
        <p className="text-white/30 text-sm">
          {loading ? 'Loading…' : 'Model registry unavailable'}
        </p>
      ) : (
        <>
          <div className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold mb-4 ${
            fromMlflow
              ? 'bg-violet-500/15 border border-violet-500/25 text-violet-300'
              : 'bg-amber-500/10 border border-amber-500/20 text-amber-400'
          }`}>
            <span className="w-1.5 h-1.5 rounded-full bg-current" />
            {fromMlflow ? 'champion alias · MLflow registry' : 'fallback · local report'}
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5">
            {kv.map(({ label, value, accent }) => (
              <div key={label} className="rounded-xl border border-white/6 bg-white/[0.02] p-3">
                <p className="text-white/35 text-xs mb-1">{label}</p>
                <p className="font-semibold text-sm truncate" style={{ color: accent ?? '#fff' }}>{value}</p>
              </div>
            ))}
          </div>
          {fromMlflow && reg.metrics && Object.keys(reg.metrics).length > 0 && (
            <details className="mt-3 text-xs text-white/40">
              <summary className="cursor-pointer hover:text-white/60 transition-colors">
                All run metrics ({Object.keys(reg.metrics).length})
              </summary>
              <div className="mt-2 grid grid-cols-2 gap-1.5">
                {Object.entries(reg.metrics).map(([k, v]) => (
                  <div key={k} className="flex justify-between px-2 py-1 rounded bg-white/[0.03] font-mono">
                    <span className="text-white/40">{k}</span>
                    <span className="text-white/70">{v}</span>
                  </div>
                ))}
              </div>
            </details>
          )}
        </>
      )}
    </Card>
  )
}
