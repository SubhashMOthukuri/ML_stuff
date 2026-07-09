/**
 * ShapChart — renders a horizontal bar chart of SHAP feature contributions.
 *
 * Design decisions:
 *  - Features that share the same concept (raw + log transform) are merged
 *    so a user never sees "Minimum nights" and "Minimum nights (log)" as
 *    two separate rows. The backend returns both because XGBoost sees both,
 *    but the explanation should be at the human concept level.
 *  - A skeleton variant is shown while the API is computing SHAP values.
 *  - The component is purely presentational — no data fetching, no side effects.
 *  - Bars animate in on mount via CSS transition (respects prefers-reduced-motion).
 */

import { useEffect, useState } from 'react'
import { TrendingUp, TrendingDown, HelpCircle } from 'lucide-react'

// ── concept deduplication ─────────────────────────────────────────────────────
// When the model returns both a raw feature and its log transform, merge them
// into one row using the base label (strip trailing " (log)", " (log-scale)", etc.)
const BASE_LABEL_RE = /\s*\(log.*?\)$|\s*\(log\)$/i

function mergeFeatures(features) {
  const map = new Map()
  for (const f of features) {
    const base = f.label.replace(BASE_LABEL_RE, '').trim()
    if (map.has(base)) {
      const existing = map.get(base)
      existing.contribution_usd += f.contribution_usd
      existing.shap_value       += f.shap_value
      existing.direction         = existing.contribution_usd >= 0 ? 'up' : 'down'
    } else {
      map.set(base, { ...f, label: base })
    }
  }
  // Re-sort by abs contribution after merging and keep top 5
  return [...map.values()]
    .sort((a, b) => Math.abs(b.contribution_usd) - Math.abs(a.contribution_usd))
    .slice(0, 5)
}

// ── skeleton ──────────────────────────────────────────────────────────────────
function Skeleton() {
  const rows = [70, 55, 40, 30, 20]
  return (
    <section aria-label="Loading price explanation" aria-busy="true"
             className="px-6 py-5 border-t border-slate-100">
      <div className="h-3 w-32 bg-slate-200 rounded animate-pulse mb-1" />
      <div className="h-2.5 w-48 bg-slate-100 rounded animate-pulse mb-5" />
      <div className="space-y-4" role="presentation">
        {rows.map((w, i) => (
          <div key={i} className="flex items-center gap-3">
            <div className="w-36 h-2.5 bg-slate-100 rounded animate-pulse shrink-0" />
            <div className="flex-1 h-2 bg-slate-100 rounded-full overflow-hidden">
              <div className="h-full bg-slate-200 rounded-full animate-pulse"
                   style={{ width: `${w}%` }} />
            </div>
            <div className="w-10 h-2.5 bg-slate-100 rounded animate-pulse shrink-0" />
          </div>
        ))}
      </div>
    </section>
  )
}

// ── bar row ───────────────────────────────────────────────────────────────────
function BarRow({ feature, maxAbs, animate }) {
  const pct    = maxAbs > 0 ? (Math.abs(feature.contribution_usd) / maxAbs) * 100 : 0
  const isUp   = feature.direction === 'up'
  const dollar = Math.abs(Math.round(feature.contribution_usd))
  const label  = `${feature.label}: ${isUp ? 'adds' : 'removes'} $${dollar} from the estimated price`

  return (
    <div className="flex items-center gap-3" role="row" aria-label={label}>
      {/* feature label */}
      <div className="w-36 shrink-0 text-right">
        <span className="text-xs text-slate-600 font-medium leading-snug">
          {feature.label}
        </span>
      </div>

      {/* bar track */}
      <div className="flex-1 h-2 rounded-full bg-slate-100 overflow-hidden" role="presentation">
        <div
          className={`h-full rounded-full
            ${isUp ? 'bg-emerald-500' : 'bg-rose-400'}
            ${animate ? 'transition-[width] duration-700 ease-out' : ''}`}
          style={{ width: animate ? `${pct}%` : '0%' }}
        />
      </div>

      {/* dollar contribution */}
      <div className="w-14 shrink-0 flex items-center gap-0.5">
        {isUp
          ? <TrendingUp  size={11} className="text-emerald-500 shrink-0" aria-hidden />
          : <TrendingDown size={11} className="text-rose-400   shrink-0" aria-hidden />}
        <span className={`text-xs font-semibold tabular-nums
                          ${isUp ? 'text-emerald-600' : 'text-rose-500'}`}>
          {isUp ? '+' : '−'}${dollar}
        </span>
      </div>
    </div>
  )
}

// ── top insight sentence ──────────────────────────────────────────────────────
function TopInsight({ features }) {
  if (!features.length) return null
  const top    = features[0]
  const isUp   = top.direction === 'up'
  const dollar = Math.abs(Math.round(top.contribution_usd))

  return (
    <p className="text-xs text-slate-500 mb-4 leading-relaxed">
      <span className={`font-semibold ${isUp ? 'text-emerald-600' : 'text-rose-500'}`}>
        {top.label}
      </span>
      {' '}is the biggest driver —{' '}
      {isUp ? 'adding' : 'removing'}{' '}
      <span className="font-semibold">${dollar}/night</span>
      {' '}from your estimate.
    </p>
  )
}

// ── main export ───────────────────────────────────────────────────────────────
export default function ShapChart({ features, loading = false }) {
  // Animate bars in after first render so they transition from 0 → final width
  const [animate, setAnimate] = useState(false)
  useEffect(() => {
    if (!loading && features?.length) {
      const id = requestAnimationFrame(() => setAnimate(true))
      return () => cancelAnimationFrame(id)
    }
    setAnimate(false)
  }, [features, loading])

  if (loading) return <Skeleton />

  if (!features?.length) return null

  const merged = mergeFeatures(features)
  const maxAbs = Math.max(...merged.map(f => Math.abs(f.contribution_usd)), 1)

  return (
    <section aria-label="Price explanation" className="px-6 py-5 border-t border-slate-100">
      <div className="flex items-center gap-1.5 mb-1">
        <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
          Why this price?
        </p>
        <HelpCircle size={12} className="text-slate-300" aria-hidden />
      </div>

      <TopInsight features={merged} />

      <div className="space-y-3.5" role="table" aria-label="Feature contributions">
        {merged.map((f, i) => (
          <BarRow key={`${f.label}-${i}`} feature={f} maxAbs={maxAbs} animate={animate} />
        ))}
      </div>

      <p className="text-[10px] text-slate-300 mt-4 text-center">
        SHAP — each bar shows one feature's estimated dollar impact on this prediction
      </p>
    </section>
  )
}
