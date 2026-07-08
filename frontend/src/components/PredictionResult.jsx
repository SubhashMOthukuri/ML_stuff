import { MapPin, Home, Users, Star, Zap, CheckCircle } from 'lucide-react'

export default function PredictionResult({ result, form }) {
  if (!result) return null

  const mae = 57.62
  const lo  = Math.max(0, result.price_usd - mae)
  const hi  = result.price_usd + mae
  const pct = ((result.price_usd - lo) / (hi - lo + 0.001)) * 100

  return (
    <div className="space-y-4">

      {/* ── price card ── */}
      <div className="relative rounded-2xl overflow-hidden border border-rose-500/20">
        <div className="absolute inset-0 bg-gradient-to-br from-rose-950 via-rose-900/80 to-slate-950" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_70%_60%_at_10%_0%,rgba(244,63,94,0.15),transparent)]" />

        <div className="relative p-6">
          {/* top row */}
          <div className="flex items-start justify-between mb-5">
            <div>
              <p className="text-rose-300/60 text-xs font-medium uppercase tracking-widest mb-1">
                Predicted nightly price
              </p>
              <p className="text-white text-5xl font-black tracking-tight leading-none">
                {result.price_str}
              </p>
            </div>
            <div className="flex flex-col items-end gap-2">
              <div className="bg-white/10 rounded-xl px-3 py-2 text-center">
                <p className="text-white text-xl font-black leading-none">{Math.round(result.r2_test * 100)}%</p>
                <p className="text-white/40 text-xs mt-0.5">R² score</p>
              </div>
              {result.cache_hit && (
                <div className="flex items-center gap-1.5 bg-emerald-500/15 border border-emerald-500/25
                                rounded-full px-2.5 py-1 text-emerald-300 text-xs font-medium">
                  <Zap size={10} />
                  Cache hit
                </div>
              )}
            </div>
          </div>

          {/* confidence range */}
          <div className="bg-black/30 border border-white/8 rounded-xl p-4">
            <p className="text-white/35 text-xs font-semibold uppercase tracking-widest mb-3">
              Confidence range · ±MAE ${mae}/night
            </p>
            <div className="flex items-center gap-3 mb-2">
              <span className="text-white/50 text-sm w-14 text-right">${Math.round(lo)}</span>
              <div className="flex-1 relative h-2 bg-white/10 rounded-full">
                <div className="absolute inset-0 bg-gradient-to-r from-rose-500/30 via-rose-400/50 to-rose-500/30 rounded-full" />
                <div
                  className="absolute top-1/2 w-3.5 h-3.5 -translate-y-1/2 -translate-x-1/2 bg-white rounded-full
                             shadow-lg shadow-rose-500/40 border-2 border-rose-300"
                  style={{ left: `${pct}%` }}
                />
              </div>
              <span className="text-white/50 text-sm w-14">${Math.round(hi)}</span>
            </div>
            <div className="flex justify-between text-white/25 text-xs px-1">
              <span>Low estimate</span>
              <span>High estimate</span>
            </div>
          </div>
        </div>
      </div>

      {/* ── listing summary ── */}
      <div className="rounded-2xl border border-white/8 bg-white/[0.025] p-5">
        <p className="text-white/30 text-xs font-semibold uppercase tracking-widest mb-4">Your listing</p>
        <div className="grid grid-cols-2 gap-2.5">
          <Chip icon={MapPin}
            label="Location"
            value={form.borough}
            sub={form.neighbourhood || 'Borough average'} />
          <Chip icon={Home}
            label="Type"
            value={form.room_type}
            sub={`${form.bedrooms} bd · ${form.bathrooms} ba`} />
          <Chip icon={Users}
            label="Capacity"
            value={`${form.accommodates} guests`}
            sub={`${form.bedrooms} bd · ${form.bathrooms} ba`} />
          <Chip icon={Star}
            label="Rating"
            value={form.review_scores_rating > 0 ? `${form.review_scores_rating.toFixed(1)} ★` : 'No reviews'}
            sub={`${form.number_of_reviews} reviews total`} />
        </div>
      </div>

      {/* ── technical detail ── */}
      <div className="flex items-center gap-3 px-4 py-2.5 rounded-xl bg-white/[0.02] border border-white/5">
        <CheckCircle size={13} className="text-emerald-400/60 flex-shrink-0" />
        <p className="text-white/25 text-xs font-mono">
          log_price={result.log_price} · rid={result.request_id?.slice(0, 8)}
        </p>
      </div>
    </div>
  )
}

function Chip({ icon: Icon, label, value, sub }) {
  return (
    <div className="flex items-center gap-3 p-3 rounded-xl bg-white/[0.035] border border-white/6">
      <div className="w-8 h-8 rounded-lg bg-rose-600/12 border border-rose-500/15 flex items-center justify-center flex-shrink-0">
        <Icon size={13} className="text-rose-400" />
      </div>
      <div className="min-w-0">
        <p className="text-white/35 text-xs">{label}</p>
        <p className="text-white text-sm font-semibold truncate leading-tight">{value}</p>
        <p className="text-white/30 text-xs truncate">{sub}</p>
      </div>
    </div>
  )
}
