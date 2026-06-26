import { MapPin, Home, Users, Star, TrendingUp, ArrowRight } from 'lucide-react'

export default function PredictionResult({ result, form }) {
  if (!result) return null

  const mae = 57.62
  const lo  = Math.max(0, result.price_usd - mae)
  const hi  = result.price_usd + mae

  return (
    <div className="space-y-4">

      {/* ── main price card ── */}
      <div className="relative rounded-3xl overflow-hidden">
        {/* background layers */}
        <div className="absolute inset-0 bg-gradient-to-br from-rose-700 via-rose-600 to-rose-800" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_50%_at_-20%_-20%,rgba(255,255,255,0.08),transparent)]" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_60%_60%_at_120%_120%,rgba(0,0,0,0.3),transparent)]" />

        <div className="relative p-7">
          <p className="text-rose-200/60 text-sm font-medium mb-1">Predicted nightly price</p>

          <div className="flex items-end justify-between mb-6">
            <div>
              <p className="text-white text-6xl font-black tracking-tight leading-none">
                {result.price_str}
              </p>
              <p className="text-rose-200/50 text-sm mt-2">
                {result.model} · log_price = {result.log_price}
              </p>
            </div>
            <div className="text-right">
              <p className="text-white text-4xl font-black">{Math.round(result.r2_test * 100)}%</p>
              <p className="text-rose-200/50 text-xs">R² score</p>
            </div>
          </div>

          {/* confidence range */}
          <div className="bg-black/20 rounded-2xl p-4">
            <p className="text-rose-200/50 text-xs font-medium mb-3">
              Confidence range · ±MAE ${mae}/night
            </p>
            <div className="flex items-center gap-4">
              <div className="text-center">
                <p className="text-white/60 text-xs">Low</p>
                <p className="text-white font-bold text-lg">${lo.toFixed(0)}</p>
              </div>
              <div className="flex-1 relative h-2 bg-white/10 rounded-full overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-r from-white/20 via-white/40 to-white/20 rounded-full" />
                <div className="absolute top-1/2 left-1/2 w-3 h-3 -translate-x-1/2 -translate-y-1/2
                                bg-white rounded-full shadow-lg" />
              </div>
              <div className="text-center">
                <p className="text-white/60 text-xs">High</p>
                <p className="text-white font-bold text-lg">${hi.toFixed(0)}</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ── listing summary ── */}
      <div className="rounded-2xl border border-white/8 bg-white/[0.03] p-5">
        <p className="text-white/30 text-xs font-semibold uppercase tracking-widest mb-4">Your listing</p>
        <div className="grid grid-cols-2 gap-3">
          <SummaryChip icon={MapPin} label="Location"
            value={form.borough}
            sub={form.neighbourhood || 'Borough average'} />
          <SummaryChip icon={Home} label="Type"
            value={form.room_type}
            sub={form.is_private_bath ? 'Private bathroom' : 'Shared bathroom'} />
          <SummaryChip icon={Users} label="Capacity"
            value={`${form.accommodates} guests`}
            sub={`${form.bedrooms} bed · ${form.bathrooms} bath`} />
          <SummaryChip icon={Star} label="Reviews"
            value={form.review_scores_rating > 0 ? `${form.review_scores_rating.toFixed(1)} ★` : 'No rating'}
            sub={`${form.number_of_reviews} total reviews`} />
        </div>
      </div>
    </div>
  )
}

function SummaryChip({ icon: Icon, label, value, sub }) {
  return (
    <div className="flex items-center gap-3 p-3 rounded-xl bg-white/4 border border-white/6">
      <div className="w-9 h-9 rounded-xl bg-rose-600/15 flex items-center justify-center flex-shrink-0">
        <Icon size={14} className="text-rose-400" />
      </div>
      <div className="min-w-0">
        <p className="text-white/40 text-xs">{label}</p>
        <p className="text-white text-sm font-semibold truncate">{value}</p>
        <p className="text-white/30 text-xs truncate">{sub}</p>
      </div>
    </div>
  )
}
