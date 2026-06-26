import { TrendingUp, MapPin, Home, Users, Star, ChevronRight } from 'lucide-react'

export default function PredictionResult({ result, form }) {
  if (!result) return null

  const mae  = 57.62
  const lo   = Math.max(0, result.price_usd - mae)
  const hi   = result.price_usd + mae
  const pct  = Math.round(result.r2_test * 100)

  return (
    <div className="space-y-4 animate-in fade-in slide-in-from-bottom-4 duration-500">

      {/* main price card */}
      <div className="rounded-3xl overflow-hidden relative">
        <div className="absolute inset-0 bg-gradient-to-br from-rose-600 to-rose-900" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,rgba(255,255,255,0.1),transparent)]" />
        <div className="relative p-6">
          <div className="flex items-start justify-between mb-4">
            <div>
              <p className="text-rose-200/70 text-sm font-medium">Predicted nightly price</p>
              <p className="text-white text-5xl font-black tracking-tight mt-1">
                {result.price_str}
              </p>
            </div>
            <div className="bg-white/15 backdrop-blur rounded-2xl px-3 py-2 text-center">
              <p className="text-white font-bold text-xl">{pct}%</p>
              <p className="text-white/70 text-xs">R² confidence</p>
            </div>
          </div>

          {/* confidence range */}
          <div className="bg-black/20 rounded-2xl p-4">
            <p className="text-rose-200/60 text-xs mb-2 font-medium">Confidence range (±MAE ${mae}/night)</p>
            <div className="flex items-center gap-3">
              <span className="text-white/70 text-sm">${lo.toFixed(0)}</span>
              <div className="flex-1 relative h-2 bg-white/10 rounded-full">
                <div className="absolute h-full bg-white/30 rounded-full"
                     style={{ left: '10%', right: '10%' }} />
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2
                                w-4 h-4 bg-white rounded-full shadow-lg shadow-black/50" />
              </div>
              <span className="text-white/70 text-sm">${hi.toFixed(0)}</span>
            </div>
          </div>
        </div>
      </div>

      {/* listing snapshot */}
      <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
        <p className="text-white/50 text-xs font-medium uppercase tracking-wider mb-3">Your listing</p>
        <div className="grid grid-cols-2 gap-2">
          <Chip icon={MapPin}  label={form.borough} sub={form.neighbourhood || 'Borough avg'} />
          <Chip icon={Home}    label={form.room_type} sub={form.is_private_bath ? 'Private bath' : 'Shared bath'} />
          <Chip icon={Users}   label={`${form.accommodates} guests`} sub={`${form.bedrooms} bed · ${form.bathrooms} bath`} />
          <Chip icon={Star}    label={form.review_scores_rating > 0 ? `${form.review_scores_rating}/5 rating` : 'No reviews yet'}
                               sub={`${form.number_of_reviews} reviews`} />
        </div>
      </div>

      {/* model detail */}
      <div className="rounded-2xl border border-white/10 bg-white/5 p-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <TrendingUp size={14} className="text-rose-400" />
          <span className="text-white/60 text-xs">
            {result.model} · log_price={result.log_price}
          </span>
        </div>
        <ChevronRight size={14} className="text-white/20" />
      </div>
    </div>
  )
}

function Chip({ icon: Icon, label, sub }) {
  return (
    <div className="flex items-center gap-3 bg-white/5 rounded-xl p-3">
      <div className="w-8 h-8 rounded-lg bg-rose-600/20 flex items-center justify-center flex-shrink-0">
        <Icon size={13} className="text-rose-400" />
      </div>
      <div className="min-w-0">
        <p className="text-white text-sm font-medium truncate">{label}</p>
        <p className="text-white/40 text-xs truncate">{sub}</p>
      </div>
    </div>
  )
}
