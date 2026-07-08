import { MapPin, Home, Users, Star } from 'lucide-react'

const RULE_LABELS = {
  shared_room_cap:     'Capped for shared room type',
  hotel_floor:         'Hotel room minimum applied',
  superhost_premium:   'Superhost premium applied',
  budget_listing_floor:'Budget listing floor applied',
}

function friendlyRule(code) {
  for (const [key, label] of Object.entries(RULE_LABELS)) {
    if (code.startsWith(key)) return label
  }
  return null
}

export default function PredictionResult({ result, form }) {
  if (!result) return null

  const { price_usd, price_low, price_high, business_rules_applied } = result
  const appliedLabels = (business_rules_applied ?? []).map(friendlyRule).filter(Boolean)
  const rangePct = ((price_usd - price_low) / (price_high - price_low + 0.01)) * 100

  return (
    <div className="bg-white rounded-2xl border border-slate-200 overflow-hidden">

      {/* price header */}
      <div className="bg-emerald-600 px-6 py-6">
        <p className="text-emerald-100 text-xs font-semibold uppercase tracking-wider mb-1">
          Estimated nightly rate
        </p>
        <div className="flex items-end gap-2 mb-3">
          <span className="text-5xl font-black text-white leading-none">
            ${Math.round(price_usd)}
          </span>
          <span className="text-emerald-200 text-lg font-medium mb-0.5">/night</span>
        </div>
        {form.minimum_nights > 1 && (
          <div className="bg-emerald-700/60 rounded-xl px-4 py-2.5 flex items-center justify-between">
            <span className="text-emerald-100 text-sm">
              Total for {form.minimum_nights} nights
            </span>
            <span className="text-white font-bold text-lg tabular-nums">
              ${Math.round(price_usd * form.minimum_nights).toLocaleString()}
            </span>
          </div>
        )}
      </div>

      {/* range bar */}
      <div className="px-6 py-5 border-b border-slate-100">
        <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-4">
          Likely range
        </p>
        <div className="flex items-center gap-4">
          <div className="text-center w-14 shrink-0">
            <p className="text-[11px] text-slate-400 mb-0.5">Low</p>
            <p className="text-slate-700 font-bold tabular-nums">${Math.round(price_low)}</p>
          </div>
          <div className="flex-1 relative h-2 rounded-full bg-slate-100">
            <div className="absolute inset-0 rounded-full bg-gradient-to-r from-slate-200 via-emerald-400 to-slate-200" />
            <div
              className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-4 h-4
                         rounded-full bg-white border-2 border-emerald-500 shadow-sm"
              style={{ left: `${rangePct}%` }}
            />
          </div>
          <div className="text-center w-14 shrink-0">
            <p className="text-[11px] text-slate-400 mb-0.5">High</p>
            <p className="text-slate-700 font-bold tabular-nums">${Math.round(price_high)}</p>
          </div>
        </div>
        <p className="text-xs text-slate-400 mt-3 text-center">
          Range reflects typical ±15% variance for this listing type in NYC
        </p>
      </div>

      {/* listing summary */}
      <div className="px-6 py-5 border-b border-slate-100">
        <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
          Your listing
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          <SummaryRow icon={MapPin}
            label="Location"
            value={form.borough}
            sub={form.neighbourhood || 'Borough average'} />
          <SummaryRow icon={Home}
            label="Type"
            value={form.room_type}
            sub={`${form.bedrooms === 0 ? 'Studio' : `${form.bedrooms} bed`} · ${form.bathrooms} bath`} />
          <SummaryRow icon={Users}
            label="Stay"
            value={`Min. ${form.minimum_nights} night${form.minimum_nights > 1 ? 's' : ''}`}
            sub="" />
          <SummaryRow icon={Star}
            label="Rating"
            value={form.review_scores_rating > 0 ? `${form.review_scores_rating.toFixed(1)} ★` : 'No reviews'}
            sub={`${form.number_of_reviews} review${form.number_of_reviews !== 1 ? 's' : ''}`} />
        </div>
      </div>

      {/* adjustments */}
      {appliedLabels.length > 0 && (
        <div className="px-6 py-4 border-b border-slate-100 bg-amber-50">
          <p className="text-xs font-semibold text-amber-700 mb-2">Adjustments applied</p>
          {appliedLabels.map((label, i) => (
            <p key={i} className="text-xs text-amber-600 flex items-center gap-2">
              <span className="w-1 h-1 rounded-full bg-amber-400 shrink-0" />
              {label}
            </p>
          ))}
        </div>
      )}

      {/* footer */}
      <div className="px-6 py-4 bg-slate-50">
        <p className="text-xs text-slate-400 text-center">
          Based on 20,000+ real NYC Airbnb listings · prices vary by season and demand
        </p>
      </div>
    </div>
  )
}

function SummaryRow({ icon: Icon, label, value, sub }) {
  return (
    <div className="flex items-center gap-3 px-3 py-2.5 rounded-xl bg-slate-50">
      <div className="w-7 h-7 rounded-lg bg-white border border-slate-200 flex items-center justify-center shrink-0">
        <Icon size={13} className="text-slate-400" />
      </div>
      <div className="min-w-0">
        <p className="text-[11px] text-slate-400">{label}</p>
        <p className="text-sm font-semibold text-slate-800 truncate leading-tight">{value}</p>
        {sub && <p className="text-[11px] text-slate-400 truncate">{sub}</p>}
      </div>
    </div>
  )
}
