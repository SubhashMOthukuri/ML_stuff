import { MapPin, Home, Users, Star } from 'lucide-react'

const RULE_LABELS = {
  shared_room_cap:         'Capped for shared room type',
  hotel_floor:             'Hotel room minimum applied',
  severe_overcrowding:     'Adjusted for high guest count',
  overcrowding:            'Adjusted for guest count',
  superhost_premium:       'Superhost premium applied',
  budget_listing_floor:    'Budget listing floor applied',
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
  const appliedLabels = (business_rules_applied ?? [])
    .map(friendlyRule)
    .filter(Boolean)

  return (
    <div className="space-y-4">

      {/* ── price card ── */}
      <div className="bg-white rounded-2xl border border-slate-200 overflow-hidden">
        <div className="bg-emerald-50 border-b border-emerald-100 px-6 py-5">
          <p className="text-emerald-700 text-xs font-semibold uppercase tracking-wider mb-1">
            Estimated nightly price
          </p>
          <p className="text-4xl font-black text-emerald-800 tracking-tight leading-none">
            ${Math.round(price_usd)}<span className="text-emerald-600 text-lg font-semibold">/night</span>
          </p>
        </div>

        {/* price range */}
        <div className="px-6 py-4">
          <p className="text-xs font-medium text-slate-500 mb-3">Your likely range</p>
          <div className="flex items-center gap-3">
            <div className="text-center">
              <p className="text-slate-400 text-[11px] mb-0.5">Low</p>
              <p className="text-slate-700 font-bold text-lg tabular-nums">${Math.round(price_low)}</p>
            </div>
            <div className="flex-1 relative h-2 rounded-full bg-slate-100">
              <div className="absolute inset-0 rounded-full bg-gradient-to-r from-slate-200 via-emerald-400 to-slate-200" />
              <div
                className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-3.5 h-3.5
                           rounded-full bg-white border-2 border-emerald-500 shadow"
                style={{ left: `${((price_usd - price_low) / (price_high - price_low + 0.01)) * 100}%` }}
              />
            </div>
            <div className="text-center">
              <p className="text-slate-400 text-[11px] mb-0.5">High</p>
              <p className="text-slate-700 font-bold text-lg tabular-nums">${Math.round(price_high)}</p>
            </div>
          </div>
          <p className="text-xs text-slate-400 mt-3">
            Range based on ±15% variance typical for this listing type.
          </p>
        </div>

        {/* adjustments applied */}
        {appliedLabels.length > 0 && (
          <div className="px-6 pb-4">
            <div className="rounded-xl bg-amber-50 border border-amber-100 px-4 py-3">
              <p className="text-xs font-semibold text-amber-700 mb-1.5">Adjustments applied</p>
              {appliedLabels.map((label, i) => (
                <p key={i} className="text-xs text-amber-600 flex items-center gap-1.5">
                  <span className="w-1 h-1 rounded-full bg-amber-400 flex-shrink-0" />
                  {label}
                </p>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* ── listing summary ── */}
      <div className="bg-white rounded-2xl border border-slate-200 px-6 py-5">
        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4">Your listing</p>
        <div className="grid grid-cols-2 gap-3">
          <SummaryRow icon={MapPin} label="Location"
            value={form.borough}
            sub={form.neighbourhood || 'Borough average'} />
          <SummaryRow icon={Home} label="Type"
            value={form.room_type}
            sub={`${form.bedrooms} bed · ${form.bathrooms} bath`} />
          <SummaryRow icon={Users} label="Capacity"
            value={`${form.accommodates} guests`}
            sub={`Min. ${form.minimum_nights} night${form.minimum_nights > 1 ? 's' : ''}`} />
          <SummaryRow icon={Star} label="Rating"
            value={form.review_scores_rating > 0 ? `${form.review_scores_rating.toFixed(1)} ★` : 'No reviews'}
            sub={`${form.number_of_reviews} review${form.number_of_reviews !== 1 ? 's' : ''}`} />
        </div>
      </div>

      <p className="text-center text-xs text-slate-400">
        Estimate based on 20,000+ real NYC Airbnb listings · prices vary by season and availability
      </p>
    </div>
  )
}

function SummaryRow({ icon: Icon, label, value, sub }) {
  return (
    <div className="flex items-start gap-3 p-3 rounded-xl bg-slate-50">
      <div className="w-7 h-7 rounded-lg bg-white border border-slate-200 flex items-center justify-center flex-shrink-0 mt-0.5">
        <Icon size={13} className="text-slate-500" />
      </div>
      <div className="min-w-0">
        <p className="text-slate-400 text-[11px]">{label}</p>
        <p className="text-slate-800 text-sm font-semibold truncate leading-tight">{value}</p>
        <p className="text-slate-400 text-[11px] truncate">{sub}</p>
      </div>
    </div>
  )
}
