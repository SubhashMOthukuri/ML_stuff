import { ChevronDown } from 'lucide-react'

const ROOM_TYPES = ['Entire home/apt', 'Private room']

const BOROUGHS = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']

const BOROUGH_NEIGHBOURHOODS = {
  Manhattan:     ['Midtown', "Hell's Kitchen", 'Harlem', 'Upper East Side', 'Upper West Side',
                  'Lower East Side', 'Chelsea', 'Greenwich Village', 'SoHo', 'Tribeca',
                  'Financial District', 'East Village', 'West Village', 'Battery Park City',
                  'Murray Hill', 'Gramercy', 'Washington Heights', 'Morningside Heights'],
  Brooklyn:      ['Williamsburg', 'Brooklyn Heights', 'Park Slope', 'Bushwick', 'DUMBO',
                  'Bedford-Stuyvesant', 'Crown Heights', 'Flatbush', 'Greenpoint',
                  'Carroll Gardens', 'Cobble Hill', 'Bay Ridge', 'Sunset Park', 'Prospect Heights'],
  Queens:        ['Astoria', 'Long Island City', 'Flushing', 'Jackson Heights',
                  'Forest Hills', 'Jamaica', 'Woodside', 'Ridgewood', 'Elmhurst', 'Bayside'],
  Bronx:         ['Fordham', 'Mott Haven', 'Riverdale', 'Pelham Bay', 'Tremont', 'Belmont'],
  'Staten Island': ['St. George', 'Stapleton', 'Tottenville', 'Great Kills', 'Arrochar'],
}

// ── atoms ─────────────────────────────────────────────────────────────────────

function Label({ children, hint }) {
  return (
    <div className="flex items-baseline justify-between mb-1.5">
      <label className="text-sm font-medium text-slate-700">{children}</label>
      {hint && <span className="text-xs text-slate-400">{hint}</span>}
    </div>
  )
}

function Select({ value, onChange, options, placeholder }) {
  return (
    <div className="relative">
      <select
        value={value}
        onChange={e => onChange(e.target.value)}
        className="w-full bg-white border border-slate-300 rounded-xl px-3.5 py-2.5
                   text-slate-900 text-sm appearance-none cursor-pointer
                   focus:outline-none focus:ring-2 focus:ring-sky-500 focus:border-transparent transition-shadow"
      >
        {placeholder && <option value="">{placeholder}</option>}
        {options.map(o => <option key={o} value={o}>{o}</option>)}
      </select>
      <ChevronDown size={14} className="absolute right-3.5 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none" />
    </div>
  )
}

function Slider({ value, onChange, min, max, step = 1, format }) {
  const pct = ((value - min) / (max - min)) * 100
  return (
    <div>
      <div className="flex justify-between items-center mb-2">
        <span className="text-xs text-slate-400">{format ? format(min) : min}</span>
        <span className="text-sm font-semibold text-slate-900 tabular-nums">
          {format ? format(value) : value}
        </span>
        <span className="text-xs text-slate-400">{format ? format(max) : max}</span>
      </div>
      <div className="relative h-5 flex items-center">
        <div className="absolute h-1.5 w-full rounded-full bg-slate-200" />
        <div className="absolute h-1.5 rounded-full bg-sky-600 pointer-events-none"
             style={{ width: `${pct}%` }} />
        <input
          type="range" min={min} max={max} step={step} value={value}
          onChange={e => onChange(Number(e.target.value))}
          className="relative w-full accent-sky-600 cursor-pointer"
          style={{ background: 'transparent' }}
        />
      </div>
    </div>
  )
}

// ── main ──────────────────────────────────────────────────────────────────────

export default function ListingForm({ form, onChange, onSubmit, loading }) {
  const neighbourhoods = BOROUGH_NEIGHBOURHOODS[form.borough] ?? []
  const set = (key, val) => onChange({ ...form, [key]: val })

  return (
    <form onSubmit={e => { e.preventDefault(); onSubmit() }}
          className="bg-white rounded-2xl border border-slate-200 divide-y divide-slate-100">

      {/* ── Location ── */}
      <div className="px-6 py-5 space-y-4">
        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Location</p>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <Label>Borough</Label>
            <Select
              value={form.borough}
              onChange={v => onChange({ ...form, borough: v, neighbourhood: '' })}
              options={BOROUGHS}
            />
          </div>
          <div>
            <Label hint="optional">Neighbourhood</Label>
            <Select
              value={form.neighbourhood}
              onChange={v => set('neighbourhood', v)}
              options={neighbourhoods}
              placeholder="Any neighbourhood"
            />
          </div>
        </div>

        <div>
          <Label>Property type</Label>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {ROOM_TYPES.map(type => (
              <button
                key={type}
                type="button"
                onClick={() => set('room_type', type)}
                className={`py-2.5 rounded-xl border text-sm font-medium transition-all
                  ${form.room_type === type
                    ? 'bg-sky-50 border-sky-400 text-sky-700'
                    : 'bg-white border-slate-200 text-slate-500 hover:border-slate-300'}`}
              >
                {type}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* ── Property details ── */}
      <div className="px-6 py-5 space-y-5">
        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Property details</p>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
          <div>
            <Label>Bedrooms</Label>
            <Slider
              value={form.bedrooms} onChange={v => set('bedrooms', v)}
              min={0} max={10}
              format={v => v === 0 ? 'Studio' : v === 1 ? '1 bed' : `${v} beds`}
            />
          </div>
          <div>
            <Label>Bathrooms</Label>
            <Slider
              value={form.bathrooms} onChange={v => set('bathrooms', v)}
              min={1} max={10} step={0.5}
              format={v => v === 1 ? '1 bath' : `${v} baths`}
            />
          </div>
          <div>
            <Label>Min. stay</Label>
            <Slider
              value={form.minimum_nights} onChange={v => set('minimum_nights', v)}
              min={1} max={30}
              format={v => v === 1 ? '1 night' : `${v} nights`}
            />
          </div>
        </div>
      </div>

      {/* ── Reviews ── */}
      <div className="px-6 py-5 space-y-5">
        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Reviews</p>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
          <div>
            <Label>Overall rating</Label>
            <Slider
              value={form.review_scores_rating} onChange={v => set('review_scores_rating', v)}
              min={0} max={5} step={0.1}
              format={v => v === 0 ? 'No reviews yet' : `${v.toFixed(1)} ★`}
            />
          </div>
          <div>
            <Label>Total reviews</Label>
            <Slider
              value={form.number_of_reviews} onChange={v => set('number_of_reviews', v)}
              min={0} max={500}
              format={v => v === 0 ? 'None yet' : `${v}`}
            />
          </div>
        </div>
      </div>

      {/* ── Submit ── */}
      <div className="px-6 py-5">
        <button
          type="submit"
          disabled={loading}
          className={`w-full py-3 rounded-xl font-semibold text-sm transition-all
            ${loading
              ? 'bg-slate-100 text-slate-400 cursor-not-allowed'
              : 'bg-sky-700 hover:bg-sky-800 text-white shadow-sm active:scale-[0.99]'}`}
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
              </svg>
              Estimating…
            </span>
          ) : 'Estimate nightly price'}
        </button>
      </div>
    </form>
  )
}
