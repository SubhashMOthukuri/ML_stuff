import { useState } from 'react'
import { ChevronDown } from 'lucide-react'

const BOROUGHS = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
const ROOM_TYPES = ['Entire home/apt', 'Private room', 'Hotel room', 'Shared room']

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

const AMENITIES = [
  { key: 'has_air_conditioning', label: 'Air conditioning' },
  { key: 'has_washer',           label: 'Washer' },
  { key: 'has_dryer',            label: 'Dryer' },
  { key: 'has_elevator',         label: 'Elevator' },
  { key: 'has_gym',              label: 'Gym' },
  { key: 'has_pool',             label: 'Pool' },
]

const BATH_BY_ROOM_TYPE = {
  'Entire home/apt': true,
  'Hotel room':      true,
  'Shared room':     false,
  'Private room':    false,
}

// ─── atoms ───────────────────────────────────────────────────────────────────

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
                   focus:outline-none focus:ring-2 focus:ring-sky-500 focus:border-transparent
                   transition-shadow"
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
  const display = format ? format(value) : value
  return (
    <div>
      <div className="flex justify-between items-center mb-2">
        <span className="text-xs text-slate-400">{format ? format(min) : min}</span>
        <span className="text-sm font-semibold text-slate-900 tabular-nums">{display}</span>
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

function Chip({ checked, onChange, label }) {
  return (
    <button
      type="button"
      onClick={() => onChange(!checked)}
      className={`px-3 py-1.5 rounded-lg border text-xs font-medium transition-all
        ${checked
          ? 'bg-sky-50 border-sky-300 text-sky-700'
          : 'bg-white border-slate-200 text-slate-500 hover:border-slate-300 hover:text-slate-700'}`}
    >
      {label}
    </button>
  )
}

function Divider() {
  return <div className="border-t border-slate-100" />
}

// ─── main ─────────────────────────────────────────────────────────────────────

export default function ListingForm({ form, onChange, onSubmit, loading }) {
  const [showHost, setShowHost] = useState(false)
  const neighbourhoods = BOROUGH_NEIGHBOURHOODS[form.borough] ?? []
  const set = (key, val) => onChange({ ...form, [key]: val })

  return (
    <form onSubmit={e => { e.preventDefault(); onSubmit() }}
          className="bg-white rounded-2xl border border-slate-200 divide-y divide-slate-100">

      {/* ── Type & Location ── */}
      <div className="px-6 py-5 space-y-4">
        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Location</p>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <Label>Room type</Label>
            <Select
              value={form.room_type}
              onChange={v => onChange({ ...form, room_type: v, is_private_bath: BATH_BY_ROOM_TYPE[v] })}
              options={ROOM_TYPES}
            />
          </div>
          <div>
            <Label>Borough</Label>
            <Select
              value={form.borough}
              onChange={v => onChange({ ...form, borough: v, neighbourhood: '' })}
              options={BOROUGHS}
            />
          </div>
        </div>

        <div>
          <Label hint="optional — improves accuracy">Neighbourhood</Label>
          <Select
            value={form.neighbourhood}
            onChange={v => set('neighbourhood', v)}
            options={neighbourhoods}
            placeholder="Use borough average"
          />
        </div>
      </div>

      {/* ── Capacity ── */}
      <div className="px-6 py-5 space-y-5">
        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Size & Capacity</p>

        <div>
          <Label>Guests</Label>
          <Slider
            value={form.accommodates} onChange={v => set('accommodates', v)}
            min={1} max={16}
            format={v => v === 1 ? '1 guest' : `${v} guests`}
          />
        </div>

        <div className="grid grid-cols-3 gap-6">
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
            <Label>Min. nights</Label>
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

        <div>
          <Label>Overall rating</Label>
          <Slider
            value={form.review_scores_rating} onChange={v => set('review_scores_rating', v)}
            min={0} max={5} step={0.1}
            format={v => v === 0 ? 'No reviews yet' : `${v.toFixed(1)} ★`}
          />
        </div>

        <div className="grid grid-cols-2 gap-6">
          <div>
            <Label>Total reviews</Label>
            <Slider value={form.number_of_reviews} onChange={v => set('number_of_reviews', v)} min={0} max={500} />
          </div>
          <div>
            <Label>Reviews / month</Label>
            <Slider value={form.reviews_per_month} onChange={v => set('reviews_per_month', v)}
                    min={0} max={15} step={0.1} format={v => v.toFixed(1)} />
          </div>
        </div>
      </div>

      {/* ── Amenities ── */}
      <div className="px-6 py-5 space-y-4">
        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Amenities</p>

        <div>
          <Label hint="total items listed on Airbnb · most NYC listings are 15–40">Total amenity count</Label>
          <Slider
            value={form.amenity_count} onChange={v => set('amenity_count', v)}
            min={5} max={100}
            format={v => {
              const tier = v <= 20 ? 'Budget' : v <= 40 ? 'Standard' : v <= 65 ? 'Premium' : 'Luxury'
              return `${v} items · ${tier}`
            }}
          />
        </div>

        <div>
          <Label>What does your listing include?</Label>
          <div className="flex flex-wrap gap-2 mt-1">
            {AMENITIES.map(({ key, label }) => (
              <Chip key={key} checked={form[key]} onChange={v => set(key, v)} label={label} />
            ))}
          </div>
        </div>
      </div>

      {/* ── Host (collapsible) ── */}
      <div>
        <button
          type="button"
          onClick={() => setShowHost(s => !s)}
          className="w-full flex items-center justify-between px-6 py-4
                     text-sm text-slate-500 hover:text-slate-700 transition-colors"
        >
          <span className="font-medium">Host details</span>
          <ChevronDown size={14} className={`transition-transform duration-200 ${showHost ? 'rotate-180' : ''}`} />
        </button>

        {showHost && (
          <div className="px-6 pb-5 space-y-5 border-t border-slate-100">
            <div className="pt-4">
              <Label>Are you a Superhost?</Label>
              <div className="flex gap-2 mt-1">
                <Chip checked={form.host_is_superhost} onChange={v => set('host_is_superhost', v)} label="Yes — Superhost" />
                <Chip checked={!form.host_is_superhost} onChange={v => set('host_is_superhost', !v)} label="No" />
              </div>
            </div>
            <div>
              <Label>Total listings you manage</Label>
              <Slider value={form.host_listings_count} onChange={v => set('host_listings_count', v)} min={1} max={50} />
            </div>
          </div>
        )}
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
