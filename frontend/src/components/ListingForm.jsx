import { useState } from 'react'
import { Home, MapPin, Users, Star, Wifi, ChevronDown } from 'lucide-react'

const BOROUGHS = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
const ROOM_TYPES = ['Entire home/apt', 'Private room', 'Hotel room', 'Shared room']

const BOROUGH_NEIGHBOURHOODS = {
  Manhattan: ['Midtown', "Hell's Kitchen", 'Harlem', 'Upper East Side', 'Upper West Side',
    'Lower East Side', 'Chelsea', 'Greenwich Village', 'SoHo', 'Tribeca',
    'Financial District', 'East Village', 'West Village', 'Morningside Heights',
    'Washington Heights', 'Inwood', 'Battery Park City', 'Murray Hill', 'Gramercy'],
  Brooklyn: ['Williamsburg', 'Brooklyn Heights', 'Park Slope', 'Bushwick',
    'Bedford-Stuyvesant', 'Crown Heights', 'Flatbush', 'DUMBO', 'Greenpoint',
    'Carroll Gardens', 'Cobble Hill', 'Bay Ridge', 'Sunset Park', 'Prospect Heights'],
  Queens: ['Astoria', 'Long Island City', 'Flushing', 'Jackson Heights',
    'Forest Hills', 'Jamaica', 'Woodside', 'Ridgewood', 'Elmhurst', 'Bayside'],
  Bronx: ['Fordham', 'Mott Haven', 'Riverdale', 'Pelham Bay', 'Tremont', 'Belmont'],
  'Staten Island': ['St. George', 'Stapleton', 'Tottenville', 'Great Kills', 'Arrochar'],
}

const AMENITIES = [
  { key: 'has_air_conditioning', label: 'A/C' },
  { key: 'has_washer',           label: 'Washer' },
  { key: 'has_dryer',            label: 'Dryer' },
  { key: 'has_elevator',         label: 'Elevator' },
  { key: 'has_gym',              label: 'Gym' },
  { key: 'has_pool',             label: 'Pool' },
]

// ── sub-components ────────────────────────────────────────────────────────────

function Label({ children, hint }) {
  return (
    <div className="flex items-center justify-between mb-1.5">
      <label className="text-white/70 text-sm font-medium">{children}</label>
      {hint && <span className="text-white/30 text-xs">{hint}</span>}
    </div>
  )
}

function Select({ value, onChange, options, placeholder }) {
  return (
    <div className="relative">
      <select
        value={value}
        onChange={e => onChange(e.target.value)}
        className="w-full bg-white/8 border border-white/10 rounded-xl px-4 py-3 text-white text-sm
                   appearance-none cursor-pointer focus:outline-none focus:border-rose-500/60
                   focus:bg-white/12 transition-all"
      >
        {placeholder && <option value="" disabled>{placeholder}</option>}
        {options.map(o => (
          <option key={o} value={o} className="bg-neutral-900">{o}</option>
        ))}
      </select>
      <ChevronDown size={14} className="absolute right-3 top-1/2 -translate-y-1/2 text-white/30 pointer-events-none" />
    </div>
  )
}

function Slider({ value, onChange, min, max, step = 1, unit = '' }) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-white/40 text-xs">{min}{unit}</span>
        <span className="text-white font-semibold text-sm bg-white/10 px-2.5 py-0.5 rounded-lg">
          {value}{unit}
        </span>
        <span className="text-white/40 text-xs">{max}{unit}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        className="w-full h-1.5 rounded-full appearance-none cursor-pointer accent-rose-500 bg-white/10"
      />
    </div>
  )
}

function Toggle({ checked, onChange, label }) {
  return (
    <button
      type="button"
      onClick={() => onChange(!checked)}
      className={`flex items-center gap-2.5 px-3.5 py-2.5 rounded-xl border text-sm font-medium transition-all
        ${checked
          ? 'bg-rose-600/25 border-rose-500/40 text-rose-300'
          : 'bg-white/5 border-white/10 text-white/50 hover:border-white/20'}`}
    >
      <div className={`w-4 h-4 rounded-md border-2 flex items-center justify-center transition-all
        ${checked ? 'bg-rose-500 border-rose-500' : 'border-white/30'}`}>
        {checked && <svg width="8" height="8" viewBox="0 0 8 8" fill="none">
          <path d="M1 4L3 6L7 2" stroke="white" strokeWidth="1.5" strokeLinecap="round"/>
        </svg>}
      </div>
      {label}
    </button>
  )
}

function Section({ icon: Icon, title, children }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/4 p-5 space-y-4">
      <div className="flex items-center gap-2.5">
        <div className="w-7 h-7 rounded-lg bg-rose-600/20 flex items-center justify-center">
          <Icon size={13} className="text-rose-400" />
        </div>
        <h3 className="text-white font-semibold text-sm">{title}</h3>
      </div>
      {children}
    </div>
  )
}

// ── main form ─────────────────────────────────────────────────────────────────

export default function ListingForm({ form, onChange, onSubmit, loading }) {
  const [showAdvanced, setShowAdvanced] = useState(false)
  const neighbourhoods = BOROUGH_NEIGHBOURHOODS[form.borough] ?? []

  const set = (key, value) => onChange({ ...form, [key]: value })

  return (
    <form onSubmit={e => { e.preventDefault(); onSubmit() }} className="space-y-4">

      {/* ── type & location ── */}
      <Section icon={MapPin} title="Type & Location">
        <div className="grid grid-cols-2 gap-3">
          <div>
            <Label>Room type</Label>
            <Select value={form.room_type} onChange={v => set('room_type', v)} options={ROOM_TYPES} />
          </div>
          <div>
            <Label>Borough</Label>
            <Select value={form.borough} onChange={v => { set('borough', v); set('neighbourhood', '') }}
                    options={BOROUGHS} />
          </div>
        </div>
        <div>
          <Label hint="optional">Neighbourhood</Label>
          <Select value={form.neighbourhood}
                  onChange={v => set('neighbourhood', v)}
                  options={['', ...neighbourhoods]}
                  placeholder="Use borough average" />
        </div>
        <div>
          <Label hint={`${form.is_private_bath ? '#1 most important feature' : 'lowers price'}`}>
            Bathroom
          </Label>
          <div className="flex gap-2">
            <Toggle checked={form.is_private_bath}  onChange={v => set('is_private_bath', v)}  label="Private" />
            <Toggle checked={!form.is_private_bath} onChange={v => set('is_private_bath', !v)} label="Shared" />
          </div>
        </div>
      </Section>

      {/* ── capacity ── */}
      <Section icon={Users} title="Capacity">
        <div>
          <Label hint={`${form.accommodates} guests`}>Guests</Label>
          <Slider value={form.accommodates} onChange={v => set('accommodates', v)} min={1} max={16} />
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <Label hint={`${form.bedrooms} bed`}>Bedrooms</Label>
            <Slider value={form.bedrooms} onChange={v => set('bedrooms', v)} min={0} max={10} />
          </div>
          <div>
            <Label hint={`${form.bathrooms} bath`}>Bathrooms</Label>
            <Slider value={form.bathrooms} onChange={v => set('bathrooms', v)} min={0} max={10} step={0.5} />
          </div>
        </div>
        <div>
          <Label>Minimum nights</Label>
          <Slider value={form.minimum_nights} onChange={v => set('minimum_nights', v)} min={1} max={30} />
        </div>
      </Section>

      {/* ── reviews ── */}
      <Section icon={Star} title="Reviews">
        <div>
          <Label hint={`${form.review_scores_rating}/5`}>Overall rating</Label>
          <Slider value={form.review_scores_rating} onChange={v => set('review_scores_rating', v)}
                  min={0} max={5} step={0.1} />
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <Label hint={form.number_of_reviews}>Total reviews</Label>
            <Slider value={form.number_of_reviews} onChange={v => set('number_of_reviews', v)} min={0} max={500} />
          </div>
          <div>
            <Label hint={`${form.reviews_per_month}/mo`}>Per month</Label>
            <Slider value={form.reviews_per_month} onChange={v => set('reviews_per_month', v)}
                    min={0} max={15} step={0.1} />
          </div>
        </div>
      </Section>

      {/* ── amenities ── */}
      <Section icon={Wifi} title="Amenities">
        <div>
          <Label hint={`${form.amenity_count} total`}>Amenity count</Label>
          <Slider value={form.amenity_count} onChange={v => set('amenity_count', v)} min={0} max={100} />
        </div>
        <div className="flex flex-wrap gap-2">
          {AMENITIES.map(({ key, label }) => (
            <Toggle key={key} checked={form[key]} onChange={v => set(key, v)} label={label} />
          ))}
        </div>
      </Section>

      {/* ── host (advanced) ── */}
      <button type="button" onClick={() => setShowAdvanced(s => !s)}
              className="w-full text-white/40 text-xs flex items-center justify-center gap-1.5 hover:text-white/60 transition-colors">
        <ChevronDown size={12} className={`transition-transform ${showAdvanced ? 'rotate-180' : ''}`} />
        {showAdvanced ? 'Hide' : 'Show'} host details
      </button>

      {showAdvanced && (
        <Section icon={Home} title="Host Details">
          <div className="flex items-center gap-3 mb-2">
            <Toggle checked={form.host_is_superhost} onChange={v => set('host_is_superhost', v)} label="Superhost" />
          </div>
          <div>
            <Label hint={form.host_listings_count}>Total host listings</Label>
            <Slider value={form.host_listings_count} onChange={v => set('host_listings_count', v)} min={1} max={50} />
          </div>
        </Section>
      )}

      {/* ── submit ── */}
      <button
        type="submit"
        disabled={loading}
        className={`w-full py-4 rounded-2xl font-bold text-base text-white transition-all
          ${loading
            ? 'bg-rose-600/40 cursor-not-allowed'
            : 'bg-rose-600 hover:bg-rose-500 active:scale-[0.98] shadow-lg shadow-rose-600/25'}`}
      >
        {loading ? (
          <span className="flex items-center justify-center gap-2">
            <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
            </svg>
            Predicting…
          </span>
        ) : 'Predict nightly price'}
      </button>
    </form>
  )
}
