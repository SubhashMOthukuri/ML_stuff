import { useState } from 'react'
import { MapPin, Users, Star, Wifi, ChevronDown, Home, Link } from 'lucide-react'

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

// ─── atoms ────────────────────────────────────────────────────────────────────

function FieldLabel({ children, badge }) {
  return (
    <div className="flex items-center justify-between mb-3">
      <span className="text-white/80 text-sm font-medium tracking-wide">{children}</span>
      {badge && (
        <span className="text-xs px-2 py-0.5 rounded-full bg-rose-500/15 text-rose-300 font-medium">
          {badge}
        </span>
      )}
    </div>
  )
}

function StyledSelect({ value, onChange, options, placeholder }) {
  return (
    <div className="relative">
      <select
        value={value}
        onChange={e => onChange(e.target.value)}
        className="w-full bg-white/6 hover:bg-white/9 border border-white/10 hover:border-white/20
                   rounded-xl px-4 py-3.5 text-white text-sm appearance-none cursor-pointer
                   focus:outline-none focus:border-rose-500/50 focus:bg-white/10 transition-all"
      >
        {placeholder && <option value="">{placeholder}</option>}
        {options.map(o => <option key={o} value={o}>{o}</option>)}
      </select>
      <ChevronDown size={15} className="absolute right-4 top-1/2 -translate-y-1/2 text-white/30 pointer-events-none" />
    </div>
  )
}

function RangeSlider({ value, onChange, min, max, step = 1, formatValue }) {
  const pct = ((value - min) / (max - min)) * 100
  const display = formatValue ? formatValue(value) : value
  return (
    <div className="space-y-3">
      <div className="flex justify-between items-center">
        <span className="text-white/30 text-xs">{formatValue ? formatValue(min) : min}</span>
        <span className="text-white font-semibold text-base bg-white/8 px-3 py-1 rounded-lg min-w-[60px] text-center">
          {display}
        </span>
        <span className="text-white/30 text-xs">{formatValue ? formatValue(max) : max}</span>
      </div>
      <div className="relative">
        <div className="absolute top-1/2 -translate-y-1/2 h-1 rounded-full bg-rose-500 pointer-events-none"
             style={{ width: `${pct}%` }} />
        <input
          type="range" min={min} max={max} step={step} value={value}
          onChange={e => onChange(Number(e.target.value))}
          className="relative w-full"
        />
      </div>
    </div>
  )
}

function ToggleChip({ checked, onChange, label }) {
  return (
    <button
      type="button"
      onClick={() => onChange(!checked)}
      className={`px-4 py-2.5 rounded-xl border text-sm font-medium transition-all duration-200 flex items-center gap-2
        ${checked
          ? 'bg-rose-600/20 border-rose-500/40 text-rose-200'
          : 'bg-white/4 border-white/10 text-white/40 hover:text-white/60 hover:border-white/20'}`}
    >
      <span className={`w-3.5 h-3.5 rounded-sm border flex items-center justify-center flex-shrink-0 transition-all
        ${checked ? 'bg-rose-500 border-rose-500' : 'border-white/25'}`}>
        {checked && (
          <svg width="9" height="7" viewBox="0 0 9 7" fill="none">
            <path d="M1 3.5L3.2 6L8 1" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        )}
      </span>
      {label}
    </button>
  )
}

function Card({ icon: Icon, title, accent, children }) {
  return (
    <div className="rounded-2xl border border-white/8 bg-white/[0.03] overflow-hidden">
      <div className="flex items-center gap-3 px-6 py-4 border-b border-white/6">
        <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${accent ?? 'bg-rose-600/20'}`}>
          <Icon size={15} className="text-rose-400" />
        </div>
        <span className="text-white font-semibold text-sm tracking-wide">{title}</span>
      </div>
      <div className="p-6 space-y-6">
        {children}
      </div>
    </div>
  )
}

// ─── main ─────────────────────────────────────────────────────────────────────

export default function ListingForm({ form, onChange, onSubmit, loading }) {
  const [showHost, setShowHost] = useState(false)
  const neighbourhoods = BOROUGH_NEIGHBOURHOODS[form.borough] ?? []
  const set = (key, val) => onChange({ ...form, [key]: val })

  return (
    <form onSubmit={e => { e.preventDefault(); onSubmit() }} className="space-y-5">

      {/* ── Type & Location ── */}
      <Card icon={MapPin} title="Type & Location">
        <div className="grid grid-cols-2 gap-5">
          <div>
            <FieldLabel>Room type</FieldLabel>
            <StyledSelect value={form.room_type} onChange={v => set('room_type', v)} options={ROOM_TYPES} />
          </div>
          <div>
            <FieldLabel>Borough</FieldLabel>
            <StyledSelect
              value={form.borough}
              onChange={v => onChange({ ...form, borough: v, neighbourhood: '' })}
              options={BOROUGHS}
            />
          </div>
        </div>

        <div>
          <FieldLabel>Neighbourhood <span className="text-white/30 text-xs font-normal ml-1">optional — improves accuracy</span></FieldLabel>
          <StyledSelect
            value={form.neighbourhood}
            onChange={v => set('neighbourhood', v)}
            options={neighbourhoods}
            placeholder="Use borough average"
          />
        </div>

        <div>
          <FieldLabel badge="#1 most important feature">Bathroom type</FieldLabel>
          <div className="flex gap-3">
            <button
              type="button"
              onClick={() => set('is_private_bath', true)}
              className={`flex-1 py-3 rounded-xl border text-sm font-medium transition-all
                ${form.is_private_bath
                  ? 'bg-rose-600/20 border-rose-500/50 text-rose-200'
                  : 'bg-white/4 border-white/10 text-white/40 hover:border-white/20'}`}
            >
              Private bath
            </button>
            <button
              type="button"
              onClick={() => set('is_private_bath', false)}
              className={`flex-1 py-3 rounded-xl border text-sm font-medium transition-all
                ${!form.is_private_bath
                  ? 'bg-white/12 border-white/25 text-white'
                  : 'bg-white/4 border-white/10 text-white/40 hover:border-white/20'}`}
            >
              Shared bath
            </button>
          </div>
        </div>
      </Card>

      {/* ── Capacity ── */}
      <Card icon={Users} title="Capacity & Stay">
        <div>
          <FieldLabel>Guests</FieldLabel>
          <RangeSlider
            value={form.accommodates}
            onChange={v => set('accommodates', v)}
            min={1} max={16}
            formatValue={v => v === 1 ? '1 guest' : `${v} guests`}
          />
        </div>

        <div className="grid grid-cols-2 gap-8">
          <div>
            <FieldLabel>Bedrooms</FieldLabel>
            <RangeSlider
              value={form.bedrooms}
              onChange={v => set('bedrooms', v)}
              min={0} max={10}
              formatValue={v => v === 0 ? 'Studio' : v === 1 ? '1 bed' : `${v} beds`}
            />
          </div>
          <div>
            <FieldLabel>Bathrooms</FieldLabel>
            <RangeSlider
              value={form.bathrooms}
              onChange={v => set('bathrooms', v)}
              min={0.5} max={10} step={0.5}
              formatValue={v => v === 1 ? '1 bath' : `${v} baths`}
            />
          </div>
        </div>

        <div>
          <FieldLabel>Minimum nights</FieldLabel>
          <RangeSlider value={form.minimum_nights} onChange={v => set('minimum_nights', v)} min={1} max={30} />
        </div>
      </Card>

      {/* ── Reviews ── */}
      <Card icon={Star} title="Reviews">
        <div>
          <FieldLabel>Overall rating</FieldLabel>
          <RangeSlider
            value={form.review_scores_rating}
            onChange={v => set('review_scores_rating', v)}
            min={0} max={5} step={0.1}
            formatValue={v => v === 0 ? 'None' : `${v.toFixed(1)} ★`}
          />
        </div>

        <div className="grid grid-cols-2 gap-8">
          <div>
            <FieldLabel>Total reviews</FieldLabel>
            <RangeSlider value={form.number_of_reviews} onChange={v => set('number_of_reviews', v)} min={0} max={500} />
          </div>
          <div>
            <FieldLabel>Reviews / month</FieldLabel>
            <RangeSlider value={form.reviews_per_month} onChange={v => set('reviews_per_month', v)}
                         min={0} max={15} step={0.1} formatValue={v => v.toFixed(1)} />
          </div>
        </div>
      </Card>

      {/* ── Amenities ── */}
      <Card icon={Wifi} title="Amenities">
        <div>
          <FieldLabel>Total amenity count</FieldLabel>
          <RangeSlider
            value={form.amenity_count}
            onChange={v => set('amenity_count', v)}
            min={5} max={100}
            formatValue={v => `${v} items`}
          />
          <div className="mt-3 flex justify-between text-[11px] text-white/25">
            <span>5 — bare minimum (bed, door)</span>
            <span className="text-white/40 font-medium">
              {form.amenity_count <= 20 ? 'Budget' :
               form.amenity_count <= 40 ? 'Standard' :
               form.amenity_count <= 65 ? 'Premium' : 'Luxury'}
            </span>
            <span>100 — every possible item</span>
          </div>
          <p className="mt-2 text-white/25 text-[11px] leading-relaxed">
            Total items listed under "Amenities" on Airbnb — includes things like WiFi, TV,
            kitchen, hangers, iron, shampoo. Most NYC listings are 15–40.
          </p>
        </div>

        <div>
          <FieldLabel>Key amenities <span className="text-white/30 font-normal text-xs ml-1">tick what applies</span></FieldLabel>
          <div className="flex flex-wrap gap-2">
            {AMENITIES.map(({ key, label }) => (
              <ToggleChip key={key} checked={form[key]} onChange={v => set(key, v)} label={label} />
            ))}
          </div>
        </div>
      </Card>

      {/* ── Host (collapsible) ── */}
      <button
        type="button"
        onClick={() => setShowHost(s => !s)}
        className="w-full flex items-center justify-center gap-2 text-white/30 hover:text-white/50
                   text-sm py-2 transition-colors"
      >
        <ChevronDown size={14} className={`transition-transform duration-200 ${showHost ? 'rotate-180' : ''}`} />
        {showHost ? 'Hide' : 'Show'} host details
      </button>

      {showHost && (
        <Card icon={Home} title="Host Details">
          <div>
            <FieldLabel>Superhost</FieldLabel>
            <div className="flex gap-3">
              <ToggleChip checked={form.host_is_superhost} onChange={v => set('host_is_superhost', v)} label="Yes, I'm a superhost" />
            </div>
          </div>
          <div>
            <FieldLabel>Total listings by this host</FieldLabel>
            <RangeSlider value={form.host_listings_count} onChange={v => set('host_listings_count', v)} min={1} max={50} />
          </div>
        </Card>
      )}

      {/* ── Listing ID (optional — enables ground truth tracking) ── */}
      <div className="rounded-2xl border border-white/6 bg-white/[0.02] px-5 py-4">
        <div className="flex items-start gap-3">
          <Link size={14} className="text-violet-400 mt-0.5 shrink-0" />
          <div className="flex-1">
            <p className="text-white/60 text-xs font-medium mb-1.5">
              Airbnb Listing ID
              <span className="text-white/25 font-normal ml-2">optional · enables accuracy tracking</span>
            </p>
            <input
              type="text"
              value={form.listing_id ?? ''}
              onChange={e => {
                const raw = e.target.value
                const m   = raw.match(/\/rooms\/(\d+)/)
                set('listing_id', m ? m[1] : raw.trim())
              }}
              placeholder="12345678 or paste an Airbnb URL"
              className="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2.5
                         text-white/80 text-sm placeholder-white/20 font-mono
                         focus:outline-none focus:border-violet-500/40 transition-colors"
            />
            <p className="text-white/20 text-[11px] mt-1.5 leading-relaxed">
              When provided, the prediction is stored with this ID so its accuracy can be verified
              against real InsideAirbnb prices later.
            </p>
          </div>
        </div>
      </div>

      {/* ── Submit ── */}
      <button
        type="submit"
        disabled={loading}
        className={`w-full py-4 rounded-2xl font-bold text-base tracking-wide transition-all duration-200
          ${loading
            ? 'bg-rose-700/40 text-white/40 cursor-not-allowed'
            : 'bg-rose-600 hover:bg-rose-500 text-white shadow-xl shadow-rose-600/20 active:scale-[0.99]'}`}
      >
        {loading ? (
          <span className="flex items-center justify-center gap-2.5">
            <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
            </svg>
            Predicting…
          </span>
        ) : 'Predict nightly price →'}
      </button>
    </form>
  )
}
