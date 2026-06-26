import { useState } from 'react'
import Header from './components/Header'
import ListingForm from './components/ListingForm'
import PredictionResult from './components/PredictionResult'
import MetricsPanel from './components/MetricsPanel'
import { useMetrics } from './hooks/useMetrics'

const BOROUGH_COORDS = {
  Manhattan:     [40.7831, -73.9712],
  Brooklyn:      [40.6782, -73.9442],
  Queens:        [40.7282, -73.7949],
  Bronx:         [40.8448, -73.8648],
  'Staten Island': [40.5795, -74.1502],
}

const DEFAULT_FORM = {
  accommodates:         2,
  bedrooms:             1,
  bathrooms:            1,
  is_private_bath:      true,
  room_type:            'Entire home/apt',
  borough:              'Brooklyn',
  neighbourhood:        'Williamsburg',
  minimum_nights:       2,
  host_is_superhost:    false,
  host_listings_count:  1,
  number_of_reviews:    30,
  reviews_per_month:    1.2,
  review_scores_rating: 4.7,
  amenity_count:        25,
  has_gym:              false,
  has_elevator:         false,
  has_dryer:            true,
  has_air_conditioning: true,
  has_washer:           true,
  has_pool:             false,
}

export default function App() {
  const [form, setForm]       = useState(DEFAULT_FORM)
  const [result, setResult]   = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)
  const { metrics, modelInfo } = useMetrics(5000)

  async function handleSubmit() {
    setLoading(true)
    setError(null)
    const [lat, lon] = BOROUGH_COORDS[form.borough]
    const payload = {
      ...form,
      latitude:              lat,
      longitude:             lon,
      number_of_reviews_ltm: Math.min(form.number_of_reviews, 12),
      review_scores_accuracy:      form.review_scores_rating,
      review_scores_cleanliness:   form.review_scores_rating,
      review_scores_checkin:       form.review_scores_rating,
      review_scores_communication: form.review_scores_rating,
      review_scores_location:      form.review_scores_rating,
      review_scores_value:         form.review_scores_rating,
    }
    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (res.ok) {
        setResult(await res.json())
      } else {
        const body = await res.json()
        setError(body.detail ?? `Error ${res.status}`)
      }
    } catch {
      setError('Cannot reach API — make sure it is running on port 8001')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-[#0c0c0e]">
      <Header modelInfo={modelInfo} apiAlive={metrics !== null} />

      <div className="max-w-6xl mx-auto px-8 py-10">
        <div className="grid grid-cols-1 xl:grid-cols-[1fr_420px] gap-10 items-start">

          {/* ── left: page title + form ── */}
          <div className="space-y-8">
            <div>
              <h1 className="text-white text-3xl font-bold tracking-tight">
                Predict nightly price
              </h1>
              <p className="text-white/40 text-base mt-2">
                Fill in your listing — feature engineering and scaling run server-side.
              </p>
            </div>

            <ListingForm
              form={form}
              onChange={setForm}
              onSubmit={handleSubmit}
              loading={loading}
            />

            {error && (
              <div className="rounded-2xl border border-red-500/25 bg-red-500/8 px-5 py-4 text-red-300 text-sm">
                {error}
              </div>
            )}
          </div>

          {/* ── right: result + metrics (sticky) ── */}
          <div className="space-y-5 xl:sticky xl:top-24">

            {result ? (
              <PredictionResult result={result} form={form} />
            ) : (
              <EmptyState />
            )}

            <MetricsPanel metrics={metrics} />
          </div>

        </div>
      </div>
    </div>
  )
}

function EmptyState() {
  return (
    <div className="rounded-3xl border border-white/8 bg-white/[0.02] p-10 text-center">
      <div className="w-16 h-16 rounded-2xl bg-rose-600/10 border border-rose-500/15
                      flex items-center justify-center mx-auto mb-5">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" className="text-rose-400">
          <path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z"
                stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          <path d="M9 22V12h6v10"
                stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </div>
      <p className="text-white/60 text-base font-medium">Your prediction will appear here</p>
      <p className="text-white/25 text-sm mt-2">
        Configure your listing and click<br/><span className="text-white/40">Predict nightly price</span>
      </p>
    </div>
  )
}
