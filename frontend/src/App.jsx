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
  accommodates:              2,
  bedrooms:                  1,
  bathrooms:                 1,
  is_private_bath:           true,
  room_type:                 'Entire home/apt',
  borough:                   'Brooklyn',
  neighbourhood:             'Williamsburg',
  minimum_nights:            2,
  host_is_superhost:         false,
  host_listings_count:       1,
  number_of_reviews:         30,
  reviews_per_month:         1.2,
  review_scores_rating:      4.7,
  amenity_count:             25,
  has_gym:                   false,
  has_elevator:              false,
  has_dryer:                 true,
  has_air_conditioning:      true,
  has_washer:                true,
  has_pool:                  false,
}

export default function App() {
  const [form, setForm]       = useState(DEFAULT_FORM)
  const [result, setResult]   = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)
  const { metrics, modelInfo } = useMetrics(5000)

  const apiAlive = metrics !== null

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
    <div className="min-h-screen bg-[#0a0a0a]">
      <Header modelInfo={modelInfo} apiAlive={apiAlive} />

      <main className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_380px] gap-8 items-start">

          {/* ── left: form ── */}
          <div className="space-y-6">
            <div>
              <h2 className="text-white text-2xl font-bold">Predict nightly price</h2>
              <p className="text-white/40 text-sm mt-1">
                Fill in your listing details — feature engineering runs server-side
              </p>
            </div>
            <ListingForm
              form={form}
              onChange={setForm}
              onSubmit={handleSubmit}
              loading={loading}
            />
            {error && (
              <div className="rounded-2xl border border-red-500/30 bg-red-500/10 p-4 text-red-300 text-sm">
                {error}
              </div>
            )}
          </div>

          {/* ── right: result + metrics ── */}
          <div className="space-y-6 lg:sticky lg:top-24">
            {result ? (
              <PredictionResult result={result} form={form} />
            ) : (
              <div className="rounded-3xl border border-white/8 bg-white/3 p-8 text-center">
                <div className="w-14 h-14 rounded-2xl bg-rose-600/15 flex items-center justify-center mx-auto mb-4">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" className="text-rose-400">
                    <path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                    <path d="M9 22V12h6v10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                  </svg>
                </div>
                <p className="text-white/50 text-sm">Fill in the form and click<br/>
                  <span className="text-white/70 font-medium">Predict nightly price</span>
                </p>
              </div>
            )}
            <MetricsPanel metrics={metrics} />
          </div>

        </div>
      </main>
    </div>
  )
}
