import { useState } from 'react'
import ListingForm from '../components/ListingForm'
import PredictionResult from '../components/PredictionResult'

const BOROUGH_COORDS = {
  Manhattan:       [40.7831, -73.9712],
  Brooklyn:        [40.6782, -73.9442],
  Queens:          [40.7282, -73.7949],
  Bronx:           [40.8448, -73.8648],
  'Staten Island': [40.5795, -74.1502],
}

const DEFAULT_FORM = {
  accommodates:         2,
  bedrooms:             1,
  bathrooms:            1.0,
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
  listing_id:           '',
  has_gym:              false,
  has_elevator:         false,
  has_dryer:            true,
  has_air_conditioning: true,
  has_washer:           true,
  has_pool:             false,
}

export default function PredictPage() {
  const [form, setForm]       = useState(DEFAULT_FORM)
  const [result, setResult]   = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)

  async function handleSubmit() {
    setLoading(true)
    setError(null)
    const coords  = BOROUGH_COORDS[form.borough] ?? [40.7128, -74.006]
    const payload = {
      ...form,
      listing_id:                  form.listing_id || undefined,
      latitude:                    coords[0],
      longitude:                   coords[1],
      number_of_reviews_ltm:       Math.min(form.number_of_reviews, 12),
      review_scores_accuracy:      form.review_scores_rating,
      review_scores_cleanliness:   form.review_scores_rating,
      review_scores_checkin:       form.review_scores_rating,
      review_scores_communication: form.review_scores_rating,
      review_scores_location:      form.review_scores_rating,
      review_scores_value:         form.review_scores_rating,
    }
    try {
      const res = await fetch('/api/predict', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(payload),
      })
      if (res.ok) {
        setResult(await res.json())
      } else {
        const body = await res.json().catch(() => ({}))
        setError(body.detail ?? `API error ${res.status}`)
      }
    } catch {
      setError('Cannot reach API — make sure it is running on port 8001')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-2">
      <div className="mb-6">
        <h1 className="text-white text-xl font-bold tracking-tight">Price Prediction</h1>
        <p className="text-white/35 text-sm mt-1">
          ONNX Runtime inference · Redis cache · request logging
        </p>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[1fr_400px] gap-8 items-start">
        <div className="space-y-4">
          <ListingForm form={form} onChange={setForm} onSubmit={handleSubmit} loading={loading} />
          {error && (
            <div className="rounded-xl border border-red-500/25 bg-red-500/8 px-4 py-3
                            text-red-300 text-sm">
              {error}
            </div>
          )}
        </div>

        <div className="space-y-4 xl:sticky xl:top-6">
          {result ? (
            <PredictionResult result={result} form={form} />
          ) : (
            <EmptyResult />
          )}
        </div>
      </div>
    </div>
  )
}

function EmptyResult() {
  return (
    <div className="rounded-2xl border border-white/8 bg-white/[0.015] p-10 text-center">
      <div className="w-14 h-14 rounded-2xl bg-rose-600/10 border border-rose-500/15
                      flex items-center justify-center mx-auto mb-4">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" className="text-rose-400">
          <path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z"
                stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          <path d="M9 22V12h6v10"
                stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </div>
      <p className="text-white/55 text-sm font-medium">Your prediction appears here</p>
      <p className="text-white/22 text-xs mt-1.5">Fill in the form and click Predict</p>
    </div>
  )
}
