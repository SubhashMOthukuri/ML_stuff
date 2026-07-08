import { useState } from 'react'
import ListingForm from '../components/ListingForm'
import PredictionResult from '../components/PredictionResult'

const BATH_BY_ROOM_TYPE = {
  'Entire home/apt': true,
  'Hotel room':      true,
  'Shared room':     false,
  'Private room':    false,
}

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
    const coords      = BOROUGH_COORDS[form.borough] ?? [40.7128, -74.006]
    const impliedBath = BATH_BY_ROOM_TYPE[form.room_type]
    const payload = {
      ...form,
      is_private_bath:             impliedBath !== null ? impliedBath : form.is_private_bath,
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
        setError(body.detail ?? `Error ${res.status} — please try again`)
      }
    } catch {
      setError('Cannot reach the API. Please try again in a moment.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-slate-900 tracking-tight">Price My Listing</h1>
        <p className="text-slate-500 mt-1 text-sm">
          Estimate what your NYC Airbnb could earn per night, based on 20,000+ real listings.
        </p>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[1fr_380px] gap-8 items-start">
        <div>
          <ListingForm form={form} onChange={setForm} onSubmit={handleSubmit} loading={loading} />
          {error && (
            <div className="mt-4 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-red-600 text-sm">
              {error}
            </div>
          )}
        </div>

        <div className="xl:sticky xl:top-24">
          {result
            ? <PredictionResult result={result} form={form} />
            : <EmptyResult />
          }
        </div>
      </div>
    </div>
  )
}

function EmptyResult() {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-10 text-center">
      <div className="w-12 h-12 rounded-full bg-slate-100 flex items-center justify-center mx-auto mb-4">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" className="text-slate-400">
          <path d="M12 2C6.477 2 2 6.477 2 12s4.477 10 10 10 10-4.477 10-10S17.523 2 12 2z"
                stroke="currentColor" strokeWidth="1.5"/>
          <path d="M12 8v4m0 4h.01" stroke="currentColor" strokeWidth="1.5"
                strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </div>
      <p className="text-slate-600 text-sm font-medium">Your estimate appears here</p>
      <p className="text-slate-400 text-xs mt-1">Fill in your listing details and click Estimate</p>
    </div>
  )
}
