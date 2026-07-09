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
  room_type:            'Entire home/apt',
  borough:              'Manhattan',
  neighbourhood:        '',
  bedrooms:             1,
  bathrooms:            1.0,
  minimum_nights:       2,
  review_scores_rating: 4.5,
  number_of_reviews:    20,
  has_air_conditioning: false,
  has_washer:           false,
  has_dryer:            false,
  has_elevator:         false,
  has_gym:              false,
  has_pool:             false,
  instant_bookable:     false,
}

export default function PredictPage() {
  const [form, setForm]     = useState(DEFAULT_FORM)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]   = useState(null)

  async function handleSubmit() {
    setLoading(true)
    setError(null)
    setResult(null)
    const coords = BOROUGH_COORDS[form.borough] ?? [40.7128, -74.006]
    const beds   = Math.max(form.bedrooms, 1)
    const payload = {
      room_type:                   form.room_type,
      borough:                     form.borough,
      neighbourhood:               form.neighbourhood || undefined,
      bedrooms:                    form.bedrooms,
      bathrooms:                   form.bathrooms,
      minimum_nights:              form.minimum_nights,
      review_scores_rating:        form.review_scores_rating,
      number_of_reviews:           form.number_of_reviews,
      accommodates:                beds * 2,
      is_private_bath:             form.room_type === 'Entire home/apt',
      latitude:                    coords[0],
      longitude:                   coords[1],
      amenity_count:               30,
      number_of_reviews_ltm:       Math.min(form.number_of_reviews, 12),
      reviews_per_month:           Math.max(form.number_of_reviews / 24, 0.1),
      review_scores_accuracy:      form.review_scores_rating,
      review_scores_cleanliness:   form.review_scores_rating,
      review_scores_checkin:       form.review_scores_rating,
      review_scores_communication: form.review_scores_rating,
      review_scores_location:      form.review_scores_rating,
      review_scores_value:         form.review_scores_rating,
      host_is_superhost:           false,
      host_listings_count:         1,
      has_gym:                     form.has_gym,
      has_elevator:                form.has_elevator,
      has_dryer:                   form.has_dryer,
      has_air_conditioning:        form.has_air_conditioning,
      has_washer:                  form.has_washer,
      has_pool:                    form.has_pool,
      instant_bookable:            form.instant_bookable ? 1 : 0,
    }
    try {
      const res = await fetch('/api/predict?explain=true', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(payload),
      })
      if (res.ok) {
        setResult(await res.json())
        setTimeout(() => {
          document.getElementById('result-anchor')?.scrollIntoView({ behavior: 'smooth' })
        }, 100)
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
    <div className="space-y-3">
      <div className="mb-6">
        <h1 className="text-xl font-bold text-slate-900">Price My Listing</h1>
        <p className="text-slate-500 text-sm mt-1">
          Estimate what your NYC Airbnb could earn per night.
        </p>
      </div>

      <ListingForm form={form} onChange={setForm} onSubmit={handleSubmit} loading={loading} />

      {error && (
        <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-red-600 text-sm">
          {error}
        </div>
      )}

      {(result || loading) && (
        <div id="result-anchor">
          <PredictionResult result={result} form={form} loading={loading} />
        </div>
      )}
    </div>
  )
}
