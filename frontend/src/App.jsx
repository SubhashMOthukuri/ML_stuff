import { useState, useEffect } from 'react'
import Sidebar from './components/Sidebar'
import PredictPage from './pages/PredictPage'
import ShadowPage from './pages/ShadowPage'
import DriftPage from './pages/DriftPage'
import AlertsPage from './pages/AlertsPage'
import OpsPage from './pages/OpsPage'

const PAGES = {
  predict: PredictPage,
  shadow:  ShadowPage,
  drift:   DriftPage,
  alerts:  AlertsPage,
  ops:     OpsPage,
}

const TITLES = {
  predict: 'Price Predict',
  shadow:  'Shadow Lab',
  drift:   'Drift Monitor',
  alerts:  'Alerts',
  ops:     'Operations',
}

export default function App() {
  // hash-based routing so each page has its own shareable URL
  const [page, setPage] = useState(() => {
    const h = window.location.hash.replace('#/', '')
    return PAGES[h] ? h : 'predict'
  })
  const [pendingAlerts, setPendingAlerts] = useState(0)

  useEffect(() => {
    window.location.hash = `/${page}`
  }, [page])

  // poll pending alert count for the sidebar badge
  useEffect(() => {
    let alive = true
    const fetchCount = async () => {
      try {
        const r = await fetch('/api/alerts?pending_only=true')
        if (r.ok && alive) {
          const body = await r.json()
          setPendingAlerts(body.stats?.pending ?? 0)
        }
      } catch {}
    }
    fetchCount()
    const id = setInterval(fetchCount, 15000)
    return () => { alive = false; clearInterval(id) }
  }, [])

  const Page = PAGES[page] ?? PredictPage

  return (
    <div className="min-h-screen bg-[#080810] flex">
      <Sidebar page={page} setPage={setPage} pendingAlerts={pendingAlerts} />

      <div className="flex-1 min-w-0">
        {/* mobile top nav */}
        <MobileNav page={page} setPage={setPage} pendingAlerts={pendingAlerts} />

        <main className="max-w-5xl mx-auto px-6 py-8">
          <Page />
        </main>
      </div>
    </div>
  )
}

function MobileNav({ page, setPage, pendingAlerts }) {
  return (
    <div className="md:hidden sticky top-0 z-40 bg-[#06060a]/95 backdrop-blur-xl
                    border-b border-white/[0.06] overflow-x-auto">
      <div className="flex gap-1 px-3 py-2 min-w-max">
        {Object.entries(TITLES).map(([id, label]) => (
          <button key={id} onClick={() => setPage(id)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-colors ${
              page === id ? 'bg-white/10 text-white' : 'text-white/40'
            }`}>
            {label}
            {id === 'alerts' && pendingAlerts > 0 && (
              <span className="ml-1.5 text-[10px] px-1 rounded-full bg-orange-500/20 text-orange-400">
                {pendingAlerts}
              </span>
            )}
          </button>
        ))}
      </div>
    </div>
  )
}
