import { useState } from 'react'
import Header from './components/Header'
import Sidebar from './components/Sidebar'
import PredictPage from './pages/PredictPage'
import AlertsPage from './pages/AlertsPage'
import DriftPage from './pages/DriftPage'
import OpsPage from './pages/OpsPage'
import ShadowPage from './pages/ShadowPage'
import { useMetrics } from './hooks/useMetrics'

export default function App() {
  const [page, setPage] = useState('predict')
  const { metrics, modelInfo } = useMetrics()

  const content = {
    predict: <PredictPage />,
    alerts:  <AlertsPage />,
    drift:   <DriftPage />,
    ops:     <OpsPage />,
    shadow:  <ShadowPage />,
  }[page] ?? <PredictPage />

  return (
    <div className="min-h-screen bg-[#0c0c0e] flex flex-col">
      <Header modelInfo={modelInfo} apiAlive={metrics !== null} />
      <div className="flex flex-1 min-h-0">
        <Sidebar page={page} setPage={setPage} />
        <main className="flex-1 overflow-y-auto p-6 lg:p-8">
          <div className="max-w-5xl w-full mx-auto">
            {content}
          </div>
        </main>
      </div>
    </div>
  )
}
