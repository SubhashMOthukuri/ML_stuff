import PredictPage from './pages/PredictPage'
import { Building2 } from 'lucide-react'

export default function App() {
  return (
    <div className="min-h-screen bg-slate-50">
      <header className="bg-white border-b border-slate-200 sticky top-0 z-50">
        <div className="max-w-4xl mx-auto px-6 h-14 flex items-center gap-3">
          <div className="w-7 h-7 rounded-lg bg-sky-700 flex items-center justify-center">
            <Building2 size={14} className="text-white" />
          </div>
          <span className="font-semibold text-slate-900 text-sm">NYC Airbnb Pricer</span>
          <span className="text-slate-300 text-sm">·</span>
          <span className="text-slate-400 text-sm">New York City</span>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-8">
        <PredictPage />
      </main>
    </div>
  )
}
