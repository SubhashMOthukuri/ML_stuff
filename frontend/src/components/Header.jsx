import { Building2, Activity } from 'lucide-react'

export default function Header({ modelInfo, apiAlive }) {
  return (
    <header className="border-b border-white/10 bg-black/40 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-rose-600 flex items-center justify-center">
            <Building2 size={18} className="text-white" />
          </div>
          <div>
            <h1 className="text-white font-semibold text-lg leading-none">NYC Airbnb Pricer</h1>
            <p className="text-white/40 text-xs mt-0.5">XGBoost · April 2026 snapshot</p>
          </div>
        </div>

        <div className="flex items-center gap-6">
          {modelInfo && (
            <>
              <Stat label="R²" value={modelInfo.r2_test} />
              <Stat label="MAE" value={`$${modelInfo.mae_dollar}`} />
              <Stat label="Features" value={modelInfo.n_features} />
            </>
          )}
          <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium
            ${apiAlive ? 'bg-emerald-500/15 text-emerald-400' : 'bg-red-500/15 text-red-400'}`}>
            <Activity size={11} />
            {apiAlive ? 'API live' : 'API offline'}
          </div>
        </div>
      </div>
    </header>
  )
}

function Stat({ label, value }) {
  return (
    <div className="text-center">
      <p className="text-white font-semibold text-sm">{value}</p>
      <p className="text-white/40 text-xs">{label}</p>
    </div>
  )
}
