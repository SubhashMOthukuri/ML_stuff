import { Building2, Activity } from 'lucide-react'

export default function Header({ modelInfo, apiAlive }) {
  return (
    <header className="border-b border-white/[0.06] bg-black/50 backdrop-blur-xl sticky top-0 z-50">
      <div className="max-w-6xl mx-auto px-8 py-4 flex items-center justify-between">

        {/* logo */}
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-rose-600 flex items-center justify-center shadow-lg shadow-rose-600/30">
            <Building2 size={17} className="text-white" />
          </div>
          <div>
            <p className="text-white font-bold text-base leading-none">NYC Airbnb Pricer</p>
            <p className="text-white/30 text-xs mt-0.5">XGBoost · April 2026 snapshot</p>
          </div>
        </div>

        {/* stats + status */}
        <div className="flex items-center gap-8">
          {modelInfo && (
            <div className="hidden md:flex items-center gap-8">
              <HeaderStat label="R² Score" value={modelInfo.r2_test} />
              <HeaderStat label="MAE" value={`$${modelInfo.mae_dollar}/night`} />
              <HeaderStat label="Features" value={`${modelInfo.n_features} engineered`} />
            </div>
          )}
          <div className={`flex items-center gap-2 px-3.5 py-1.5 rounded-full text-xs font-semibold
            ${apiAlive
              ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
              : 'bg-red-500/10 text-red-400 border border-red-500/20'}`}>
            <span className={`w-1.5 h-1.5 rounded-full ${apiAlive ? 'bg-emerald-400' : 'bg-red-400'} animate-pulse`} />
            {apiAlive ? 'API live' : 'API offline'}
          </div>
        </div>
      </div>
    </header>
  )
}

function HeaderStat({ label, value }) {
  return (
    <div>
      <p className="text-white font-semibold text-sm">{value}</p>
      <p className="text-white/30 text-xs">{label}</p>
    </div>
  )
}
