import { Building2 } from 'lucide-react'

export default function Header({ modelInfo, apiAlive }) {
  return (
    <header className="border-b border-white/[0.07] bg-black/60 backdrop-blur-xl sticky top-0 z-50">
      {/* accent line */}
      <div className="h-px bg-gradient-to-r from-transparent via-rose-500/50 to-transparent" />

      <div className="max-w-6xl mx-auto px-8 h-16 flex items-center justify-between">
        {/* logo */}
        <div className="flex items-center gap-3.5">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-rose-500 to-rose-700
                          flex items-center justify-center shadow-lg shadow-rose-600/30">
            <Building2 size={17} className="text-white" />
          </div>
          <div>
            <p className="text-white font-bold text-[15px] leading-none">NYC Airbnb Pricer</p>
            <p className="text-white/30 text-xs mt-0.5">XGBoost · ONNX Runtime · April 2026</p>
          </div>
        </div>

        {/* model stats + status */}
        <div className="flex items-center gap-6">
          {modelInfo && (
            <div className="hidden md:flex items-center divide-x divide-white/[0.07]">
              <Stat label="R²" value={modelInfo.r2_test} />
              <Stat label="MAE" value={`$${modelInfo.mae_dollar}`} />
              <Stat label="Features" value={modelInfo.n_features} />
              <Stat label="Backend" value={modelInfo.inference_backend?.replace('ExecutionProvider', '')} />
            </div>
          )}

          <div className={`flex items-center gap-2 px-3.5 py-1.5 rounded-full text-xs font-semibold border transition-colors ${
            apiAlive
              ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'
              : 'bg-red-500/10 border-red-500/20 text-red-400'
          }`}>
            <span className={`w-1.5 h-1.5 rounded-full animate-pulse ${apiAlive ? 'bg-emerald-400' : 'bg-red-400'}`} />
            {apiAlive ? 'API live' : 'API offline'}
          </div>
        </div>
      </div>
    </header>
  )
}

function Stat({ label, value }) {
  return (
    <div className="px-5 first:pl-0 last:pr-0 text-center">
      <p className="text-white font-bold text-sm leading-none">{value ?? '—'}</p>
      <p className="text-white/30 text-[11px] mt-0.5">{label}</p>
    </div>
  )
}
