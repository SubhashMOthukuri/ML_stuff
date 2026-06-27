import { Sparkles, FlaskConical, Activity, Bell, Server, Building2 } from 'lucide-react'

const NAV = [
  { id: 'predict', label: 'Price Predict',  icon: Sparkles,     accent: '#e11d48', dim: 'rgba(225,29,72,0.12)'  },
  { id: 'shadow',  label: 'Shadow Lab',     icon: FlaskConical, accent: '#7c3aed', dim: 'rgba(124,58,237,0.12)' },
  { id: 'drift',   label: 'Drift Monitor',  icon: Activity,     accent: '#d97706', dim: 'rgba(217,119,6,0.12)'  },
  { id: 'alerts',  label: 'Alerts',         icon: Bell,         accent: '#f97316', dim: 'rgba(249,115,22,0.12)' },
  { id: 'ops',     label: 'Operations',     icon: Server,       accent: '#0ea5e9', dim: 'rgba(14,165,233,0.12)' },
]

export default function Sidebar({ page, setPage, pendingAlerts = 0 }) {
  return (
    <aside className="hidden md:flex flex-col w-[220px] shrink-0 border-r border-white/[0.06]
                       bg-[#06060a] min-h-screen sticky top-0 h-screen">
      {/* logo */}
      <div className="h-16 flex items-center gap-3 px-5 border-b border-white/[0.06]">
        <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-rose-500 to-rose-700
                        flex items-center justify-center shadow-lg shadow-rose-600/25">
          <Building2 size={15} className="text-white" />
        </div>
        <div>
          <p className="text-white font-bold text-[13px] leading-none">NYC Pricer</p>
          <p className="text-white/30 text-[10px] mt-0.5">MLOps Platform</p>
        </div>
      </div>

      {/* nav */}
      <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
        {NAV.map(({ id, label, icon: Icon, accent, dim }) => {
          const active = page === id
          return (
            <button
              key={id}
              onClick={() => setPage(id)}
              className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-left
                         transition-all duration-150 group relative"
              style={{
                background: active ? dim : 'transparent',
                color:      active ? '#fff' : 'rgba(255,255,255,0.45)',
              }}
            >
              {/* active indicator */}
              {active && (
                <span className="absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-5 rounded-r-full"
                      style={{ background: accent }} />
              )}
              <Icon
                size={16}
                style={{ color: active ? accent : 'rgba(255,255,255,0.35)' }}
                className="shrink-0 transition-colors group-hover:opacity-80"
              />
              <span className="text-[13px] font-medium">{label}</span>
              {id === 'alerts' && pendingAlerts > 0 && (
                <span className="ml-auto text-[10px] font-bold px-1.5 py-0.5 rounded-full
                                 bg-orange-500/20 text-orange-400 border border-orange-500/25">
                  {pendingAlerts}
                </span>
              )}
            </button>
          )
        })}
      </nav>

      {/* footer */}
      <div className="px-5 py-4 border-t border-white/[0.06]">
        <p className="text-white/20 text-[10px]">v3.0.0 · ONNX Runtime</p>
      </div>
    </aside>
  )
}
