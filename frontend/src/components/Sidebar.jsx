import { Sparkles, Bell, Activity, Server, FlaskConical, Building2 } from 'lucide-react'

const NAV = [
  { id: 'predict', label: 'Price My Listing', icon: Sparkles     },
  { id: 'alerts',  label: 'Alerts',           icon: Bell         },
  { id: 'drift',   label: 'Drift Monitor',    icon: Activity     },
  { id: 'ops',     label: 'Operations',       icon: Server       },
  { id: 'shadow',  label: 'Shadow Lab',       icon: FlaskConical },
]

export default function Sidebar({ page, setPage }) {
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
          <p className="text-white font-bold text-[13px] leading-none">NYC Airbnb Pricer</p>
          <p className="text-white/30 text-[10px] mt-0.5">New York City</p>
        </div>
      </div>

      {/* nav */}
      <nav className="flex-1 px-3 py-4 space-y-0.5">
        {NAV.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setPage(id)}
            className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-left
                       transition-all duration-150 relative"
            style={{
              background: page === id ? 'rgba(225,29,72,0.12)' : 'transparent',
              color:      page === id ? '#fff' : 'rgba(255,255,255,0.45)',
            }}
          >
            {page === id && (
              <span className="absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-5 rounded-r-full bg-rose-500" />
            )}
            <Icon size={16}
              style={{ color: page === id ? '#e11d48' : 'rgba(255,255,255,0.35)' }}
              className="shrink-0" />
            <span className="text-[13px] font-medium">{label}</span>
          </button>
        ))}
      </nav>
    </aside>
  )
}
