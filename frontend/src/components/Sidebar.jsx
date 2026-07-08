import { Sparkles, Building2 } from 'lucide-react'

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
      <nav className="flex-1 px-3 py-4">
        <button
          onClick={() => setPage('predict')}
          className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-left
                     transition-all duration-150 relative"
          style={{
            background: page === 'predict' ? 'rgba(225,29,72,0.12)' : 'transparent',
            color:      page === 'predict' ? '#fff' : 'rgba(255,255,255,0.45)',
          }}
        >
          {page === 'predict' && (
            <span className="absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-5 rounded-r-full bg-rose-500" />
          )}
          <Sparkles size={16}
            style={{ color: page === 'predict' ? '#e11d48' : 'rgba(255,255,255,0.35)' }}
            className="shrink-0" />
          <span className="text-[13px] font-medium">Price My Listing</span>
        </button>
      </nav>
    </aside>
  )
}
