import { useState, useEffect, useCallback, useRef } from 'react'

export function usePolling(url, intervalMs = 10000) {
  const [data, setData]         = useState(null)
  const [loading, setLoading]   = useState(true)
  const [error, setError]       = useState(null)
  const [lastFetch, setLastFetch] = useState(null)
  const mountedRef = useRef(true)

  const refresh = useCallback(async () => {
    try {
      const r = await fetch(url)
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      const json = await r.json()
      if (mountedRef.current) {
        setData(json)
        setError(null)
        setLastFetch(new Date())
      }
    } catch (e) {
      if (mountedRef.current) setError(e.message)
    } finally {
      if (mountedRef.current) setLoading(false)
    }
  }, [url])

  useEffect(() => {
    mountedRef.current = true
    refresh()
    if (!intervalMs) return
    const id = setInterval(refresh, intervalMs)
    return () => {
      mountedRef.current = false
      clearInterval(id)
    }
  }, [refresh, intervalMs])

  return { data, loading, error, refresh, lastFetch }
}
