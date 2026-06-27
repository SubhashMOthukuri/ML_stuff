import { useState, useEffect, useCallback } from 'react'

export function useMetrics(intervalMs = 5000) {
  const [metrics, setMetrics] = useState(null)
  const [modelInfo, setModelInfo] = useState(null)

  const fetchMetrics = useCallback(async () => {
    try {
      const r = await fetch('/api/metrics')
      if (r.ok) setMetrics(await r.json())
    } catch {}
  }, [])

  const fetchModelInfo = useCallback(async () => {
    try {
      const r = await fetch('/api/model-info')
      if (r.ok) setModelInfo(await r.json())
    } catch {}
  }, [])

  useEffect(() => {
    fetchModelInfo()
    fetchMetrics()
    const id = setInterval(fetchMetrics, intervalMs)
    return () => clearInterval(id)
  }, [fetchMetrics, fetchModelInfo, intervalMs])

  return { metrics, modelInfo }
}
