"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { AlertCircle } from "lucide-react"
import WaveformViewer from "@/components/waveform-viewer"
import PerformanceMetrics from "@/components/performance-metrics"
import DistributionPlot from "@/components/distribution-plot"

const exampleNetlist = `V1 in 0 SIN(0 10 50)
R1 in 1 1k
R2 1 2 1k
L1 1 0 2H
C1 2 0 6u
.tran 20m 0 0.01m
.END`

export default function FaultDetectionApp() {
  const [netlist, setNetlist] = useState(exampleNetlist)
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [error, setError] = useState(null)

  async function handleClassify() {
    setRunning(true)
    setError(null)

    try {
      const payload = { netlist }
      const resp = await fetch("http://127.0.0.1:8000/api/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      })

      if (!resp.ok) {
        const errorData = await resp.json().catch(() => ({ detail: "Unknown error" }))
        throw new Error(errorData.detail || `Backend error: ${resp.status}`)
      }

      const resJson = await resp.json()
      setResult(resJson)

    } catch (err) {
      console.error("Classification error:", err)
      setError(err.message || "Failed to classify fault")
    } finally {
      setRunning(false)
    }
  }

  if (!result) {
    return (
      <main className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-950 dark:to-slate-900 flex items-center justify-center p-4">
        <Card className="w-full max-w-2xl shadow-lg">
          <CardHeader className="border-b">
            <CardTitle className="text-3xl">RLC Fault Detection</CardTitle>
            <CardDescription className="text-base mt-2">
              IEEE Paper Implementation - Table I Fault Classes
            </CardDescription>
          </CardHeader>
          <CardContent className="pt-6 space-y-6">
            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Error</AlertTitle>
                <AlertDescription>
                  {error}
                  <br />
                  <strong>Ensure backend is running:</strong>
                  <code className="block mt-1 bg-black/10 p-1 rounded">python -m uvicorn backend.main:app --reload --port 8000</code>
                </AlertDescription>
              </Alert>
            )}

            <div className="space-y-2">
              <label className="block text-sm font-medium">SPICE Netlist</label>
              <textarea
                value={netlist}
                onChange={(e) => setNetlist(e.target.value)}
                className="w-full h-40 p-3 border rounded-lg font-mono text-sm focus:outline-none focus:ring-2 focus:ring-primary"
              />
            </div>

            <div className="flex gap-3">
              <Button onClick={handleClassify} disabled={running} size="lg" className="flex-1">
                {running ? "Classifying..." : "Classify Fault"}
              </Button>
            </div>

            <div className="pt-4 text-xs text-muted-foreground border-t space-y-1">
              <p><strong>Fault Classes (IEEE Table I):</strong></p>
              <ul className="list-disc pl-4 space-y-1">
                <li>E0: Normal (Tolerance ±5%)</li>
                <li>E1: R1 High (&gt;150%)</li>
                <li>E2: R1 Low (-50%)</li>
                <li>E3: R2 High (&gt;150%)</li>
                <li>E4: R2 Low (-50%)</li>
                <li>E5: C1 Low (-50%)</li>
              </ul>
            </div>
          </CardContent>
        </Card>
      </main>
    )
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-950 dark:to-slate-900 p-4">
      <div className="max-w-7xl mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <Button onClick={() => setResult(null)} variant="outline" size="sm">← New Classification</Button>
          <div className="text-right">
            <p className="text-sm text-muted-foreground">Result</p>
            <p className="text-2xl font-bold">{result.classification?.label} <span className="text-sm font-normal">({(result.classification?.confidence * 100).toFixed(1)}%)</span></p>
          </div>
        </div>

        <Card>
          <CardHeader><CardTitle>Waveforms</CardTitle></CardHeader>
          <CardContent><WaveformViewer waveforms={result.waveforms} /></CardContent>
        </Card>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card>
            <CardHeader><CardTitle>Distribution (V0 Pk-Pk vs RMS)</CardTitle></CardHeader>
            <CardContent><DistributionPlot data={result.distribution} /></CardContent>
          </Card>
          <Card>
            <CardHeader><CardTitle>Metrics</CardTitle></CardHeader>
            <CardContent><PerformanceMetrics metrics={result.metrics} /></CardContent>
          </Card>
        </div>
      </div>
    </main>
  )
}
