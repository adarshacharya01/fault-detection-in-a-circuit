"use client"

import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

interface WaveformData {
  t: number[]
  vin: number[]
  v0_healthy: number[]
  v0_E1: number[]
  v0_E2: number[]
  v0_E3: number[]
  v0_E4: number[]
  v0_E5: number[]
  v1_healthy: number[]
  v1_E1: number[]
  v1_E2: number[]
  v1_E3: number[]
  v1_E4: number[]
  v1_E5: number[]
}

function SimpleLineChart({ data, label }: { data: number[]; label: string }) {
  if (!Array.isArray(data) || data.length === 0) {
    return (
      <div className="h-64 bg-slate-50 rounded flex items-center justify-center text-muted-foreground">No data</div>
    )
  }

  const min = Math.min(...data)
  const max = Math.max(...data)
  const range = max - min || 1
  const padding = 40
  const width = 600
  const height = 280

  const points = data.map((y, i) => ({
    x: (i / (data.length - 1)) * (width - 2 * padding) + padding,
    y: height - ((y - min) / range) * (height - 2 * padding) - padding,
  }))

  const pathD = points.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`).join(" ")

  return (
    <div className="w-full overflow-x-auto">
      <svg width={width} height={height} className="mx-auto">
        <line
          x1={padding}
          y1={height - padding}
          x2={width - padding}
          y2={height - padding}
          stroke="currentColor"
          strokeWidth="1"
          opacity="0.2"
        />
        <line
          x1={padding}
          y1={padding}
          x2={padding}
          y2={height - padding}
          stroke="currentColor"
          strokeWidth="1"
          opacity="0.2"
        />
        <path d={pathD} stroke="hsl(var(--primary))" strokeWidth="2" fill="none" />
        <text x={width / 2} y={height - 5} textAnchor="middle" fontSize="12" fill="currentColor" opacity="0.6">
          Time (ms)
        </text>
        <text
          x={15}
          y={height / 2}
          textAnchor="middle"
          fontSize="12"
          fill="currentColor"
          opacity="0.6"
          transform={`rotate(-90 15 ${height / 2})`}
        >
          {label}
        </text>
      </svg>
    </div>
  )
}

export default function WaveformViewer({ waveforms }: { waveforms: WaveformData }) {
  const [selectedFault, setSelectedFault] = useState("healthy")

  return (
    <Tabs defaultValue="vin" className="w-full">
      <TabsList className="grid w-full grid-cols-3">
        <TabsTrigger value="vin">Input (Vin)</TabsTrigger>
        <TabsTrigger value="v0">Output V0 (C1)</TabsTrigger>
        <TabsTrigger value="v1">Output V1 (R2)</TabsTrigger>
      </TabsList>

      <TabsContent value="vin" className="space-y-4">
        <SimpleLineChart data={waveforms.vin || []} label="Vin (V)" />
      </TabsContent>

      <TabsContent value="v0" className="space-y-4">
        <div className="flex gap-2 flex-wrap">
          {["healthy", "E1", "E2", "E3", "E4", "E5"].map((fault) => (
            <button
              key={fault}
              onClick={() => setSelectedFault(fault)}
              className={`px-3 py-1 rounded text-sm transition ${
                selectedFault === fault
                  ? "bg-primary text-primary-foreground"
                  : "bg-secondary text-secondary-foreground hover:bg-slate-300"
              }`}
            >
              {fault}
            </button>
          ))}
        </div>
        <SimpleLineChart data={waveforms[`v0_${selectedFault}`] || []} label="V0 (V)" />
      </TabsContent>

      <TabsContent value="v1" className="space-y-4">
        <div className="flex gap-2 flex-wrap">
          {["healthy", "E1", "E2", "E3", "E4", "E5"].map((fault) => (
            <button
              key={fault}
              onClick={() => setSelectedFault(fault)}
              className={`px-3 py-1 rounded text-sm transition ${
                selectedFault === fault
                  ? "bg-primary text-primary-foreground"
                  : "bg-secondary text-secondary-foreground hover:bg-slate-300"
              }`}
            >
              {fault}
            </button>
          ))}
        </div>
        <SimpleLineChart data={waveforms[`v1_${selectedFault}`] || []} label="V1 (V)" />
      </TabsContent>
    </Tabs>
  )
}
