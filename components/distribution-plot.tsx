"use client"

interface DistributionData {
  E0?: number[][]
  E1?: number[][]
  E2?: number[][]
  E3?: number[][]
  E4?: number[][]
  E5?: number[][]
}

const CLASS_COLORS = {
  E0: '#10b981', // green
  E1: '#3b82f6', // blue
  E2: '#8b5cf6', // purple
  E3: '#f59e0b', // amber
  E4: '#ef4444', // red
  E5: '#ec4899', // pink
}

const CLASS_LABELS = {
  E0: 'Healthy',
  E1: 'R1 Open',
  E2: 'R2 Open',
  E3: 'L1 Fault',
  E4: 'C1 Fault',
  E5: 'Short',
}

export default function DistributionPlot({ data }: { data: DistributionData }) {
  if (!data || Object.keys(data).length === 0) {
    return (
      <div className="h-96 bg-slate-50 rounded flex items-center justify-center text-muted-foreground">
        No distribution data available
      </div>
    )
  }

  // Find min/max for scaling
  let minX = Infinity, maxX = -Infinity
  let minY = Infinity, maxY = -Infinity

  Object.values(data).forEach((points) => {
    if (Array.isArray(points)) {
      points.forEach(([x, y]) => {
        if (typeof x === 'number' && typeof y === 'number') {
          minX = Math.min(minX, x)
          maxX = Math.max(maxX, x)
          minY = Math.min(minY, y)
          maxY = Math.max(maxY, y)
        }
      })
    }
  })

  // Add padding
  const padX = (maxX - minX) * 0.1
  const padY = (maxY - minY) * 0.1
  minX -= padX
  maxX += padX
  minY -= padY
  maxY += padY

  const width = 600
  const height = 400
  const margin = { top: 20, right: 20, bottom: 50, left: 60 }
  const plotWidth = width - margin.left - margin.right
  const plotHeight = height - margin.top - margin.bottom

  const scaleX = (x: number) => margin.left + ((x - minX) / (maxX - minX)) * plotWidth
  const scaleY = (y: number) => height - margin.bottom - ((y - minY) / (maxY - minY)) * plotHeight

  return (
    <div className="w-full space-y-4">
      <svg width={width} height={height} className="bg-white rounded border">
        {/* Grid lines */}
        <g opacity="0.2">
          {[0, 0.25, 0.5, 0.75, 1].map((frac) => {
            const x = margin.left + frac * plotWidth
            const y = height - margin.bottom - frac * plotHeight
            return (
              <g key={frac}>
                <line x1={x} y1={margin.top} x2={x} y2={height - margin.bottom} stroke="#ccc" />
                <line x1={margin.left} y1={y} x2={width - margin.right} y2={y} stroke="#ccc" />
              </g>
            )
          })}
        </g>

        {/* Axes */}
        <line
          x1={margin.left}
          y1={height - margin.bottom}
          x2={width - margin.right}
          y2={height - margin.bottom}
          stroke="black"
          strokeWidth="2"
        />
        <line
          x1={margin.left}
          y1={margin.top}
          x2={margin.left}
          y2={height - margin.bottom}
          stroke="black"
          strokeWidth="2"
        />

        {/* Axis labels */}
        <text
          x={width / 2}
          y={height - 10}
          textAnchor="middle"
          className="text-sm fill-slate-700"
        >
          V₀ Real Part (V)
        </text>
        <text
          x={15}
          y={height / 2}
          textAnchor="middle"
          transform={`rotate(-90, 15, ${height / 2})`}
          className="text-sm fill-slate-700"
        >
          V₀ Imaginary Part (V)
        </text>

        {/* Plot points for each class */}
        {Object.entries(data).map(([className, points]) => {
          if (!Array.isArray(points)) return null

          const color = CLASS_COLORS[className as keyof typeof CLASS_COLORS] || '#666'

          return (
            <g key={className}>
              {points.map(([x, y], idx) => {
                if (typeof x !== 'number' || typeof y !== 'number') return null

                return (
                  <circle
                    key={idx}
                    cx={scaleX(x)}
                    cy={scaleY(y)}
                    r="3"
                    fill={color}
                    opacity="0.6"
                  />
                )
              })}
            </g>
          )
        })}
      </svg>

      {/* Legend */}
      <div className="flex flex-wrap gap-4 text-sm">
        {Object.keys(data).map((className) => {
          const color = CLASS_COLORS[className as keyof typeof CLASS_COLORS] || '#666'
          const label = CLASS_LABELS[className as keyof typeof CLASS_LABELS] || className
          const count = data[className as keyof DistributionData]?.length || 0

          return (
            <div key={className} className="flex items-center gap-2">
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: color }}
              />
              <span className="font-medium">{className}:</span>
              <span className="text-muted-foreground">{label} ({count} points)</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
