"use client"

interface MetricsData {
  accuracy?: number
  per_class?: Record<string, number>
  confusion?: number[][]
}

export default function PerformanceMetrics({ metrics }: { metrics: MetricsData }) {
  const accuracy = metrics?.accuracy || 0
  const perClass = metrics?.per_class || {}
  const confusion = Array.isArray(metrics?.confusion) ? metrics.confusion : []

  return (
    <div className="space-y-6">
      {/* Overall Accuracy */}
      <div>
        <div className="text-sm font-medium mb-2">Overall Accuracy</div>
        <div className="flex items-center gap-3">
          <div className="flex-1 bg-slate-200 rounded-full h-3 overflow-hidden">
            <div className="bg-primary h-full transition-all" style={{ width: `${accuracy * 100}%` }} />
          </div>
          <span className="text-lg font-bold">{(accuracy * 100).toFixed(1)}%</span>
        </div>
      </div>

      {/* Per-Class Metrics */}
      <div>
        <div className="text-sm font-medium mb-3">Per-Class Precision</div>
        <div className="grid grid-cols-2 gap-3">
          {Object.entries(perClass).map(([label, score]) => (
            <div key={label} className="space-y-1">
              <div className="flex justify-between text-xs">
                <span>{label}</span>
                <span className="font-medium">{(score * 100).toFixed(0)}%</span>
              </div>
              <div className="bg-slate-200 rounded h-2 overflow-hidden">
                <div className="bg-primary h-full transition-all" style={{ width: `${score * 100}%` }} />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Confusion Matrix */}
      {confusion.length > 0 && (
        <div>
          <div className="text-sm font-medium mb-4">Confusion Matrix</div>

          <div className="flex flex-col items-center">
            {/* X-Axis Label */}
            <div className="text-xs font-semibold mb-2 text-muted-foreground">Predicted Class</div>

            <div className="flex">
              {/* Y-Axis Label */}
              <div className="flex items-center justify-center mr-2">
                <div className="text-xs font-semibold text-muted-foreground -rotate-90 whitespace-nowrap">
                  Actual Class
                </div>
              </div>

              {/* Matrix */}
              <div className="overflow-x-auto">
                <table className="text-xs border-collapse">
                  <thead>
                    <tr>
                      <th className="p-1"></th>
                      {['E0', 'E1', 'E2', 'E3', 'E4', 'E5'].map(cls => (
                        <th key={cls} className="p-1 font-medium text-muted-foreground">{cls}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {confusion.map((row, i) => {
                      const rowLabel = ['E0', 'E1', 'E2', 'E3', 'E4', 'E5'][i]
                      return (
                        <tr key={i}>
                          <td className="p-1 font-medium text-muted-foreground text-right pr-2">{rowLabel}</td>
                          {row.map((cell, j) => {
                            // Color logic
                            let bgColor = "transparent"
                            let textColor = "inherit"

                            if (i === j) {
                              // Diagonal (Correct)
                              if (cell > 0) {
                                bgColor = `rgba(16, 185, 129, ${Math.min(cell / 60, 1)})` // Green
                                textColor = cell > 30 ? "white" : "black"
                              } else {
                                bgColor = "rgba(16, 185, 129, 0.1)"
                              }
                            } else {
                              // Off-Diagonal (Mistake)
                              if (cell > 0) {
                                bgColor = `rgba(239, 68, 68, ${Math.min(cell / 10, 1)})` // Red
                                textColor = "white"
                              }
                            }

                            return (
                              <td
                                key={j}
                                className="border border-slate-200 p-2 text-center w-10 h-10"
                                style={{ backgroundColor: bgColor, color: textColor }}
                                title={`Actual: ${rowLabel}, Predicted: ${['E0', 'E1', 'E2', 'E3', 'E4', 'E5'][j]}, Count: ${cell}`}
                              >
                                {cell > 0 ? cell : ""}
                              </td>
                            )
                          })}
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="mt-4 flex gap-4 text-xs text-muted-foreground justify-center">
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 bg-emerald-500 rounded"></div>
              <span>Correct</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 bg-red-500 rounded"></div>
              <span>Misclassified</span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
