"use client"

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"

interface BarChartData {
  ticker: string
  return: number
}

interface BarChartComponentProps {
  data: BarChartData[]
  size?: "normal" | "small"
  onClick?: (data: string) => void
}

export function BarChartComponent({ data, size = "normal", onClick }: BarChartComponentProps) {
  const height = size === "small" ? 80 : 160 // h-20 or h-40
  const padding = size === "small" ? "p-2" : "p-4"
  const fontSize = size === "small" ? 8 : 10
  const tooltipFontSize = size === "small" ? "9px" : "11px"
  return (
    <div className={`glass-card rounded-xl ${padding}`}>
      <div style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 166, 234, 0.2)" />
            <XAxis dataKey="ticker" stroke="#575758" fontSize={fontSize} fontFamily="Plus Jakarta Sans" />
            <YAxis
              stroke="#575758"
              fontSize={fontSize}
              fontFamily="Plus Jakarta Sans"
              tickFormatter={(value) => `${value}%`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "rgba(255, 255, 255, 0.95)",
                backdropFilter: "blur(10px)",
                border: "1px solid rgba(255, 166, 234, 0.2)",
                borderRadius: "8px",
                fontSize: tooltipFontSize,
                fontFamily: "Plus Jakarta Sans",
                boxShadow: "0 8px 32px rgba(255, 166, 234, 0.1)",
              }}
              formatter={(value: number) => [`${value.toFixed(1)}%`, "Return"]}
            />
            <Bar onClick={(data, index) => {
              if (size === "normal") {
                // @ts-ignore
                console.log(data.payload, "clicked")
                // @ts-ignore
                onClick?.(data.payload.ticker as string)
              }
            }} dataKey="return" fill="url(#gradient)" radius={[2, 2, 0, 0]} />
            <defs>
              <linearGradient id="gradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#FF64C8" />
                <stop offset="100%" stopColor="#32C8FF" />
                {/* <stop offset="100%" stopColor="#B478FF" /> */}
              </linearGradient>
            </defs>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
