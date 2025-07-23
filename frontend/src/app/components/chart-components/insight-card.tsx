interface Insight {
  title: string
  description: string
  emoji: string
}

interface InsightCardComponentProps {
  insight: Insight
  type: "bull" | "bear"
}

export function InsightCardComponent({ insight, type }: InsightCardComponentProps) {
  const getTypeStyles = () => {
    switch (type) {
      case "bull":
        return "border-l-4 border-l-[#00d237] glass-card neon-glow"
      case "bear":
        return "border-l-4 border-l-red-500 glass-card"
      default:
        return "border-l-4 border-l-[#D8D8E5] glass-card"
    }
  }

  return (
    <div className={`rounded-xl p-3 ${getTypeStyles()}`}>
      <div className="flex items-start gap-2">
        <span className="text-lg">{insight.emoji}</span>
        <div className="flex-1 min-w-0">
          <h3 className="text-sm font-semibold gradient-text font-['Roobert'] mb-1">{insight.title}</h3>
          <p className="text-xs text-[#575758] font-['Plus_Jakarta_Sans'] leading-relaxed">{insight.description}</p>
        </div>
      </div>
    </div>
  )
}
