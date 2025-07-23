interface SectionTitleProps {
  title: string
}

export function SectionTitle({ title }: SectionTitleProps) {
  return (
    <div className="border-b border-[#D8D8E5] pb-1">
      <h2 className="text-lg font-semibold gradient-text font-['Roobert']">{title}</h2>
    </div>
  )
}
