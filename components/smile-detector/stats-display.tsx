import { Progress } from "../ui/progress"
import { formatTime } from "../../src/lib/utils"

type Stats = {
  totalTime: number
  faceTime: number
  smileTime: number
  engagement: number
}

type StatsDisplayProps = {
  stats: Stats
}

export function StatsDisplay({ stats }: StatsDisplayProps) {
  return (
    <div className="w-full space-y-6 pt-4 border-t border-border">
      <div className="pt-6">
        <div className="flex justify-between items-center mb-2">
          <span className="text-base font-medium text-muted-foreground">Engagement Score</span>
          <span className="text-xl font-bold">{stats.engagement.toFixed(1)}%</span>
        </div>
        <Progress value={stats.engagement} className="w-full h-3" />
      </div>
      <div className="grid grid-cols-3 gap-4 text-center">
        <div>
          <p className="text-base text-muted-foreground">Total Time</p>
          <p className="text-3xl font-semibold tracking-tighter">{formatTime(stats.totalTime)}</p>
        </div>
        <div>
          <p className="text-base text-muted-foreground">Face Time</p>
          <p className="text-3xl font-semibold tracking-tighter">{formatTime(stats.faceTime)}</p>
        </div>
        <div>
          <p className="text-base text-muted-foreground">Smile Time</p>
          <p className="text-3xl font-semibold tracking-tighter">{formatTime(stats.smileTime)}</p>
        </div>
      </div>
    </div>
  )
}
