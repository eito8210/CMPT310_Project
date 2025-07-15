"use client"

import { Button } from "../ui/button"
import { Play, Pause, Video, VideoOff } from "lucide-react"

type ControlPanelProps = {
  isDetecting: boolean
  videoStream: MediaStream | null
  modelsLoaded: boolean
  modelError: string | null
  toggleWebcam: () => void
  startDetection: () => void
  stopDetection: () => void
}

export function ControlPanel({
  isDetecting,
  videoStream,
  modelsLoaded,
  modelError,
  toggleWebcam,
  startDetection,
  stopDetection,
}: ControlPanelProps) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
      <Button
        onClick={toggleWebcam}
        variant="outline"
        disabled={!modelsLoaded || !!modelError}
        className="py-6 text-lg bg-transparent"
      >
        <VideoOff className="mr-2 h-5 w-5" />
        {videoStream ? "Stop Cam" : "Start Cam"}
      </Button>
      <Button onClick={startDetection} disabled={!videoStream || isDetecting || !!modelError} className="py-6 text-lg">
        <Play className="mr-2 h-5 w-5" />
        Start
      </Button>
      <Button onClick={stopDetection} disabled={!isDetecting} variant="destructive" className="py-6 text-lg">
        <Pause className="mr-2 h-5 w-5" />
        Stop
      </Button>
    </div>
  )
}
