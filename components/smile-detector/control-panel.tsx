"use client"

import { Button } from "../ui/button"
import { Play, Pause } from "lucide-react"

type ControlPanelProps = {
  isDetecting: boolean
  modelsLoaded: boolean
  modelError: string | null
  startDetection: () => void
  stopDetection: () => void
}

export function ControlPanel({
  isDetecting,
  modelsLoaded,
  modelError,
  startDetection,
  stopDetection,
}: ControlPanelProps) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
      <Button 
        onClick={startDetection} 
        disabled={isDetecting || !modelsLoaded || !!modelError} 
        className="py-6 text-lg"
      >
        <Play className="mr-2 h-5 w-5" />
        Start
      </Button>
      <Button 
        onClick={stopDetection} 
        disabled={!isDetecting} 
        variant="destructive" 
        className="py-6 text-lg"
      >
        <Pause className="mr-2 h-5 w-5" />
        Stop
      </Button>
    </div>
  )
}