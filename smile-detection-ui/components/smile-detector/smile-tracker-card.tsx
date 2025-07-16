"use client"

import { useRef } from "react"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "../ui/card"
import { useSmileDetector } from "../hooks/use-smile-detector"
import { StatusAlert } from "./status-alert"
import { WebcamDisplay } from "./webcam-display"
import { ControlPanel } from "./control-panel"
import { StatsDisplay } from "./stats-display"
import { Separator } from "../ui/separator"
import { ThemeToggleButton } from "../theme-toggle-button"

export function SmileTrackerCard() {
  const videoRef = useRef<HTMLVideoElement>(null!)
  const canvasRef = useRef<HTMLCanvasElement>(null!)

  const { modelsLoaded, isDetecting, videoStream, modelError, stats, toggleWebcam, startDetection, stopDetection } =
    useSmileDetector(videoRef, canvasRef)

  return (
    <Card className="w-full max-w-6xl mx-auto shadow-2xl overflow-hidden">
      <CardHeader className="relative text-center p-6 border-b">
        <div className="absolute top-4 right-4">
          <ThemeToggleButton />
        </div>
        <CardTitle className="text-3xl font-bold">Smile Engagement Tracker</CardTitle>
        <CardDescription className="text-lg text-muted-foreground pt-2">
          Turn on your webcam to analyze your engagement.
        </CardDescription>
      </CardHeader>
      <div className="grid md:grid-cols-2">
        {/* Left Column: Webcam */}
        <div className="p-4 md:p-6">
          <WebcamDisplay videoRef={videoRef} canvasRef={canvasRef} videoStream={videoStream} />
        </div>

        {/* Right Column: Controls and Stats */}
        <div className="flex flex-col justify-between p-4 md:p-6 border-t md:border-t-0 md:border-l border-border">
          <div className="flex-grow">
            <StatusAlert modelError={modelError} modelsLoaded={modelsLoaded} />
          </div>

          <div className="flex flex-col gap-6 mt-6">
            <ControlPanel
              isDetecting={isDetecting}
              videoStream={videoStream}
              modelsLoaded={modelsLoaded}
              modelError={modelError}
              toggleWebcam={toggleWebcam}
              startDetection={startDetection}
              stopDetection={stopDetection}
            />
            <Separator />
            <StatsDisplay stats={stats} />
          </div>
        </div>
      </div>
    </Card>
  )
}
