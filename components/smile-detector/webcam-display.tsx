import type React from "react"
import { VideoOff } from "lucide-react"

type WebcamDisplayProps = {
  videoRef: React.RefObject<HTMLVideoElement>
  canvasRef: React.RefObject<HTMLCanvasElement>
  videoStream: MediaStream | null
}

export function WebcamDisplay({ videoRef, canvasRef, videoStream }: WebcamDisplayProps) {
  return (
    <div className="relative w-full aspect-video bg-muted rounded-md overflow-hidden mt-4">
      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        className="w-full h-full object-cover"
        onLoadedMetadata={() => {
          if (videoRef.current && canvasRef.current) {
            canvasRef.current.width = videoRef.current.videoWidth
            canvasRef.current.height = videoRef.current.videoHeight
          }
        }}
      />
      <canvas ref={canvasRef} className="absolute top-0 left-0" />
      {!videoStream && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/50">
          <VideoOff className="w-16 h-16 text-white" />
        </div>
      )}
    </div>
  )
}
