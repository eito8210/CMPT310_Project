import { VideoOff } from "lucide-react"
import { Alert, AlertDescription, AlertTitle } from "../ui/alert"
import { Terminal } from "lucide-react"

type StatusAlertProps = {
  modelError: string | null
  modelsLoaded: boolean
}

export function StatusAlert({ modelError, modelsLoaded }: StatusAlertProps) {
  if (modelError) {
    return (
      <Alert variant="destructive">
        <Terminal className="h-4 w-4" />
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>{modelError}</AlertDescription>
      </Alert>
    )
  }

  if (!modelsLoaded) {
    return (
      <Alert>
        <Terminal className="h-4 w-4" />
        <AlertTitle>Loading Models</AlertTitle>
        <AlertDescription>Please wait while we load the face recognition models...</AlertDescription>
      </Alert>
    )
  }

  return null
}

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
