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
        className="w-full h-full object-contain"
        onLoadedMetadata={() => {
          if (videoRef.current && canvasRef.current) {
            const video = videoRef.current
            const canvas = canvasRef.current
            
            // 表示サイズを取得
            const displayWidth = video.clientWidth
            const displayHeight = video.clientHeight
            
            // キャンバスサイズを表示サイズに合わせる（重要な修正点）
            canvas.width = displayWidth
            canvas.height = displayHeight
            
            // キャンバスの表示サイズも同じに設定
            canvas.style.width = `${displayWidth}px`
            canvas.style.height = `${displayHeight}px`
            
            console.log("Video display setup:", {
              videoActual: { width: video.videoWidth, height: video.videoHeight },
              display: { width: displayWidth, height: displayHeight },
              canvas: { width: canvas.width, height: canvas.height }
            })
          }
        }}
      />
      <canvas 
        ref={canvasRef} 
        className="absolute top-0 left-0 pointer-events-none"
      />
      {!videoStream && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/50">
          <VideoOff className="w-16 h-16 text-white" />
        </div>
      )}
    </div>
  )
}