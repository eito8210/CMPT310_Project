import type React from "react"
import { VideoOff, Play } from "lucide-react"

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
            
            // キャンバスサイズを表示サイズに合わせる
            canvas.width = displayWidth
            canvas.height = displayHeight
            
            // キャンバスの表示サイズも同じに設定
            canvas.style.width = `${displayWidth}px`
            canvas.style.height = `${displayHeight}px`
          }
        }}
      />
      <canvas 
        ref={canvasRef} 
        className="absolute top-0 left-0 pointer-events-none"
      />
      {!videoStream && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/50 text-white">
          <Play className="w-16 h-16 mb-4 opacity-70" />
          <p className="text-lg font-medium">Press Start to begin</p>
          <p className="text-sm opacity-70 mt-1">Camera will activate automatically</p>
        </div>
      )}
    </div>
  )
}