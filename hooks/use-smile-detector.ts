"use client"

import type React from "react"
import { useState, useRef, useEffect, useCallback } from "react"

export function useSmileDetector(
  videoRef: React.RefObject<HTMLVideoElement>,
  canvasRef: React.RefObject<HTMLCanvasElement>,
) {
  const [modelsLoaded, setModelsLoaded] = useState(false)
  const [isDetecting, setIsDetecting] = useState(false)
  const [videoStream, setVideoStream] = useState<MediaStream | null>(null)
  const [modelError, setModelError] = useState<string | null>(null)
  const [stats, setStats] = useState({
    totalTime: 0,
    faceTime: 0,
    smileTime: 0,
    engagement: 0,
  })

  // 統計保持フラグを追加
  const [preserveStats, setPreserveStats] = useState(false)

  const wsRef = useRef<WebSocket | null>(null)
  const frameIntervalRef = useRef<number | undefined>(undefined)
  const reconnectTimeoutRef = useRef<number | undefined>(undefined)
  const isConnectingRef = useRef(false)

  // Check Python backend health and model status
  useEffect(() => {
    let isMounted = true

    const checkBackend = async () => {
      try {
        console.log("Checking backend health...")
        const response = await fetch("http://localhost:8000/health")
        const data = await response.json()
        console.log("Backend response:", data)

        if (isMounted) {
          if (data.models_loaded) {
            setModelsLoaded(true)
            setModelError(null)
          } else {
            setModelError("Python models not loaded on backend")
          }
        }
      } catch (error) {
        console.error("Backend connection error:", error)
        if (isMounted) {
          setModelError("Cannot connect to Python backend. Please ensure FastAPI server is running on port 8000.")
        }
      }
    }

    checkBackend()

    return () => {
      isMounted = false
    }
  }, [])

  // Initialize WebSocket connection with improved stability
  const initWebSocket = useCallback(() => {
    if (isConnectingRef.current || wsRef.current?.readyState === WebSocket.OPEN) {
      console.log("WebSocket already connected or connecting")
      return
    }

    isConnectingRef.current = true
    console.log("Attempting to connect to WebSocket...")

    try {
      wsRef.current = new WebSocket("ws://localhost:8000/ws/smile-detection")

      wsRef.current.onopen = () => {
        console.log("✅ WebSocket connected successfully")
        isConnectingRef.current = false
        setModelError(null)
      }

      wsRef.current.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data)
          console.log("📨 Received WebSocket message:", message.type, message)

          if (message.type === "detection" && message.data) {
            const { detections, stats: newStats } = message.data
            console.log("🎯 Detection data received:", { detections, stats: newStats })

            // Update stats from Python backend
            if (newStats) {
              console.log("📊 Updating stats:", newStats)
              setStats({
                totalTime: newStats.totalTime,
                faceTime: newStats.faceTime,
                smileTime: newStats.smileTime,
                engagement: newStats.engagement,
              })
            }

            // Draw detections on canvas
            if (canvasRef.current && detections && detections.length > 0) {
              const canvas = canvasRef.current
              const ctx = canvas.getContext("2d")
              if (ctx) {
                console.log("🎨 Drawing", detections.length, "detections")
                ctx.clearRect(0, 0, canvas.width, canvas.height)

                detections.forEach((detection: any, index: number) => {
                  const { x, y, width, height, isSmiling, label } = detection
                  console.log(`✏️ Drawing detection ${index + 1}:`, { x, y, width, height, isSmiling, label })

                  // Draw rectangle
                  ctx.strokeStyle = isSmiling ? "#00ff00" : "#ff0000"
                  ctx.lineWidth = 3
                  ctx.strokeRect(x, y, width, height)

                  // Draw label with background
                  ctx.fillStyle = isSmiling ? "#00ff00" : "#ff0000"
                  ctx.font = "16px Arial"

                  const textMetrics = ctx.measureText(label)
                  ctx.fillRect(x, y - 25, textMetrics.width + 10, 20)

                  ctx.fillStyle = "white"
                  ctx.fillText(label, x + 5, y - 10)
                })
              }
            } else {
              console.log("❌ No detections or canvas not available")
            }
          } else if (message.type === "stopped" && message.finalStats) {
            console.log("⏹️ Detection stopped, final stats:", message.finalStats)
            // 最終統計が有効な場合のみ更新（0でない場合）
            if (message.finalStats.totalTime > 0) {
              setStats({
                totalTime: message.finalStats.totalTime,
                faceTime: message.finalStats.faceTime,
                smileTime: message.finalStats.smileTime,
                engagement: message.finalStats.engagement,
              })
            }
            // 統計が0の場合は現在の値を保持
          } else if (message.type === "started") {
            console.log("🚀 Detection started confirmation received")
          }
        } catch (error) {
          console.error("Error parsing WebSocket message:", error)
        }
      }

      wsRef.current.onerror = (error) => {
        console.error("❌ WebSocket error:", error)
        isConnectingRef.current = false
        setModelError("WebSocket connection failed. Check if FastAPI server is running.")
      }

      wsRef.current.onclose = (event) => {
        console.log("WebSocket closed:", event.code, event.reason)
        isConnectingRef.current = false
      }
    } catch (error) {
      console.error("Failed to create WebSocket:", error)
      isConnectingRef.current = false
      setModelError("Failed to initialize WebSocket connection")
    }
  }, [canvasRef])

  const stopWebcam = useCallback(() => {
    console.log("Stopping webcam...")
    setIsDetecting(false)

    if (videoStream) {
      videoStream.getTracks().forEach((track) => track.stop())
      setVideoStream(null)
    }

    if (frameIntervalRef.current) {
      window.clearInterval(frameIntervalRef.current)
      frameIntervalRef.current = undefined
    }

    if (reconnectTimeoutRef.current) {
      window.clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = undefined
    }

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.close(1000, "Stopping webcam")
    }
  }, [videoStream])

  useEffect(() => {
    return () => {
      console.log("Component unmounting - cleaning up...")
      stopWebcam()
    }
  }, [stopWebcam])

  const startWebcam = async () => {
    try {
      console.log("Starting webcam...")
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
      })
      setVideoStream(stream)
      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }
      console.log("✅ Webcam started successfully")
    } catch (error) {
      console.error("Failed to access webcam:", error)
      setModelError("Failed to access webcam. Please allow camera permissions.")
    }
  }

  const toggleWebcam = async () => {
    if (videoStream) {
      stopWebcam()
    } else {
      await startWebcam()
    }
  }

  // Capture frame and send to Python backend
  const captureFrame = useCallback(() => {
    if (!videoRef.current || !canvasRef.current || !wsRef.current) {
      console.log("Missing refs for frame capture")
      return
    }
    if (wsRef.current.readyState !== WebSocket.OPEN) {
      console.log("WebSocket not open for frame capture")
      return
    }

    const video = videoRef.current
    const canvas = canvasRef.current

    // Check if video is ready
    if (video.videoWidth === 0 || video.videoHeight === 0) {
      console.log("Video not ready yet")
      return
    }

    // Set canvas size to match video
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    // Capture frame
    const ctx = canvas.getContext("2d")
    if (ctx) {
      ctx.drawImage(video, 0, 0)
      const frameData = canvas.toDataURL("image/jpeg", 0.8)

      console.log("Sending frame to backend, size:", frameData.length)

      // Send frame to Python backend
      try {
        wsRef.current.send(
          JSON.stringify({
            type: "frame",
            data: frameData,
          }),
        )
      } catch (error) {
        console.error("Error sending frame:", error)
      }
    }
  }, [videoRef, canvasRef])

  const startDetection = () => {
    console.log("Start detection called", {
      videoStream: !!videoStream,
      modelsLoaded,
      wsState: wsRef.current?.readyState,
    })

    if (!videoStream || !modelsLoaded) {
      console.log("Cannot start detection: videoStream =", !!videoStream, "modelsLoaded =", modelsLoaded)
      return
    }

    // WebSocket接続を確立
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      console.log("Initializing WebSocket for detection...")
      initWebSocket()

      // WebSocket接続を待ってから再試行
      setTimeout(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          startDetection()
        }
      }, 1000)
      return
    }

    console.log("Starting detection...")
    setIsDetecting(true)
    setPreserveStats(false) // 保持フラグをリセット

    // 新しい検出開始時のみ統計をリセット
    console.log("🔄 Resetting stats for new detection session")
    setStats({ totalTime: 0, faceTime: 0, smileTime: 0, engagement: 0 })

    // Send start signal to Python backend
    try {
      wsRef.current.send(JSON.stringify({ type: "start" }))
      console.log("Sent start signal to backend")
    } catch (error) {
      console.error("Error sending start signal:", error)
      return
    }

    // Start capturing frames and sending to Python backend
    frameIntervalRef.current = window.setInterval(captureFrame, 500) // 2 FPS for debugging
    console.log("Started frame capture interval")
  }

  const stopDetection = () => {
    console.log("Stopping detection...")
    setIsDetecting(false)
    setPreserveStats(true) // 統計保持フラグを設定

    if (frameIntervalRef.current) {
      window.clearInterval(frameIntervalRef.current)
      frameIntervalRef.current = undefined
    }

    // Send stop signal to Python backend
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(JSON.stringify({ type: "stop" }))
        console.log("Sent stop signal to backend")
      } catch (error) {
        console.error("Error sending stop signal:", error)
      }
    }

    // Clear canvas but preserve stats
    if (canvasRef.current) {
      const context = canvasRef.current.getContext("2d")
      context?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
    }

    // 統計は保持される - リセットしない
    console.log("📊 Stats preserved after stop:", stats)
  }

  return {
    modelsLoaded,
    isDetecting,
    videoStream,
    modelError,
    stats,
    toggleWebcam,
    startDetection,
    stopDetection,
  }
}
