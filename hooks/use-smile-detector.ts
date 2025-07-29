"use client"

import type React from "react"
import { useState, useEffect, useCallback, useRef } from "react"

type Detection = {
  faceDetected: boolean
  smiling: boolean
  confidence: number
}

type Stats = {
  totalTime: number
  faceTime: number
  smileTime: number
  engagement: number
}

export function useSmileDetector(
  videoRef: React.RefObject<HTMLVideoElement>,
  canvasRef: React.RefObject<HTMLCanvasElement>,
) {
  const [modelsLoaded, setModelsLoaded] = useState(false)
  const [modelError, setModelError] = useState<string | null>(null)
  const [isDetecting, setIsDetecting] = useState(false)
  const [videoStream, setVideoStream] = useState<MediaStream | null>(null)
  const [lastDetection, setLastDetection] = useState<Detection | null>(null)
  const [stats, setStats] = useState<Stats>({
    totalTime: 0,
    faceTime: 0,
    smileTime: 0,
    engagement: 0,
  })

  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null)

  // Initialize Python models
  useEffect(() => {
    const initializeModels = async () => {
      try {
        setModelError("Connecting to Python server...")

        const response = await fetch("/api/python/initialize", {
          method: "POST",
        })

        const data = await response.json()

        if (data.success) {
          setModelsLoaded(true)
          setModelError(null)
        } else {
          setModelError("Failed to initialize face recognition models")
        }
      } catch (error) {
        setModelError("Cannot connect to Python server. Please check if face_detection_api.py is running.")
      }
    }

    initializeModels()
  }, [])

  // Poll detection data
  const pollDetectionData = useCallback(async () => {
    try {
      // Get current detection results
      const detectionResponse = await fetch("/api/python/detection")
      const detectionData = await detectionResponse.json()

      // Get current statistics
      const statsResponse = await fetch("/api/python/stats")
      const statsData = await statsResponse.json()

      setLastDetection({
        faceDetected: detectionData.faceDetected,
        smiling: detectionData.smiling,
        confidence: detectionData.confidence,
      })

      setStats({
        totalTime: statsData.totalTime,
        faceTime: statsData.faceTime,
        smileTime: statsData.smileTime,
        engagement: statsData.engagement,
      })

      // Draw detection results on canvas
      if (canvasRef.current && videoRef.current && detectionData.faceDetected) {
        const canvas = canvasRef.current
        const ctx = canvas.getContext("2d")
        const video = videoRef.current
        
        if (ctx) {
          ctx.clearRect(0, 0, canvas.width, canvas.height)

          // Draw rectangle based on display size
          const displayWidth = canvas.width
          const displayHeight = canvas.height
          
          // Calculate actual display area considering video aspect ratio
          const videoAspectRatio = video.videoWidth / video.videoHeight
          const displayAspectRatio = displayWidth / displayHeight
          
          let actualVideoWidth, actualVideoHeight, offsetX, offsetY
          
          if (videoAspectRatio > displayAspectRatio) {
            // Video is wider
            actualVideoWidth = displayWidth
            actualVideoHeight = displayWidth / videoAspectRatio
            offsetX = 0
            offsetY = (displayHeight - actualVideoHeight) / 2
          } else {
            // Video is taller
            actualVideoWidth = displayHeight * videoAspectRatio
            actualVideoHeight = displayHeight
            offsetX = (displayWidth - actualVideoWidth) / 2
            offsetY = 0
          }

          // Draw face detection rectangle within actual video display area
          const x = offsetX + actualVideoWidth * 0.3
          const y = offsetY + actualVideoHeight * 0.2
          const width = actualVideoWidth * 0.4
          const height = actualVideoHeight * 0.6

          ctx.strokeStyle = detectionData.smiling ? "#22c55e" : "#3b82f6"
          ctx.lineWidth = 3
          ctx.strokeRect(x, y, width, height)

          // Draw label
          ctx.fillStyle = detectionData.smiling ? "#22c55e" : "#3b82f6"
          ctx.font = "16px Arial"
          const label = detectionData.smiling
            ? `Smile Detected (${(detectionData.confidence * 100).toFixed(0)}%)`
            : `Face Detected (${(detectionData.confidence * 100).toFixed(0)}%)`
          ctx.fillText(label, x, y - 10)
        }
      }
    } catch (error) {
      console.error("Error fetching detection data:", error)
    }
  }, [canvasRef, videoRef])

  // Start webcam function
  const startWebcam = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { 
          width: { ideal: 640, max: 1280 },
          height: { ideal: 480, max: 720 },
          frameRate: { ideal: 30, max: 30 },
          aspectRatio: { ideal: 4/3 }
        }
      })
      setVideoStream(stream)
      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }
      return true
    } catch (error) {
      console.error("Webcam error:", error)
      setModelError("Could not access webcam")
      return false
    }
  }, [videoRef])

  // Stop webcam function
  const stopWebcam = useCallback(() => {
    if (videoStream) {
      videoStream.getTracks().forEach((track) => track.stop())
      setVideoStream(null)
      if (videoRef.current) {
        videoRef.current.srcObject = null
      }
    }
  }, [videoStream, videoRef])

  // Start detection (start camera + start detection)
  const startDetection = useCallback(async () => {
    if (isDetecting) return

    try {
      // 1. First start webcam
      const cameraStarted = await startWebcam()
      if (!cameraStarted) return

      // 2. Reset statistics before starting
      const resetResponse = await fetch("/api/python/reset-stats", {
        method: "POST",
      })
      
      if (resetResponse.ok) {
        // Reset local stats as well
        setStats({
          totalTime: 0,
          faceTime: 0,
          smileTime: 0,
          engagement: 0,
        })
        setLastDetection(null)
      }

      // 3. Start Python detection
      const response = await fetch("/api/python/start-detection", {
        method: "POST",
      })
      const data = await response.json()

      if (data.success) {
        setIsDetecting(true)
        // Start polling detection data
        pollingIntervalRef.current = setInterval(pollDetectionData, 100)
      } else {
        setModelError("Failed to start detection")
        // Also stop camera
        stopWebcam()
      }
    } catch (error) {
      console.error("Detection start error:", error)
      setModelError("Could not connect to Python server")
      // Also stop camera
      stopWebcam()
    }
  }, [isDetecting, startWebcam, stopWebcam, pollDetectionData])

  // Stop detection (stop detection + stop camera)
  const stopDetection = useCallback(async () => {
    try {
      // 1. Stop Python detection
      await fetch("/api/python/stop-detection", {
        method: "POST",
      })

      // 2. Stop polling
      setIsDetecting(false)
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current)
        pollingIntervalRef.current = null
      }

      // 3. Clear canvas
      if (canvasRef.current) {
        const ctx = canvasRef.current.getContext("2d")
        if (ctx) {
          ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
        }
      }

      // 4. Stop webcam
      stopWebcam()
      
    } catch (error) {
      console.error("Detection stop error:", error)
    }
  }, [canvasRef, stopWebcam])

  // Cleanup
  useEffect(() => {
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current)
      }
      if (videoStream) {
        videoStream.getTracks().forEach((track) => track.stop())
      }
      // Stop detection on unmount
      fetch("/api/python/stop-detection", { method: "POST" }).catch(() => {})
    }
  }, [videoStream])

  return {
    lastDetection,
    modelsLoaded,
    isDetecting,
    videoStream,
    modelError,
    stats,
    startDetection,
    stopDetection,
  }
}