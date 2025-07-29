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

  const detectionIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const startTimeRef = useRef<number>(0)
  const faceTimeRef = useRef<number>(0)
  const smileTimeRef = useRef<number>(0)

  // Simulate model loading
  useEffect(() => {
    const loadModels = async () => {
      try {
        // Simulate loading time
        await new Promise((resolve) => setTimeout(resolve, 2000))
        setModelsLoaded(true)
      } catch (error) {
        setModelError("Failed to load face recognition models")
      }
    }
    loadModels()
  }, [])

  const toggleWebcam = useCallback(async () => {
    if (videoStream) {
      // Stop webcam
      videoStream.getTracks().forEach((track) => track.stop())
      setVideoStream(null)
      if (videoRef.current) {
        videoRef.current.srcObject = null
      }
    } else {
      // Start webcam
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480 },
        })
        setVideoStream(stream)
        if (videoRef.current) {
          videoRef.current.srcObject = stream
        }
      } catch (error) {
        setModelError("Failed to access webcam")
      }
    }
  }, [videoStream, videoRef])

  const simulateDetection = useCallback(() => {
    // Simulate face and smile detection with random values
    const faceDetected = Math.random() > 0.2 // 80% chance of face detection
    const smiling = faceDetected && Math.random() > 0.4 // 60% chance of smiling when face detected
    const confidence = faceDetected ? 0.7 + Math.random() * 0.3 : 0

    const detection: Detection = {
      faceDetected,
      smiling,
      confidence,
    }

    setLastDetection(detection)

    // Update stats
    const currentTime = Date.now()
    const elapsed = (currentTime - startTimeRef.current) / 1000

    if (faceDetected) {
      faceTimeRef.current += 1
    }
    if (smiling) {
      smileTimeRef.current += 1
    }

    const newStats: Stats = {
      totalTime: elapsed,
      faceTime: faceTimeRef.current,
      smileTime: smileTimeRef.current,
      engagement: elapsed > 0 ? (smileTimeRef.current / elapsed) * 100 : 0,
    }

    setStats(newStats)

    // Draw detection on canvas
    if (canvasRef.current && videoRef.current) {
      const canvas = canvasRef.current
      const ctx = canvas.getContext("2d")
      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height)

        if (faceDetected) {
          // Draw a simple rectangle to simulate face detection
          const x = canvas.width * 0.3
          const y = canvas.height * 0.2
          const width = canvas.width * 0.4
          const height = canvas.height * 0.6

          ctx.strokeStyle = smiling ? "#22c55e" : "#3b82f6"
          ctx.lineWidth = 3
          ctx.strokeRect(x, y, width, height)

          // Draw label
          ctx.fillStyle = smiling ? "#22c55e" : "#3b82f6"
          ctx.font = "16px Arial"
          ctx.fillText(
            smiling
              ? `Smiling (${(confidence * 100).toFixed(0)}%)`
              : `Face Detected (${(confidence * 100).toFixed(0)}%)`,
            x,
            y - 10,
          )
        }
      }
    }
  }, [canvasRef, videoRef])

  const startDetection = useCallback(() => {
    if (!videoStream || isDetecting) return

    setIsDetecting(true)
    startTimeRef.current = Date.now()
    faceTimeRef.current = 0
    smileTimeRef.current = 0

    detectionIntervalRef.current = setInterval(simulateDetection, 100)
  }, [videoStream, isDetecting, simulateDetection])

  const stopDetection = useCallback(() => {
    setIsDetecting(false)
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current)
      detectionIntervalRef.current = null
    }

    // Clear canvas
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext("2d")
      if (ctx) {
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
      }
    }
  }, [canvasRef])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (detectionIntervalRef.current) {
        clearInterval(detectionIntervalRef.current)
      }
      if (videoStream) {
        videoStream.getTracks().forEach((track) => track.stop())
      }
    }
  }, [videoStream])

  return {
    lastDetection,
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
