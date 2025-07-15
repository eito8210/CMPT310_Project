"use client"

import type React from "react"

import { useState, useRef, useEffect, useCallback } from "react"
import * as faceapi from "face-api.js"

const MODEL_URL = "/models"

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

  const lastFrameTimeRef = useRef(Date.now())
  const detectionIntervalRef = useRef<number | undefined>(undefined)
  const timerRef = useRef<number | undefined>(undefined)

  useEffect(() => {
    const loadModels = async () => {
      try {
        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
          faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
          faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL),
        ])
        setModelsLoaded(true)
      } catch (error) {
        console.error("Failed to load models: ", error)
        setModelError("Failed to load models. Please ensure model files are placed in the public/models directory.")
      }
    }
    loadModels()
  }, [])

  const stopWebcam = useCallback(() => {
    if (isDetecting) {
      setIsDetecting(false)
    }
    if (videoStream) {
      videoStream.getTracks().forEach((track) => track.stop())
      setVideoStream(null)
    }
  }, [isDetecting, videoStream])

  useEffect(() => {
    return () => {
      stopWebcam()
      if (detectionIntervalRef.current) {
        window.clearInterval(detectionIntervalRef.current)
      }
      if (timerRef.current) {
        window.clearInterval(timerRef.current)
      }
    }
  }, [stopWebcam])

  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: {} })
      setVideoStream(stream)
      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }
    } catch (error) {
      console.error("Failed to access webcam: ", error)
    }
  }

  const toggleWebcam = async () => {
    if (videoStream) {
      stopWebcam()
    } else {
      await startWebcam()
    }
  }

  const startDetection = () => {
    if (!videoStream || !modelsLoaded) return
    setIsDetecting(true)
    lastFrameTimeRef.current = Date.now()
    setStats({ totalTime: 0, faceTime: 0, smileTime: 0, engagement: 0 })

    timerRef.current = window.setInterval(() => {
      setStats((prev) => ({ ...prev, totalTime: prev.totalTime + 1 }))
    }, 1000)
  }

  const stopDetection = () => {
    setIsDetecting(false)
    if (detectionIntervalRef.current) {
      window.clearInterval(detectionIntervalRef.current)
    }
    if (timerRef.current) {
      window.clearInterval(timerRef.current)
    }
    if (canvasRef.current) {
      const context = canvasRef.current.getContext("2d")
      context?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
    }
  }

  const handleDetection = useCallback(async () => {
    if (!isDetecting || !videoRef.current || videoRef.current.paused || videoRef.current.ended) {
      return
    }

    const video = videoRef.current
    const canvas = canvasRef.current
    if (!canvas) return

    const displaySize = { width: video.videoWidth, height: video.videoHeight }
    faceapi.matchDimensions(canvas, displaySize)

    const detections = await faceapi
      .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceExpressions()
    const resizedDetections = faceapi.resizeResults(detections, displaySize)

    const context = canvas.getContext("2d")
    context?.clearRect(0, 0, canvas.width, canvas.height)

    const currentTime = Date.now()
    const deltaTime = (currentTime - lastFrameTimeRef.current) / 1000
    lastFrameTimeRef.current = currentTime

    let faceDetectedInFrame = false
    let smileDetectedInFrame = false

    if (resizedDetections.length > 0) {
      faceDetectedInFrame = true
      resizedDetections.forEach((detection) => {
        const happiness = detection.expressions.happy
        const label =
          happiness > 0.5 ? `Smiling (${happiness.toFixed(2)})` : `Not Smiling (${(1 - happiness).toFixed(2)})`
        const color = happiness > 0.5 ? "green" : "red"
        if (happiness > 0.5) smileDetectedInFrame = true

        new faceapi.draw.DrawBox(detection.detection.box, {
          label,
          boxColor: color,
          drawLabelOptions: { fontColor: "white", backgroundColor: color },
        }).draw(canvas)
      })
    }

    setStats((prev) => {
      const newFaceTime = prev.faceTime + (faceDetectedInFrame ? deltaTime : 0)
      const newSmileTime = prev.smileTime + (smileDetectedInFrame ? deltaTime : 0)
      const newEngagement = newFaceTime > 0 ? (newSmileTime / newFaceTime) * 100 : 0
      return { ...prev, faceTime: newFaceTime, smileTime: newSmileTime, engagement: newEngagement }
    })
  }, [isDetecting, videoRef, canvasRef])

  useEffect(() => {
    if (isDetecting) {
      detectionIntervalRef.current = window.setInterval(handleDetection, 200)
    } else {
      if (detectionIntervalRef.current) {
        window.clearInterval(detectionIntervalRef.current)
      }
    }
    return () => {
      if (detectionIntervalRef.current) {
        window.clearInterval(detectionIntervalRef.current)
      }
    }
  }, [isDetecting, handleDetection])

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
