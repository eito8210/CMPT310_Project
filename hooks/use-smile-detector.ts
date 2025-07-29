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

  // Pythonモデルの初期化
  useEffect(() => {
    const initializeModels = async () => {
      try {
        setModelError("Pythonサーバーに接続中...")

        const response = await fetch("/api/python/initialize", {
          method: "POST",
        })

        const data = await response.json()

        if (data.success) {
          setModelsLoaded(true)
          setModelError(null)
        } else {
          setModelError("顔認識モデルの初期化に失敗しました")
        }
      } catch (error) {
        setModelError("Pythonサーバーに接続できません。face_detection_api.pyが実行されているか確認してください。")
      }
    }

    initializeModels()
  }, [])

  // 検出データのポーリング
  const pollDetectionData = useCallback(async () => {
    try {
      // 現在の検出結果を取得
      const detectionResponse = await fetch("/api/python/detection")
      const detectionData = await detectionResponse.json()

      // 現在の統計を取得
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

      // キャンバスに検出結果を描画
      if (canvasRef.current && videoRef.current && detectionData.faceDetected) {
        const canvas = canvasRef.current
        const ctx = canvas.getContext("2d")
        const video = videoRef.current
        
        if (ctx) {
          ctx.clearRect(0, 0, canvas.width, canvas.height)

          // 表示サイズに基づいて矩形を描画
          const displayWidth = canvas.width
          const displayHeight = canvas.height
          
          // ビデオのアスペクト比を考慮した実際の表示領域を計算
          const videoAspectRatio = video.videoWidth / video.videoHeight
          const displayAspectRatio = displayWidth / displayHeight
          
          let actualVideoWidth, actualVideoHeight, offsetX, offsetY
          
          if (videoAspectRatio > displayAspectRatio) {
            // ビデオが横長の場合
            actualVideoWidth = displayWidth
            actualVideoHeight = displayWidth / videoAspectRatio
            offsetX = 0
            offsetY = (displayHeight - actualVideoHeight) / 2
          } else {
            // ビデオが縦長の場合
            actualVideoWidth = displayHeight * videoAspectRatio
            actualVideoHeight = displayHeight
            offsetX = (displayWidth - actualVideoWidth) / 2
            offsetY = 0
          }

          // 顔検出矩形を実際のビデオ表示領域内に描画
          const x = offsetX + actualVideoWidth * 0.3
          const y = offsetY + actualVideoHeight * 0.2
          const width = actualVideoWidth * 0.4
          const height = actualVideoHeight * 0.6

          ctx.strokeStyle = detectionData.smiling ? "#22c55e" : "#3b82f6"
          ctx.lineWidth = 3
          ctx.strokeRect(x, y, width, height)

          // ラベルを描画
          ctx.fillStyle = detectionData.smiling ? "#22c55e" : "#3b82f6"
          ctx.font = "16px Arial"
          const label = detectionData.smiling
            ? `笑顔検出 (${(detectionData.confidence * 100).toFixed(0)}%)`
            : `顔検出 (${(detectionData.confidence * 100).toFixed(0)}%)`
          ctx.fillText(label, x, y - 10)
        }
      }
    } catch (error) {
      console.error("検出データの取得エラー:", error)
    }
  }, [canvasRef, videoRef])

  const toggleWebcam = useCallback(async () => {
    if (videoStream) {
      // ウェブカメラを停止
      videoStream.getTracks().forEach((track) => track.stop())
      setVideoStream(null)
      if (videoRef.current) {
        videoRef.current.srcObject = null
      }
    } else {
      // ウェブカメラを開始
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
      } catch (error) {
        setModelError("ウェブカメラにアクセスできませんでした")
      }
    }
  }, [videoStream, videoRef])

  const startDetection = useCallback(async () => {
    if (!videoStream || isDetecting) return

    try {
      const response = await fetch("/api/python/start-detection", {
        method: "POST",
      })
      const data = await response.json()

      if (data.success) {
        setIsDetecting(true)
        // 検出データのポーリングを開始
        pollingIntervalRef.current = setInterval(pollDetectionData, 100)
      } else {
        setModelError("検出を開始できませんでした")
      }
    } catch (error) {
      setModelError("Pythonサーバーに接続できませんでした")
    }
  }, [videoStream, isDetecting, pollDetectionData])

  const stopDetection = useCallback(async () => {
    try {
      await fetch("/api/python/stop-detection", {
        method: "POST",
      })

      setIsDetecting(false)
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current)
        pollingIntervalRef.current = null
      }

      // キャンバスをクリア
      if (canvasRef.current) {
        const ctx = canvasRef.current.getContext("2d")
        if (ctx) {
          ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
        }
      }
    } catch (error) {
      console.error("検出停止エラー:", error)
    }
  }, [canvasRef])

  // クリーンアップ
  useEffect(() => {
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current)
      }
      if (videoStream) {
        videoStream.getTracks().forEach((track) => track.stop())
      }
      // アンマウント時に検出を停止
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
    toggleWebcam,
    startDetection,
    stopDetection,
  }
}