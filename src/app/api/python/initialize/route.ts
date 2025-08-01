import { NextResponse } from "next/server"

const PYTHON_API_URL = "http://localhost:5001"

export async function POST() {
  try {
    console.log("Attempting to connect to Python API...")
    
    const response = await fetch(`${PYTHON_API_URL}/api/initialize`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      // Next.js 14でのfetch設定
      cache: 'no-store',
    })

    console.log("Response status:", response.status)
    
    if (!response.ok) {
      const errorText = await response.text()
      console.error("Error response:", errorText)
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const data = await response.json()
    console.log("Success response:", data)
    
    return NextResponse.json({ ...data, success: true })
  } catch (error) {
    console.error("Python API connection error:", error)
    
    // エラーの詳細情報を返す
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : "Unknown error",
        success: false,
        detail: "Cannot connect to Python server at " + PYTHON_API_URL,
      },
      { status: 500 }
    )
  }
}