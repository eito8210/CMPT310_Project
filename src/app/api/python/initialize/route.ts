import { type NextRequest, NextResponse } from "next/server"

const PYTHON_API_URL = "http://localhost:5001"

export async function POST(request: NextRequest) {
  try {
    const response = await fetch(`${PYTHON_API_URL}/api/initialize`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
    })

    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error("Python API connection error:", error)
    return NextResponse.json(
      {
        error: "Pythonサーバーに接続できません。start_server.pyを実行してください。",
        success: false,
      },
      { status: 500 },
    )
  }
}
