import { type NextRequest, NextResponse } from "next/server"

const PYTHON_API_URL = "http://localhost:5001"

export async function GET(request: NextRequest) {
  try {
    const response = await fetch(`${PYTHON_API_URL}/api/stats`, {
      cache: "no-store",
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error("Stats fetch error:", error)
    return NextResponse.json(
      {
        totalTime: 0,
        faceTime: 0,
        smileTime: 0,
        engagement: 0,
        error: "統計データを取得できませんでした",
      },
      { status: 500 },
    )
  }
}
