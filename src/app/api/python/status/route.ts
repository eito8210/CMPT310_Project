import { type NextRequest, NextResponse } from "next/server"

const PYTHON_API_URL = "http://localhost:5001"

export async function GET(request: NextRequest) {
  try {
    const response = await fetch(`${PYTHON_API_URL}/api/status`, {
      cache: "no-store",
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error("Status fetch error:", error)
    return NextResponse.json(
      {
        modelsLoaded: false,
        detecting: false,
        error: "ステータスを取得できませんでした",
      },
      { status: 500 },
    )
  }
}
