import { type NextRequest, NextResponse } from "next/server"

const PYTHON_API_URL = "http://localhost:5001"

export async function POST(request: NextRequest) {
  try {
    const response = await fetch(`${PYTHON_API_URL}/api/reset-stats`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error("Reset stats error:", error)
    return NextResponse.json({ error: "Failed to reset statistics", success: false }, { status: 500 })
  }
}