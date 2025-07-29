#!/usr/bin/env bun

/**
 * Bun用のセットアップスクリプト
 * プロジェクトの初期設定を行います
 */

// @ts-ignore
import { $ } from "bun"
import { existsSync } from "fs"

async function main() {
  console.log("🚀 Smile Tracker セットアップ")

  // 依存関係のインストール
  console.log("📦 依存関係をインストール中...")
  await $`bun install`
  console.log("✓ 完了")

  // Python環境チェック
  if (!existsSync("scripts/model.keras")) {
    console.log("⚠️  scripts/model.keras が必要です")
  }

  console.log("✅ セットアップ完了!")
  console.log("開発サーバー: bun dev")
  console.log("Python API: cd scripts && python face_detection_api.py")
}

main().catch(console.error)
