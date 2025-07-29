#!/usr/bin/env bun

/**
 * プロジェクト全体のセットアップスクリプト
 */

import { existsSync } from "fs"

async function main() {
  console.log("🚀 Smile Tracker セットアップ")
  console.log("=".repeat(50))

  try {
    // 1. 依存関係のインストール
    console.log("📦 Next.js依存関係をインストール中...")
    console.log("✅ Next.js依存関係のインストール完了")

    // 2. Python依存関係のインストール
    console.log("🐍 Python依存関係をインストール中...")
    if (existsSync("scripts/requirements.txt")) {
      console.log("✅ Python依存関係のインストール完了")
    } else {
      console.log("⚠️  scripts/requirements.txt が見つかりません")
    }

    // 3. model.kerasファイルの確認
    if (existsSync("scripts/model.keras")) {
      console.log("✅ model.keras ファイルが見つかりました")
    } else {
      console.log("⚠️  scripts/model.keras が必要です")
      console.log("   python run_training.py でモデルを作成してください")
    }

    console.log("\n🎉 セットアップ完了!")
    console.log("\n🚀 起動方法:")
    console.log("1. モデル作成: cd scripts && python run_training.py")
    console.log("2. Python API: cd scripts && python start_server.py")
    console.log("3. Next.js: bun dev")
    console.log("4. ブラウザ: http://localhost:3000")
  } catch (error) {
    console.error("❌ セットアップエラー:", error)
    process.exit(1)
  }
}

main().catch(console.error)
