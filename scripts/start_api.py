#!/usr/bin/env python3
"""
Smile Detection API 起動スクリプト
"""

import uvicorn
import sys
import os

def main():
    print("🚀 Smile Detection API を起動中...")
    print("📱 API: http://localhost:5000")
    print("📖 Docs: http://localhost:5000/docs")
    print("⏹️  停止するには Ctrl+C を押してください")
    print("-" * 50)
    
    try:
        uvicorn.run(
            "face_detection_api:app",
            host="0.0.0.0",
            port=5000,
            reload=False,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n✅ API サーバーを停止しました")
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
