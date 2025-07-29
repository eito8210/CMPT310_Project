
#!/usr/bin/env python3
"""
利用可能なポートでAPIサーバーを起動するスクリプト
"""

import socket
import uvicorn
import sys

def find_available_port(start_port=5001, max_attempts=10):
    """利用可能なポートを見つける"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    return None

def main():
    # 利用可能なポートを検索
    port = find_available_port()
    
    if not port:
        print("❌ 利用可能なポートが見つかりませんでした")
        sys.exit(1)
    
    print("🚀 Smile Detection API を起動中...")
    print(f"📱 API: http://localhost:{port}")
    print(f"📖 Docs: http://localhost:{port}/docs")
    print("⏹️  停止するには Ctrl+C を押してください")
    print("-" * 50)
    
    try:
        uvicorn.run(
            "face_detection_api:app",
            host="0.0.0.0",
            port=port,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n✅ API サーバーを停止しました")
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
