#!/usr/bin/env python3
"""
訓練プロセス全体を管理するスクリプト
"""

import os
import sys
from pathlib import Path
import subprocess

def check_requirements():
    """必要な依存関係を確認"""
    print("🔍 依存関係を確認中...")
    
    required_packages = [
        'tensorflow',
        'scikit-learn',
        'matplotlib',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 不足しているパッケージをインストール中...")
        for package in missing_packages:
            subprocess.run([sys.executable, "-m", "pip", "install", package])
    
    return len(missing_packages) == 0

def check_dataset_structure():
    """データセット構造を確認"""
    print("\n📁 データセット構造を確認中...")
    
    smiles_dir = Path("SMILEs")
    if not smiles_dir.exists():
        print("❌ SMILEsフォルダが見つかりません")
        print("💡 setup_dataset.py を実行してフォルダ構造を作成してください")
        return False
    
    positives_dir = smiles_dir / "positives"
    negatives_dir = smiles_dir / "negatives"
    
    if not positives_dir.exists() or not negatives_dir.exists():
        print("❌ positives または negatives フォルダが見つかりません")
        return False
    
    # 画像ファイル数をカウント
    pos_files = list(positives_dir.rglob("*.jpg")) + list(positives_dir.rglob("*.png")) + list(positives_dir.rglob("*.jpeg"))
    neg_files = list(negatives_dir.rglob("*.jpg")) + list(negatives_dir.rglob("*.png")) + list(negatives_dir.rglob("*.jpeg"))
    
    print(f"😊 Positives: {len(pos_files)}枚")
    print(f"😐 Negatives: {len(neg_files)}枚")
    
    if len(pos_files) < 10 or len(neg_files) < 10:
        print("⚠️  各クラス最低10枚の画像が必要です")
        return False
    
    print("✅ データセット構造OK")
    return True

def run_training():
    """訓練スクリプトを実行"""
    print("\n🏋️ モデル訓練を開始...")
    
    try:
        # training_smiles.py を実行
        result = subprocess.run([sys.executable, "training_smiles.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 訓練完了!")
            
            # model.keras ファイルの確認
            if Path("model.keras").exists():
                print("✅ model.keras が作成されました")
                return True
            else:
                print("❌ model.keras が作成されませんでした")
                return False
        else:
            print("❌ 訓練中にエラーが発生しました:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ 訓練実行エラー: {e}")
        return False

def main():
    print("🚀 笑顔検出モデル訓練システム")
    print("=" * 50)
    
    # 現在のディレクトリを確認
    print(f"作業ディレクトリ: {Path.cwd()}")
    
    # 1. 依存関係チェック
    if not check_requirements():
        print("❌ 依存関係の問題があります")
        return
    
    # 2. データセット構造チェック
    if not check_dataset_structure():
        print("❌ データセットの準備が必要です")
        print("\n📋 次のステップ:")
        print("1. python setup_dataset.py を実行")
        print("2. 画像データを適切なフォルダに配置")
        print("3. 再度このスクリプトを実行")
        return
    
    # 3. 訓練実行
    if run_training():
        print("\n🎉 すべて完了!")
        print("次のステップ:")
        print("1. python start_server.py でAPIサーバーを起動")
        print("2. bun dev でNext.jsアプリを起動")
    else:
        print("\n❌ 訓練に失敗しました")

if __name__ == "__main__":
    main()
