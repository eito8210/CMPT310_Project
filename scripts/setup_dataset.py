#!/usr/bin/env python3
"""
データセット構造をセットアップし、zipファイルを自動解凍するスクリプト
"""

import os
import zipfile
from pathlib import Path
import shutil

def create_dataset_structure():
    """SMILEsデータセット用のフォルダ構造を作成"""
    print("📁 データセット構造を作成中...")
    
    # 基本構造を作成
    base_dir = Path("SMILEs")
    base_dir.mkdir(exist_ok=True)
    
    # positives と negatives フォルダを作成
    positives_dir = base_dir / "positives" / "images"
    negatives_dir = base_dir / "negatives" / "images"
    
    positives_dir.mkdir(parents=True, exist_ok=True)
    negatives_dir.mkdir(parents=True, exist_ok=True)
    
    print("✅ フォルダ構造を作成しました:")
    print(f"  📂 {positives_dir}")
    print(f"  📂 {negatives_dir}")
    
    return positives_dir, negatives_dir

def extract_zip_files():
    """zipファイルを自動解凍"""
    print("📦 zipファイルを確認中...")
    
    base_dir = Path("SMILEs")
    extracted_files = 0
    
    # positives.zipの処理
    positives_zip = base_dir / "positives" / "positives.zip"
    positives_images_dir = base_dir / "positives" / "images"
    
    if positives_zip.exists():
        print(f"📦 {positives_zip} を解凍中...")
        try:
            with zipfile.ZipFile(positives_zip, 'r') as zip_ref:
                # 画像ファイルのみを抽出
                for file_info in zip_ref.filelist:
                    if file_info.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        # ファイル名のみを取得（パスを除去）
                        filename = os.path.basename(file_info.filename)
                        if filename:  # 空でない場合
                            # 解凍先パスを設定
                            extract_path = positives_images_dir / filename
                            
                            # ファイルを解凍
                            with zip_ref.open(file_info.filename) as source:
                                with open(extract_path, 'wb') as target:
                                    target.write(source.read())
                            extracted_files += 1
            
            print(f"✅ positives.zip から {extracted_files}枚の画像を解凍しました")
        except Exception as e:
            print(f"❌ positives.zip の解凍エラー: {e}")
    else:
        print("⚠️  positives.zip が見つかりません")
    
    # negatives.zipの処理
    negatives_zip = base_dir / "negatives" / "negatives.zip"
    negatives_images_dir = base_dir / "negatives" / "images"
    
    if negatives_zip.exists():
        print(f"📦 {negatives_zip} を解凍中...")
        try:
            neg_extracted = 0
            with zipfile.ZipFile(negatives_zip, 'r') as zip_ref:
                # 画像ファイルのみを抽出
                for file_info in zip_ref.filelist:
                    if file_info.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        # ファイル名のみを取得（パスを除去）
                        filename = os.path.basename(file_info.filename)
                        if filename:  # 空でない場合
                            # 解凍先パスを設定
                            extract_path = negatives_images_dir / filename
                            
                            # ファイルを解凍
                            with zip_ref.open(file_info.filename) as source:
                                with open(extract_path, 'wb') as target:
                                    target.write(source.read())
                            neg_extracted += 1
            
            print(f"✅ negatives.zip から {neg_extracted}枚の画像を解凍しました")
            extracted_files += neg_extracted
        except Exception as e:
            print(f"❌ negatives.zip の解凍エラー: {e}")
    else:
        print("⚠️  negatives.zip が見つかりません")
    
    return extracted_files

def check_dataset():
    """データセットの存在確認"""
    print("🔍 データセットを確認中...")
    
    base_dir = Path("SMILEs")
    if not base_dir.exists():
        print("❌ SMILEsフォルダが見つかりません")
        return False
    
    positives_dir = base_dir / "positives" / "images"
    negatives_dir = base_dir / "negatives" / "images"
    
    pos_count = 0
    neg_count = 0
    
    if positives_dir.exists():
        pos_files = list(positives_dir.glob("*.jpg")) + list(positives_dir.glob("*.png")) + list(positives_dir.glob("*.jpeg"))
        pos_count = len(pos_files)
    
    if negatives_dir.exists():
        neg_files = list(negatives_dir.glob("*.jpg")) + list(negatives_dir.glob("*.png")) + list(negatives_dir.glob("*.jpeg"))
        neg_count = len(neg_files)
    
    print(f"📊 データセット統計:")
    print(f"  😊 Positives (笑顔): {pos_count}枚")
    print(f"  😐 Negatives (非笑顔): {neg_count}枚")
    print(f"  📈 合計: {pos_count + neg_count}枚")
    
    if pos_count == 0 or neg_count == 0:
        print("⚠️  データが不足しています")
        return False
    
    if pos_count < 10 or neg_count < 10:
        print("⚠️  各クラス最低10枚の画像を推奨します")
        return False
    
    return True

def cleanup_zip_files():
    """解凍後にzipファイルを削除するかユーザーに確認"""
    base_dir = Path("SMILEs")
    zip_files = []
    
    positives_zip = base_dir / "positives" / "positives.zip"
    negatives_zip = base_dir / "negatives" / "negatives.zip"
    
    if positives_zip.exists():
        zip_files.append(positives_zip)
    if negatives_zip.exists():
        zip_files.append(negatives_zip)
    
    if zip_files:
        print(f"\n🗑️  解凍が完了しました。zipファイルを削除しますか？")
        for zip_file in zip_files:
            print(f"   {zip_file}")
        
        response = input("削除する場合は 'y' を入力してください (y/n): ").lower().strip()
        
        if response == 'y':
            for zip_file in zip_files:
                try:
                    zip_file.unlink()
                    print(f"✅ {zip_file.name} を削除しました")
                except Exception as e:
                    print(f"❌ {zip_file.name} の削除エラー: {e}")
        else:
            print("📦 zipファイルを保持します")

def main():
    print("🚀 データセットセットアップツール（zip対応版）")
    print("=" * 50)
    
    # 現在のディレクトリを確認
    current_dir = Path.cwd()
    print(f"現在のディレクトリ: {current_dir}")
    
    # 1. データセット構造を作成
    positives_dir, negatives_dir = create_dataset_structure()
    
    # 2. zipファイルを自動解凍
    extracted_count = extract_zip_files()
    
    # 3. データセットの確認
    has_data = check_dataset()
    
    # 4. zipファイルのクリーンアップ
    if extracted_count > 0:
        cleanup_zip_files()
    
    if not has_data:
        print("\n📋 次のステップ:")
        print("1. 笑顔の画像またはpositives.zipを以下のフォルダに配置:")
        print(f"   {positives_dir.parent}")
        print("2. 非笑顔の画像またはnegatives.zipを以下のフォルダに配置:")
        print(f"   {negatives_dir.parent}")
        print("3. 再度このスクリプトを実行")
        print("4. python training_smiles.py を実行")
    else:
        print("\n✅ データセットの準備完了!")
        print("次のステップ:")
        print("1. python training_smiles.py でモデルを訓練")
        print("2. python start_server.py でAPIサーバーを起動")

if __name__ == "__main__":
    main()
