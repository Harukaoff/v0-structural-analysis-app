# 構造力学解析アプリケーション

手書きの梁構造図面を画像認識で自動解析し、剛性マトリクス法を用いて構造解析を行うWebアプリケーションです。

## 機能

1. **画像認識 (YOLOv8)**
   - 手書き図面から梁、支点、荷重を自動検出
   - 4種類の支点: ピンローラー、ピン、固定、ヒンジ
   - 4種類の荷重: 点荷重、等分布荷重、モーメント(左右)

2. **清書・正規化**
   - 角度を15度刻みに補正
   - 支点のY座標を整列
   - 近接要素の自動接続
   - 節点グラフの生成

3. **構造解析 (剛性マトリクス法)**
   - 節点変位の計算
   - 支点反力の算出
   - 部材内力(せん断力・曲げモーメント)の計算

4. **応力図の生成**
   - 変形図
   - せん断力図
   - 曲げモーメント図

## セットアップ

### 1. 必要なファイルの配置

YOLOモデルとテンプレート画像を配置してください:

\`\`\`
C:\Users\morim\Downloads\graduation\
├── runs\obb\train31\weights\best.pt  # YOLOモデル
└── templates\
    ├── hinge.png
    ├── pin.png
    ├── roller.png
    ├── fixed.png
    ├── UDL.png
    ├── load.png
    ├── beam.png
    ├── momentR.png
    └── momentL.png
\`\`\`

### 2. 依存パッケージのインストール

\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 3. アプリの起動

\`\`\`bash
streamlit run app.py
\`\`\`

ブラウザで `http://localhost:8501` にアクセスしてください。

## 使い方

1. **画像アップロード**: 手書き梁図面の画像をアップロード
2. **要素検出**: YOLOモデルで梁、支点、荷重を検出
3. **清書**: 検出した要素を正規化して清書
4. **構造解析**: 剛性マトリクス法で変位・反力を計算
5. **応力図生成**: 変形図、せん断力図、曲げモーメント図を表示

## パラメータ設定

サイドバーから以下のパラメータを調整できます:

- **材料特性**: ヤング率、断面二次モーメント
- **荷重設定**: 点荷重、等分布荷重、モーメントの大きさ
- **検出パラメータ**: 信頼度閾値、接続判定距離

## 技術スタック

- **フロントエンド**: Streamlit
- **画像認識**: YOLOv8 (Ultralytics)
- **構造解析**: NumPy, SciPy
- **可視化**: Matplotlib, Pillow

## ライセンス

教育目的での使用を想定しています。
\`\`\`

```text file=".streamlit/config.toml"
[theme]
primaryColor = "#0066cc"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f8ff"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 10
enableXsrfProtection = true
