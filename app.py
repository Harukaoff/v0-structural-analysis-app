"""
構造力学解析アプリケーション (Streamlit版)
手書き梁図面から構造解析と応力図を自動生成
"""

import streamlit as st
import numpy as np
from PIL import Image
import json
import base64
from io import BytesIO
import sys
from pathlib import Path

# スクリプトのインポート
sys.path.append(str(Path(__file__).parent / "scripts"))

from scripts.yolo_detection import detect_elements
from scripts.template_cleanup import normalize_elements, draw_normalized_structure
from scripts.structural_analysis import StructuralAnalyzer, prepare_analysis_data
from scripts.generate_diagrams import generate_all_diagrams

# ページ設定
st.set_page_config(
    page_title="構造力学解析アプリ",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0066cc;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
        background: linear-gradient(135deg, #e8f4f8 0%, #f0f8ff 100%);
        border-radius: 10px;
        border-left: 5px solid #0066cc;
    }
    .step-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #0052a3;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        padding: 0.5rem;
        background-color: #f0f8ff;
        border-left: 4px solid #0066cc;
        border-radius: 5px;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #0066cc;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #0052a3;
        box-shadow: 0 4px 8px rgba(0,102,204,0.3);
    }
    </style>
""", unsafe_allow_html=True)

def image_to_base64(image):
    """PIL ImageをBase64文字列に変換"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def base64_to_image(base64_string):
    """Base64文字列をPIL Imageに変換"""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data))

# セッション状態の初期化
if 'detection_result' not in st.session_state:
    st.session_state.detection_result = None
if 'normalized_result' not in st.session_state:
    st.session_state.normalized_result = None
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'diagram_result' not in st.session_state:
    st.session_state.diagram_result = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None

# ヘッダー
st.markdown('<div class="main-header">🏗️ 構造力学解析アプリケーション</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
<b>このアプリについて:</b><br>
手書きの梁構造図面を画像認識で自動解析し、剛性マトリクス法を用いて構造解析を行います。<br>
変形図、せん断力図、曲げモーメント図を自動生成します。
</div>
""", unsafe_allow_html=True)

# サイドバー - パラメータ設定
with st.sidebar:
    st.header("⚙️ 解析パラメータ")
    
    st.subheader("材料特性")
    E = st.number_input(
        "ヤング率 E (GPa)",
        min_value=1.0,
        max_value=500.0,
        value=200.0,
        step=10.0,
        help="材料のヤング率 (鋼材: 200 GPa)"
    )
    
    I = st.number_input(
        "断面二次モーメント I (×10⁻⁵ m⁴)",
        min_value=0.1,
        max_value=100.0,
        value=1.0,
        step=0.1,
        help="梁の断面二次モーメント"
    )
    
    st.subheader("荷重設定")
    default_point_load = st.number_input(
        "点荷重の大きさ (kN)",
        min_value=0.1,
        max_value=1000.0,
        value=10.0,
        step=1.0
    )
    
    default_udl = st.number_input(
        "等分布荷重の大きさ (kN/m)",
        min_value=0.1,
        max_value=1000.0,
        value=5.0,
        step=0.5
    )
    
    default_moment = st.number_input(
        "モーメント荷重の大きさ (kN·m)",
        min_value=0.1,
        max_value=1000.0,
        value=5.0,
        step=0.5
    )
    
    st.subheader("検出パラメータ")
    confidence_threshold = st.slider(
        "検出信頼度閾値",
        min_value=0.1,
        max_value=0.9,
        value=0.25,
        step=0.05,
        help="YOLOモデルの検出信頼度の閾値"
    )
    
    connection_threshold = st.slider(
        "接続判定距離 (pixel)",
        min_value=10,
        max_value=100,
        value=30,
        step=5,
        help="要素同士を接続と判定する距離の閾値"
    )

# メインコンテンツ
st.markdown('<div class="step-header">📤 STEP 1: 画像アップロード</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "手書き梁図面の画像をアップロードしてください",
    type=['png', 'jpg', 'jpeg'],
    help="梁、支点、荷重が描かれた手書き図面の写真をアップロード"
)

if uploaded_file is not None:
    # 画像を読み込み
    image = Image.open(uploaded_file).convert('RGB')
    st.session_state.uploaded_image = image
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="アップロードされた画像", use_container_width=True)
    
    with col2:
        st.markdown("""
        **検出対象要素:**
        - 🟦 梁 (Beam)
        - 🔴 ピンローラー支点 (Roller Support)
        - 🟢 ピン支点 (Pin Support)
        - 🟥 固定支点 (Fixed Support)
        - 🟡 ヒンジ (Hinge)
        - 🔻 点荷重 (Point Load)
        - ↓↓↓ 等分布荷重 (UDL)
        - ↻ モーメント荷重 (Moment)
        """)
    
    # STEP 2: 要素検出
    st.markdown('<div class="step-header">🔍 STEP 2: 要素検出</div>', unsafe_allow_html=True)
    
    if st.button("🚀 要素検出を実行", key="detect_btn"):
        with st.spinner("YOLOモデルで要素を検出中..."):
            try:
                # 画像をBase64に変換
                image_base64 = "data:image/png;base64," + image_to_base64(image)
                
                # YOLO検出実行
                detection_result = detect_elements(image_base64)
                
                if "error" in detection_result:
                    st.error(f"❌ エラー: {detection_result['error']}")
                elif detection_result.get("success"):
                    st.session_state.detection_result = detection_result
                    
                    st.markdown('<div class="success-box">✅ 要素検出が完了しました!</div>', unsafe_allow_html=True)
                    
                    # 検出結果のサマリー
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("梁", detection_result['counts']['beam'])
                    with col2:
                        st.metric("支点", detection_result['counts']['supports'])
                    with col3:
                        st.metric("荷重", detection_result['counts']['loads'])
                    
                    # 検出された要素のリスト表示
                    with st.expander("📋 検出された要素の詳細"):
                        for element in detection_result['elements']:
                            st.write(f"**{element['type']}** - ID: {element['id']}, 信頼度: {element['confidence']:.2%}")
                else:
                    st.error("検出に失敗しました")
                    
            except Exception as e:
                st.error(f"❌ エラーが発生しました: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # STEP 3: 清書と正規化
    if st.session_state.detection_result is not None:
        st.markdown('<div class="step-header">✏️ STEP 3: 清書と正規化</div>', unsafe_allow_html=True)
        
        if st.button("📐 清書を実行", key="normalize_btn"):
            with st.spinner("要素を正規化して清書中..."):
                try:
                    # 正規化実行
                    normalized_result = normalize_elements(st.session_state.detection_result.copy())
                    
                    # 荷重の大きさを設定
                    for element in normalized_result['elements']:
                        if element['type'] == 'load':
                            element['magnitude'] = default_point_load * 1000  # kN -> N
                        elif element['type'] == 'UDL':
                            element['magnitude'] = default_udl * 1000  # kN/m -> N/m
                        elif element['type'] in ['momentL', 'momentR']:
                            element['magnitude'] = default_moment * 1000  # kN·m -> N·m
                    
                    st.session_state.normalized_result = normalized_result
                    
                    # 清書画像の生成
                    image_base64 = "data:image/png;base64," + image_to_base64(st.session_state.uploaded_image)
                    normalized_image = draw_normalized_structure(normalized_result, image_base64)
                    
                    st.markdown('<div class="success-box">✅ 清書が完了しました!</div>', unsafe_allow_html=True)
                    
                    # 清書結果の表示
                    st.image(base64_to_image(normalized_image), caption="清書された構造図", use_container_width=True)
                    
                    # 節点情報の表示
                    with st.expander("🔗 節点情報"):
                        st.write(f"**総節点数:** {len(normalized_result['nodes'])}")
                        for node in normalized_result['nodes']:
                            st.write(f"節点 {node['id']}: ({node['x']:.1f}, {node['y']:.1f}) - タイプ: {node.get('type', 'beam_end')}")
                    
                except Exception as e:
                    st.error(f"❌ エラーが発生しました: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # STEP 4: 構造解析
    if st.session_state.normalized_result is not None:
        st.markdown('<div class="step-header">🧮 STEP 4: 構造解析 (剛性マトリクス法)</div>', unsafe_allow_html=True)
        
        if st.button("⚡ 構造解析を実行", key="analyze_btn"):
            with st.spinner("剛性マトリクス法で解析中..."):
                try:
                    # 解析データの準備
                    nodes, elements, supports, loads = prepare_analysis_data(st.session_state.normalized_result.copy())
                    
                    # 材料特性の設定
                    material_props = {
                        'E': E * 1e9,  # GPa -> Pa
                        'I': I * 1e-5  # ×10⁻⁵ m⁴ -> m⁴
                    }
                    
                    # 構造解析実行
                    analyzer = StructuralAnalyzer(nodes, elements, supports, loads, material_props)
                    analysis_result = analyzer.solve()
                    
                    if "error" in analysis_result:
                        st.error(f"❌ 解析エラー: {analysis_result['error']}")
                    elif analysis_result.get("success"):
                        st.session_state.analysis_result = analysis_result
                        
                        st.markdown('<div class="success-box">✅ 構造解析が完了しました!</div>', unsafe_allow_html=True)
                        
                        # 結果の表示
                        tab1, tab2, tab3 = st.tabs(["📊 変位", "⚡ 反力", "🔧 部材力"])
                        
                        with tab1:
                            st.subheader("節点変位")
                            disp_data = []
                            for disp in analysis_result['displacements']:
                                disp_data.append({
                                    "節点ID": disp['node_id'],
                                    "水平変位 u (mm)": f"{disp['u']*1000:.4f}",
                                    "鉛直変位 v (mm)": f"{disp['v']*1000:.4f}",
                                    "回転角 θ (rad)": f"{disp['theta']:.6f}"
                                })
                            st.table(disp_data)
                        
                        with tab2:
                            st.subheader("支点反力")
                            react_data = []
                            for react in analysis_result['reactions']:
                                if abs(react['Rx']) > 1e-6 or abs(react['Ry']) > 1e-6 or abs(react['M']) > 1e-6:
                                    react_data.append({
                                        "節点ID": react['node_id'],
                                        "水平反力 Rx (kN)": f"{react['Rx']/1000:.2f}",
                                        "鉛直反力 Ry (kN)": f"{react['Ry']/1000:.2f}",
                                        "モーメント M (kN·m)": f"{react['M']/1000:.2f}"
                                    })
                            if react_data:
                                st.table(react_data)
                            else:
                                st.info("有意な反力はありません")
                        
                        with tab3:
                            st.subheader("部材力")
                            force_data = []
                            for force in analysis_result['element_forces']:
                                force_data.append({
                                    "部材ID": force['element_id'],
                                    "節点1→2": f"{force['node1_id']}→{force['node2_id']}",
                                    "せん断力 V1 (kN)": f"{force['V1']/1000:.2f}",
                                    "せん断力 V2 (kN)": f"{force['V2']/1000:.2f}",
                                    "曲げモーメント M1 (kN·m)": f"{force['M1']/1000:.2f}",
                                    "曲げモーメント M2 (kN·m)": f"{force['M2']/1000:.2f}",
                                    "部材長 (m)": f"{force['length']:.3f}"
                                })
                            st.table(force_data)
                    else:
                        st.error("解析に失敗しました")
                        
                except Exception as e:
                    st.error(f"❌ エラーが発生しました: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # STEP 5: 応力図の生成
    if st.session_state.analysis_result is not None:
        st.markdown('<div class="step-header">📈 STEP 5: 応力図の生成</div>', unsafe_allow_html=True)
        
        if st.button("📊 応力図を生成", key="diagram_btn"):
            with st.spinner("応力図を生成中..."):
                try:
                    # 応力図生成
                    diagram_result = generate_all_diagrams(
                        st.session_state.normalized_result.copy(),
                        st.session_state.analysis_result
                    )
                    
                    if "error" in diagram_result:
                        st.error(f"❌ エラー: {diagram_result['error']}")
                    elif diagram_result.get("success"):
                        st.session_state.diagram_result = diagram_result
                        
                        st.markdown('<div class="success-box">✅ 応力図の生成が完了しました!</div>', unsafe_allow_html=True)
                        
                        # 応力図の表示
                        st.subheader("📉 変形図")
                        st.image(base64_to_image(diagram_result['deformation_diagram']), use_container_width=True)
                        
                        st.subheader("🔴 せん断力図")
                        st.image(base64_to_image(diagram_result['shear_diagram']), use_container_width=True)
                        
                        st.subheader("🔵 曲げモーメント図")
                        st.image(base64_to_image(diagram_result['moment_diagram']), use_container_width=True)
                    else:
                        st.error("応力図の生成に失敗しました")
                        
                except Exception as e:
                    st.error(f"❌ エラーが発生しました: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

# フッター
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
<b>構造力学解析アプリ</b> | YOLOv8 + 剛性マトリクス法<br>
画像認識による構造解析の自動化システム
</div>
""", unsafe_allow_html=True)
