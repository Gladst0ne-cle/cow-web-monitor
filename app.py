import streamlit as st
import os
import time
import json
import queue
import tempfile
from datetime import datetime

# 尝试导入关键库，若缺少则友好提示（应对环境依赖问题）
try:
    import cv2
    import numpy as np
    import torch
    import paho.mqtt.client as mqtt
    import pandas as pd
    import altair as alt
    from ultralytics import YOLO
except ImportError as e:
    st.error(f"❌ 关键依赖库导入失败: {e}")
    st.info("💡 请确保 requirements.txt 中包含 ultralytics 和 opencv-python-headless，并尝试在 Streamlit 设置中将 Python 切换至 3.10 或 3.11。")
    st.stop()

# --- 1. 路径与模型加载 (使用相对路径，彻底解决 FileNotFoundError) ---
st.set_page_config(page_title="CAU 智慧牧场监控终端", layout="wide")
st.title("🐄 智慧牧场：环境监测与行为感知平台")

# 获取当前脚本所在目录，确保路径在 Linux 云端绝对正确
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DET_MODEL_PATH = os.path.join(BASE_DIR, 'runs', 'detect', 'yolov8_cattle_detection_1', 'weights', 'best.pt')
POSE_MODEL_PATH = os.path.join(BASE_DIR, 'runs', 'pose', 'cattle_pose_v19', 'weights', 'best.pt')

@st.cache_resource
def load_models():
    # 检查模型文件是否存在（LFS 上传校验）
    if not os.path.exists(DET_MODEL_PATH):
        st.error(f"找不到检测模型: {DET_MODEL_PATH}")
        st.info("请检查 GitHub 仓库中的 runs/ 文件夹是否包含 .pt 文件，并确认 LFS 已成功推送。")
        st.stop()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return YOLO(DET_MODEL_PATH).to(device), YOLO(POSE_MODEL_PATH).to(device)

# 初始化模型
det_model, pose_model = load_models()

# --- 2. 数据通信逻辑 (MQTT) ---
@st.cache_resource
def get_msg_queue():
    return queue.Queue()

msg_queue = get_msg_queue()
if 'history' not in st.session_state:
    st.session_state.history = []

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        # 字段兼容性处理
        if 'nh3' in data: data['ammonia'] = data['nh3']
        if 'lux' in data: data['light'] = data['lux']
        data['timestamp'] = datetime.now()
        msg_queue.put(data)
    except: pass

@st.cache_resource
def init_mqtt():
    try:
        s = st.secrets
        client = mqtt.Client(transport="websockets")
        client.tls_set()
        client.ws_set_options(path="/mqtt")
        client.username_pw_set(s["MQTT_USER"], s["MQTT_PWD"])
        client.on_message = on_message
        client.connect(s["MQTT_BROKER"], 8884, 60)
        client.subscribe("cow-web-monitor")
        client.loop_start()
        return client
    except Exception as e:
        st.sidebar.error(f"MQTT 连接失败: {e}")
        return None

mqtt_client = init_mqtt()

# --- 3. 视觉识别核心 ---
def judge_behavior(kpts, kpt_confs, bw, bh):
    aspect_ratio = bw / bh
    if kpts is not None and np.mean(kpt_confs) > 0.25:
        try:
            nose_y, back_y = kpts[0][1], kpts[4][1]
            if nose_y > back_y + (bh * 0.18): return "Eating"
        except: pass
    return "Lying" if aspect_ratio > 1.85 else "Standing"

def process_frame(frame, conf_val):
    results = det_model.track(frame, persist=True, conf=0.3, verbose=False)
    if not results or results[0].boxes is None or results[0].boxes.id is None:
        return frame

    for box, conf, obj_id in zip(results[0].boxes.xyxy.cpu().numpy(), 
                                 results[0].boxes.conf.cpu().numpy(), 
                                 results[0].boxes.id.cpu().numpy()):
        if conf < conf_val: continue
        x1, y1, x2, y2 = map(int, box)
        behavior = "Unknown"
        
        crop = frame[max(0,y1):y2, max(0,x1):x2]
        if crop.size > 0:
            p_res = pose_model.predict(crop, conf=0.2, verbose=False)
            if p_res and p_res[0].keypoints is not None:
                kpts = p_res[0].keypoints.xy.cpu().numpy()[0]
                k_confs = p_res[0].keypoints.conf.cpu().numpy()[0]
                behavior = judge_behavior(kpts, k_confs, x2-x1, y2-y1)

        color = (0, 255, 0) if behavior == "Standing" else (255, 165, 0)
        if behavior == "Eating": color = (255, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID:{int(obj_id)} {behavior}", (x1, y1-10), 0, 0.6, color, 2)
    return frame

# --- 4. 界面展示 ---
tab_env, tab_vision = st.tabs(["📊 环境监控", "📷 行为识别"])

with tab_env:
    while not msg_queue.empty():
        st.session_state.history.append(msg_queue.get())
        if len(st.session_state.history) > 100: st.session_state.history.pop(0)

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        latest = df.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("温度", f"{latest.get('temp', 0):.1f} ℃")
        c2.metric("湿度", f"{latest.get('humi', 0):.1f} %")
        c3.metric("氨气", f"{latest.get('ammonia', 0):.2f} ppm")
        c4.metric("光照", f"{latest.get('light', 0):.0f} Lux")
        
        # 趋势图
        df_melt = df.melt('timestamp', value_vars=['temp', 'humi', 'ammonia'])
        chart = alt.Chart(df_melt).mark_line().encode(
            x='timestamp:T', y='value:Q', color='variable:N'
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("📡 等待 MQTT 数据...")

with tab_vision:
    st.sidebar.header("控制面板")
    run_vision = st.sidebar.checkbox("开启 AI 识别引擎")
    vision_conf = st.sidebar.slider("识别阈值", 0.1, 1.0, 0.4)
    
    if run_vision:
        v_file = st.sidebar.file_uploader("上传监控视频", type=["mp4", "avi"])
        video_placeholder = st.empty()
        if v_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(v_file.read())
            cap = cv2.VideoCapture(tfile.name)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                frame = process_frame(frame, vision_conf)
                video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            os.unlink(tfile.name) # 清理临时文件

if not run_vision:
    time.sleep(2)
    st.rerun()
