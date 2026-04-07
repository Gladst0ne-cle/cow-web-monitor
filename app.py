import streamlit as st
import paho.mqtt.client as mqtt
import json
import pandas as pd
import numpy as np
import time
import queue
import altair as alt
from datetime import datetime
import cv2
import torch
from ultralytics import YOLO
import tempfile
import os

# --- 1. 核心配置与模型加载 ---
st.set_page_config(page_title="智慧牧场集成监控终端", layout="wide")
st.title("🐄 智慧牧场：环境监测与行为感知一体化平台")

# 【核心修复】使用相对路径！不要包含 D:/urp/...
# 只要你的 runs 文件夹在 GitHub 仓库根目录，这样写就能找到
DET_MODEL_PATH = 'runs/detect/yolov8_cattle_detection_1/weights/best.pt'
POSE_MODEL_PATH = 'runs/pose/cattle_pose_v19/weights/best.pt'

@st.cache_resource
def load_models():
    # 自动检测设备，云端通常使用 cpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 额外检查文件是否存在，防止再次报错
    if not os.path.exists(DET_MODEL_PATH):
        st.error(f"找不到模型文件: {DET_MODEL_PATH}。请检查 GitHub 仓库中是否存在该路径。")
        st.stop()
        
    return YOLO(DET_MODEL_PATH).to(device), YOLO(POSE_MODEL_PATH).to(device)

det_model, pose_model = load_models()

# --- 2. MQTT 数据处理 (保持你的原逻辑) ---
@st.cache_resource
def get_msg_queue():
    return queue.Queue()

msg_queue = get_msg_queue()
if 'history' not in st.session_state:
    st.session_state.history = []

def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode()
        if msg.topic == "cow-web-monitor":
            data = json.loads(payload)
            if 'nh3' in data: data['ammonia'] = data['nh3']
            if 'lux' in data: data['light'] = data['lux']
            if any(k in data for k in ['temp', 'humi', 'ammonia', 'light']):
                data['timestamp'] = datetime.now()
                msg_queue.put(data)
    except: pass

@st.cache_resource
def init_mqtt():
    try:
        c = st.secrets
        client = mqtt.Client(transport="websockets")
        client.tls_set()
        client.ws_set_options(path="/mqtt")
        client.username_pw_set(c["MQTT_USER"], c["MQTT_PWD"])
        client.on_message = on_message
        client.connect(c["MQTT_BROKER"], 8884, 60)
        client.subscribe("cow-web-monitor")
        client.loop_start()
        return client
    except: return None

mqtt_client = init_mqtt()

# --- 3. 行为识别算法 ---
def judge_behavior(kpts, kpt_confs, bw, bh):
    aspect_ratio = bw / bh
    if kpts is not None and np.mean(kpt_confs) > 0.25:
        try:
            # 索引根据你的 cattle_pose_v19 模型定义可能需要微调
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
        x1, y1, x2, y2 = map(int, box)
        behavior = "Unknown"
        if conf >= conf_val:
            crop = frame[max(0,y1):y2, max(0,x1):x2]
            if crop.size > 0:
                p_res = pose_model.predict(crop, conf=0.2, verbose=False)
                if p_res and p_res[0].keypoints is not None:
                    kpts = p_res[0].keypoints.xy.cpu().numpy()[0]
                    k_confs = p_res[0].keypoints.conf.cpu().numpy()[0]
                    behavior = judge_behavior(kpts, k_confs, x2-x1, y2-y1)

        color = (0, 255, 0) if behavior == "Standing" else (255, 165, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID:{int(obj_id)} {behavior}", (x1, y1-10), 0, 0.6, color, 2)
    return frame

# --- 4. UI 布局 ---
tab_env, tab_vision = st.tabs(["📊 环境监控", "📷 视觉 AI 识别"])

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
        
        chart = alt.Chart(df.melt('timestamp')).mark_line().encode(
            x='timestamp:T', y='value:Q', color='variable:N'
        ).interactive()
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("📡 等待传感器数据...")

with tab_vision:
    source_type = st.sidebar.radio("视频源", ["本地文件", "摄像头"])
    vision_conf = st.sidebar.slider("识别阈值", 0.1, 1.0, 0.4)
    run_vision = st.sidebar.checkbox("开启识别引擎")
    
    video_placeholder = st.empty()

    if run_vision:
        cap = None
        if source_type == "本地文件":
            v_file = st.sidebar.file_uploader("上传视频", type=["mp4", "avi"])
            if v_file:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(v_file.read())
                cap = cv2.VideoCapture(tfile.name)
        else:
            cap = cv2.VideoCapture(0)

        if cap:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                frame = process_frame(frame, vision_conf)
                video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()

if not run_vision:
    time.sleep(2)
    st.rerun()
