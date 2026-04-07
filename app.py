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

# --- 1. 核心配置与路径管理 ---
st.set_page_config(page_title="CAU 智慧牧场集成系统", layout="wide", initial_sidebar_state="expanded")
st.title("🐄 智慧牧场：环境监测与行为感知一体化平台")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DET_PATH = os.path.join(BASE_DIR, 'runs/detect/yolov8_cattle_detection_1/weights/best.pt')
POSE_PATH = os.path.join(BASE_DIR, 'runs/pose/cattle_pose_v19/weights/best.pt')

@st.cache_resource
def load_yolo_models():
    if not os.path.exists(DET_PATH) or not os.path.exists(POSE_PATH):
        return None, None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    det = YOLO(DET_PATH).to(device)
    pose = YOLO(POSE_PATH).to(device)
    return det, pose

det_model, pose_model = load_yolo_models()

# --- 2. MQTT 通信模块 ---
@st.cache_resource
def get_msg_queue():
    return queue.Queue()

msg_queue = get_msg_queue()

if 'history' not in st.session_state:
    st.session_state.history = []

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        if 'nh3' in payload:
            payload['ammonia'] = payload['nh3']
        if 'lux' in payload:
            payload['light'] = payload['lux']

        if any(k in payload for k in ['temp', 'humi', 'ammonia', 'light']):
            payload['timestamp'] = datetime.now()
            msg_queue.put(payload)
    except:
        pass

@st.cache_resource
def init_mqtt_connection():
    try:
        c = st.secrets
        client = mqtt.Client(transport="websockets")
        client.tls_set()
        client.ws_set_options(path="/mqtt")
        client.username_pw_set(c["MQTT_USER"], c["MQTT_PWD"])
        client.on_message = on_message
        client.connect(c["MQTT_BROKER"], 8884, 60)
        client.subscribe([("cow-web-monitor", 0), ("cowshed/control/manual", 0)])
        client.loop_start()
        return client
    except:
        return None

mqtt_client = init_mqtt_connection()

# --- 3. 侧边栏：远程控制 ---
with st.sidebar:
    st.header("🎮 设备远程控制")

    def send_mqtt_cmd(device, action):
        if mqtt_client:
            cmd = json.dumps({"device": device, "action": action, "time": int(time.time())})
            mqtt_client.publish("cowshed/control/manual", cmd)
            st.toast(f"✅ 指令已送达: {device} -> {action.upper()}")
        else:
            st.error("未连接到服务器")

    col_f, col_h = st.columns(2)
    with col_f:
        st.write("**排风扇**")
        if st.button("开启", key="f_on"):
            send_mqtt_cmd("fan", "on")
        if st.button("关闭", key="f_off"):
            send_mqtt_cmd("fan", "off")

    with col_h:
        st.write("**加热器**")
        if st.button("开启", key="h_on"):
            send_mqtt_cmd("heater", "on")
        if st.button("关闭", key="h_off"):
            send_mqtt_cmd("heater", "off")

    st.divider()
    st.header("⚙️ 视觉引擎参数")
    vision_conf = st.slider("识别置信度阈值", 0.1, 1.0, 0.4)

# --- 4. 核心功能函数 ---
def create_center_chart(data, col, title, color):
    if data.empty or col not in data.columns:
        return None

    v_min, v_max = data[col].min(), data[col].max()
    margin = (v_max - v_min) * 0.2 if v_max != v_min else 2.0

    return alt.Chart(data).mark_line(color=color, strokeWidth=3).encode(
        x=alt.X('timestamp:T', axis=alt.Axis(title=None, format='%H:%M:%S')),
        y=alt.Y(f'{col}:Q', title=title,
                scale=alt.Scale(domain=[v_min - margin, v_max + margin], nice=True)),
        tooltip=[
            alt.Tooltip('timestamp:T', format='%H:%M:%S'),
            alt.Tooltip(f'{col}:Q', title=title)
        ]
    ).properties(height=240).interactive()

def judge_cow_behavior(kpts, kpt_confs, bw, bh):
    ratio = bw / bh
    if kpts is not None and np.mean(kpt_confs) > 0.2:
        try:
            if kpts[0][1] > kpts[4][1] + (bh * 0.15):
                return "Eating"
        except:
            pass
    return "Lying" if ratio > 1.8 else "Standing"

def process_vision_frame(frame, conf):
    if det_model is None:
        return frame

    results = det_model(frame, conf=conf, verbose=False)

    if not results or results[0].boxes is None:
        return frame

    boxes = results[0].boxes.xyxy.cpu().numpy()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        behavior = "Detecting..."

        crop = frame[max(0, y1):y2, max(0, x1):x2]

        if crop.size > 0 and pose_model is not None:
            p_res = pose_model.predict(crop, conf=0.2, verbose=False)
            if p_res and p_res[0].keypoints is not None:
                kpts = p_res[0].keypoints.xy.cpu().numpy()[0]
                k_confs = p_res[0].keypoints.conf.cpu().numpy()[0]
                behavior = judge_cow_behavior(kpts, k_confs, x2 - x1, y2 - y1)

        color = (0, 255, 0) if behavior == "Standing" else (255, 165, 0)
        if behavior == "Eating":
            color = (255, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Cow {i+1} {behavior}", (x1, y1 - 10), 0, 0.7, color, 2)

    return frame

# --- 5. 数据同步逻辑 ---
while not msg_queue.empty():
    st.session_state.history.append(msg_queue.get())
    if len(st.session_state.history) > 200:
        st.session_state.history.pop(0)

# --- 6. 核心页面布局 ---
tab_realtime, tab_ai, tab_history = st.tabs(["📊 实时环境中心", "📷 AI 行为感知", "📑 数据管理中心"])

with tab_realtime:
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        latest = df.iloc[-1]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("环境温度", f"{latest.get('temp', 0):.1f} ℃")
        m2.metric("相对湿度", f"{latest.get('humi', 0):.1f} %")
        m3.metric("氨气浓度", f"{latest.get('ammonia', 0):.2f} ppm")
        m4.metric("光照强度", f"{latest.get('light', 0):.0f} Lux")

        st.divider()

        r1_l, r1_r = st.columns(2)
        r2_l, r2_r = st.columns(2)

        with r1_l:
            st.altair_chart(create_center_chart(df, 'temp', '温度趋势 (℃)', '#FF4B4B'), use_container_width=True)
        with r1_r:
            st.altair_chart(create_center_chart(df, 'humi', '湿度趋势 (%)', '#0068C9'), use_container_width=True)
        with r2_l:
            st.altair_chart(create_center_chart(df, 'ammonia', '氨气趋势 (ppm)', '#29B09D'), use_container_width=True)
        with r2_r:
            st.altair_chart(create_center_chart(df, 'light', '光照趋势 (Lux)', '#FFD700'), use_container_width=True)
    else:
        st.info("📡 等待数据...")

with tab_ai:
    st.subheader("📹 监控点 AI 行为实时分析")

    v_source = st.radio("选择视频源", ["本地文件上传", "实时摄像头"], horizontal=True)
    v_display = st.empty()

    if 'playing' not in st.session_state:
        st.session_state.playing = False

    if v_source == "本地文件上传":
        f = st.file_uploader("上传录像", type=['mp4', 'avi', 'mov'])

        if f:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(f.read())
            tfile.close()

            col1, col2 = st.columns(2)

            with col1:
                if st.button("▶ 播放视频"):
                    st.session_state.playing = True

            with col2:
                if st.button("⏹ 停止播放"):
                    st.session_state.playing = False

            if st.session_state.playing:
                cap = cv2.VideoCapture(tfile.name)

                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 25

                frame_delay = 1.0 / fps

                while cap.isOpened() and st.session_state.playing:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = cv2.resize(frame, (640, 480))
                    processed = process_vision_frame(frame, vision_conf)

                    v_display.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))

                    time.sleep(frame_delay)

                cap.release()
                os.unlink(tfile.name)
                st.session_state.playing = False

    else:
        st.info("📷 摄像头模式暂未实现（可扩展为 RTSP / USB 摄像头）")

with tab_history:
    st.subheader("📋 历史记录存档")

    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)
    else:
        st.warning("暂无数据")
