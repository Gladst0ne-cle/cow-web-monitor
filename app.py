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

# --- 1. 核心配置与模型初始化 (新增 CV 部分) ---
st.set_page_config(page_title="智慧牧场监控", layout="wide")
st.title("🐄 环境监测与 AI 行为感知中心")

# 使用相对路径，确保云端兼容
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DET_MODEL_PATH = os.path.join(BASE_DIR, 'runs/detect/yolov8_cattle_detection_1/weights/best.pt')
POSE_MODEL_PATH = os.path.join(BASE_DIR, 'runs/pose/cattle_pose_v19/weights/best.pt')

@st.cache_resource
def load_models():
    if not os.path.exists(DET_MODEL_PATH):
        # 如果模型文件没传，先跳过不报错，保证环境监测部分能跑
        return None, None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return YOLO(DET_MODEL_PATH).to(device), YOLO(POSE_MODEL_PATH).to(device)

det_model, pose_model = load_models()

@st.cache_resource
def get_msg_queue():
    return queue.Queue()

msg_queue = get_msg_queue()
if 'history' not in st.session_state:
    st.session_state.history = []

# --- 2. MQTT 数据处理 (完全保留你的逻辑) ---
def on_message(client, userdata, msg):
    try:
        if msg.topic == "cowshed/control/manual":
            pass
        data = json.loads(msg.payload.decode())
        if 'nh3' in data: data['ammonia'] = data['nh3']
        if 'lux' in data: data['light'] = data['lux']
        if 'temp' in data or 'humi' in data:
            data['timestamp'] = datetime.now()
            msg_queue.put(data)
    except:
        pass

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
        client.subscribe([("cow-web-monitor", 0), ("cowshed/control/manual", 0)])
        client.loop_start()
        return client
    except:
        return None

mqtt_client = init_mqtt()

# --- 3. 侧边栏：远程控制 (完全保留你的逻辑) ---
with st.sidebar:
    st.header("🎮 远程控制")
    
    def send_cmd(device, action):
        if mqtt_client:
            cmd = json.dumps({
                "device": device, 
                "action": action, 
                "time": int(time.time())
            })
            mqtt_client.publish("cowshed/control/manual", cmd)
            st.toast(f"已下发指令: {device} -> {action.upper()}")
        else:
            st.error("MQTT 未连接")

    c1, c2 = st.columns(2)
    with c1:
        st.write("**排风扇**")
        if st.button("开启", key="f_on"): send_cmd("fan", "on")
        if st.button("关闭", key="f_off"): send_cmd("fan", "off")
    with c2:
        st.write("**加热器**")
        if st.button("开启", key="h_on"): send_cmd("heater", "on")
        if st.button("关闭", key="h_off"): send_cmd("heater", "off")

    st.divider()
    # 新增摄像头控制开关
    st.header("📹 摄像头控制")
    cam_on = st.toggle("开启实时监控/AI识别", value=False)
    cam_source = st.radio("选择视频源", ["本地摄像头", "上传文件"], horizontal=True)

# --- 4. CV 处理算法 ---
def process_frame(frame):
    if det_model is None: return frame
    results = det_model.track(frame, persist=True, conf=0.3, verbose=False)
    if not results or results[0].boxes is None or results[0].boxes.id is None:
        return frame
    
    for box, obj_id in zip(results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.id.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box)
        # 简单绘制检测框，不改变太多 UI
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Cow ID:{int(obj_id)}", (x1, y1-10), 0, 0.6, (0, 255, 0), 2)
    return frame

# --- 5. 数据同步 (完全保留你的逻辑) ---
while not msg_queue.empty():
    st.session_state.history.append(msg_queue.get())
    if len(st.session_state.history) > 60:
        st.session_state.history.pop(0)

# --- 6. 动态居中图表函数 (修复 NameError 隐患) ---
def create_center_chart(data, col, title, color):
    if data[col].isnull().all(): return None
    v_min, v_max = data[col].min(), data[col].max()
    margin = (v_max - v_min) * 0.2 if v_max != v_min else 2
    # 删除了引起错误的 padding 引用，统一使用 margin
    y_domain = [v_min - margin, v_max + margin]

    return alt.Chart(data).mark_line(color=color, strokeWidth=3).encode(
        x=alt.X('timestamp:T', axis=alt.Axis(title=None, format='%H:%M:%S')),
        y=alt.Y(f'{col}:Q', title=title, scale=alt.Scale(domain=y_domain, nice=True)),
        tooltip=[alt.Tooltip('timestamp:T', format='%H:%M:%S'), f'{col}:Q']
    ).properties(height=230).interactive()

# --- 7. 主界面渲染 ---
# A. 环境监测看板 (完全保留原样)
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    latest = df.iloc[-1]
    
    idx = st.columns(4)
    idx[0].metric("温度", f"{latest.get('temp', 0)} ℃")
    idx[1].metric("湿度", f"{latest.get('humi', 0)} %")
    idx[2].metric("氨气", f"{latest.get('ammonia', 0)}")
    idx[3].metric("光照", f"{latest.get('light', 0)}")

    st.divider()

    r1_l, r1_r = st.columns(2)
    r2_l, r2_r = st.columns(2)

    with r1_l:
        st.altair_chart(create_center_chart(df, 'temp', '温度', '#FF4B4B'), use_container_width=True)
    with r1_r:
        st.altair_chart(create_center_chart(df, 'humi', '湿度', '#0068C9'), use_container_width=True)
    with r2_l:
        st.altair_chart(create_center_chart(df, 'ammonia', '氨气', '#29B09D'), use_container_width=True)
    with r2_r:
        st.altair_chart(create_center_chart(df, 'light', '光照', '#FFD700'), use_container_width=True)
else:
    st.info("📡 正在等待传感器数据上传...")

# B. 新增：摄像头区块 (在底部以 Expander 形式存在，不挤占原空间)
if cam_on:
    with st.expander("📷 实时视觉监控 (点击折叠)", expanded=True):
        v_placeholder = st.empty()
        
        if cam_source == "本地摄像头":
            cap = cv2.VideoCapture(0) # 0 为默认摄像头
            while cam_on:
                ret, frame = cap.read()
                if not ret: break
                processed = process_frame(frame)
                v_placeholder.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), channels="RGB")
                # 循环内同步数据防止主界面卡顿
                while not msg_queue.empty():
                    st.session_state.history.append(msg_queue.get())
            cap.release()
        else:
            v_file = st.file_uploader("上传视频文件", type=['mp4', 'avi'])
            if v_file:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(v_file.read())
                cap = cv2.VideoCapture(tfile.name)
                while cap.isOpened() and cam_on:
                    ret, frame = cap.read()
                    if not ret: break
                    processed = process_frame(frame)
                    v_placeholder.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), channels="RGB")
                    while not msg_queue.empty():
                        st.session_state.history.append(msg_queue.get())
                cap.release()
                os.unlink(tfile.name)

# --- 8. 自动刷新 ---
if not cam_on:
    time.sleep(1.5)
    st.rerun()
