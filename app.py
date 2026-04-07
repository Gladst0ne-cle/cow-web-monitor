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

# --- 1. 核心配置与模型初始化 ---
st.set_page_config(page_title="智慧牧场监控", layout="wide")
st.title("🐄 环境监测与控制中心")

@st.cache_resource
def get_msg_queue():
    return queue.Queue()

msg_queue = get_msg_queue()
if 'history' not in st.session_state:
    st.session_state.history = []

# --- 2. MQTT 数据处理 ---
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

# --- 3. 侧边栏：远程控制逻辑 ---
with st.sidebar:
    st.header("🎮 远程控制")
    def send_cmd(device, action):
        if mqtt_client:
            cmd = json.dumps({"device": device, "action": action, "time": int(time.time())})
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

# --- 4. 数据同步 ---
while not msg_queue.empty():
    st.session_state.history.append(msg_queue.get())
    if len(st.session_state.history) > 60:
        st.session_state.history.pop(0)

# --- 5. 动态居中图表函数 (修正了 padding 报错逻辑) ---
def create_center_chart(data, col, title, color):
    if data[col].isnull().all(): return None
    v_min, v_max = data[col].min(), data[col].max()
    margin = (v_max - v_min) * 0.2 if v_max != v_min else 2
    y_domain = [v_min - margin, v_max + margin] 

    return alt.Chart(data).mark_line(color=color, strokeWidth=3).encode(
        x=alt.X('timestamp:T', axis=alt.Axis(title=None, format='%H:%M:%S')),
        y=alt.Y(f'{col}:Q', title=title, scale=alt.Scale(domain=y_domain, nice=True)),
        tooltip=[alt.Tooltip('timestamp:T', format='%H:%M:%S'), f'{col}:Q']
    ).properties(height=250).interactive()

# --- 6. 主界面渲染 ---
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    latest = df.iloc[-1]
    
    idx = st.columns(4)
    idx[0].metric("温度", f"{latest.get('temp', 0)} ℃")
    idx[1].metric("湿度", f"{latest.get('humi', 0)} %")
    idx[2].metric("氨气", f"{latest.get('ammonia', 0)} ppm")
    idx[3].metric("光照", f"{latest.get('light', 0)} Lux")

    st.divider()

    r1_l, r1_r = st.columns(2)
    r2_l, r2_r = st.columns(2)

    with r1_l:
        st.altair_chart(create_center_chart(df, 'temp', '温度 (℃)', '#FF4B4B'), use_container_width=True)
    with r1_r:
        st.altair_chart(create_center_chart(df, 'humi', '湿度 (%)', '#0068C9'), use_container_width=True)
    with r2_l:
        st.altair_chart(create_center_chart(df, 'ammonia', '氨气 (ppm)', '#29B09D'), use_container_width=True)
    with r2_r:
        st.altair_chart(create_center_chart(df, 'light', '光照 (Lux)', '#FFD700'), use_container_width=True)
else:
    st.info("📡 正在连接数据源并等待传感器上传...")

# --- 7. 摄像头/视频监测区块 (仅修改此处的播放逻辑) ---
st.divider()
st.subheader("📷 实时视觉监测")
v_file = st.file_uploader("上传监控视频文件", type=["mp4", "avi"])
video_placeholder = st.empty()

if v_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(v_file.read())
    tfile.close() # 必须关闭句柄，cv2 才能读取
    
    cap = cv2.VideoCapture(tfile.name)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 将 BGR 转为 RGB 供 Streamlit 显示
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, use_container_width=True)
        
        # 视觉运行时顺便刷新 MQTT 数据防止卡顿
        while not msg_queue.empty():
            st.session_state.history.append(msg_queue.get())
        
        time.sleep(0.01) # 关键：给浏览器渲染时间
    
    cap.release()
    os.unlink(tfile.name)

# --- 8. 自动刷新 ---
if not v_file:
    time.sleep(1.5)
    st.rerun()
