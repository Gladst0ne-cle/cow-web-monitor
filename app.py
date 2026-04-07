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
st.set_page_config(page_title="智慧牧场集成监控", layout="wide")
st.title("🐄 环境监测与 AI 行为感知中心")

# 模型路径（确保在 GitHub 仓库中存在此路径）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DET_MODEL_PATH = os.path.join(BASE_DIR, 'runs/detect/yolov8_cattle_detection_1/weights/best.pt')
POSE_MODEL_PATH = os.path.join(BASE_DIR, 'runs/pose/cattle_pose_v19/weights/best.pt')

@st.cache_resource
def load_models():
    if not os.path.exists(DET_MODEL_PATH):
        st.error(f"找不到模型文件: {DET_MODEL_PATH}")
        st.stop()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 加载检测模型和姿态模型
    return YOLO(DET_MODEL_PATH).to(device), YOLO(POSE_MODEL_PATH).to(device)

det_model, pose_model = load_models()

@st.cache_resource
def get_msg_queue():
    return queue.Queue()

msg_queue = get_msg_queue()
if 'history' not in st.session_state:
    st.session_state.history = []

# --- 2. MQTT 数据处理 (保持原逻辑) ---
def on_message(client, userdata, msg):
    try:
        if msg.topic == "cowshed/control/manual":
            pass
            
        data = json.loads(msg.payload.decode())
        if 'nh3' in data: data['ammonia'] = data['nh3']
        if 'lux' in data: data['light'] = data['lux']
        
        if 'temp' in data or 'humi' in data or 'ammonia' in data:
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

# --- 3. 侧边栏：远程控制 (保持原逻辑) ---
with st.sidebar:
    st.header("🎮 远程设备控制")
    
    def send_cmd(device, action):
        if mqtt_client:
            cmd = json.dumps({
                "device": device, 
                "action": action, 
                "time": int(time.time())
            })
            mqtt_client.publish("cowshed/control/manual", cmd)
            st.toast(f"指令已发送: {device} -> {action.upper()}")
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
    st.header("⚙️ AI 视觉设置")
    vision_enabled = st.checkbox("开启实时 AI 识别")
    conf_threshold = st.slider("识别置信度", 0.1, 1.0, 0.4)

# --- 4. CV 处理算法 (新增) ---
def judge_behavior(kpts, kpt_confs, bw, bh):
    aspect_ratio = bw / bh
    if kpts is not None and np.mean(kpt_confs) > 0.25:
        try:
            # 根据关键点判断行为 (鼻尖 vs 后背高度)
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
        
        # 裁剪个体进行姿态识别
        crop = frame[max(0,y1):y2, max(0,x1):x2]
        if crop.size > 0:
            p_res = pose_model.predict(crop, conf=0.2, verbose=False)
            if p_res and p_res[0].keypoints is not None:
                kpts = p_res[0].keypoints.xy.cpu().numpy()[0]
                k_confs = p_res[0].keypoints.conf.cpu().numpy()[0]
                behavior = judge_behavior(kpts, k_confs, x2-x1, y2-y1)

        # 绘制 UI
        color = (0, 255, 0) if behavior == "Standing" else (255, 165, 0)
        if behavior == "Eating": color = (255, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID:{int(obj_id)} {behavior}", (x1, y1-10), 0, 0.6, color, 2)
    return frame

# --- 5. 图表与数据同步逻辑 (修复了 padding 报错隐患) ---
while not msg_queue.empty():
    st.session_state.history.append(msg_queue.get())
    if len(st.session_state.history) > 60:
        st.session_state.history.pop(0)

def create_center_chart(data, col, title, color):
    if data[col].isnull().all(): return None
    v_min, v_max = data[col].min(), data[col].max()
    margin = (v_max - v_min) * 0.2 if v_max != v_min else 2.0
    y_domain = [v_min - margin, v_max + margin] # 移除了未定义的 padding 引用

    return alt.Chart(data).mark_line(color=color, strokeWidth=3).encode(
        x=alt.X('timestamp:T', axis=alt.Axis(title=None, format='%H:%M:%S')),
        y=alt.Y(f'{col}:Q', title=title, scale=alt.Scale(domain=y_domain, nice=True)),
        tooltip=[alt.Tooltip('timestamp:T', format='%H:%M:%S'), f'{col}:Q']
    ).properties(height=250).interactive()

# --- 6. 主界面渲染 (采用 Tabs 布局，不影响原 UI) ---
tab_env, tab_vision = st.tabs(["📊 环境监测与控制", "📷 AI 视觉监控"])

with tab_env:
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        latest = df.iloc[-1]
        
        idx = st.columns(4)
        idx[0].metric("温度", f"{latest.get('temp', 0):.1f} ℃")
        idx[1].metric("湿度", f"{latest.get('humi', 0):.1f} %")
        idx[2].metric("氨气", f"{latest.get('ammonia', 0):.2f} ppm")
        idx[3].metric("光照", f"{latest.get('light', 0):.0f} Lux")

        st.divider()

        r1_l, r1_r = st.columns(2)
        r2_l, r2_r = st.columns(2)

        with r1_l:
            st.altair_chart(create_center_chart(df, 'temp', '温度趋势', '#FF4B4B'), use_container_width=True)
        with r1_r:
            st.altair_chart(create_center_chart(df, 'humi', '湿度趋势', '#0068C9'), use_container_width=True)
        with r2_l:
            st.altair_chart(create_center_chart(df, 'ammonia', '氨气趋势', '#29B09D'), use_container_width=True)
        with r2_r:
            st.altair_chart(create_center_chart(df, 'light', '光照趋势', '#FFD700'), use_container_width=True)
    else:
        st.info("📡 正在等待传感器上传环境数据...")

with tab_vision:
    if vision_enabled:
        video_source = st.file_uploader("上传牧场监控视频", type=['mp4', 'avi'])
        video_placeholder = st.empty()
        
        if video_source:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_source.read())
            cap = cv2.VideoCapture(tfile.name)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # AI 处理
                processed_frame = process_frame(frame, conf_threshold)
                # 显示
                video_placeholder.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB")
                
                # 视觉处理时顺便处理 MQTT 队列，防止切回 Tab 时数据堆积
                while not msg_queue.empty():
                    st.session_state.history.append(msg_queue.get())
            
            cap.release()
            os.unlink(tfile.name)
        else:
            st.info("请在侧边栏开启 AI 引擎并上传视频文件。")
    else:
        st.warning("AI 视觉引擎已关闭，请在侧边栏开启。")

# --- 7. 自动刷新逻辑 ---
# 当视觉引擎未开启时，执行页面重绘以更新传感器图表
if not vision_enabled:
    time.sleep(1.5)
    st.rerun()
