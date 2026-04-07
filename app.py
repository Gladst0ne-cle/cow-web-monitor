import streamlit as st
import paho.mqtt.client as mqtt
import json
import pandas as pd
import numpy as np  # 必须导入，用于 judge_behavior
import time
import queue
import altair as alt
from datetime import datetime
import cv2
import torch
from ultralytics import YOLO
import tempfile

# --- 1. 核心配置与模型加载 ---
st.set_page_config(page_title="智慧牧场集成监控终端", layout="wide")
st.title("🐄 智慧牧场：环境监测与行为感知一体化平台")

# 【重要修改】使用相对路径，确保云端兼容
DET_MODEL_PATH = 'runs/detect/yolov8_cattle_detection_1/weights/best.pt'
POSE_MODEL_PATH = 'runs/pose/cattle_pose_v19/weights/best.pt'

@st.cache_resource
def load_models():
    # 强制使用 CPU 运行（云端通常没有 GPU，避免报错）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return YOLO(DET_MODEL_PATH).to(device), YOLO(POSE_MODEL_PATH).to(device)

try:
    det_model, pose_model = load_models()
except Exception as e:
    st.error(f"模型加载失败，请检查路径或 LFS 上传状态: {e}")
    st.stop()

@st.cache_resource
def get_msg_queue():
    return queue.Queue()

msg_queue = get_msg_queue()
if 'history' not in st.session_state:
    st.session_state.history = []

# --- 2. MQTT 数据处理 ---
def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode()
        if msg.topic == "cow-web-monitor":
            data = json.loads(payload)
            # 兼容不同传感器字段名
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
        # Streamlit Cloud 专用 WebSockets 连接方式
        client = mqtt.Client(transport="websockets")
        client.tls_set()
        client.ws_set_options(path="/mqtt")
        client.username_pw_set(c["MQTT_USER"], c["MQTT_PWD"])
        client.on_message = on_message
        client.connect(c["MQTT_BROKER"], 8884, 60)
        client.subscribe("cow-web-monitor")
        client.loop_start()
        return client
    except Exception as e:
        st.sidebar.warning(f"MQTT 连接未就绪: {e}")
        return None

mqtt_client = init_mqtt()

# --- 3. 行为识别核心算法 ---
def judge_behavior(kpts, kpt_confs, bw, bh):
    aspect_ratio = bw / bh
    # 如果关键点置信度达标，优先判定“采食”
    if kpts is not None and np.mean(kpt_confs) > 0.25:
        try:
            # 关键点 0 为鼻镜，4 为背部/肩部（需根据你的 v19 定义调整索引）
            nose_y, back_y = kpts[0][1], kpts[4][1]
            if nose_y > back_y + (bh * 0.18): 
                return "Eating"
        except: pass
    
    # 备选方案：通过宽高比判定“躺卧”或“站立”
    return "Lying" if aspect_ratio > 1.85 else "Standing"

def process_frame(frame, conf_val):
    # 执行目标检测与追踪
    results = det_model.track(frame, persist=True, conf=0.3, verbose=False)
    
    if not results or results[0].boxes is None or results[0].boxes.id is None:
        return frame

    for box, conf, obj_id in zip(results[0].boxes.xyxy.cpu().numpy(), 
                                 results[0].boxes.conf.cpu().numpy(), 
                                 results[0].boxes.id.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box)
        behavior = "Unknown"
        
        if conf >= conf_val:
            # 截取牛只区域进行姿态分析
            crop = frame[max(0,y1):y2, max(0,x1):x2]
            if crop.size > 0:
                p_res = pose_model.predict(crop, conf=0.2, verbose=False)
                kpts, k_confs = None, [0]
                if p_res and p_res[0].keypoints is not None:
                    # 获取该区域内最显著的一只牛的关键点
                    kpts = p_res[0].keypoints.xy.cpu().numpy()[0]
                    k_confs = p_res[0].keypoints.conf.cpu().numpy()[0]
                    behavior = judge_behavior(kpts, k_confs, x2-x1, y2-y1)

        # 绘制 UI
        color = (0, 255, 0) if behavior == "Standing" else (255, 165, 0) # 站立绿，其他橙
        if behavior == "Eating": color = (255, 255, 0) # 采食黄
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID:{int(obj_id)} {behavior}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

# --- 4. 主界面 Tab 分类 ---
tab_env, tab_vision = st.tabs(["📊 环境监控", "📷 视觉 AI 识别"])

# --- TAB 1: 环境监控 ---
with tab_env:
    # 提取队列数据并更新 session_state
    while not msg_queue.empty():
        st.session_state.history.append(msg_queue.get())
        if len(st.session_state.history) > 100: 
            st.session_state.history.pop(0)

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        latest = df.iloc[-1]
        
        # 指标展示
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("温度", f"{latest.get('temp', 0):.1f} ℃")
        c2.metric("湿度", f"{latest.get('humi', 0):.1f} %")
        c3.metric("氨气", f"{latest.get('ammonia', 0):.2f} ppm")
        c4.metric("光照", f"{latest.get('light', 0):.0f} Lux")
        
        # 趋势图表
        st.subheader("环境变化趋势 (实时)")
        chart_data = df.melt('timestamp', var_name='Metric', value_name='Value')
        line_chart = alt.Chart(chart_data).mark_line().encode(
            x='timestamp:T',
            y='Value:Q',
            color='Metric:N',
            tooltip=['timestamp', 'Metric', 'Value']
        ).interactive()
        st.altair_chart(line_chart, use_container_width=True)
    else:
        st.info("📡 等待传感器数据上传 (Topic: cow-web-monitor)...")

# --- TAB 2: 视觉 AI 识别 ---
with tab_vision:
    st.sidebar.header("📹 视觉设置")
    source_type = st.sidebar.radio("选择视频源", ["本地文件", "实时摄像头"])
    vision_conf = st.sidebar.slider("AI 识别阈值", 0.1, 1.0, 0.4)
    run_vision = st.sidebar.checkbox("开启识别引擎", value=False)

    video_placeholder = st.empty()

    if run_vision:
        cap = None
        if source_type == "本地文件":
            v_file = st.sidebar.file_uploader("上传视频", type=["mp4", "avi", "mov"])
            if v_file:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(v_file.read())
                cap = cv2.VideoCapture(tfile.name)
            else:
                st.warning("请先上传视频文件")
                st.stop()
        else:
            # 提示：Streamlit Cloud 服务器端无法访问你的本地摄像头，
            # 实时摄像头功能通常仅在本地运行或使用 WebRTC 插件时有效。
            cap = cv2.VideoCapture(0)

        if cap:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # AI 处理
                processed_frame = process_frame(frame, vision_conf)
                # 转为 RGB 供 Streamlit 显示
                video_placeholder.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB")
                
                # 视觉运行时后台继续收集环境数据，防止断联
                while not msg_queue.empty():
                    st.session_state.history.append(msg_queue.get())
            
            cap.release()

# --- 5. 自动刷新逻辑 ---
if not run_vision:
    time.sleep(2)  # 刷新频率设为 2s 降低云端负载
    st.rerun()
