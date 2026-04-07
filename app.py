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
import gc

# --- 1. 核心配置与路径管理 ---
st.set_page_config(page_title="智能牛舍环境监测与调控系统", layout="wide", initial_sidebar_state="expanded")
st.title("🐄 智能牛舍环境监测与调控系统")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DET_PATH = os.path.join(BASE_DIR, 'runs/detect/yolov8_cattle_detection_1/weights/best.pt')
POSE_PATH = os.path.join(BASE_DIR, 'runs/pose/cattle_pose_v19/weights/best.pt')

@st.cache_resource
def load_yolo_models():
    if not os.path.exists(DET_PATH):
        return None, None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    det = YOLO(DET_PATH).to(device)
    pose = YOLO(POSE_PATH).to(device) if os.path.exists(POSE_PATH) else None
    return det, pose

det_model, pose_model = load_yolo_models()

# --- 2. 环境评价逻辑 (参考 4月春季肉牛养殖规范) ---
def get_status_config(value, thresholds, reverse=False):
    """
    根据阈值返回程度和颜色标签
    reverse=True 用于光照等越大越好的指标
    """
    if not reverse:
        if value <= thresholds['good']: return "优", "green", 0
        elif value <= thresholds['normal']: return "良", "blue", 1
        elif value <= thresholds['warning']: return "警告", "orange", 2
        else: return "异常", "red", 3
    else:
        if value >= thresholds['good']: return "优", "green", 0
        elif value >= thresholds['normal']: return "良", "blue", 1
        elif value >= thresholds['warning']: return "警告", "orange", 2
        else: return "异常", "red", 3

# 4月春季阈值设定 (参考 GB/T 30767-2014 肉牛养殖环境)
ENV_THRESHOLDS = {
    'temp': {'good': 18, 'normal': 24, 'warning': 28},    # 春季适宜温度 10-20℃
    'humi': {'good': 60, 'normal': 70, 'warning': 85},    # 湿度不宜过高，防止病菌滋生
    'ammonia': {'good': 5, 'normal': 12, 'warning': 20},  # 氨气是春季通风的关键指标
    'light': {'good': 150, 'normal': 80, 'warning': 40},  # 保证春季光照时长
}

# --- 3. MQTT 通信模块 ---
@st.cache_resource
def get_msg_queue():
    return queue.Queue()

msg_queue = get_msg_queue()
if 'history' not in st.session_state:
    st.session_state.history = []

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        if 'nh3' in payload: payload['ammonia'] = payload['nh3']
        if 'lux' in payload: payload['light'] = payload['lux']
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

# --- 4. 侧边栏与辅助函数 ---
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
        if st.button("开启", key="f_on"): send_mqtt_cmd("fan", "on")
        if st.button("关闭", key="f_off"): send_mqtt_cmd("fan", "off")
    with col_h:
        st.write("**加热器**")
        if st.button("开启", key="h_on"): send_mqtt_cmd("heater", "on")
        if st.button("关闭", key="h_off"): send_mqtt_cmd("heater", "off")

    st.divider()
    st.header("🚀 性能平衡参数")
    skip_frames = st.slider("处理跳帧", 1, 10, 3)
    pose_every_n_frames = st.slider("姿态分析频率", 5, 50, 15)
    max_cows = st.slider("最大处理目标数", 1, 10, 4)

def create_center_chart(data, col, title, color):
    if data.empty or col not in data.columns: return None
    v_min, v_max = data[col].min(), data[col].max()
    margin = (v_max - v_min) * 0.2 if v_max != v_min else 2.0
    return alt.Chart(data).mark_line(color=color, strokeWidth=3).encode(
        x=alt.X('timestamp:T', axis=alt.Axis(title=None, format='%H:%M:%S')),
        y=alt.Y(f'{col}:Q', title=title, scale=alt.Scale(domain=[v_min - margin, v_max + margin], nice=True)),
        tooltip=[alt.Tooltip('timestamp:T', format='%H:%M:%S'), alt.Tooltip(f'{col}:Q', title=title)]
    ).properties(height=220).interactive()

# --- 5. 视觉处理逻辑 ---
def judge_cow_behavior(kpts, kpt_confs, bw, bh):
    ratio = float(bw) / float(bh + 1e-6)
    if kpts is not None and kpt_confs is not None and len(kpt_confs) > 0:
        if float(np.mean(kpt_confs)) > 0.2:
            try:
                if kpts[0][1] > kpts[4][1] + (bh * 0.15): return "Eating"
            except: pass
    return "Lying" if ratio > 1.8 else "Standing"

def process_vision_frame(frame, frame_id):
    if det_model is None: return frame
    results = det_model(frame, conf=0.4, verbose=False)
    if not results or results[0].boxes is None: return frame
    
    boxes = results[0].boxes.xyxy.cpu().numpy()
    if len(boxes) > max_cows: boxes = boxes[:max_cows]
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        bw, bh = x2 - x1, y2 - y1
        if bw < 40 or bh < 40: continue
        
        behavior = "Standing"
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        
        if crop.size > 0 and pose_model and (frame_id % pose_every_n_frames == 0):
            p_res = pose_model.predict(crop, conf=0.25, verbose=False)
            if p_res and p_res[0].keypoints is not None:
                kp = p_res[0].keypoints
                if kp.xy is not None and len(kp.xy) > 0:
                    behavior = judge_cow_behavior(kp.xy.cpu().numpy()[0], kp.conf.cpu().numpy()[0], bw, bh)
        
        color = (255, 165, 0) if behavior == "Lying" else (0, 255, 0)
        if behavior == "Eating": color = (255, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Cow{i+1} {behavior}", (x1, y1 - 10), 0, 0.7, color, 2)
    return frame

# --- 6. 数据流更新 ---
while not msg_queue.empty():
    st.session_state.history.append(msg_queue.get())
    if len(st.session_state.history) > 100: st.session_state.history.pop(0)

# --- 7. 页面布局 ---
tab_realtime, tab_ai, tab_history = st.tabs(["📊 实时环境监测", "📷 牛只行为识别", "📑 数据管理中心"])

with tab_realtime:
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        latest = df.iloc[-1]
        
        # 计算各维度状态
        t_label, t_color, t_score = get_status_config(latest.get('temp', 0), ENV_THRESHOLDS['temp'])
        h_label, h_color, h_score = get_status_config(latest.get('humi', 0), ENV_THRESHOLDS['humi'])
        a_label, a_color, a_score = get_status_config(latest.get('ammonia', 0), ENV_THRESHOLDS['ammonia'])
        l_label, l_color, l_score = get_status_config(latest.get('light', 0), ENV_THRESHOLDS['light'], reverse=True)

        # 指标展示
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("环境温度", f"{latest.get('temp', 0):.1f} ℃")
            st.markdown(f":{t_color}[● {t_label}]")
        with m2:
            st.metric("相对湿度", f"{latest.get('humi', 0):.1f} %")
            st.markdown(f":{h_color}[● {h_label}]")
        with m3:
            st.metric("氨气浓度", f"{latest.get('ammonia', 0):.2f} ppm")
            st.markdown(f":{a_color}[● {a_label}]")
        with m4:
            st.metric("光照强度", f"{latest.get('light', 0):.0f} Lux")
            st.markdown(f":{l_color}[● {l_label}]")

        # 总体综合评价 (取最差项分数为基准)
        st.divider()
        final_score = max(t_score, h_score, a_score, l_score)
        eval_map = {
            0: ("优", "green", "当前牛舍微气候极佳。4月春季气温适中，建议维持当前通风频率，保持光照充足。"),
            1: ("良", "blue", "环境参数处于正常范围。春季昼夜温差较大，建议加强夜间保暖措施。"),
            2: ("警告", "orange", "部分环境指标出现波动！请检查牛舍通风设备或遮阳设施是否正常运行。"),
            3: ("异常", "red", "核心指标严重超标！可能导致肉牛产生应激反应或呼吸道疾病，请立即干预。")
        }
        res_tag, res_cls, res_text = eval_map[final_score]
        
        st.success(f"🗓 **4月春季环境综合评价：{res_tag}** \n {res_text}") if final_score <= 1 else st.error(f"🚨 **4月春季环境综合评价：{res_tag}** \n {res_text}")

        st.divider()
        r1_l, r1_r = st.columns(2); r2_l, r2_r = st.columns(2)
        with r1_l: st.altair_chart(create_center_chart(df, 'temp', '温度趋势 (℃)', '#FF4B4B'), use_container_width=True)
        with r1_r: st.altair_chart(create_center_chart(df, 'humi', '湿度趋势 (%)', '#0068C9'), use_container_width=True)
        with r2_l: st.altair_chart(create_center_chart(df, 'ammonia', '氨气趋势 (ppm)', '#29B09D'), use_container_width=True)
        with r2_r: st.altair_chart(create_center_chart(df, 'light', '光照趋势 (Lux)', '#FFD700'), use_container_width=True)
    else:
        st.info("📡 等待传感器上传数据...")

with tab_ai:
    st.subheader("📹 监控点行为分析")
    v_mode = st.radio("选择视频来源", ["本地文件上传", "开启本地摄像头"], horizontal=True)
    v_display = st.empty()
    
    if 'playing' not in st.session_state: st.session_state.playing = False
    if 'frame_id' not in st.session_state: st.session_state.frame_id = 0

    cap = None
    if v_mode == "本地文件上传":
        f = st.file_uploader("上传录像文件", type=['mp4', 'avi', 'mov'])
        if f:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(f.read())
            tfile.close()
            if st.button("▶ 开始分析"):
                st.session_state.playing = True
                cap = cv2.VideoCapture(tfile.name)
    else:
        if st.button("▶ 开启摄像头实时检测"):
            st.session_state.playing = True
            cap = cv2.VideoCapture(0)

    if st.session_state.playing and cap is not None:
        stop_btn = st.button("⏹ 停止播放")
        count = 0
        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret: break
            count += 1
            if count % skip_frames != 0: continue
            
            st.session_state.frame_id += 1
            h, w = frame.shape[:2]
            scale = 800.0 / float(w)
            frame = cv2.resize(frame, (800, int(h * scale)))
            
            processed = process_vision_frame(frame, st.session_state.frame_id)
            v_display.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            while not msg_queue.empty(): st.session_state.history.append(msg_queue.get())
            if stop_btn: break
        
        cap.release()
        st.session_state.playing = False
        gc.collect()

with tab_history:
    st.subheader("📋 历史记录存档")
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
    else:
        st.warning("暂无历史数据")

if not st.session_state.playing:
    time.sleep(1.5)
    st.rerun()
