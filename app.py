import streamlit as st
import paho.mqtt.client as mqtt
import json
import pandas as pd
import numpy as np
import queue
import altair as alt
from datetime import datetime, timedelta, timezone
import tempfile
import os
import gc
import uuid
from streamlit_autorefresh import st_autorefresh

try:
    import cv2
    import torch
    from ultralytics import YOLO
except:
    cv2 = None
    torch = None
    YOLO = None

def get_local_now():
    return datetime.now(timezone(timedelta(hours=8)))

st.set_page_config(page_title="智能牛舍环境监测与调控系统", layout="wide", initial_sidebar_state="expanded")
st.title("🐄 智能牛舍环境监测与调控系统")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DET_PATH = os.path.join(BASE_DIR, 'runs/detect/yolov8_cattle_detection_1/weights/best.pt')
POSE_PATH = os.path.join(BASE_DIR, 'runs/pose/cattle_pose_v19/weights/best.pt')

@st.cache_resource
def load_yolo_models():
    if cv2 is None or YOLO is None:
        return None, None

    if not os.path.exists(DET_PATH):
        return None, None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    det = YOLO(DET_PATH).to(device)

    pose = None
    if os.path.exists(POSE_PATH):
        pose = YOLO(POSE_PATH).to(device)

    return det, pose

det_model, pose_model = load_yolo_models()

def get_hourly_thresholds():
    h = get_local_now().hour

    if 0 <= h < 6:
        ts = {
            'temp': {'good': 18, 'normal': 24, 'warning': 30},
            'humi': {'good': 55, 'normal': 75, 'warning': 90},
            'ammonia': {'good': 300, 'normal': 380, 'warning': 500},
            'light': {'good': 0, 'normal': 0, 'warning': 0}
        }
    elif 6 <= h < 10:
        ts = {
            'temp': {'good': 20, 'normal': 26, 'warning': 32},
            'humi': {'good': 60, 'normal': 80, 'warning': 95},
            'ammonia': {'good': 320, 'normal': 400, 'warning': 520},
            'light': {'good': 100, 'normal': 60, 'warning': 20}
        }
    elif 10 <= h < 16:
        ts = {
            'temp': {'good': 24, 'normal': 30, 'warning': 36},
            'humi': {'good': 65, 'normal': 85, 'warning': 98},
            'ammonia': {'good': 340, 'normal': 430, 'warning': 550},
            'light': {'good': 180, 'normal': 120, 'warning': 60}
        }
    elif 16 <= h < 20:
        ts = {
            'temp': {'good': 20, 'normal': 26, 'warning': 32},
            'humi': {'good': 62, 'normal': 82, 'warning': 95},
            'ammonia': {'good': 310, 'normal': 390, 'warning': 520},
            'light': {'good': 70, 'normal': 30, 'warning': 10}
        }
    else:
        ts = {
            'temp': {'good': 18, 'normal': 24, 'warning': 30},
            'humi': {'good': 58, 'normal': 78, 'warning': 92},
            'ammonia': {'good': 290, 'normal': 370, 'warning': 480},
            'light': {'good': 10, 'normal': 0, 'warning': 0}
        }

    return ts

def get_status_config(value, thresholds, mode='normal'):
    if mode == 'light':
        return "优", "green", 0

    if value <= thresholds['good']:
        return "优", "green", 0
    elif value <= thresholds['normal']:
        return "良", "blue", 1
    elif value <= thresholds['warning']:
        return "警告", "orange", 2
    else:
        return "异常", "red", 3

@st.cache_resource
def get_msg_queue():
    return queue.Queue()

msg_queue = get_msg_queue()

if "history" not in st.session_state:
    st.session_state.history = []

if "mqtt_client" not in st.session_state:
    st.session_state.mqtt_client = None

def on_message(client, userdata, msg):
    try:
        raw = msg.payload.decode(errors="ignore")
        payload = json.loads(raw)

        if "nh3" in payload:
            payload["ammonia"] = payload["nh3"]
        if "lux" in payload:
            payload["light"] = payload["lux"]

        if any(k in payload for k in ["temp", "humi", "ammonia", "light"]):
            payload["timestamp"] = get_local_now()
            msg_queue.put(payload)

    except:
        pass

def connect_mqtt():
    try:
        c = st.secrets

        client_id = "streamlit_" + str(uuid.uuid4())[:8]
        client = mqtt.Client(client_id=client_id, transport="websockets")

        client.username_pw_set(c["MQTT_USER"], c["MQTT_PWD"])
        client.tls_set()
        client.reconnect_delay_set(min_delay=1, max_delay=10)

        client.on_message = on_message

        client.connect(c["MQTT_BROKER"], 8884, 30)
        client.subscribe("cow-web-monitor", 0)
        client.subscribe("cowshed/control/manual", 0)

        client.loop_start()
        return client

    except:
        return None

if st.session_state.mqtt_client is None:
    st.session_state.mqtt_client = connect_mqtt()

mqtt_client = st.session_state.mqtt_client

with st.sidebar:
    st.header("🎮 设备远程控制")

    def send_mqtt_cmd(device, action):
        if mqtt_client:
            cmd = json.dumps({"device": device, "action": action, "time": int(datetime.now().timestamp())})
            mqtt_client.publish("cowshed/control/manual", cmd)
            st.toast(f"✅ 指令已送达: {device} -> {action.upper()}")
        else:
            st.error("MQTT 未连接")

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
    st.header("🚀 视觉性能调节")
    skip_frames = st.slider("处理跳帧", 1, 10, 3)
    pose_every_n_frames = st.slider("姿态分析频率", 5, 50, 15)
    max_cows = st.slider("最大处理数", 1, 10, 4)

def create_center_chart(data, col, title, color):
    if data.empty or col not in data.columns:
        return None

    v_min, v_max = data[col].min(), data[col].max()
    margin = (v_max - v_min) * 0.2 if v_max != v_min else 2.0

    return alt.Chart(data).mark_line(color=color, strokeWidth=3).encode(
        x=alt.X('timestamp:T', axis=alt.Axis(title=None, format='%H:%M:%S')),
        y=alt.Y(f'{col}:Q', title=title,
                scale=alt.Scale(domain=[v_min - margin, v_max + margin], nice=True)),
        tooltip=[alt.Tooltip('timestamp:T', format='%H:%M:%S'),
                 alt.Tooltip(f'{col}:Q', title=title)]
    ).properties(height=220).interactive()

def judge_cow_behavior(kpts, kpt_confs, bw, bh):
    ratio = float(bw) / float(bh + 1e-6)

    if kpts is not None and kpt_confs is not None and len(kpt_confs) > 0:
        if float(np.mean(kpt_confs)) > 0.2:
            try:
                if kpts[0][1] > kpts[4][1] + (bh * 0.15):
                    return "Eating"
            except:
                pass

    return "Lying" if ratio > 1.8 else "Standing"

def process_vision_frame(frame, frame_id):
    if det_model is None:
        return frame

    results = det_model(frame, conf=0.4, verbose=False)
    if not results or results[0].boxes is None:
        return frame

    boxes = results[0].boxes.xyxy.cpu().numpy()
    if len(boxes) > max_cows:
        boxes = boxes[:max_cows]

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        bw, bh = x2 - x1, y2 - y1

        if bw < 40 or bh < 40:
            continue

        behavior = "Standing"
        crop = frame[max(0, y1):y2, max(0, x1):x2]

        if crop.size > 0 and pose_model and (frame_id % pose_every_n_frames == 0):
            p_res = pose_model.predict(crop, conf=0.25, verbose=False)
            if p_res and p_res[0].keypoints is not None:
                kp = p_res[0].keypoints
                if kp.xy is not None and len(kp.xy) > 0:
                    behavior = judge_cow_behavior(
                        kp.xy.cpu().numpy()[0],
                        kp.conf.cpu().numpy()[0],
                        bw, bh
                    )

        color = (255, 165, 0) if behavior == "Lying" else (0, 255, 0)
        if behavior == "Eating":
            color = (255, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Cow{i+1} {behavior}", (x1, y1 - 10), 0, 0.7, color, 2)

    return frame

while not msg_queue.empty():
    st.session_state.history.append(msg_queue.get())
    if len(st.session_state.history) > 2000:
        st.session_state.history.pop(0)

tab_realtime, tab_ai, tab_history = st.tabs(["📊 实时监测", "📷 行为识别", "📑 数据中心"])

with tab_realtime:
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        latest = df.iloc[-1]

        local_now = get_local_now()
        ts = get_hourly_thresholds()
        h_now = local_now.hour

        t_l, t_c, t_s = get_status_config(latest.get('temp', 0), ts['temp'])
        h_l, h_c, h_s = get_status_config(latest.get('humi', 0), ts['humi'])
        a_l, a_c, a_s = get_status_config(latest.get('ammonia', 0), ts['ammonia'])
        l_l, l_c, l_s = get_status_config(latest.get('light', 0), ts['light'], mode='light')

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("环境温度", f"{latest.get('temp', 0):.1f} ℃")
            st.markdown(f":{t_c}[● {t_l}]")
        with m2:
            st.metric("相对湿度", f"{latest.get('humi', 0):.1f} %")
            st.markdown(f":{h_c}[● {h_l}]")
        with m3:
            st.metric("氨气浓度", f"{latest.get('ammonia', 0):.2f} ppm")
            st.markdown(f":{a_c}[● {a_l}]")
        with m4:
            st.metric("光照强度", f"{latest.get('light', 0):.0f} Lux")
            st.markdown(f":{l_c}[● {l_l}]")

        st.divider()

        final_score = max(t_s, h_s, a_s, l_s)

        if 0 <= h_now < 6:
            time_tag = "凌晨睡眠期"
        elif 6 <= h_now < 10:
            time_tag = "早晨活跃期"
        elif 10 <= h_now < 16:
            time_tag = "日间高峰期"
        elif 16 <= h_now < 20:
            time_tag = "傍晚采食期"
        else:
            time_tag = "夜间休整期"

        eval_map = {
            0: ("优", f"当前时间 {local_now.strftime('%H:%M')}，处于{time_tag}，环境指标处于动态最优区间。"),
            1: ("良", f"当前时间 {local_now.strftime('%H:%M')}，处于{time_tag}，环境参数稳定。"),
            2: ("警告", f"警告：时段 {time_tag} 某些参数偏离标准，请注意自动通风控制。"),
            3: ("异常", f"严重异常：检测到恶劣环境数据波动，请立即核查现场！")
        }

        res_tag, res_text = eval_map[final_score]

        if final_score <= 1:
            st.success(f"🗓 **{local_now.strftime('%H:%M')} {time_tag}综合评价：{res_tag}** \n\n {res_text}")
        elif final_score == 2:
            st.warning(f"⚠️ **{local_now.strftime('%H:%M')} {time_tag}综合评价：{res_tag}** \n\n {res_text}")
        else:
            st.error(f"🚨 **{local_now.strftime('%H:%M')} {time_tag}综合评价：{res_tag}** \n\n {res_text}")

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
        st.info("📡 等待传感器数据...")

with tab_ai:
    st.subheader("📹 牛只状况检测工作区")

    if cv2 is None:
        st.error("当前环境无法使用 OpenCV，AI 视频模块不可用。")
        st.stop()

    if "video_path" not in st.session_state:
        st.session_state.video_path = None
    if "cap" not in st.session_state:
        st.session_state.cap = None
    if "playing" not in st.session_state:
        st.session_state.playing = False
    if "frame_id" not in st.session_state:
        st.session_state.frame_id = 0

    v_mode = st.radio("视频源", ["文件上传", "摄像头"], horizontal=True)

    if v_mode == "文件上传":
        f = st.file_uploader("上传录像", type=['mp4', 'avi'])
        if f:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(f.read())
            tfile.close()
            st.session_state.video_path = tfile.name

        if st.button("开始分析"):
            if st.session_state.video_path:
                st.session_state.cap = cv2.VideoCapture(st.session_state.video_path)
                st.session_state.playing = True
                st.session_state.frame_id = 0
            else:
                st.warning("请先上传视频文件")
    else:
        if st.button("开启摄像头"):
            st.session_state.cap = cv2.VideoCapture(0)
            st.session_state.playing = True
            st.session_state.frame_id = 0

    stop_btn = st.button("⏹ 停止")
    v_display = st.empty()

    if stop_btn:
        st.session_state.playing = False
        if st.session_state.cap:
            st.session_state.cap.release()
        st.session_state.cap = None
        gc.collect()
        st.info("已停止播放")

    if st.session_state.playing and st.session_state.cap is not None:
        cap = st.session_state.cap

        for _ in range(skip_frames):
            ret, frame = cap.read()

        if not ret:
            st.session_state.playing = False
            cap.release()
            st.session_state.cap = None
            st.warning("视频播放结束")
        else:
            st.session_state.frame_id += 1

            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (800, int(h * (800.0 / w))))

            processed = process_vision_frame(frame, st.session_state.frame_id)
            v_display.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), use_container_width=True)

            st_autorefresh(interval=50, key="video_refresh")

with tab_history:
    st.subheader("📋 历史记录")
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)

if not st.session_state.get("playing", False):
    st_autorefresh(interval=1500, key="refresh_main")
