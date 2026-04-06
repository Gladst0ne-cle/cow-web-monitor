import streamlit as st
import paho.mqtt.client as mqtt
import json
import pandas as pd
import time
import queue
from datetime import datetime

# --- 1. 页面配置 ---
st.set_page_config(page_title="CAU 智慧牧场监控", layout="wide")
st.title("🐄 智慧牧场 - 全环境监测与控制中心")

# --- 2. 异步队列初始化 ---
@st.cache_resource
def get_msg_queue():
    return queue.Queue()

msg_queue = get_msg_queue()

if 'history' not in st.session_state:
    st.session_state.history = []

# --- 3. MQTT 回调逻辑 ---
def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        # 字段兼容：gateway 可能发送 nh3 或 ammonia
        if 'nh3' in data: data['ammonia'] = data['nh3']
        data['timestamp'] = datetime.now().strftime("%H:%M:%S")
        # 存入队列
        msg_queue.put(data)
    except:
        pass

# --- 4. 建立 MQTT 连接 (Websockets 8884) ---
@st.cache_resource
def init_mqtt():
    try:
        conf = st.secrets
        client = mqtt.Client(transport="websockets")
        client.tls_set()
        client.ws_set_options(path="/mqtt")
        client.username_pw_set(conf["MQTT_USER"], conf["MQTT_PWD"])
        client.on_message = on_message
        client.connect(conf["MQTT_BROKER"], 8884, 60)
        # 修正订阅主题
        client.subscribe("cow-web-monitor")
        client.loop_start()
        return client
    except Exception as e:
        st.error(f"连接失败: {e}")
        return None

mqtt_client = init_mqtt()

# --- 5. 侧边栏：远程控制 (修正控制主题) ---
with st.sidebar:
    st.header("⚡ 链路状态")
    if mqtt_client:
        st.success("MQTT 已在线")
    else:
        st.error("MQTT 未连接")

    st.divider()
    st.header("🎮 远程干预")
    
    def send_cmd(device, action):
        # 修正控制指令主题
        topic = "cowshed/control/manual"
        payload = json.dumps({"device": device, "action": action, "time": time.time()})
        mqtt_client.publish(topic, payload)
        st.toast(f"指令已发: {device} -> {action}")

    # 控制按钮
    c_fan, c_win = st.columns(2)
    with c_fan:
        st.write("风扇控制")
        if st.button("开启风扇"): send_cmd("fan", "on")
        if st.button("关闭风扇"): send_cmd("fan", "off")
    with c_win:
        st.write("窗户控制")
        if st.button("开启窗户"): send_cmd("window", "open")
        if st.button("关闭窗户"): send_cmd("window", "close")
    
    if st.button("开启加热器", use_container_width=True): send_cmd("heater", "on")

# --- 6. 数据处理循环 ---
while not msg_queue.empty():
    st.session_state.history.append(msg_queue.get())
    if len(st.session_state.history) > 60:
        st.session_state.history.pop(0)

# --- 7. UI 渲染：分层曲线 ---
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    latest = df.iloc[-1]
    
    # 顶部数据卡片
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("温度", f"{latest.get('temp', 0)} ℃")
    m2.metric("湿度", f"{latest.get('humi', 0)} %")
    m3.metric("氨气", f"{latest.get('ammonia', 0)} ppm")
    m4.metric("光照", f"{latest.get('light', 0)} lx")

    st.divider()
    st.subheader("📊 环境趋势监测")

    # 分层布局：解决数值差异大的问题
    row1_left, row1_right = st.columns(2)
    row2_left, row2_right = st.columns(2)

    with row1_left:
        st.caption("温度趋势 (°C)")
        st.line_chart(df.set_index('timestamp')['temp'], color="#FF4B4B")
    with row1_right:
        st.caption("湿度趋势 (%)")
        st.line_chart(df.set_index('timestamp')['humi'], color="#0068C9")
    with row2_left:
        st.caption("氨气浓度 (ppm)")
        st.line_chart(df.set_index('timestamp')['ammonia'], color="#29B09D")
    with row2_right:
        st.caption("光照强度 (lx)")
        st.line_chart(df.set_index('timestamp')['light'], color="#FFD700")
else:
    st.warning("📡 等待数据接入... 请确保本地网关正向主题 'cow-web-monitor' 发送数据。")

# --- 8. 自动刷新 ---
time.sleep(1.5)
st.rerun()
