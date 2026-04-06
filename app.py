import streamlit as st
import paho.mqtt.client as mqtt
import json
import pandas as pd
import time
import queue
import altair as alt
from datetime import datetime

# --- 1. 页面配置 ---
st.set_page_config(page_title="CAU 智慧牧场监控", layout="wide")
st.title("🐄 智慧牧场 - 环境监测与控制中心")

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
        if 'nh3' in data: data['ammonia'] = data['nh3']
        data['timestamp'] = datetime.now() # 绘图需要 datetime 对象
        msg_queue.put(data)
    except:
        pass

# --- 4. 建立 MQTT 连接 ---
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
        client.subscribe("cow-web-monitor")
        client.loop_start()
        return client
    except Exception as e:
        st.error(f"连接失败: {e}")
        return None

mqtt_client = init_mqtt()

# --- 5. 侧边栏：设备控制 (风扇与加热器并列) ---
with st.sidebar:
    st.header("⚡ 运行状态")
    if mqtt_client:
        st.success("MQTT 已在线")
    else:
        st.error("MQTT 未连接")

    st.divider()
    st.header("🎮 远程控制")
    
    def send_cmd(device, action):
        topic = "cowshed/control/manual"
        payload = json.dumps({"device": device, "action": action, "time": time.time()})
        mqtt_client.publish(topic, payload)
        st.toast(f"指令已发: {device} -> {action}")

    # 并列布局：风扇与加热器
    col_fan, col_heat = st.columns(2)
    with col_fan:
        st.write("**排风扇**")
        if st.button("开启风扇"): send_cmd("fan", "on")
        if st.button("关闭风扇"): send_cmd("fan", "off")
    
    with col_heat:
        st.write("**加热器**")
        if st.button("开启加热"): send_cmd("heater", "on")
        if st.button("关闭加热"): send_cmd("heater", "off")

# --- 6. 数据处理 ---
while not msg_queue.empty():
    st.session_state.history.append(msg_queue.get())
    if len(st.session_state.history) > 50:
        st.session_state.history.pop(0)

# --- 7. UI 渲染：分层曲线 (强制 Y 轴范围) ---
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    latest = df.iloc[-1]
    
    # 顶部 Metric
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("温度", f"{latest.get('temp', 0)} ℃")
    m2.metric("湿度", f"{latest.get('humi', 0)} %")
    m3.metric("氨气", f"{latest.get('ammonia', 0)} ppm")
    m4.metric("光照", f"{latest.get('light', 0)} lx")

    st.divider()
    st.subheader("📈 实时趋势 (数值居中模式)")

    # 绘图辅助函数：锁定 Y 轴
    def create_chart(data, column, title, y_range, color):
        chart = alt.Chart(data).mark_line(color=color).encode(
            x=alt.X('timestamp:T', title='时间'),
            y=alt.Y(f'{column}:Q', title=title, scale=alt.Scale(domain=y_range)),
            tooltip=['timestamp', column]
        ).properties(height=250)
        return chart

    row1_l, row1_r = st.columns(2)
    row2_l, row2_r = st.columns(2)

    with row1_l:
        st.altair_chart(create_chart(df, 'temp', '温度 (°C)', [0, 50], '#FF4B4B'), use_container_width=True)
    with row1_r:
        st.altair_chart(create_chart(df, 'humi', '湿度 (%)', [0, 50], '#0068C9'), use_container_width=True)
    with row2_l:
        # 160 居中，范围设为 [120, 200]
        st.altair_chart(create_chart(df, 'ammonia', '氨气 (ppm)', [120, 200], '#29B09D'), use_container_width=True)
    with row2_r:
        st.altair_chart(create_chart(df, 'light', '光照 (lx)', [0, 50], '#FFD700'), use_container_width=True)

else:
    st.warning("📡 正在同步云端数据...")

# --- 8. 自动刷新 ---
time.sleep(1.5)
st.rerun()
