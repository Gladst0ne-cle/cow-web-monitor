import streamlit as st
import paho.mqtt.client as mqtt
import json
import pandas as pd
import time
import queue
import altair as alt
from datetime import datetime

# --- 1. 核心配置 ---
st.set_page_config(page_title="高精度牧场监控", layout="wide")
st.title("🐄 环境高精度监测中心")

@st.cache_resource
def get_msg_queue():
    return queue.Queue()

msg_queue = get_msg_queue()
if 'history' not in st.session_state:
    st.session_state.history = []

# --- 2. MQTT 数据处理 ---
def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        if 'nh3' in data: data['ammonia'] = data['nh3']
        if 'lux' in data: data['light'] = data['lux']
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
        client.subscribe("cow-web-monitor")
        client.loop_start()
        return client
    except:
        return None

mqtt_client = init_mqtt()

# --- 3. 侧边栏：极简控制 ---
with st.sidebar:
    st.header("🎮 远程控制")
    
    def send_cmd(device, action):
        topic = "cowshed/control/manual"
        cmd = json.dumps({"device": device, "action": action, "time": time.time()})
        mqtt_client.publish(topic, cmd)
        st.toast(f"已发送: {device} {action}")

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

# --- 5. 精准刻度动态图表函数 ---
def create_precise_chart(data, col, title, color, tick_step=None):
    if data[col].isnull().all(): return None
    
    v_min, v_max = data[col].min(), data[col].max()
    diff = v_max - v_min
    
    # 动态边距：波动极小时保持 0.01 基础空间，波动大时保持 5% 空间
    padding = diff * 0.05 if diff > 0.1 else 0.05
    y_domain = [v_min - padding, v_max + padding]

    # 配置刻度间距
    y_axis_config = alt.Axis(title=title, grid=True)
    if tick_step:
        y_axis_config = alt.Axis(title=title, grid=True, tickMinStep=tick_step)

    return alt.Chart(data).mark_line(
        color=color, 
        strokeWidth=2.5,
        interpolate='monotone'
    ).encode(
        x=alt.X('timestamp:T', axis=alt.Axis(title=None, format='%H:%M:%S')),
        y=alt.Y(f'{col}:Q', 
                title=title, 
                scale=alt.Scale(domain=y_domain, nice=False),
                axis=y_axis_config),
        tooltip=[alt.Tooltip('timestamp:T', format='%H:%M:%S'), alt.Tooltip(f'{col}:Q', format='.2f')]
    ).properties(height=260).interactive()

# --- 6. 主界面渲染 ---
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    latest = df.iloc[-1]
    
    # 指标行
    idx = st.columns(4)
    idx[0].metric("温度", f"{latest.get('temp', 0):.2f} ℃")
    idx[1].metric("湿度", f"{latest.get('humi', 0):.2f} %")
    idx[2].metric("氨气", f"{latest.get('ammonia', 0):.1f}")
    idx[3].metric("光照", f"{latest.get('light', 0):.0f}")

    st.divider()

    r1_l, r1_r = st.columns(2)
    r2_l, r2_r = st.columns(2)

    with r1_l:
        # 温度间距 1
        st.altair_chart(create_precise_chart(df, 'temp', '温度 (°C)', '#FF4B4B', tick_step=1), use_container_width=True)
    with r1_r:
        st.altair_chart(create_precise_chart(df, 'humi', '湿度 (%)', '#0068C9'), use_container_width=True)
    with r2_l:
        # 氨气间距 5
        st.altair_chart(create_precise_chart(df, 'ammonia', '氨气 (ppm)', '#29B09D', tick_step=5), use_container_width=True)
    with r2_r:
        st.altair_chart(create_precise_chart(df, 'light', '光照强度', '#FFD700'), use_container_width=True)
else:
    st.info("📡 等待高灵敏数据流...")

# --- 7. 自动刷新 ---
time.sleep(1)
st.rerun()
