import streamlit as st
import paho.mqtt.client as mqtt
import json
import pandas as pd
import time
import queue
import altair as alt
from datetime import datetime

# --- 1. 核心配置 ---
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
        # 新增：判断是否是控制指令的反馈
        if msg.topic == "cowshed/control/manual":
            # 这里可以扩展 UI 显示设备当前状态
            pass
            
        data = json.loads(msg.payload.decode())
        # 统一字段映射
        if 'nh3' in data: data['ammonia'] = data['nh3']
        if 'lux' in data: data['light'] = data['lux']
        
        # 只有包含环境数据时才放入队列绘图
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
        
        # --- 核心更改：同时订阅数据主题和控制主题 ---
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
            # 构造符合 gateway.py 解析标准的 JSON 指令
            cmd = json.dumps({
                "device": device, 
                "action": action, 
                "time": int(time.time())
            })
            # 发布到控制主题
            mqtt_client.publish("cow-web-control", cmd)
            st.toast(f"已下发指令: {device} -> {action.upper()}")
        else:
            st.error("MQTT 未连接，无法发送指令")

    # 左右并列排布
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

# --- 5. 动态居中图表函数 (保持不变) ---
def create_center_chart(data, col, title, color):
    if data[col].isnull().all(): return None
    v_min, v_max = data[col].min(), data[col].max()
    margin = (v_max - v_min) * 0.2 if v_max != v_min else 2
    y_domain = [v_min - margin, v_max + padding] if 'padding' not in locals() else [v_min - margin, v_max + margin]
    y_domain = [v_min - margin, v_max + margin]

    return alt.Chart(data).mark_line(color=color, strokeWidth=3).encode(
        x=alt.X('timestamp:T', axis=alt.Axis(title=None, format='%H:%M:%S')),
        y=alt.Y(f'{col}:Q', title=title, scale=alt.Scale(domain=y_domain, nice=True)),
        tooltip=[alt.Tooltip('timestamp:T', format='%H:%M:%S'), f'{col}:Q']
    ).properties(height=250).interactive()

# --- 6. 主界面渲染 (保持不变) ---
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
    st.info("📡 正在连接数据源并等待传感器上传...")

# --- 7. 自动刷新 ---
time.sleep(1.5)
st.rerun()
