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

# 初始化消息队列和历史记录
@st.cache_resource
def get_msg_queue():
    return queue.Queue()

msg_queue = get_msg_queue()
if 'history' not in st.session_state:
    st.session_state.history = []

# --- 2. MQTT 数据处理逻辑 ---
def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode()
        # 只处理传感器数据主题，控制主题仅发送
        if msg.topic == "cow-web-monitor":
            data = json.loads(payload)
            # 统一字段名映射 (适配 Arduino 不同版本的 JSON Key)
            if 'nh3' in data: data['ammonia'] = data['nh3']
            if 'lux' in data: data['light'] = data['lux']
            
            # 只要包含任一核心指标即视为有效数据
            if any(k in data for k in ['temp', 'humi', 'ammonia', 'light']):
                data['timestamp'] = datetime.now()
                msg_queue.put(data)
    except Exception as e:
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
        
        # 订阅传感器数据主题
        client.subscribe("cow-web-monitor")
        
        client.loop_start()
        return client
    except Exception as e:
        st.error(f"MQTT 连接失败: {e}")
        return None

mqtt_client = init_mqtt()

# --- 3. 侧边栏：远程控制 (更新至 cow-web-control) ---
with st.sidebar:
    st.header("🎮 远程控制")
    
    def send_cmd(device, action):
        if mqtt_client:
            # 构造指令 JSON
            cmd = json.dumps({
                "device": device, 
                "action": action, 
                "time": int(time.time())
            })
            # --- 关键修改：发布到 cow-web-control ---
            mqtt_client.publish("cow-web-control", cmd)
            st.toast(f"✅ 已下发: {device} {action}")
        else:
            st.error("MQTT 未连接，无法发送指令")

    # 左右并列排布控制按钮
    c1, c2 = st.columns(2)
    with c1:
        st.write("**排风扇**")
        if st.button("开启", key="f_on"): send_cmd("fan", "on")
        if st.button("关闭", key="f_off"): send_cmd("fan", "off")
    with c2:
        st.write("**加热器**")
        if st.button("开启", key="h_on"): send_cmd("heater", "on")
        if st.button("关闭", key="h_off"): send_cmd("heater", "off")

# --- 4. 数据同步处理 ---
while not msg_queue.empty():
    st.session_state.history.append(msg_queue.get())
    # 保持最近 60 条数据（约 1 分钟的历史）
    if len(st.session_state.history) > 60:
        st.session_state.history.pop(0)

# --- 5. 动态居中图表函数 (已修复 NameError) ---
def create_center_chart(data, col, title, color):
    if data[col].isnull().all(): return None
    
    # 获取当前窗口极值
    v_min, v_max = data[col].min(), data[col].max()
    diff = v_max - v_min
    
    # 动态边距算法：确保曲线始终垂直居中
    # 如果数据恒定（diff=0），则预留上下各 1.0 的冗余空间
    margin = diff * 0.2 if diff > 0 else 1.0
    y_domain = [v_min - margin, v_max + margin]

    return alt.Chart(data).mark_line(
        color=color, 
        strokeWidth=3,
        interpolate='monotone'
    ).encode(
        x=alt.X('timestamp:T', axis=alt.Axis(title=None, format='%H:%M:%S')),
        y=alt.Y(f'{col}:Q', 
                title=title, 
                scale=alt.Scale(domain=y_domain, nice=True)),
        tooltip=[alt.Tooltip('timestamp:T', format='%H:%M:%S'), f'{col}:Q']
    ).properties(height=250).interactive()

# --- 6. 主界面渲染 ---
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    latest = df.iloc[-1]
    
    # 顶部实时指标
    idx = st.columns(4)
    idx[0].metric("温度", f"{latest.get('temp', 0):.2f} ℃")
    idx[1].metric("湿度", f"{latest.get('humi', 0):.2f} %")
    idx[2].metric("氨气", f"{latest.get('ammonia', 0):.1f}")
    idx[3].metric("光照", f"{latest.get('light', 0):.0f}")

    st.divider()

    # 四路动态曲线布局
    r1_l, r1_r = st.columns(2)
    r2_l, r2_r = st.columns(2)

    with r1_l:
        st.altair_chart(create_center_chart(df, 'temp', '温度趋势', '#FF4B4B'), use_container_width=True)
    with r1_r:
        st.altair_chart(create_center_chart(df, 'humi', '湿度趋势', '#0068C9'), use_container_width=True)
    with r2_l:
        st.altair_chart(create_center_chart(df, 'ammonia', '氨气浓度', '#29B09D'), use_container_width=True)
    with r2_r:
        st.altair_chart(create_center_chart(df, 'light', '光照强度趋势', '#FFD700'), use_container_width=True)
else:
    st.info("📡 正在等待传感器数据上传...")

# --- 7. 自动刷新 ---
time.sleep(1.2)
st.rerun()
