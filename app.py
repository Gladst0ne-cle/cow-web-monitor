import streamlit as st
import paho.mqtt.client as mqtt
import json
import pandas as pd
import time
from datetime import datetime
from threading import enumerate

# --- 1. 核心修复：保持线程上下文兼容性（解决云端运行报错） ---
try:
    from streamlit.runtime.scriptrunner import add_script_run_context
except ImportError:
    try:
        from streamlit.scriptrunner import add_script_run_context
    except ImportError:
        def add_script_run_context(thread):
            pass

# --- 2. 页面配置 ---
st.set_page_config(page_title="中农大智慧牧场 V2", layout="wide")
st.title("🐄 智慧牧场实时看板 (cow-web-monitor)")

# 初始化 Session State
if 'history' not in st.session_state:
    st.session_state.history = []
if 'connected' not in st.session_state:
    st.session_state.connected = False

# --- 3. MQTT 回调函数 ---
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        # 订阅你的 gateway 发送的主题
        client.subscribe("cow-web-monitor")
        st.session_state.connected = True
    else:
        st.session_state.connected = False

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        # 字段兼容处理
        if 'nh3' in payload: payload['ammonia'] = payload['nh3']
        payload['time'] = datetime.now().strftime("%H:%M:%S")
        
        st.session_state.history.append(payload)
        if len(st.session_state.history) > 50:
            st.session_state.history.pop(0)
    except:
        pass

# --- 4. 侧边栏控制 ---
with st.sidebar:
    st.header("⚡ 系统接入")
    if st.button("🚀 启动监控", use_container_width=True):
        conf = st.secrets
        # 网页端必须用 websockets 和 8884
        client = mqtt.Client(transport="websockets")
        client.tls_set()
        client.ws_set_options(path="/mqtt")
        client.username_pw_set(conf["MQTT_USER"], conf["MQTT_PWD"])
        client.on_connect = on_connect
        client.on_message = on_message
        
        client.connect(conf["MQTT_BROKER"], 8884, 60)
        client.loop_start()
        
        # 核心：将 Paho 线程绑定到 Streamlit 上下文
        for thread in enumerate():
            if thread.name.startswith("paho-mqtt"):
                add_script_run_context(thread)
        st.sidebar.success("✅ 链路已激活")

# --- 5. 主界面展示 ---
status_text = "🟢 已连接" if st.session_state.connected else "🔴 未连接"
st.write(f"**当前状态：** {status_text} | **已接收：** {len(st.session_state.history)} 条")

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    latest = df.iloc[-1]
    
    # 指标卡片
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("温度", f"{latest.get('temp', '--')} ℃")
    c2.metric("湿度", f"{latest.get('humi', '--')} %")
    c3.metric("氨气", f"{latest.get('ammonia', latest.get('nh3', '--'))} ppm")
    c4.metric("光照", f"{latest.get('light', '--')} lx")

    # 实时趋势图
    st.subheader("📈 实时环境趋势图")
    # 动态筛选列名，防止之前报错的截断问题
    plot_cols = [c for c in ['temp', 'humi', 'ammonia'] if c in df.columns]
    if plot_cols:
        st.line_chart(df.set_index('time')[plot_cols])
else:
    st.warning("📡 等待数据中... 请确保 gateway.py 正在运行。")

# --- 6. 自动刷新 ---
time.sleep(1.5)
st.rerun()
