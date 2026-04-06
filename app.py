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

# --- 2. 页面配置与样式 ---
st.set_page_config(page_title="中农大智慧牧场 V2", layout="wide")
st.title("🐄 智慧牧场实时看板 (cow-web-monitor 版)")

# 初始化 Session State
if 'history' not in st.session_state:
    st.session_state.history = []
if 'connected' not in st.session_state:
    st.session_state.connected = False

# --- 3. MQTT 回调函数 ---
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        # 必须与你的 gateway.py 中的 MQTT_TOPIC 保持完全一致
        client.subscribe("cow-web-monitor")
        st.session_state.connected = True
        print("✅ 已成功订阅: cow-web-monitor")

def on_message(client, userdata, msg):
    try:
        # 解析来自网关的 JSON 数据
        payload = json.loads(msg.payload.decode())
        
        # 字段清洗：确保能够读取到数据
        # 兼容 gateway 可能发送的不同字段名 (nh3 -> ammonia)
        if 'nh3' in payload: payload['ammonia'] = payload['nh3']
        
        # 添加时间戳用于绘图
        payload['time'] = datetime.now().strftime("%H:%M:%S")
        
        # 存入历史记录
        st.session_state.history.append(payload)
        
        # 限制缓冲区大小，防止内存溢出
        if len(st.session_state.history) > 50:
            st.session_state.history.pop(0)
    except Exception as e:
        print(f"数据解析错误: {e}")

# --- 4. 侧边栏：启动逻辑 ---
with st.sidebar:
    st.header("⚡ 系统接入")
    if st.button("🚀 启动实时监控", use_container_width=True):
        # 从 Streamlit Secrets 读取凭据
        conf = st.secrets
        
        # 网页端 Streamlit 必须使用 websockets 协议和 8884 端口
        client = mqtt.Client(transport="websockets")
        client.tls_set()
        client.ws_set_options(path="/mqtt")
        client.username_pw_set(conf["MQTT_USER"], conf["MQTT_PWD"])
        
        client.on_connect = on_connect
        client.on_message = on_message
        
        # 连接 HiveMQ Cloud
        client.connect(conf["MQTT_BROKER"], 8884, 60)
        client.loop_start()
        
        # --- 核心：绑定 Paho 线程到 Streamlit 上下文 ---
        # 这一步能确保回调函数里的 st.session_state 操作生效
        for thread in enumerate():
            if thread.name.startswith("paho-mqtt"):
                add_script_run_context(thread)
        
        st.sidebar.success("✅ 链路已激活")

# --- 5. 主界面展示 ---
# 诊断状态栏
status_text = "🟢 已连接" if st.session_state.connected else "🔴 未连接"
st.write(f"**当前状态：** {status_text} | **已接收数据：** {len(st.session_state.history)} 条")

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    latest = df.iloc[-1]
    
    # A. 核心指标卡片
    c1, c2, c3, c4 = st.columns(4)
    # 使用 .get(key, default) 提高代码健壮性
    c1.metric("🌡️ 温度", f"{latest.get('temp', '--')} ℃")
    c2.metric("💧 湿度", f"{latest.get('humi', '--')} %")
    c3.metric("🧪 氨气", f"{latest.get('ammonia', latest.get('nh3', '--'))} ppm")
    c4.metric("☀️ 光照", f"{latest.get('light', '--')} lx")

    # B. 实时趋势图表
    st.divider()
    st.subheader("📈 实时环境趋势图")
    # 动态筛选出 DataFrame 中存在的数值列进行绘图
    plot_cols = [col for col in ['temp', 'humi', 'am
