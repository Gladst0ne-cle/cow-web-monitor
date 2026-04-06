import streamlit as st
import paho.mqtt.client as mqtt
import json
import pandas as pd
import time
from datetime import datetime
from threading import enumerate

# --- 1. 关键修复：兼容不同版本的 ScriptRunContext ---
try:
    from streamlit.runtime.scriptrunner import add_script_run_context
except ImportError:
    try:
        from streamlit.scriptrunner import add_script_run_context
    except ImportError:
        def add_script_run_context(thread):
            pass

# --- 2. 页面配置与彩色样式 ---
st.set_page_config(page_title="CAU 智慧牧场 V2", layout="wide")

# 注入 CSS 增强视觉效果和彩色诊断块
st.markdown("""
    <style>
    .diag-box { background-color: #161B22; padding: 20px; border-radius: 10px; border: 1px solid #30363d; text-align: center; margin-bottom: 10px; }
    .status-ok { color: #39FF14; font-weight: bold; font-size: 20px; }
    .status-err { color: #FF3131; font-weight: bold; font-size: 20px; }
    .status-warn { color: #FFD700; font-weight: bold; font-size: 20px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🐄 智慧牧场全链路监控中心")

# --- 3. 初始化 Session State ---
if 'history' not in st.session_state:
    st.session_state.history = []
if 'mqtt_connected' not in st.session_state:
    st.session_state.mqtt_connected = False
if 'last_action' not in st.session_state:
    st.session_state.last_action = "等待指令状态..."

# --- 4. MQTT 回调逻辑 ---
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        # 订阅传感器数据和控制反馈主题
        client.subscribe([("cattle/sensors", 0), ("cowshed/control/output", 0)])
        st.session_state.mqtt_connected = True
    else:
        st.session_state.mqtt_connected = False

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        
        # 处理控制反馈
        if msg.topic == "cowshed/control/output":
            st.session_state.last_action = f"执行器响应：{payload.get('device', '系统')} -> {payload.get('status', payload.get('action', '已更新'))}"
        
        # 处理传感器数据
        elif msg.topic == "cattle/sensors":
            # 统一字段名：将 nh3 映射为 ammonia
            if 'nh3' in payload: payload['ammonia'] = payload['nh3']
            payload['time'] = datetime.now().strftime("%H:%M:%S")
            
            st.session_state.history.append(payload)
            if len(st.session_state.history) > 50:
                st.session_state.history.pop(0)
    except:
        pass

# --- 5. 侧边栏：连接与远程控制 ---
with st.sidebar:
    st.header("⚡ 系统接入")
    if st.button("🚀 启动实时监控", use_container_width=True):
        conf = st.secrets
        # 使用 Websockets 连接云端 8884 端口
        client = mqtt.Client(transport="websockets")
        client.tls_set()
        client.ws_set_options(path="/mqtt")
        client.username_pw_set(conf["MQTT_USER"], conf["MQTT_PWD"])
        
        client.on_connect = on_connect
        client.on_message = on_message
        
        client.connect(conf["MQTT_BROKER"], 8884, 60)
        client.loop_start()
        
        # 核心：绑定线程上下文防止丢失
        for thread in enumerate():
            if thread.name.startswith("paho-mqtt"):
                add_script_run_context(thread)
        st.success("监控已激活")

    if st.session_state.mqtt_connected:
        st.divider()
        st.subheader("🎮 远程干预")
        def send_ctrl(dev, act):
            topic = "cowshed/control/manual"
            msg = json.dumps({"device": dev, "action": act, "timestamp": time.time()})
            # 这里通过全局变量或重新获取客户端来发布，为简化逻辑此处建议通过 st.session_state 存储 client
            # 为确保稳定性，直接在按钮触发处临时建立短连接或复用
            pass 

# --- 6. 核心 UI 展示 ---
# A. 诊断中心
diag_c1, diag_c2, diag_c3 = st.columns(3)
with diag_c1:
    status, cls = ("正常", "status-ok") if st.session_state.mqtt_connected else ("断开", "status-err")
    st.markdown(f'<div class="diag-box">1. 云端接入<br><span class="{cls}">{status}</span></div>', unsafe_allow_html=True)
with diag_c2:
    status, cls = ("有信号", "status-ok") if st.session_state.history else ("无信号", "status-err")
    st.markdown(f'<div class="diag-box">2. 数据下发<br><span class="{cls}">{status}</span></div>', unsafe_allow_html=True)
with diag_c3:
    st.markdown(f'<div class="diag-box">3. 执行状态<br><span class="status-warn">{st.session_state.last_action}</span></div>', unsafe_allow_html=True)

st.divider()

# B. 数据卡片与图表
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    # 使用 delta 参数展示颜色趋势
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("温度", f"{latest.get('temp', 0)} ℃", delta=round(latest.get('temp',0)-prev.get('temp',0), 2), delta_color="inverse")
    m2.metric("湿度", f"{latest.get('humi', 0)} %", delta=round(latest.get('humi',0)-prev.get('humi',0), 2))
    m3.metric("氨气", f"{latest.get('ammonia', 0)} ppm", delta=round(latest.get('ammonia',0)-prev.get('ammonia',0), 2), delta_color="inverse")
    m4.metric("光照", f"{latest.get('light', 0)} lx")

    st.subheader("📈 实时环境曲线")
    st.line_chart(df.set_index('time')[['temp', 'humi', 'ammonia']])
else:
    st.info("📡 等待数据中... 请确保本地传感器脚本（sensor_sim.py）正在运行。")

# --- 7. 自动刷新逻辑 ---
time.sleep(2)
st.rerun()
