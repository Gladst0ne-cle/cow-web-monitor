import streamlit as st
import paho.mqtt.client as mqtt
import json
import pandas as pd
import time
import queue
from datetime import datetime

# --- 1. 页面配置与美化 ---
st.set_page_config(page_title="中农大智慧牧场监控", layout="wide")
st.title("🐄 智慧牧场 - 全环境监测与控制中心")

# --- 2. 全局状态与队列初始化 ---
@st.cache_resource
def get_msg_queue():
    return queue.Queue()

msg_queue = get_msg_queue()

if 'history' not in st.session_state:
    st.session_state.history = []

# --- 3. MQTT 回调：只负责入队 ---
def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        # 统一字段名映射
        if 'nh3' in data: data['ammonia'] = data['nh3']
        if 'lux' in data: data['light'] = data['lux']
        
        data['timestamp'] = datetime.now().strftime("%H:%M:%S")
        msg_queue.put(data)
    except:
        pass

# --- 4. 缓存 MQTT 客户端 (Websockets 8884) ---
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

# --- 5. 侧边栏：控制逻辑区 ---
with st.sidebar:
    st.header("⚡ 系统接入状态")
    if mqtt_client:
        st.success("MQTT 链路已激活")
    else:
        st.error("MQTT 未连接")

    st.divider()
    st.header("🎮 远程设备干预")
    st.info("指令将发送至主题: cowshed/control/manual")

    def send_cmd(device, action):
        cmd = json.dumps({
            "device": device, 
            "action": action, 
            "sender": "Web_Dashboard",
            "timestamp": time.time()
        })
        mqtt_client.publish("cowshed/control/manual", cmd)
        st.toast(f"已下发：{device} -> {action}")

    # 控制按钮组
    col_fan, col_window = st.columns(2)
    with col_fan:
        st.write("**排风扇**")
        if st.button("开启风扇"): send_cmd("fan", "on")
        if st.button("关闭风扇"): send_cmd("fan", "off")
    
    with col_window:
        st.write("**内窗排气**")
        if st.button("开启窗户"): send_cmd("window", "open")
        if st.button("关闭窗户"): send_cmd("window", "close")

    st.write("**环境补偿**")
    if st.button("开启加热器", use_container_width=True): send_cmd("heater", "on")
    if st.button("关闭加热器", use_container_width=True): send_cmd("heater", "off")

# --- 6. 数据同步：将队列转入界面 ---
while not msg_queue.empty():
    st.session_state.history.append(msg_queue.get())
    if len(st.session_state.history) > 100: # 增加到100个点，曲线更细腻
        st.session_state.history.pop(0)

# --- 7. 主界面：分层曲线展示 ---
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    latest = df.iloc[-1]
    
    # A. 顶层大指标
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("温度", f"{latest.get('temp', 0)} ℃")
    c2.metric("湿度", f"{latest.get('humi', 0)} %")
    c3.metric("氨气", f"{latest.get('ammonia', 0)} ppm")
    c4.metric("光照", f"{latest.get('light', 0)} lx")

    st.divider()
    
    # B. 层次化曲线：使用 Tabs 切换或 Columns 平铺
    # 这里建议用 Columns 平铺，视觉冲击力更强
    st.subheader("📈 环境实时趋势图 (分层监测)")
    
    row1_c1, row1_c2 = st.columns(2)
    row2_c1, row2_c2 = st.columns(2)

    with row1_c1:
        st.write("🌡️ **温度曲线**")
        st.line_chart(df.set_index('timestamp')['temp'], color="#FF4B4B")
    
    with row1_c2:
        st.write("💧 **湿度曲线**")
        st.line_chart(df.set_index('timestamp')['humi'], color="#0068C9")
    
    with row2_c1:
        st.write("🧪 **氨气浓度 (NH3)**")
        st.line_chart(df.set_index('timestamp')['ammonia'], color="#29B09D")
    
    with row2_c2:
        st.write("☀️ **光照强度**")
        st.line_chart(df.set_index('timestamp')['light'], color="#FFD700")

else:
    st.warning("📡 正在等待数据注入... 请运行本地网关。")

# --- 8. 自动刷新 ---
time.sleep(1.5)
st.rerun()
