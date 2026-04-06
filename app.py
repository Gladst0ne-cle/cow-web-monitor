import streamlit as st
import paho.mqtt.client as mqtt
import json
import pandas as pd
import time
import queue
from datetime import datetime

# --- 1. 页面基本配置 ---
st.set_page_config(page_title="中农大牧场监控V2", layout="wide")
st.title("🐄 智慧牧场 - 实时数据中心")

# --- 2. 创建一个全局中转站（队列） ---
# 使用 st.cache_resource 确保这个队列在页面刷新时不会消失
@st.cache_resource
def get_message_queue():
    return queue.Queue()

msg_queue = get_message_queue()

# 初始化 Session State 用于存储绘图历史
if 'history' not in st.session_state:
    st.session_state.history = []

# --- 3. MQTT 回调（只负责放数据到队列） ---
def on_message(client, userdata, msg):
    try:
        # 解析数据
        data = json.loads(msg.payload.decode())
        # 兼容字段名：将 nh3 映射到 ammonia
        if 'nh3' in data: data['ammonia'] = data['nh3']
        data['timestamp'] = datetime.now().strftime("%H:%M:%S")
        
        # 【关键】把解析好的字典扔进队列，不操作任何 Streamlit 组件
        msg_queue.put(data)
    except:
        pass

# --- 4. 启动并缓存 MQTT 客户端 ---
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

# 自动在后台运行连接
mqtt_client = init_mqtt()

# --- 5. 主循环：从“中转站”取数据到“展示区” ---
# 每次页面刷新或循环，都会把队列里堆积的数据全部取出来
while not msg_queue.empty():
    new_msg = msg_queue.get()
    st.session_state.history.append(new_msg)
    # 保持历史记录在 50 条以内
    if len(st.session_state.history) > 50:
        st.session_state.history.pop(0)

# --- 6. UI 界面渲染 ---
st.sidebar.markdown(f"### 📡 系统状态")
if mqtt_client:
    st.sidebar.success("✅ MQTT 已在线")
else:
    st.sidebar.error("❌ MQTT 未连接")

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    latest = df.iloc[-1]
    
    # 顶部指标卡
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("温度", f"{latest.get('temp', '--')} ℃")
    c2.metric("湿度", f"{latest.get('humi', '--')} %")
    c3.metric("氨气浓度", f"{latest.get('ammonia', '--')} ppm")
    c4.metric("光照强度", f"{latest.get('light', '--')} lx")

    # 环境曲线图
    st.divider()
    st.subheader("📊 实时变化趋势")
    plot_cols = [c for c in ['temp', 'humi', 'ammonia'] if c in df.columns]
    st.line_chart(df.set_index('timestamp')[plot_cols])
else:
    st.warning("📡 正在同步云端数据...")
    st.info("提示：请确保本地 gateway.py 正在运行且串口 COM5 已连接硬件。")

# --- 7. 刷新驱动 ---
time.sleep(1.5)
st.rerun()
