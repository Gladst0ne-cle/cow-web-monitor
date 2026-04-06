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

# --- 3. MQTT 回调逻辑 (純數據處理) ---
def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        # 统一兼容：nh3 -> ammonia, lux -> light
        if 'nh3' in data: data['ammonia'] = data['nh3']
        if 'lux' in data: data['light'] = data['lux']
        
        # 绘图需要真实 datetime 对象
        data['timestamp'] = datetime.now()
        msg_queue.put(data)
    except:
        pass

# --- 4. 缓存并自动启动 MQTT 客户端 (Websockets 8884) ---
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
    except:
        return None

# 自动连接
mqtt_client = init_mqtt()

# --- 5. 侧边栏：精简的设备控制 (风扇与加热器并列) ---
with st.sidebar:
    st.header("🎮 远程控制")
    
    # 建立短连接用于发布，防止 loop_start 线程冲突
    def send_cmd(device, action):
        cmd = json.dumps({"device": device, "action": action, "time": time.time()})
        mqtt_client.publish("cowshed/control/manual", cmd)
        st.toast(f"指令已发: {device} -> {action}")

    # 并列布局
    col_fan, col_heat = st.columns(2)
    with col_fan:
        st.write("**排风扇**")
        if st.button("开启风扇"): send_cmd("fan", "on")
        if st.button("关闭风扇"): send_cmd("fan", "off")
    
    with col_heat:
        st.write("**加热器**")
        if st.button("开启加热"): send_cmd("heater", "on")
        if st.button("关闭加热"): send_cmd("heater", "off")

# --- 6. 数据处理循环 ---
# 将队列堆积的数据全部取出转入 session_state
while not msg_queue.empty():
    st.session_state.history.append(msg_queue.get())
    # 保持历史记录在 60 条以内，保证绘图流畅度
    if len(st.session_state.history) > 60:
        st.session_state.history.pop(0)

# --- 7. UI 渲染：动态居中的 Altair 图表 ---
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    latest = df.iloc[-1]
    
    # A. 顶部 Metric展示 (删除了 lx 等描述)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("温度", f"{latest.get('temp', 0)} ℃")
    m2.metric("湿度", f"{latest.get('humi', 0)} %")
    m3.metric("氨气浓度", f"{latest.get('ammonia', 0)} ppm")
    m4.metric("光照强度", f"{latest.get('light', 0)}")

    st.divider()
    st.subheader("📈 环境趋势 Monitoring (自动居中模式)")

    # --- B. 关键修复：动态计算 Y轴范围的Altair绘图函数 ---
    def create_dynamic_chart(data, column, title, color):
        if data[column].isnull().all():
            return None
        
        # 1. 计算当前数据的最大和最小值
        min_val = data[column].min()
        max_val = data[column].max()
        
        # 2. 如果数据平直，强制设置范围，防止 domain=[0,0] 报错
        if min_val == max_val:
            domain = [min_val - 1, max_val + 1]
        else:
            # 3. 如果数据存在陡变，自动居中算法：
            # 在上下预留 15% 的“呼吸空间”，确保中间 70% 的区域用于显示曲线
            padding = (max_val - min_val) * 0.15
            domain = [min_val - padding, max_val + padding]

        # 4. 创建锁定 Y轴动态范围的 Altair Chart
        chart = alt.Chart(data).mark_line(color=color).encode(
            # X轴显示时间
            x=alt.X('timestamp:T', title='时间', axis=alt.Axis(format='%H:%M')),
            # Y轴锁定动态计算出的 domain
            y=alt.Y(f'{column}:Q', title=title, scale=alt.Scale(domain=domain)),
            tooltip=['timestamp', column]
        ).properties(height=280)
        return chart

    # 绘制温、湿、气、光四个图表，布局对齐
    row1_l, row1_r = st.columns(2)
    row2_l, row2_r = st.columns(2)

    with row1_l:
        st.altair_chart(create_dynamic_chart(df, 'temp', '温度 (°C)', '#FF4B4B'), use_container_width=True)
    with row1_r:
        st.altair_chart(create_dynamic_chart(df, 'humi', '湿度 (%)', '#0068C9'), use_container_width=True)
    with row2_l:
        st.altair_chart(create_dynamic_chart(df, 'ammonia', '氨气 (ppm)', '#29B09D'), use_container_width=True)
    with row2_r:
        st.altair_chart(create_dynamic_chart(df, 'light', '光照', '#FFD700'), use_container_width=True)

# --- 8. 刷新驱动 ---
# 使用 st.rerun() 驱动界面根据最新数据重新计算绘图范围
time.sleep(1.5)
st.rerun()
