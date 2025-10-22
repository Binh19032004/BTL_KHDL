import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
# from streamlit_option_menu import option_menu  <- Đã xóa
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')



st.set_page_config(page_title="BTC Analysis Pro", page_icon="📈", layout="wide", initial_sidebar_state="collapsed") # Thu gọn sidebar (mặc dù ta không dùng)

st.markdown("""
<style>
    .main { padding-top: 1.5rem; }
    h1 { color: #1f77b4; margin-bottom: 0.5rem; }
    .insight { background: rgba(31,119,180,0.15); padding: 15px; border-left: 5px solid #1f77b4; border-radius: 6px; margin-bottom: 1.5rem; font-size: 14px; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# CÁC HÀM LẤY DỮ LIỆU, XỬ LÝ VÀ VẼ BIỂU ĐỒ
# ==============================================================================

@st.cache_data
def fetch_binance(sym='BTCUSDT', intv='1h', lim=500):
    url = 'https://api.binance.com/api/v3/klines'
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        r = requests.get(url, params={'symbol': sym, 'interval': intv, 'limit': lim}, 
                        timeout=15, headers=headers)
        r.raise_for_status()
        cols = ['t', 'o', 'h', 'l', 'c', 'v', 'ct', 'qav', 'nt', 'tbbav', 'tbqav', 'ig']
        df = pd.DataFrame(r.json(), columns=cols)
        return df
    except Exception as e:
        st.error(f"❌ Lỗi API Binance: {e}")
        return None

@st.cache_data
def fetch_news(key='fb371b39780a94f8a3500184fcdd2aa0326ebc66'):
    url = 'https://cryptopanic.com/api/v1/posts/'
    try:
        r = requests.get(url, params={'auth_token': key, 'kind': 'news', 'filter': 'trending', 'limit': 20}, timeout=15)
        r.raise_for_status()
        data = r.json()
        news = []
        if 'results' in data:
            for item in data['results']:
                title = item.get('title', '').lower()
                if any(word in title for word in ['bitcoin', 'btc', 'crypto', 'btc', 'ethereum']):
                    news.append({
                        'tiêu_đề': item.get('title', 'N/A'),
                        'loại': item.get('kind', 'news'),
                        'nguồn': item.get('source', {}).get('title', 'Unknown') if item.get('source') else 'Unknown'
                    })
        return news[:20]
    except Exception as e:
        st.warning(f"⚠️ Lỗi CryptoPanic: {e}")
        return []

@st.cache_data
def clean_binance(df):
    d = df.copy()
    
    num_cols = ['o', 'h', 'l', 'c', 'v', 'qav']
    for col in num_cols:
        d[col] = pd.to_numeric(d[col], errors='coerce')
    
    d['t'] = pd.to_datetime(d['t'], unit='ms')
    d = d.dropna(subset=['c', 'h', 'l', 'o', 'v'])
    d = d[d['v'] > 0].reset_index(drop=True)
    
    d['ret'] = d['c'].pct_change() * 100
    d['rng'] = d['h'] - d['l']
    d['bod'] = abs(d['c'] - d['o'])
    d['vol7'] = d['c'].rolling(7).std()
    d['vol14'] = d['c'].rolling(14).std()
    d['ma7'] = d['c'].rolling(7).mean()
    d['ma21'] = d['c'].rolling(21).mean()
    d['ma50'] = d['c'].rolling(50).mean()
    d['rsi'] = calc_rsi(d['c'])
    d['macd'], d['signal'] = calc_macd(d['c'])
    d['bb_up'], d['bb_down'] = calc_bb(d['c'])
    
    d['dir'] = d['ret'].apply(lambda x: 'TĂNG' if x > 0 else ('GIẢM' if x < 0 else 'NGANG'))
    d['wd'] = d['t'].dt.day_name()
    d['wd_vn'] = d['t'].dt.day_name().map({
        'Monday': 'Thứ Hai', 'Tuesday': 'Thứ Ba', 'Wednesday': 'Thứ Tư',
        'Thursday': 'Thứ Năm', 'Friday': 'Thứ Sáu', 'Saturday': 'Thứ Bảy', 'Sunday': 'Chủ Nhật'
    })
    d['hr'] = d['t'].dt.hour
    d['dy'] = d['t'].dt.date
    d['wk'] = d['t'].dt.isocalendar().week
    d['vol_norm'] = (d['v'] - d['v'].mean()) / d['v'].std()
    
    return d.reset_index(drop=True)

def calc_rsi(pr, p=14):
    d = pr.diff()
    g = d.where(d > 0, 0)
    l = -d.where(d < 0, 0)
    rs = g.rolling(p).mean() / l.rolling(p).mean()
    return 100 - (100 / (1 + rs))

def calc_macd(pr, f=12, s=26, sig=9):
    m = pr.ewm(span=f).mean() - pr.ewm(span=s).mean()
    s_line = m.ewm(span=sig).mean()
    return m, s_line

def calc_bb(pr, p=20, dev=2):
    ma = pr.rolling(p).mean()
    std = pr.rolling(p).std()
    return ma + (std * dev), ma - (std * dev)

# --- (CÁC HÀM CHART ĐÃ ĐƯỢC ĐỔI TÊN TITLE) ---
def chart_hist(d):
    fig = go.Figure()
    ret = d['ret'].dropna()
    fig.add_trace(go.Histogram(x=ret, nbinsx=40, marker=dict(color='rgba(31,119,180,0.7)', line=dict(color='rgba(31,119,180,1)', width=1)), hovertemplate='Lợi suất: %{x:.2f}%<br>Tần suất: %{y}<extra></extra>'))
    fig.add_vline(x=ret.mean(), line_dash="dash", line_color="red", annotation_text=f"Trung bình: {ret.mean():.3f}%", annotation_position="top right")
    # ĐÃ THAY ĐỔI TITLE
    fig.update_layout(title='Phân Tích Tần Suất Lợi Nhuận (Histogram)', xaxis_title='Lợi Suất (%)', yaxis_title='Tần Suất', template='plotly_white', height=500, hovermode='x unified')
    return fig

def chart_box(d):
    fig = go.Figure()
    for dr in ['GIẢM', 'NGANG', 'TĂNG']:
        sub = d[d['dir'] == dr]['rng']
        if len(sub) > 0:
            fig.add_trace(go.Box(y=sub, name=dr, boxmean='sd', marker_color={'TĂNG': '#00cc96', 'GIẢM': '#ef553b', 'NGANG': '#636efb'}[dr]))
    # ĐÃ THAY ĐỔI TITLE
    fig.update_layout(title='Phạm Vi Biến Động Giá (Boxplot)', yaxis_title='Biên Độ ($)', template='plotly_white', height=500)
    return fig

def chart_violin(d):
    fig = go.Figure()
    wd_ord = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for wd in wd_ord:
        sub = d[d['wd'] == wd]['vol7'].fillna(d['vol7'].mean())
        if len(sub) > 0:
            wd_vn = d[d['wd'] == wd]['wd_vn'].iloc[0]
            fig.add_trace(go.Violin(y=sub, name=wd_vn, box_visible=True, meanline_visible=True))
    # ĐÃ THAY ĐỔI TITLE
    fig.update_layout(title='Phân Phối Biến Động (Violin) theo Ngày', yaxis_title='Độ Biến Động 7 Ngày ($)', template='plotly_white', height=500)
    return fig

def chart_line(d):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d['t'], y=d['c'], mode='lines', name='Giá Đóng Cửa', line=dict(color='#1f77b4', width=2), hovertemplate='%{x}<br>$%{y:.2f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=d['t'], y=d['ma7'], mode='lines', name='MA7 (Xu Hướng Ngắn)', line=dict(color='#ff7f0e', width=1, dash='dash')))
    fig.add_trace(go.Scatter(x=d['t'], y=d['ma21'], mode='lines', name='MA21 (Xu Hướng Trung)', line=dict(color='#d62728', width=1, dash='dot')))
    # ĐÃ THAY ĐỔI TITLE
    fig.update_layout(title='Phân Tích Xu Hướng Giá (MA7, MA21)', xaxis_title='Thời Gian', yaxis_title='Giá ($)', template='plotly_white', height=500, hovermode='x unified')
    return fig

def chart_area(d):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d['t'], y=d['c'], fill='tozeroy', name='Giá', line=dict(color='#1f77b4'), hovertemplate='%{x}<br>$%{y:.2f}<extra></extra>'))
    # ĐÃ THAY ĐỔI TITLE
    fig.update_layout(title='Biểu Đồ Vùng: Giá Đóng Cửa Theo Thời Gian', xaxis_title='Thời Gian', yaxis_title='Giá ($)', template='plotly_white', height=500, hovermode='x unified')
    return fig

def chart_scatter(d):
    clean_d = d.dropna(subset=['ret', 'vol7'])
    x = np.array(range(len(clean_d))).reshape(-1, 1)
    y = clean_d['ret'].values
    model = LinearRegression().fit(x, y)
    y_pred = model.predict(x)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=clean_d['t'], y=clean_d['ret'], mode='markers', name='Lợi Suất', marker=dict(size=6, color='#1f77b4', opacity=0.6), hovertemplate='%{x}<br>Lợi Suất: %{y:.2f}%<extra></extra>'))
    fig.add_trace(go.Scatter(x=clean_d['t'], y=y_pred, mode='lines', name='Xu Hướng', line=dict(color='red', width=2)))
    # ĐÃ THAY ĐỔI TITLE
    fig.update_layout(title='Hồi Quy Tuyến Tính: Xu Hướng Lợi Suất', xaxis_title='Thời Gian', yaxis_title='Lợi Suất (%)', template='plotly_white', height=500)
    return fig

def chart_heatmap(d):
    cols = ['c', 'o', 'h', 'l', 'v', 'ret', 'rng', 'vol7', 'rsi']
    corr = d[cols].corr()
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='RdBu', zmid=0, hovertemplate='%{x} - %{y}: %{z:.2f}<extra></extra>'))
    # ĐÃ THAY ĐỔI TITLE
    fig.update_layout(title='Bản Đồ Nhiệt: Tương Quan Dữ Liệu', height=600, width=700)
    return fig

def chart_treemap(d):
    d_agg = d.groupby('dy').agg({'ret': 'sum', 'c': 'last', 'v': 'sum'}).reset_index()
    d_agg.columns = ['Ngày', 'Lợi Suất Tổng', 'Giá Cuối', 'Khối Lượng']
    d_agg['Abs Return'] = abs(d_agg['Lợi Suất Tổng'])
    d_agg['Ngày Str'] = d_agg['Ngày'].astype(str)
    d_agg['parent'] = ''
    
    if len(d_agg) == 0:
        st.warning("❌ Không có dữ liệu Treemap")
        return None
    
    fig = go.Figure(go.Treemap(
        labels=d_agg['Ngày Str'],
        parents=d_agg['parent'],
        values=d_agg['Abs Return'],
        marker=dict(
            colors=d_agg['Lợi Suất Tổng'],
            colorscale='RdYlGn',
            cmid=0,
            line=dict(color='white', width=2)
        ),
        hovertemplate='<b>%{label}</b><br>Lợi Suất: %{customdata[0]:.3f}%<br>Giá: $%{customdata[1]:.2f}<br>Khối Lượng: %{customdata[2]:.0f}<extra></extra>',
        customdata=d_agg[['Lợi Suất Tổng', 'Giá Cuối', 'Khối Lượng']].values
    ))
    # ĐÃ THAY ĐỔI TITLE
    fig.update_layout(title='Treemap: Lợi Suất Ròng Hàng Ngày', height=600)
    return fig

def chart_sunburst(d):
    d_agg = d.groupby('dy').agg({'ret': 'sum', 'v': 'sum', 'hr': 'first'}).reset_index()
    d_agg.columns = ['Ngày', 'Lợi Suất', 'Khối Lượng', 'Giờ']
    d_agg['Ngày Str'] = d_agg['Ngày'].astype(str)
    
    if len(d_agg) == 0:
        st.warning("❌ Không có dữ liệu Sunburst")
        return None
    
    labels = ['Tổng Cộng'] + d_agg['Ngày Str'].tolist()
    parents = [''] + ['Tổng Cộng'] * len(d_agg)
    values = [d_agg['Lợi Suất'].sum()] + d_agg['Lợi Suất'].tolist()
    colors = [0] + d_agg['Lợi Suất'].tolist()
    
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(
            colors=colors,
            colorscale='RdYlGn',
            cmid=0,
            showscale=True
        ),
        hovertemplate='<b>%{label}</b><br>Lợi Suất: %{value:.3f}%<extra></extra>'
    ))
    # ĐÃ THAY ĐỔI TITLE
    fig.update_layout(title='Sunburst: Phân Cấp Lợi Suất Hàng Ngày', height=600)
    return fig

def chart_wordcloud(news):
    if not news:
        st.warning("⚠️ Không có dữ liệu tin tức")
        return None
    
    txt = ' '.join([item['tiêu_đề'] for item in news])
    if len(txt) < 20:
        st.warning("⚠️ Dữ liệu văn bản quá ít cho WordCloud")
        return None
    
    wc = WordCloud(width=1200, height=500, background_color='white', colormap='viridis', prefer_horizontal=0.7).generate(txt)
    fig, ax = plt.subplots(figsize=(12, 5))
    # ĐÃ THAY ĐỔI TITLE
    ax.set_title('Các Chủ Đề Nóng (WordCloud)')
    ax.axis('off')
    return fig

def chart_network(d):
    d_pivot = d.pivot_table(values='ret', index='wd_vn', columns='hr', aggfunc='mean').fillna(0)
    wd_ord_vn = ['Thứ Hai', 'Thứ Ba', 'Thứ Tư', 'Thứ Năm', 'Thứ Sáu', 'Thứ Bảy', 'Chủ Nhật']
    d_pivot = d_pivot.reindex([w for w in wd_ord_vn if w in d_pivot.index])
    
    fig = go.Figure(data=go.Heatmap(z=d_pivot.values, x=d_pivot.columns, y=d_pivot.index, colorscale='RdBu', zmid=0, hovertemplate='%{x}h - %{y}: %{z:.3f}%<extra></extra>'))
    # ĐÃ THAY ĐỔI TITLE
    fig.update_layout(title='Heatmap: Lợi Suất Trung Bình (Giờ vs. Ngày)', xaxis_title='Giờ Trong Ngày', yaxis_title='Ngày Trong Tuần', height=500)
    return fig

# ==============================================================================
# HÀM STORYTELLING (SẼ ĐƯỢC GỌI Ở CUỐI)
# ==============================================================================

def render_story(d, news):
    # ĐÃ THAY ĐỔI TITLE CHÍNH
    st.markdown("# 📖 Báo Cáo Tổng Hợp & Kết Luận Của Chuyên Gia")
    
    pr_chg = ((d['c'].iloc[-1] - d['c'].iloc[0]) / d['c'].iloc[0]) * 100
    avg_ret = d['ret'].mean()
    vol_avg = d['vol7'].mean()
    max_rng = d['rng'].max()
    bull_cnt = len(d[d['dir'] == 'TĂNG'])
    bear_cnt = len(d[d['dir'] == 'GIẢM'])
    
    st.markdown(f"""
    ## 📊 Tóm Tắt 
    
    **Biến Động Giá:** Bitcoin đã thay đổi **{pr_chg:+.2f}%** trong khoảng thời gian phân tích.
    
    **Tâm Lý Thị Trường:** Có **{bull_cnt}** giờ tăng so với **{bear_cnt}** giờ giảm (Tỷ Lệ: {bull_cnt/(bear_cnt+1):.2f}:1).
    
    **Phân Tích Độ Biến Động:** Độ biến động trung bình 7 ngày đạt **${vol_avg:.2f}**. 
    
    **Hồ Sơ Lợi Suất:** Lợi suất trung bình mỗi giờ **{avg_ret:+.3f}%**
    """)
    
    st.markdown("---")
    
    st.markdown("## 💡 Những Hiểu Biết Chính (Insights)")
    st.markdown("""
    1. **Hành Động Giá:** Theo dõi xu hướng với các mức hỗ trợ/kháng cự rõ ràng được xác định qua Đường Trung Bình Động
    2. **Phân Bố Lợi Suất:** Histogram cho thấy phân bố gần như chuẩn nhưng có đuôi lệch, chỉ ra các sự kiện cực đoan ít xảy ra
    3. **Mô Hình Theo Ngày:** Một số ngày trong tuần có mức biến động cao hơn, có thể do tin tức hoặc hoạt động giao dịch
    4. **Mối Tương Quan:** Khối lượng thường tăng trong các ngày biến động cao, cho thấy tham gia của nhà đầu tư
    5. **Cơ Hội Giao Dịch:** Bản đồ mô hình tiết lộ các cửa sổ giao dịch tối ưu theo giờ và ngày
    """)
    
    st.markdown("---")
    
    st.markdown("## 🔔 Tin Tức Thị Trường Gần Đây")
    st.markdown("**Tin Tức Quan Trọng Về Bitcoin:**")
    
    if news:
        for i, item in enumerate(news[:15], 1):
            st.markdown(f"**{i}.** {item['tiêu_đề'][:100]}... *(Nguồn: {item['nguồn']})*")
    else:
        st.markdown("ℹ️ Không có tin tức quan trọng mới")
    
    st.markdown("---")
    
    # === PHẦN KẾT LUẬN CHUYÊN GIA MỚI ===
    st.markdown("## 🎓 Kết Luận & Cơ Hội Giao Dịch Của Chuyên Gia")
    
    # 1. Phân tích MA
    try:
        last_ma7 = d['ma7'].iloc[-1]
        last_ma21 = d['ma21'].iloc[-1]
        prev_ma7 = d['ma7'].iloc[-2]
        prev_ma21 = d['ma21'].iloc[-2]
        
        ma_signal = ""
        if last_ma7 > last_ma21 and prev_ma7 <= prev_ma21:
            ma_signal = f"**TÍN HIỆU MUA (Bullish Crossover):** Đường MA7 (ngắn hạn) vừa cắt lên trên MA21 (trung hạn). Giá hiện tại: **${d['c'].iloc[-1]:.2f}**."
        elif last_ma7 < last_ma21 and prev_ma7 >= prev_ma21:
            ma_signal = f"**TÍN HIỆU BÁN (Bearish Crossover):** Đường MA7 (ngắn hạn) vừa cắt xuống dưới MA21 (trung hạn). Giá hiện tại: **${d['c'].iloc[-1]:.2f}**."
        elif last_ma7 > last_ma21:
            ma_signal = f"**Xu hướng TĂNG:** Giá đang trong xu hướng tăng ngắn hạn (MA7 > MA21). Chờ tín hiệu mua khi giá điều chỉnh về gần MA7 (**~${last_ma7:.2f}**) hoặc MA21 (**~${last_ma21:.2f}**)."
        else:
            ma_signal = f"**Xu hướng GIẢM:** Giá đang trong xu hướng giảm ngắn hạn (MA7 < MA21). Chờ tín hiệu bán khi giá hồi phục về gần MA7 (**~${last_ma7:.2f}**) hoặc MA21 (**~${last_ma21:.2f}**)."
        
        st.markdown(f"**1. Phân Tích Xu Hướng (MA):**\n{ma_signal}")
    except Exception as e:
        st.markdown("**1. Phân Tích Xu Hướng (MA):**\nKhông đủ dữ liệu để phân tích đường MA.")


    # 2. Phân tích RSI
    try:
        last_rsi = d['rsi'].iloc[-1]
        rsi_signal = ""
        if last_rsi > 70:
            rsi_signal = f"**CẢNH BÁO QUÁ MUA (Overbought):** RSI hiện tại là **{last_rsi:.2f}** (> 70). Thị trường đang hưng phấn quá mức, rủi ro điều chỉnh giảm là cao. Cân nhắc chốt lời hoặc đứng ngoài."
        elif last_rsi < 30:
            rsi_signal = f"**CƠ HỘI QUÁ BÁN (Oversold):** RSI hiện tại là **{last_rsi:.2f}** (< 30). Thị trường đang bi quan, đây có thể là cơ hội MUA vào nếu kết hợp với các tín hiệu hỗ trợ khác."
        else:
            rsi_signal = f"**Trung tính:** Động lượng thị trường đang ở mức trung tính (RSI: **{last_rsi:.2f}**). Giao dịch nên dựa vào xu hướng (MA) hoặc mô hình giá."
            
        st.markdown(f"**2. Phân Tích Động Lượng (RSI):**\n{rsi_signal}")
    except Exception as e:
        st.markdown("**2. Phân Tích Động Lượng (RSI):**\nKhông đủ dữ liệu để phân tích RSI.")

    # 3. Phân tích Biến động & Thời gian (Từ Violin & Heatmap)
    try:
        vol_by_day = d.groupby('wd_vn')['vol7'].mean().sort_values(ascending=False)
        most_vol_day = vol_by_day.index[0]
        
        ret_by_hour = d.groupby('hr')['ret'].mean()
        best_hour = ret_by_hour.idxmax()
        worst_hour = ret_by_hour.idxmin()
        
        time_signal = f"**Ngày biến động nhất** (rủi ro & cơ hội cao) là **{most_vol_day}**. Dựa trên dữ liệu 500 giờ qua, **khung giờ {best_hour}:00** có lợi suất trung bình cao nhất, trong khi **khung giờ {worst_hour}:00** có lợi suất trung bình thấp nhất."
        st.markdown(f"**3. Phân Tích Mô Hình (Thời Gian):**\n{time_signal}")
    except Exception as e:
        st.markdown("**3. Phân Tích Mô Hình (Thời Gian):**\nKhông đủ dữ liệu để phân tích mô hình thời gian.")

    
    st.markdown(f"""
    ---
    ## 🎓 Tổng Kết Của Chuyên Gia
    
    Kết hợp các yếu tố trên:
    * **Xu hướng chính:** {ma_signal.split(':', 1)[-1].strip()}
    * **Động lượng:** {rsi_signal.split(':', 1)[-1].strip()}
    * **Thời điểm:** {time_signal.split(':', 1)[-1].strip()}
    
    **Khuyến nghị:** Đây là tài sản biến động cao. Giao dịch nên tuân thủ quản lý rủi ro nghiêm ngặt.
    """)
    
    # Giữ lại các metrics cuối cùng
    st.columns(5) # Tạo khoảng trống
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("💰 Giá Hiện Tại", f"${d['c'].iloc[-1]:.2f}")
    with col2:
        st.metric("📊 Thay Đổi (500h)", f"{pr_chg:+.2f}%")
    with col3:
        st.metric("📈 Lợi Suất TB/h", f"{avg_ret:+.3f}%")
    with col4:
        st.metric("🟢 Giờ Tăng", f"{bull_cnt}")
    with col5:
        st.metric("📉 BĐ TB (7h)", f"${vol_avg:.2f}")

# ==============================================================================
# HÀM MAIN() - ĐÃ SẮP XẾP LẠI
# ==============================================================================

def main():
    st.markdown("<h1>📈 Dashboard Phân Tích Bitcoin Pro</h1>", unsafe_allow_html=True)
    st.markdown("Phân Tích Nâng Cao | Trực Quan Hóa Tương Tác Thời Gian Thực | Insights Chuyên Nghiệp")
    st.markdown("---")
    
    with st.spinner("⏳ Đang tải dữ liệu..."):
        df_raw = fetch_binance(sym='BTCUSDT', intv='1h', lim=500)
        news = fetch_news(key='fb371b39780a94f8a3500184fcdd2aa0326ebc66')
        
        if df_raw is None:
            st.error("Không thể tải dữ liệu từ Binance. Vui lòng thử lại sau.")
            st.stop()
        
        df = clean_binance(df_raw)
    
    # Hiển thị các chỉ số chính (KPIs)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("💰 Giá", f"${df['c'].iloc[-1]:.2f}")
    with col2:
        st.metric("📈 Giờ Này", f"{df['ret'].iloc[-1]:+.2f}%")
    with col3:
        st.metric("📊 TB", f"{df['ret'].mean():+.2f}%")
    with col4:
        st.metric("Biên Độ", f"${df['rng'].mean():.2f}")
    with col5:
        st.metric("RSI", f"{df['rsi'].iloc[-1]:.1f}")
    
    st.markdown("---")
    
    # === PHẦN 1: PHÂN TÍCH BIỂU ĐỒ CHI TIẾT ===
    st.markdown("## 📈 Phân Tích Biểu Đồ Chi Tiết")
    
    # ĐÃ THAY ĐỔI TIÊU ĐỀ INSIGHT
    st.markdown('<div class="insight"><b>📈 Phân Tích Xu Hướng Giá (MA7, MA21):</b><br>Hiển thị giá đóng cửa (xanh) cùng với 2 đường trung bình động. MA7 (cam nét) thể hiện xu hướng ngắn hạn (7 giờ). MA21 (đỏ chấm) thể hiện xu hướng trung hạn (21 giờ). Khi MA7 cắt lên trên MA21 = tín hiệu mua. Khi cắt xuống = tín hiệu bán. Đây là chiến lược giao dịch cơ bản.</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_line(df), use_container_width=True)
    
    # ĐÃ THAY ĐỔI TIÊU ĐỀ INSIGHT
    st.markdown('<div class="insight"><b>📉 Biểu Đồ Vùng: Giá Đóng Cửa Theo Thời Gian:</b><br>Tương tự biểu đồ đường nhưng vùng dưới đường được tô màu xanh. Trực quan hóa hành động giá theo thời gian, nhất là để thấy rõ mức độ "bật" của giá. Diện tích càng lớn = giá càng cao. Giúp nắm bắt nhanh xu hướng tổng quát của giá trong giai đoạn dài.</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_area(df), use_container_width=True)

    # Chia cột cho các biểu đồ nhỏ hơn
    c1, c2 = st.columns(2)
    with c1:
        # ĐÃ THAY ĐỔI TIÊU ĐỀ INSIGHT
        st.markdown('<div class="insight"><b>📊 Phân Tích Tần Suất Lợi Nhuận (Histogram):</b><br>Hiển thị tần suất xuất hiện của mỗi mức lợi suất hàng giờ. Giúp xác định mô hình lợi suất và các ngoại lệ. Nếu biểu đồ có hình chuông (phân bố chuẩn), thì thị trường đang hoạt động theo quy luật. Đường đỏ ngang là giá trị trung bình - nếu lệch trái có nghĩa lợi suất âm chiếm đa số, lệch phải là lợi suất dương chiếm đa số.</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_hist(df), use_container_width=True)
    with c2:
        # ĐÃ THAY ĐỔI TIÊU ĐỀ INSIGHT
        st.markdown('<div class="insight"><b>📦 Phạm Vi Biến Động Giá (Boxplot):</b><br>So sánh phạm vi dao động (Cao - Thấp) giữa các ngày tăng/giảm/đi ngang. Hộp càng to = biên độ càng lớn = bất ổn định. Đường trong hộp = trung vị (50% dữ liệu). Các chấm ngoài = ngoại lệ. Giúp nhận biết khi nào thị trường "sôi động" hoặc "yên tĩnh".</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_box(df), use_container_width=True)

    # ĐÃ THAY ĐỔI TIÊU ĐỀ INSIGHT
    st.markdown('<div class="insight"><b>🎻 Phân Phối Biến Động (Violin) theo Ngày:</b><br>Hiển thị phân bố độ biến động (volatility) cho mỗi ngày trong tuần. Hình bầu dục rộng = biến động cao và không ổn định. Hình hẹp = biến động thấp và ổn định. Có thể phát hiện ngày nào trong tuần giao dịch "nóng" nhất. Ví dụ: Thứ Sáu có thể biến động hơn Thứ Hai.</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_violin(df), use_container_width=True)
    
    # ĐÃ THAY ĐỔI TIÊU ĐỀ INSIGHT
    st.markdown('<div class="insight"><b>🔵 Hồi Quy Tuyến Tính: Xu Hướng Lợi Suất:</b><br>Mỗi chấm xanh = lợi suất 1 giờ. Đường đỏ = đường hồi quy tuyến tính thể hiện xu hướng tổng thể của lợi suất. Nếu đường đỏ đi lên = lợi suất có xu hướng tăng. Nếu đi xuống = xu hướng giảm. Độ dốc của đường = tốc độ thay đổi. Giúp xác định momentum thị trường.</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_scatter(df), use_container_width=True)
    
    # ĐÃ THAY ĐỔI TIÊU ĐỀ INSIGHT
    st.markdown('<div class="insight"><b>🔥 Bản Đồ Nhiệt: Tương Quan Dữ Liệu:</b><br>Hiển thị mối quan hệ giữa các biến (giá, khối lượng, biến động, RSI). Màu xanh = tương quan dương (cùng tăng giảm). Màu đỏ = tương quan âm (ngược nhau). Càng đậm = tương quan càng mạnh. Ví dụ: Nếu khối lượng & biến động là xanh đậm = khi khối lượng lớn thì biến động cũng lớn.</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_heatmap(df), use_container_width=True)
    
    # ĐÃ THAY ĐỔI TIÊU ĐỀ INSIGHT
    st.markdown('<div class="insight"><b>🌳 Treemap: Lợi Suất Ròng Hàng Ngày:</b><br>Mỗi hình chữ nhật = 1 ngày. Kích thước hình = lợi suất tuyệt đối (càng to = dao động càng lớn). Màu xanh = ngày tăng (lợi suất dương). Màu đỏ = ngày giảm (lợi suất âm). Xem nhanh ngày nào "sôi động" nhất và ngày nào lợi suất tốt nhất. Điều này giúp phát hiện mô hình giao dịch theo ngày.</div>', unsafe_allow_html=True)
    fig_tree = chart_treemap(df)
    if fig_tree:
        st.plotly_chart(fig_tree, use_container_width=True)
        
    # ĐÃ THAY ĐỔI TIÊU ĐỀ INSIGHT
    st.markdown('<div class="insight"><b>☀️ Sunburst: Phân Cấp Lợi Suất Hàng Ngày:</b><br>Hiển thị lợi suất phân cấp theo Tháng → Tuần. Vòng giữa = tháng, vòng ngoài = tuần. Kích thước cung = lợi suất tuyệt đối. Màu = tích cực/tiêu cực. Click vào cung để zoom vào chi tiết. Giúp xác định tháng & tuần nào hoạt động tốt nhất. Phát hiện mô hình theo thời gian lớn.</div>', unsafe_allow_html=True)
    fig_sun = chart_sunburst(df)
    if fig_sun:
        st.plotly_chart(fig_sun, use_container_width=True)
        
    # ĐÃ THAY ĐỔI TIÊU ĐỀ INSIGHT
    st.markdown('<div class="insight"><b>💬 Các Chủ Đề Nóng (WordCloud):</b><br>Dữ liệu từ CryptoPanic API (tin tức tiền điện tử). Từ càng to = xuất hiện trong tin tức càng nhiều. Giúp xác định chủ đề đang bị nhà đầu tư chú ý. Ví dụ: Nếu "ETF" to = có tin ETF Bitcoin, có thể ảnh hưởng đến giá. Hữu ích để hiểu "tâm lý thị trường" lúc này.</div>', unsafe_allow_html=True)
    fig_wc = chart_wordcloud(news)
    if fig_wc:
        st.pyplot(fig_wc, use_container_width=True)
        
    # ĐÃ THAY ĐỔI TIÊU ĐỀ INSIGHT
    st.markdown('<div class="insight"><b>🕸️ Heatmap: Lợi Suất Trung Bình (Giờ vs. Ngày):</b><br>Ma trận với hàng = ngày trong tuần, cột = giờ trong ngày. Mỗi ô = lợi suất trung bình. Xanh đậm = giờ/ngày giao dịch lợi suất cao. Đỏ đậm = giờ/ngày lợi suất thấp. Giúp "nhà giao dịch nhạy cảm thời gian" tìm giờ vàng để giao dịch. Ví dụ: Có thể thấy Thứ Sáu 14h luôn "sôi động".</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_network(df), use_container_width=True)
    
    st.markdown("---")
    
    # === PHẦN 2: STORYTELLING & KẾT LUẬN (ĐÃ CHUYỂN XUỐNG CUỐI) ===
    # Gọi hàm render_story trực tiếp
    render_story(df, news)
    
    st.markdown("---")
    st.markdown(f"*📅 Cập Nhật: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 📊 Dữ Liệu: 500 nến 1 giờ từ Binance | 🔔 Tin Tức: CryptoPanic API*")

if __name__ == "__main__":
    main()