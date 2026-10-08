# Code cells from notebooks/Toronto_Notebook.json, Toronto (Trương Tấn Sang).
# Copied cell by cell; only the "%pyspark" line at the top of each cell is removed.
# Markdown, %sh and %angular cells are left out. Some cells use Zeppelin's z object.

# %% Cell 1
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ĐỌC DỮ LIỆU 
URL = "https://drive.google.com/uc?export=download&id=1Bz_qOlaKE7aIrW_QpPAFJLT1Nn8PgEq4"
df = pd.read_csv(URL, engine='python', on_bad_lines='skip')

# Xử lý cột thời gian
time_cols = ['Ext_agent_app_or_defer_time', 'Fire_Under_Control_Time',
             'Last_TFS_Unit_Clear_Time', 'TFS_Alarm_Time', 'TFS_Arrival_Time']
for col in time_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Tính Response Time (phút)
df['Response_Time_Min'] = (df['TFS_Arrival_Time'] - df['TFS_Alarm_Time']).dt.total_seconds() / 60.0

# Trích năm / tháng / giờ
df['Year']  = df['TFS_Alarm_Time'].dt.year
df['Month'] = df['TFS_Alarm_Time'].dt.month
df['Hour']  = df['TFS_Alarm_Time'].dt.hour

print(f"Kích thước dữ liệu: {df.shape[0]:,} dòng x {df.shape[1]} cột")
print(f"Phạm vi thời gian: {df['TFS_Alarm_Time'].min()} → {df['TFS_Alarm_Time'].max()}")

# %% Cell 3
sample_columns = [
    "Area_of_Origin",
    "Possible_Cause",
    "Property_Use",
    "Estimated_Dollar_Loss",
    "Civilian_Casualties",
    "Count_of_Persons_Rescued",
    "TFS_Alarm_Time",
    "TFS_Arrival_Time"
]
existing_columns = [col for col in sample_columns if col in df.columns]
sample_data = df[existing_columns].sample(n=min(10, len(df)), random_state=42).copy()

for col in ['TFS_Alarm_Time', 'TFS_Arrival_Time']:
    if col in sample_data.columns:
        if not pd.api.types.is_datetime64_any_dtype(sample_data[col]):
            sample_data[col] = pd.to_datetime(sample_data[col], errors="coerce")
        sample_data[col] = sample_data[col].dt.strftime('%Y-%m-%d %H:%M:%S')

def clean_text(x):
    if pd.isna(x):
        return ""
    if isinstance(x, (int, float)):
        return f"{x:,.0f}" if x == int(x) or pd.isna(x) else f"{x:,.2f}"
    return str(x).replace('\t', ' ').replace('\n', ' ')

sample_data_cleaned = sample_data.applymap(clean_text)
header = "\t".join(sample_data_cleaned.columns)
rows = "\n".join(["\t".join(row) for row in sample_data_cleaned.values])

print(f"%table {header}\n{rows}")

# %% Cell 5
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
import numpy as np
import io, base64

# Thiết lập style
sns.set_theme(style="whitegrid")
BLUE, ORANGE, RED, SALMON, GREEN, GRAY = "#2F7FB1", "#D9992F", "#E74C3C", "#E8897C", "#58A65C", "#555555"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 11, "axes.labelsize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8, "figure.dpi": 120})

all_html = ""


# PHẦN 1: BOXPLOT THIỆT HẠI TÀI SẢN THEO 3 PHÂN KHÚC

loss = df["Estimated_Dollar_Loss"].dropna()
loss = loss[loss >= 0]
seg1 = loss[loss <= 20000]
seg2 = loss[(loss > 20000) & (loss <= 200000)]
seg3 = loss[loss > 200000]
total_cases = len(loss)

fig1 = plt.figure(figsize=(10, 6))
gs = GridSpec(2, 4, figure=fig1)
ax1 = fig1.add_subplot(gs[0, 0:2])
ax2 = fig1.add_subplot(gs[0, 2:4])
ax3 = fig1.add_subplot(gs[1, 1:3])
fmt = FuncFormatter(lambda x, pos: f"{x:,.0f}")

for ax, data, color, title in zip([ax1, ax2, ax3], [seg1, seg2, seg3], [BLUE, ORANGE, RED], 
                                  ['Phân khúc 1: 0–20,000', 'Phân khúc 2: 20,000–200,000', 'Phân khúc 3: > 200,000']):
    sns.boxplot(x=data, color=color, linewidth=1.2, fliersize=2, ax=ax)
    pct = (len(data)/total_cases*100) if total_cases > 0 else 0
    ax.set_title(f"Boxplot thiệt hại tài sản\n{title} CAD ({len(data):,} vụ = {pct:.1f}%)")
    ax.set_xlabel("Estimated Dollar Loss (CAD)")
    ax.grid(axis="x", alpha=0.3)
    ax.xaxis.set_major_formatter(fmt)

plt.tight_layout()
buf1 = io.BytesIO(); fig1.savefig(buf1, format='png', dpi=120); buf1.seek(0)
b64_1 = base64.b64encode(buf1.read()).decode()
plt.close(fig1)

all_html += f"""
<div style='text-align:center;margin:15px'>
  <img src='data:image/png;base64,{b64_1}' style='max-width:800px'><br>
  <a href='data:image/png;base64,{b64_1}' download='3_boxplots_segments.png' style='color:#2196F3;font-size:12px'> Tải: Boxplot 3 Phân khúc</a>
</div><hr>"""

# PHẦN 2: BOXPLOT + PHÂN PHỐI (LOG SCALE) CHO 3 BIẾN

def plot_log_charts(col, color, title, xlabel, fname_prefix):
    val = df[col].dropna()
    val = val[val >= 0]
    val_log = np.log1p(val)
    html_out = ""
    
    # Boxplot
    f_box, a_box = plt.subplots(figsize=(6, 3.5))
    sns.boxplot(x=val_log, color=color, linewidth=1.2, fliersize=3, ax=a_box)
    a_box.set_title(f"Boxplot {title} (log scale)")
    a_box.set_xlabel(xlabel)
    a_box.grid(axis="x", alpha=0.35)
    plt.tight_layout()
    
    buf_b = io.BytesIO(); f_box.savefig(buf_b, format='png', dpi=120); buf_b.seek(0)
    b64_box = base64.b64encode(buf_b.read()).decode()
    plt.close(f_box)
    
    html_out += f"""
    <div style='display:inline-block;text-align:center;margin:10px'>
      <img src='data:image/png;base64,{b64_box}' style='max-width:420px'><br>
      <a href='data:image/png;base64,{b64_box}' download='{fname_prefix}_boxplot_log.png' style='color:#2196F3;font-size:12px'> Tải: Boxplot {title}</a>
    </div>"""

    # Histogram
    f_hist, a_hist = plt.subplots(figsize=(6, 3.5))
    sns.histplot(val_log, bins=40, kde=True, color=color, edgecolor=GRAY, alpha=0.65, ax=a_hist)
    a_hist.set_title(f"Phân phối {title} (log scale)")
    a_hist.set_xlabel(xlabel)
    a_hist.set_ylabel("Số vụ")
    a_hist.grid(axis="y", alpha=0.35)
    plt.tight_layout()
    
    buf_h = io.BytesIO(); f_hist.savefig(buf_h, format='png', dpi=120); buf_h.seek(0)
    b64_hist = base64.b64encode(buf_h.read()).decode()
    plt.close(f_hist)
    
    html_out += f"""
    <div style='display:inline-block;text-align:center;margin:10px'>
      <img src='data:image/png;base64,{b64_hist}' style='max-width:420px'><br>
      <a href='data:image/png;base64,{b64_hist}' download='{fname_prefix}_dist_log.png' style='color:#2196F3;font-size:12px'> Tải: Phân phối {title}</a>
    </div><br>"""
    
    return html_out

all_html += plot_log_charts("Estimated_Dollar_Loss", BLUE, "thiệt hại tài sản ước tính", "log(1 + Estimated Dollar Loss)", "08_loss")
all_html += plot_log_charts("Civilian_Casualties", SALMON, "thương vong dân sự", "log(1 + Civilian Casualties)", "09_casualties")
all_html += plot_log_charts("Estimated_Number_Of_Persons_Displaced", GREEN, "số người phải di dời", "log(1 + Estimated Number Of Persons Displaced)", "10_displaced")

print("%html <div style='text-align:center'>" + all_html + "</div>")

# %% Cell 7
import numpy as np

# TÍNH TOÁN MOMENT STATISTICS
sns.set_theme(style="whitegrid")
LIGHT_BLUE = "#5DADE2"
SALMON = "#E8897C"
GRAY = "#555555"

stat_cols = ['Estimated_Dollar_Loss', 'Civilian_Casualties',
             'Estimated_Number_Of_Persons_Displaced', 'Response_Time_Min']

labels_dict = {
    'Estimated_Dollar_Loss':                 'Thiệt hại tài sản',
    'Civilian_Casualties':                   'Thương vong',
    'Estimated_Number_Of_Persons_Displaced': 'Sơ tán',
    'Response_Time_Min':                     'T.G Phản ứng',
}

moment_data = []

for col in stat_cols:
    vals = df[col].dropna().values.astype(float)
    n = len(vals)
    mu = np.mean(vals)
    sigma = np.std(vals, ddof=1)
    variance = np.var(vals, ddof=1)
    
    z_score = (vals - mu) / sigma
    skewness = (n / ((n-1)*(n-2))) * np.sum(z_score**3) if n > 2 else 0
    ex_kurtosis = np.mean(z_score**4) - 3
    hyperskewness = np.mean(z_score**5)
    hypertailedness = np.mean(z_score**6)
    
    moment_data.append({
        "Variable": labels_dict[col],
        "skewness": skewness,
        "excess_kurtosis": ex_kurtosis
    })
    
    print(f"\n {labels_dict[col]}")
    print(f"   Mean              = {mu:>20,.2f}")
    print(f"   Variance          = {variance:>20,.2f}")
    print(f"   Std Dev           = {sigma:>20,.2f}")
    print(f"   Skewness          = {skewness:>20,.2f}")
    print(f"   Excess Kurtosis   = {ex_kurtosis:>20,.2f}")
    print(f"   Hyperskewness     = {hyperskewness:>20,.2f}")
    print(f"   Hypertailedness   = {hypertailedness:>20,.2f}")

moment_plot = pd.DataFrame(moment_data)


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Biểu đồ 1: Skewness

ax1 = axes[0]
sns.barplot(data=moment_plot, x="Variable", y="skewness", color=LIGHT_BLUE, edgecolor=GRAY, ax=ax1)
ax1.set_title("Độ lệch phân phối (Skewness)", fontsize=11)
ax1.set_xlabel("Biến", fontsize=10)
ax1.set_ylabel("Skewness", fontsize=10)
ax1.tick_params(axis="x", labelsize=9)
ax1.grid(axis="y", alpha=0.35)

for container in ax1.containers:
    ax1.bar_label(container, fmt="%.2f", fontsize=9, padding=3)


# Biểu đồ 2: Excess Kurtosis

ax2 = axes[1]
sns.barplot(data=moment_plot, x="Variable", y="excess_kurtosis", color=SALMON, edgecolor=GRAY, ax=ax2)
ax2.set_title("Độ nhọn vượt mức (Excess Kurtosis)", fontsize=11)
ax2.set_xlabel("Biến", fontsize=10)
ax2.set_ylabel("Excess Kurtosis", fontsize=10)
ax2.tick_params(axis="x", labelsize=9)
ax2.grid(axis="y", alpha=0.35)

for container in ax2.containers:
    ax2.bar_label(container, fmt="%.2f", fontsize=9, padding=3)

plt.tight_layout()

# Chuyển đổi sang Base64 để hiển thị HTML trong Zeppelin
buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=120, bbox_inches='tight'); buf.seek(0)
b64 = base64.b64encode(buf.read()).decode()
plt.close(fig)

all_html = f"""
<div style='text-align:center;margin:15px'>
  <img src='data:image/png;base64,{b64}' style='max-width:850px'><br>
  <a href='data:image/png;base64,{b64}' download='08_skewness_kurtosis_combined.png' style='color:#2196F3;font-size:12px'> Tải: Biểu đồ Skewness & Kurtosis</a>
</div>
"""
print("%html " + all_html)

# %% Cell 9
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import io, base64

# THỐNG KÊ THEO PHẠM VI (SCOPED STATISTICS)


sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "DejaVu Sans", 
    "axes.titlesize": 14, 
    "axes.labelsize": 11, 
    "xtick.labelsize": 9, 
    "ytick.labelsize": 10, 
    "figure.dpi": 120
})

BLUE = "#2F7FB1"
SALMON = "#E8897C"
GRAY = "#555555"

all_html = ""

def fig_to_html(fig, filename, title_link, max_width="900px"):
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"""
        <div style='text-align:center;margin:15px'>
            <img src='data:image/png;base64,{b64}' style='max-width:850px'><br>
            <a href='data:image/png;base64,{b64}' download='08_skewness_kurtosis_combined.png' style='color:#2196F3;font-size:12px'> Tải: Biểu đồ Skewness & Kurtosis</a>
        </div>
        """

# 1. TOP LOẠI CÔNG TRÌNH CÓ THIỆT HẠI TRUNG BÌNH CAO NHẤT

property_df = df[
    df["Property_Use"].notna() & 
    df["Estimated_Dollar_Loss"].notna() & 
    (df["Estimated_Dollar_Loss"] >= 0)
].copy()

property_stats = (
    property_df.groupby("Property_Use")
    .agg(count=("Estimated_Dollar_Loss", "size"), mean_loss=("Estimated_Dollar_Loss", "mean"))
    .query("count >= 50")
    .sort_values("mean_loss", ascending=False)
)
top_property = property_stats.head(10).sort_values("mean_loss")

fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1 = top_property["mean_loss"].plot(kind="barh", color=BLUE, edgecolor=GRAY, ax=ax1)
ax1.set_title("Top loại công trình có thiệt hại trung bình cao nhất", fontweight="bold")
ax1.set_xlabel("Average Estimated Dollar Loss (CAD)")
ax1.set_ylabel("Property Use")
ax1.grid(axis="x", alpha=0.35)

for i, value in enumerate(top_property["mean_loss"]):
    ax1.text(value, i, f" {value:,.0f}", va="center", fontsize=9)

all_html += fig_to_html(fig1, "01_property_use_highest_average_loss.png", "Biểu đồ Property Use")


# 2. TOP 10 NGUYÊN NHÂN CÓ THỂ GÂY CHÁY

top_causes = (
    df["Possible_Cause"]
    .fillna("Không xác định")
    .value_counts()
    .head(10)
    .sort_values()
)

fig2, ax2 = plt.subplots(figsize=(12, 7))
bars = ax2.barh(
    top_causes.index,
    top_causes.values,
    color=sns.color_palette("flare", len(top_causes))
)
ax2.bar_label(bars, padding=4, fontsize=9)
ax2.set_title("Top 10 nguyên nhân có thể gây cháy", fontsize=16, fontweight="bold")
ax2.set_xlabel("Số vụ cháy")
ax2.set_ylabel("Nguyên nhân")

all_html += fig_to_html(fig2, "02_top_possible_causes.png", "Biểu đồ Possible Cause")



# 3. BOXPLOT: PHÂN BỐ THIỆT HẠI THEO QUY MÔ DƯỚI 100K

plot_df = df[
    df["Extent_Of_Fire"].notna() &
    df["Estimated_Dollar_Loss"].notna() &
    (df["Estimated_Dollar_Loss"] >= 0) &
    (df["Estimated_Dollar_Loss"] <= 100000)
].copy()

# Lấy danh sách các nhóm đã sắp xếp theo trung vị thiệt hại
extent_order = (
    plot_df.groupby("Extent_Of_Fire")["Estimated_Dollar_Loss"]
    .median()
    .sort_values()
    .index
)

# Chia đôi danh sách thành 2 nửa
mid_idx = len(extent_order) // 2 + len(extent_order) % 2  # Ưu tiên phần 1 nhiều hơn 1 xíu nếu là số lẻ
order_part1 = extent_order[:mid_idx]
order_part2 = extent_order[mid_idx:]

# === ẢNH 1: NỬA ĐẦU CỦA DANH SÁCH ===
plot_df_1 = plot_df[plot_df["Extent_Of_Fire"].isin(order_part1)]

fig3a, ax3a = plt.subplots(figsize=(14, 6))
sns.boxplot(
    data=plot_df_1, x="Extent_Of_Fire", y="Estimated_Dollar_Loss",
    order=order_part1, palette="Set2", linewidth=1.2, fliersize=3, ax=ax3a
)
ax3a.set_title("Boxplot: Phân bố thiệt hại theo Quy mô đám cháy (Dưới 100K CAD) - Nhóm 1", fontsize=14, fontweight="bold")
ax3a.set_xlabel("Quy mô đám cháy")
ax3a.set_ylabel("Ước tính thiệt hại (CAD)")
ax3a.set_xticklabels(ax3a.get_xticklabels(), rotation=35, ha="right")
ax3a.grid(axis="y", alpha=0.35)

all_html += fig_to_html(fig3a, "03a_boxplot_loss_by_extent_part1.png", "Biểu đồ Extent of Fire (Nhóm 1)", max_width="950px")

# === ẢNH 2: NỬA SAU CỦA DANH SÁCH ===
plot_df_2 = plot_df[plot_df["Extent_Of_Fire"].isin(order_part2)]

fig3b, ax3b = plt.subplots(figsize=(14, 6))
sns.boxplot(
    data=plot_df_2, x="Extent_Of_Fire", y="Estimated_Dollar_Loss",
    order=order_part2, palette="Set2", linewidth=1.2, fliersize=3, ax=ax3b
)
ax3b.set_title("Boxplot: Phân bố thiệt hại theo Quy mô đám cháy (Dưới 100K CAD) - Nhóm 2", fontsize=14, fontweight="bold")
ax3b.set_xlabel("Quy mô đám cháy")
ax3b.set_ylabel("Ước tính thiệt hại (CAD)")
ax3b.set_xticklabels(ax3b.get_xticklabels(), rotation=35, ha="right")
ax3b.grid(axis="y", alpha=0.35)

all_html += fig_to_html(fig3b, "03b_boxplot_loss_by_extent_part2.png", "Biểu đồ Extent of Fire (Nhóm 2)", max_width="950px")


# 4. TÌNH TRẠNG KHI ĐẾN NƠI (SỐ VỤ & THIỆT HẠI)

status_df = df[
    df["Status_of_Fire_On_Arrival"].notna() &
    df["Estimated_Dollar_Loss"].notna() &
    (df["Estimated_Dollar_Loss"] >= 0)
].copy()

status_stats = (
    status_df.groupby("Status_of_Fire_On_Arrival")
    .agg(count=("Estimated_Dollar_Loss", "size"), mean_loss=("Estimated_Dollar_Loss", "mean"))
    .sort_values("count", ascending=False)
)
top_status = status_stats.head(8).copy()
count_plot = top_status.sort_values("count", ascending=True)
loss_plot = top_status.sort_values("mean_loss", ascending=True)

fig4, axes = plt.subplots(1, 2, figsize=(16, 6))

# Biểu đồ trái: Số vụ
axes[0].barh(count_plot.index, count_plot["count"], color=BLUE, edgecolor=GRAY)
axes[0].set_title("Số vụ cháy theo tình trạng khi TFS đến nơi", fontweight="bold")
axes[0].set_xlabel("Số vụ")
axes[0].set_ylabel("Status of Fire On Arrival")
axes[0].grid(axis="x", alpha=0.35)
for i, value in enumerate(count_plot["count"]):
    axes[0].text(value, i, f" {value:,.0f}", va="center", fontsize=9)

# Biểu đồ phải: Thiệt hại trung bình
axes[1].barh(loss_plot.index, loss_plot["mean_loss"], color=SALMON, edgecolor=GRAY)
axes[1].set_title("Thiệt hại trung bình theo tình trạng khi TFS đến nơi", fontweight="bold")
axes[1].set_xlabel("Average Estimated Dollar Loss (CAD)")
axes[1].set_ylabel("")
axes[1].grid(axis="x", alpha=0.35)
for i, value in enumerate(loss_plot["mean_loss"]):
    axes[1].text(value, i, f" {value:,.0f}", va="center", fontsize=9)

all_html += fig_to_html(fig4, "04_status_on_arrival_count_and_loss.png", "Biểu đồ Status of Fire on Arrival", max_width="1100px")


print("%html <div style='font-family:DejaVu Sans; max-width: 1200px; margin: auto;'>" + all_html + "</div>")

# %% Cell 10
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 12, "axes.labelsize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8, "figure.dpi": 120})
BLUE, ORANGE = "#2F7FB1", "#D9992F"

extent_order = [
    'Confined to object of origin',
    'Confined to part of room/area of origin',
    'Spread to entire room of origin',
    'Spread beyond room of origin, same floor',
    'Spread to other floors, confined to building',
    'Entire Structure'
]
short_labels = ['Vật thể gốc', 'Phần phòng', 'Toàn phòng', 'Ngoài phòng\ncùng tầng', 'Nhiều tầng', 'Toàn bộ\ncông trình']

df_box = df[df['Extent_Of_Fire'].isin(extent_order)].copy()
ext_map = dict(zip(extent_order, short_labels))
df_box['Extent_Short'] = df_box['Extent_Of_Fire'].map(ext_map)

all_html = ""

#  Zoom in <= 100K 
df_zoom = df_box[df_box['Estimated_Dollar_Loss'] <= 100000]
fig1, ax1 = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df_zoom, x="Extent_Short", y="Estimated_Dollar_Loss", order=short_labels, color=BLUE, linewidth=1.2, fliersize=2, ax=ax1)
ax1.set_title("Boxplot: Thiệt hại theo mức lan rộng (≤ 100.000 CAD)", fontweight="bold")
ax1.set_xlabel("Mức độ lan rộng")
ax1.set_ylabel("Estimated Dollar Loss (CAD)")
ax1.grid(axis="y", alpha=0.35)
plt.tight_layout()

buf1 = io.BytesIO(); fig1.savefig(buf1, format='png', dpi=120, bbox_inches='tight'); buf1.seek(0)
b64_1 = base64.b64encode(buf1.read()).decode()
plt.close(fig1)

all_html += f"""
<div style='text-align:center;margin:15px'>
  <img src='data:image/png;base64,{b64_1}' style='max-width:800px'><br>
  <a href='data:image/png;base64,{b64_1}' download='boxplot_extent_zoom100k.png' style='color:#2196F3;font-size:12px'>Tải: Boxplot Extent (≤100K)</a>
</div><hr>"""

# 2. Log Scale 
df_log = df_box[df_box['Estimated_Dollar_Loss'] > 0]
fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df_log, x="Extent_Short", y="Estimated_Dollar_Loss", order=short_labels, color=ORANGE, linewidth=1.2, fliersize=2, ax=ax2)
ax2.set_yscale('log')
ax2.set_title("Boxplot: Thiệt hại theo mức lan rộng (Log-scale - Toàn bộ dữ liệu)", fontweight="bold")
ax2.set_xlabel("Mức độ lan rộng")
ax2.set_ylabel("Estimated Dollar Loss (Log CAD)")
ax2.grid(axis="y", alpha=0.35)
plt.tight_layout()

buf2 = io.BytesIO(); fig2.savefig(buf2, format='png', dpi=120, bbox_inches='tight'); buf2.seek(0)
b64_2 = base64.b64encode(buf2.read()).decode()
plt.close(fig2)

all_html += f"""
<div style='text-align:center;margin:15px'>
  <img src='data:image/png;base64,{b64_2}' style='max-width:800px'><br>
  <a href='data:image/png;base64,{b64_2}' download='boxplot_extent_log.png' style='color:#2196F3;font-size:12px'>Tải: Boxplot Extent (Log)</a>
</div>"""

print("%html " + all_html)

# %% Cell 11
# Tạo SEVERITY từ Estimated_Dollar_Loss (Q1/Q3 chỉ tính trên loss > 0)
loss_positive = df[df['Estimated_Dollar_Loss'] > 0]['Estimated_Dollar_Loss']
Q1 = loss_positive.quantile(0.25)
Q3 = loss_positive.quantile(0.75)
print(f"Q1 (loss > 0) = {Q1:,.0f} CAD")
print(f"Q3 (loss > 0) = {Q3:,.0f} CAD")

def assign_severity(loss):
    if pd.isna(loss) or loss == 0:
        return 'No Loss'
    elif loss <= Q1:
        return 'Minor'
    elif loss <= Q3:
        return 'Moderate'
    else:
        return 'Major'

df['SEVERITY'] = df['Estimated_Dollar_Loss'].apply(assign_severity)
SEV_ORDER = ['No Loss', 'Minor', 'Moderate', 'Major']

print(f"\nPhân phối SEVERITY:")
print(df['SEVERITY'].value_counts().reindex(SEV_ORDER))
print(f"\nTỷ lệ %:")
print((df['SEVERITY'].value_counts(normalize=True) * 100).reindex(SEV_ORDER).round(2))

# %% Cell 12
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df_ml = df.copy()

feature_cols_cat = ['Possible_Cause', 'Property_Use', 'Area_of_Origin',
                    'Extent_Of_Fire', 'Status_of_Fire_On_Arrival']
feature_cols_num = ['Response_Time_Min']

# Fill missing
for c in feature_cols_cat:
    df_ml[c] = df_ml[c].astype(str).replace('nan', 'Not Reported')

for c in feature_cols_num:
    median_val = df_ml[c].median()
    df_ml[c] = df_ml[c].fillna(median_val)
    print(f"{c}: median = {median_val:.2f}")

# Encode categorical
encoders = {}
for c in feature_cols_cat:
    le = LabelEncoder()
    df_ml[c + '_ENC'] = le.fit_transform(df_ml[c])
    encoders[c] = le
    print(f"  {c}: {len(le.classes_)} categories")

encoded_cols = [c + '_ENC' for c in feature_cols_cat]
all_feature_cols = encoded_cols + feature_cols_num

# Target
df_ml['LOG_TOTAL_LOSS'] = np.log1p(df_ml['Estimated_Dollar_Loss'])

# Train/test split
X = df_ml[all_feature_cols]
y_reg = df_ml['LOG_TOTAL_LOSS']
y_clf = df_ml['SEVERITY']

X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf,
    test_size=0.2, random_state=42, stratify=y_clf
)
print(f"\nTrain: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
print(f"\nPhân phối SEVERITY train:")
print(y_clf_train.value_counts().reindex(SEV_ORDER))

# %% Cell 13
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_recall_fscore_support,
                             classification_report, confusion_matrix)

# Train models
lr_clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr_clf.fit(X_train, y_clf_train)

rf_clf = RandomForestClassifier(n_estimators=100, max_depth=15,
                                class_weight='balanced',
                                random_state=42, n_jobs=-1)
rf_clf.fit(X_train, y_clf_train)
print("Đã train xong 2 mô hình classification:")
print("  1. Logistic Regression")
print("  2. Random Forest Classifier")

# Predict
pred_clf = {
    'Logistic Regression': lr_clf.predict(X_test),
    'Random Forest':       rf_clf.predict(X_test),
}

# Metrics
rows = []
for model_name, y_pred in pred_clf.items():
    acc     = accuracy_score(y_clf_test, y_pred)
    bal_acc = balanced_accuracy_score(y_clf_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_clf_test, y_pred, average='macro')
    rows.append({
        'Model': model_name,
        'Accuracy': round(acc, 4),
        'Balanced Accuracy': round(bal_acc, 4),
        'Precision (macro)': round(prec, 4),
        'Recall (macro)': round(rec, 4),
        'F1 (macro)': round(f1, 4),
    })

metrics_clf_df = pd.DataFrame(rows)
print("\nSo sánh metrics:")
print(metrics_clf_df.to_string(index=False))

for model_name, y_pred in pred_clf.items():
    print(f"\nClassification Report — {model_name}")
    print(classification_report(y_clf_test, y_pred, labels=SEV_ORDER, digits=3))

# %% Cell 14
import io, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

all_html = ""

# Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, (name, y_pred) in zip(axes, pred_clf.items()):
    cm = confusion_matrix(y_clf_test, y_pred, labels=SEV_ORDER)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=SEV_ORDER, yticklabels=SEV_ORDER,
                ax=ax, cbar_kws={'label': '% theo hàng (Actual)'})
    ax.set_title(f'Confusion Matrix — {name}\n(% theo Actual class)',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()

buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=120, bbox_inches='tight'); buf.seek(0)
b64 = base64.b64encode(buf.read()).decode()
plt.close(fig)
all_html += f"""
<div style='text-align:center;margin:15px'>
  <img src='data:image/png;base64,{b64}' style='max-width:900px'><br>
  <a href='data:image/png;base64,{b64}' download='confusion_matrix.png' style='color:#2196F3;font-size:12px'>Tải: Confusion Matrix</a>
</div><hr>"""

# Feature Importance
importances = pd.Series(rf_clf.feature_importances_, index=all_feature_cols).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(importances.index, importances.values, color='#9b59b6')
ax.set_title('Feature Importance — Random Forest Classifier (SEVERITY)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Importance')
for i, v in enumerate(importances.values):
    ax.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=9)
plt.tight_layout()

buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=120, bbox_inches='tight'); buf.seek(0)
b64 = base64.b64encode(buf.read()).decode()
plt.close(fig)
all_html += f"""
<div style='text-align:center;margin:15px'>
  <img src='data:image/png;base64,{b64}' style='max-width:800px'><br>
  <a href='data:image/png;base64,{b64}' download='feature_importance_clf.png' style='color:#2196F3;font-size:12px'>Tải: Feature Importance Classifier</a>
</div>"""

print("%html <div style='font-family:Arial;max-width:1100px;margin:auto'>" + all_html + "</div>")

# %% Cell 15
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Chỉ train trên rows có loss > 0
mask_loss = df_ml['Estimated_Dollar_Loss'] > 0
df_loss = df_ml[mask_loss]
print(f"Rows dùng train regression: {len(df_loss):,} ({mask_loss.mean()*100:.1f}%)")

X_loss = df_loss[all_feature_cols]
y_loss = df_loss['LOG_TOTAL_LOSS']

X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
    X_loss, y_loss, test_size=0.2, random_state=42
)

lr_reg = LinearRegression()
lr_reg.fit(X_tr_r, y_tr_r)

rf_reg = RandomForestRegressor(n_estimators=100, max_depth=15,
                               random_state=42, n_jobs=-1)
rf_reg.fit(X_tr_r, y_tr_r)
print("Đã train xong 2 mô hình regression:")
print("  1. Linear Regression")
print("  2. Random Forest Regressor")

# Metrics
pred_reg = {
    'Linear Regression': {'train': lr_reg.predict(X_tr_r), 'test': lr_reg.predict(X_te_r)},
    'Random Forest':     {'train': rf_reg.predict(X_tr_r), 'test': rf_reg.predict(X_te_r)},
}

rows = []
for model_name, p in pred_reg.items():
    for split, y_true_log, y_pred_log in [
        ('Train', y_tr_r, p['train']),
        ('Test',  y_te_r, p['test']),
    ]:
        mae_log  = mean_absolute_error(y_true_log, y_pred_log)
        rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
        r2_log   = r2_score(y_true_log, y_pred_log)
        y_true_usd = np.expm1(y_true_log)
        y_pred_usd = np.expm1(np.clip(y_pred_log, 0, None))
        mae_usd  = mean_absolute_error(y_true_usd, y_pred_usd)
        rmse_usd = np.sqrt(mean_squared_error(y_true_usd, y_pred_usd))
        r2_usd   = r2_score(y_true_usd, y_pred_usd)
        rows.append({
            'Model': model_name, 'Split': split,
            'MAE (log)': round(mae_log, 4), 'RMSE (log)': round(rmse_log, 4), 'R² (log)': round(r2_log, 4),
            'MAE ($)': round(mae_usd, 0), 'RMSE ($)': round(rmse_usd, 0), 'R² ($)': round(r2_usd, 4),
        })

metrics_df = pd.DataFrame(rows)
print("\nSo sánh metrics Regression:")
print("%table\n" + metrics_df.to_csv(index=False, sep="\t"))

# %% Cell 16
all_html = ""

# Scatter Actual vs Predicted
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, (model_name, p) in zip(axes, pred_reg.items()):
    ax.scatter(y_te_r, p['test'], s=5, alpha=0.1, color='#3498DB')
    lims = [0, y_te_r.max()]
    ax.plot(lims, lims, 'r--', lw=1.5, label='Đường lý tưởng (y=x)')
    ax.set_title(f'{model_name}\nActual vs Predicted (test set)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Actual log(Estimated_Dollar_Loss+1)')
    ax.set_ylabel('Predicted log(Estimated_Dollar_Loss+1)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=120, bbox_inches='tight'); buf.seek(0)
b64 = base64.b64encode(buf.read()).decode()
plt.close(fig)
all_html += f"""
<div style='text-align:center;margin:15px'>
  <img src='data:image/png;base64,{b64}' style='max-width:900px'><br>
  <a href='data:image/png;base64,{b64}' download='actual_vs_predicted.png' style='color:#2196F3;font-size:12px'>Tải: Actual vs Predicted</a>
</div><hr>"""

# Feature Importance Regressor
importances = pd.Series(rf_reg.feature_importances_, index=all_feature_cols).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(importances.index, importances.values, color='#27AE60', edgecolor='white')
ax.set_title('Feature Importance — Random Forest Regressor (Estimated_Dollar_Loss)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Importance')
for bar, v in zip(bars, importances.values):
    ax.text(v + 0.002, bar.get_y() + bar.get_height()/2, f'{v:.3f}', va='center', fontsize=9)
plt.tight_layout()

buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=120, bbox_inches='tight'); buf.seek(0)
b64 = base64.b64encode(buf.read()).decode()
plt.close(fig)
all_html += f"""
<div style='text-align:center;margin:15px'>
  <img src='data:image/png;base64,{b64}' style='max-width:800px'><br>
  <a href='data:image/png;base64,{b64}' download='feature_importance_reg.png' style='color:#2196F3;font-size:12px'>Tải: Feature Importance Regressor</a>
</div>"""

print("%html <div style='font-family:Arial;max-width:1100px;margin:auto'>" + all_html + "</div>")

# %% Cell 17
import joblib, os, json

MODEL_DIR = '/tmp/toronto_models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Tạo code_lookup (dùng cho UI predict)
code_lookup = {}
for c in feature_cols_cat:
    le = encoders[c]
    code_lookup[c] = {str(label): int(idx) for idx, label in enumerate(le.classes_)}

# Tạo text_options (danh sách dropdown cho UI)
text_options = {}
for c in feature_cols_cat:
    text_options[c] = list(encoders[c].classes_)

joblib.dump(lr_reg,          os.path.join(MODEL_DIR, 'lr_reg.pkl'))
joblib.dump(rf_reg,          os.path.join(MODEL_DIR, 'rf_reg.pkl'))
joblib.dump(lr_clf,          os.path.join(MODEL_DIR, 'lr_clf.pkl'))
joblib.dump(rf_clf,          os.path.join(MODEL_DIR, 'rf_clf.pkl'))
joblib.dump(encoders,        os.path.join(MODEL_DIR, 'encoders.pkl'))
joblib.dump(code_lookup,     os.path.join(MODEL_DIR, 'code_lookup.pkl'))
joblib.dump(all_feature_cols,os.path.join(MODEL_DIR, 'all_feature_cols.pkl'))
joblib.dump(text_options,    os.path.join(MODEL_DIR, 'text_options.pkl'))

print("Đã lưu model vào:", MODEL_DIR)
print(os.listdir(MODEL_DIR))

# %% Cell 18
import joblib, json
import pandas as pd
import numpy as np

MODEL_DIR = '/tmp/toronto_models'
lr_reg          = joblib.load(os.path.join(MODEL_DIR, 'lr_reg.pkl'))
rf_reg          = joblib.load(os.path.join(MODEL_DIR, 'rf_reg.pkl'))
lr_clf          = joblib.load(os.path.join(MODEL_DIR, 'lr_clf.pkl'))
rf_clf          = joblib.load(os.path.join(MODEL_DIR, 'rf_clf.pkl'))
encoders        = joblib.load(os.path.join(MODEL_DIR, 'encoders.pkl'))
code_lookup     = joblib.load(os.path.join(MODEL_DIR, 'code_lookup.pkl'))
all_feature_cols= joblib.load(os.path.join(MODEL_DIR, 'all_feature_cols.pkl'))
text_options    = joblib.load(os.path.join(MODEL_DIR, 'text_options.pkl'))

feature_cols_cat = list(text_options.keys())
SEV_ORDER = ['No Loss', 'Minor', 'Moderate', 'Major']

default_inputs = {
    c: ('Not Reported' if 'Not Reported' in text_options[c] else text_options[c][0])
    for c in feature_cols_cat
}
default_inputs['Response_Time_Min'] = 5.0

z.z.angularBind("feature_cols_js", feature_cols_cat)
z.z.angularBind("text_options_js",  {c: text_options[c] for c in feature_cols_cat})
z.z.angularBind("selected_inputs",  default_inputs)
z.z.angularBind("predict_result",   "Chưa có kết quả")
z.z.angularBind("selected_json",    json.dumps(default_inputs))

print("Load model xong!")
print("Features:", all_feature_cols)

# %% Cell 19
selected_raw = z.z.angular("selected_json")
selected = json.loads(selected_raw)

X_new_dict = {
    c + '_ENC': code_lookup[c][str(selected[c]).strip()]
    for c in feature_cols_cat
}
X_new_dict['Response_Time_Min'] = float(selected['Response_Time_Min'])
X_new = pd.DataFrame([X_new_dict])[all_feature_cols]

# Classification
pred_clf_lr = lr_clf.predict(X_new)[0]
pred_clf_rf = rf_clf.predict(X_new)[0]
proba_lr = dict(zip(lr_clf.classes_, lr_clf.predict_proba(X_new)[0].round(3)))
proba_rf = dict(zip(rf_clf.classes_, rf_clf.predict_proba(X_new)[0].round(3)))

# Regression (two-stage)
pred_log_lr = lr_reg.predict(X_new)[0]
pred_log_rf = rf_reg.predict(X_new)[0]
pred_usd_lr = 0 if pred_clf_lr == 'No Loss' else np.expm1(max(pred_log_lr, 0))
pred_usd_rf = 0 if pred_clf_rf == 'No Loss' else np.expm1(max(pred_log_rf, 0))

def fmt_proba(d):
    icons = {'Major': '🔴', 'Moderate': '🟠', 'Minor': '🟡', 'No Loss': '🟢'}
    lines = []
    for label, prob in sorted(d.items(), key=lambda x: -x[1]):
        bar = '█' * int(prob * 20)
        pct = f"{prob*100:.1f}%"
        icon = icons.get(label, '⚪')
        lines.append(f"  {icon} {label:<10} {pct:>6}  [{bar:<20}]")
    return '\n'.join(lines)

result = f"""
KẾT QUẢ DỰ ĐOÁN (Toronto Fire Services)
{'─' * 45}
⚠️  MỨC ĐỘ NGHIÊM TRỌNG (SEVERITY)
  • Logistic Regression : {pred_clf_lr}
  • Random Forest       : {pred_clf_rf}

📊 XÁC SUẤT - Logistic Regression
{fmt_proba(proba_lr)}

📊 XÁC SUẤT - Random Forest
{fmt_proba(proba_rf)}

💰 THIỆT HẠI ƯỚC TÍNH (Estimated Dollar Loss)
  • Logistic Regression : ${pred_usd_lr:>12,.0f} CAD
  • Random Forest       : ${pred_usd_rf:>12,.0f} CAD
  ⚠️ No Loss → mặc định $0
""".strip()

z.z.angularBind("predict_result", result)
print(result)
