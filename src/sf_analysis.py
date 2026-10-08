# Code cells from notebooks/SF_Notebook.json, San Francisco (Huỳnh Thanh Nhân).
# Copied cell by cell; only the "%pyspark" line at the top of each cell is removed.
# Markdown, %sh and %angular cells are left out. Some cells use Zeppelin's z object.

# %% Cell 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy import stats

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)

# %% Cell 4
import gdown

file_id = "1AcFqbPjLyecpVCrxV_QrN7faJg6c9pZq"
url = f"https://drive.google.com/uc?id={file_id}"

gdown.download(
    url,
    "Fire_Incidents_20260509.csv",
    quiet=False
)

# %% Cell 5
df = pd.read_csv(
    "Fire_Incidents_20260509.csv"
)

print("Shape:", df.shape)
print("-" * 30)

print("%table " + df.head().to_csv(index=False, sep='\t'))

# %% Cell 7
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

unique_info = pd.DataFrame({
    'Column': df.columns,
    'Data Type': df.dtypes.astype(str),
    'Unique Values': df.nunique(dropna=False),
    'Missing Values': df.isnull().sum(),
    'Missing (%)': round(df.isnull().mean()*100, 2)
})

unique_info = unique_info.sort_values(by='Unique Values', ascending=False)
print("%table " + unique_info.to_csv(index=False, sep='\t'))

# %% Cell 9
top_missing = (
    unique_info
    .sort_values('Missing (%)', ascending=False)
    .head(15)
)

plt.figure(figsize=(12,7))

ax = sns.barplot(
    data=top_missing,
    x='Missing (%)',
    y='Column'
)

for i, value in enumerate(top_missing['Missing (%)']):
    ax.text(
        value + 0.2,
        i,
        f'{value:.2f}%',
        va='center'
    )

plt.title(
    'Top 15 Fields with Highest Missing Values'
)

plt.tight_layout()
plt.show()

# %% Cell 11
df['Situation_Code'] = (
    df['Primary Situation']
    .astype(str)
    .str.extract(r'^(\d+)')
)

df['Situation_Desc'] = (
    df['Primary Situation']
    .astype(str)
    .str.replace(
        r'^\d+\s*-?\s*',
        '',
        regex=True
    )
)

print("%table " + df[['Primary Situation', 'Situation_Code', 'Situation_Desc']].head(20).to_csv(index=False, sep='\t'))

# %% Cell 13
top_situations = (
    df[['Situation_Code','Situation_Desc']]
    .value_counts()
    .reset_index(name='Count')
    .head(10)
    .sort_values('Count')
)

top_situations['Label'] = (
    top_situations['Situation_Code']
    + ' - '
    + top_situations['Situation_Desc']
)

# %% Cell 15
plt.figure(figsize=(14,8))

ax = sns.barplot(
    data=top_situations,
    x='Count',
    y='Label'
)

for i, v in enumerate(top_situations['Count']):
    ax.text(
        v,
        i,
        f' {v:,}',
        va='center'
    )

plt.tight_layout()
plt.show()

# %% Cell 17
df['Situation_Code'] = pd.to_numeric(
    df['Situation_Code'],
    errors='coerce'
)

fire_codes = [
    111,112,113,114,
    115,116,117,118,
    120,121,122,123
]

df_fire = df[
    df['Situation_Code'].isin(fire_codes)
].copy()

print(df.shape)
print(df_fire.shape)

# %% Cell 19
plt.figure(figsize=(10, 6))

ax = sns.countplot(
    data=df_fire,
    y='Situation_Code',
    order=sorted(df_fire['Situation_Code'].unique())
)

# Tự động thêm số liệu vào đầu mỗi cột
ax.bar_label(ax.containers[0], padding=2)

current_max = ax.get_xlim()[1]
ax.set_xlim(0, current_max * 1.15)

plt.tight_layout()
plt.show()

# %% Cell 21
# 1. Tính số lượng và tỷ lệ phần trăm missing
missing = df_fire.isna().sum().to_frame('missing_count')
missing['missing_pct'] = round((missing['missing_count'] / len(df_fire)) * 100, 2)

# 2. Sắp xếp giảm dần theo tỷ lệ missing
missing = missing.sort_values('missing_pct', ascending=False)

# 3. Đưa tên cột từ Index ra thành một cột dữ liệu chuẩn để hiện lên bảng
missing = missing.reset_index().rename(columns={'index': 'Column'})

# 4. In dạng %table của Zeppelin
print("%table " + missing.to_csv(index=False, sep='\t'))

# %% Cell 22
missing = missing.set_index('Column')
df_plot = missing[missing['missing_count'] > 0]

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(14,10),
    sharex=True
)

# Missing Count
df_plot['missing_count'].plot(
    kind='bar',
    ax=ax1,
    color='royalblue'
)

ax1.set_ylabel("Count Missing")
ax1.set_title("Missing Values Count")
ax1.grid(
    axis='y',
    linestyle='--',
    alpha=0.7
)

# Missing Percentage
df_plot['missing_pct'].plot(
    kind='bar',
    ax=ax2,
    color='darkorange'
)

ax2.set_ylabel("% Missing")
ax2.set_title("Missing Values Percentage")
ax2.grid(
    axis='y',
    linestyle='--',
    alpha=0.7
)

plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

# %% Cell 24
df_fire['Estimated Property Loss'].dropna().unique()[:1000]

# %% Cell 25
loss = (
    df_fire['Estimated Property Loss']
    .str.replace(',', '', regex=False)
    .astype(float)
)

(loss < 0).sum()

# %% Cell 27
print("%table " + loss[loss < 0].value_counts().to_csv(sep='\t'))

# %% Cell 29
df_fire['Estimated Property Loss'] = pd.to_numeric(
    df_fire['Estimated Property Loss']
        .astype(str)
        .str.replace(',', ''),
    errors='coerce'
)

df_fire.loc[
    df_fire['Estimated Property Loss'] < 0,
    'Estimated Property Loss'
] = np.nan

# %% Cell 30
(df_fire['Estimated Property Loss'] < 0).sum()

# %% Cell 32
df_fire['Estimated Contents Loss'] = (
    df_fire['Estimated Contents Loss']
    .astype(str)
    .str.replace(',', '', regex=False)
)

df_fire['Estimated Contents Loss'] = pd.to_numeric(
    df_fire['Estimated Contents Loss'],
    errors='coerce'
)

df_fire.loc[
    df_fire['Estimated Contents Loss'] < 0,
    'Estimated Contents Loss'
] = np.nan

(df_fire['Estimated Contents Loss'] < 0).sum()

# %% Cell 34
def common_stats(series, name):

    x = series.dropna()

    q1 = x.quantile(0.25)
    q2 = x.quantile(0.50)
    q3 = x.quantile(0.75)

    print("%table")
    print("Statistic\tValue")
    print(f"Count (Size)\t{x.count()}")
    print(f"Minimum\t{x.min()}")
    print(f"Maximum\t{x.max()}")
    print(f"Mean\t{x.mean()}")
    print(f"Median\t{q2}")
    print(f"Standard Deviation\t{x.std()}")
    print(f"Q1 (25th percentile)\t{q1}")
    print(f"Q2 (50th percentile)\t{q2}")
    print(f"Q3 (75th percentile)\t{q3}")
    print(f"IQR\t{q3 - q1}")

# %% Cell 36
common_stats(
    df_fire['Estimated Property Loss'],
    "Estimated Property Loss"
)

# %% Cell 37
common_stats(
    df_fire['Estimated Contents Loss'],
    "Estimated Contents Loss"
)

# %% Cell 39
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

for ax, col in zip(
    axes,
    [
        'Estimated Property Loss',
        'Estimated Contents Loss'
    ]
):

    x = np.log1p(df_fire[col].dropna())

    sns.boxplot(x=x, ax=ax)

    ax.set_title(
        f'Biểu đồ hộp - {col}\n(thang log)',
        fontweight='bold'
    )

    ax.set_xlabel(
        'log(1 + giá trị tổn thất)'
    )

plt.tight_layout()
plt.show()

# %% Cell 41
fig, axes = plt.subplots(1, 2, figsize=(14,5))

for ax, col in zip(
    axes,
    [
        'Estimated Property Loss',
        'Estimated Contents Loss'
    ]
):

    sns.histplot(
        np.log1p(
            df_fire[col].dropna()
        ),
        bins=40,
        kde=True,
        ax=ax
    )

    ax.set_title(
        f'Phân phối Tổn thất {col} (log scale)'
    )

plt.tight_layout()
plt.show()

# %% Cell 43
df_fire['Alarm DtTm'] = pd.to_datetime(
    df_fire['Alarm DtTm'],
    format='%Y/%m/%d %I:%M:%S %p',
    errors='coerce'
)

df_fire['Arrival DtTm'] = pd.to_datetime(
    df_fire['Arrival DtTm'],
    format='%Y/%m/%d %I:%M:%S %p',
    errors='coerce'
)

df_fire['Response Time Min'] = (
    df_fire['Arrival DtTm']
    - df_fire['Alarm DtTm']
).dt.total_seconds() / 60.0

df_fire.loc[
    df_fire['Response Time Min'] < 0,
    'Response Time Min'
] = np.nan

# %% Cell 44
common_stats(
    df_fire['Response Time Min'],
    "Response Time (Minutes)"
)

# %% Cell 46
rt = df_fire['Response Time Min'].dropna()

p99 = rt.quantile(0.99)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.boxplot(
    x=rt,
    ax=axes[0],
    color='steelblue'
)

axes[0].set_title(
    'Toàn bộ dữ liệu',
    fontweight='bold'
)

axes[0].set_xlabel(
    'Thời gian phản ứng (phút)'
)

sns.boxplot(
    x=rt.clip(upper=p99),
    ax=axes[1],
    color='cornflowerblue'
)

axes[1].set_title(
    f'Clip tại P99 ({p99:.1f} phút)',
    fontweight='bold'
)

axes[1].set_xlabel(
    'Thời gian phản ứng (phút)'
)

fig.suptitle(
    'Response Time Distribution',
    fontsize=14,
    fontweight='bold'
)

plt.tight_layout()
plt.show()

# %% Cell 47
fig, ax = plt.subplots(figsize=(8, 4))

sns.histplot(
    rt,
    bins=60,
    kde=True,
    log_scale=(True, False),
    color='steelblue',
    ax=ax
)

ax.set_title(
    'Phân phối Response Time (log scale)',
    fontweight='bold'
)

ax.set_xlabel(
    'Thời gian phản ứng (phút, log scale)'
)

plt.tight_layout()
plt.show()

# %% Cell 49
df_fire['Total Casualties'] = (
    df_fire['Civilian Fatalities']
    + df_fire['Civilian Injuries']
    + df_fire['Fire Fatalities']
    + df_fire['Fire Injuries']
)

# %% Cell 50
common_stats(df_fire['Total Casualties'], 'Total Casualties')

# %% Cell 52
casualty_dist = (
    df_fire['Total Casualties']
    .value_counts()
    .sort_index()
)

fig, ax = plt.subplots(figsize=(10, 5))

sns.barplot(
    x=casualty_dist.index,
    y=casualty_dist.values,
    ax=ax,
    color='steelblue'
)

for i, v in enumerate(casualty_dist.values):

    ax.text(
        i,
        v + 200,
        f'{v:,}',
        ha='center',
        va='bottom',
        fontsize=8
    )

ax.set_title(
    'Phân phối Total Casualties (toàn bộ giá trị)',
    fontweight='bold'
)

ax.set_xlabel(
    'Number of Casualties'
)

ax.set_ylabel(
    'Number of Incidents'
)

plt.tight_layout()
plt.show()

# %% Cell 53
casualty_nonzero = df_fire.loc[
    df_fire['Total Casualties'] > 0,
    'Total Casualties'
]

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 4)
)

sns.boxplot(
    x=df_fire['Total Casualties'],
    ax=axes[0],
    color='lightcoral'
)

axes[0].set_title(
    'Toàn bộ dữ liệu',
    fontweight='bold'
)

axes[0].set_xlabel(
    'Number of Casualties'
)

sns.boxplot(
    x=casualty_nonzero,
    ax=axes[1],
    color='salmon'
)

axes[1].set_title(
    f'Chỉ non-zero (n={len(casualty_nonzero):,})',
    fontweight='bold'
)

axes[1].set_xlabel(
    'Number of Casualties'
)

fig.suptitle(
    'Boxplot Total Casualties',
    fontsize=14,
    fontweight='bold'
)

plt.tight_layout()

plt.show()

# %% Cell 55
def moment_stats(series, name):

    x = series.dropna()

    mean = x.mean()
    variance = x.var()

    skewness = stats.skew(x)

    kurtosis = stats.kurtosis(x)

    hyperskew = (
        stats.moment(x, moment=5)
        / (x.std() ** 5)
    )

    hypertail = (
        stats.moment(x, moment=6)
        / (x.std() ** 6)
    )

    return {
        'Variable': name,
        'Mean': mean,
        'Variance': variance,
        'Skewness': skewness,
        'Kurtosis': kurtosis,
        'Hyperskewness': hyperskew,
        'Hypertailedness': hypertail
    }

# %% Cell 56
moment_df = pd.DataFrame([

    moment_stats(
        df_fire['Estimated Property Loss'],
        'Estimated Property Loss'
    ),

    moment_stats(
        df_fire['Estimated Contents Loss'],
        'Estimated Contents Loss'
    ),

    moment_stats(
        df_fire['Total Casualties'],
        'Total Casualties'
    ),

    moment_stats(
        df_fire['Response Time Min'],
        'Response Time Min'
    )

])

moment_df

# %% Cell 57
print(
    "%table " +
    moment_df.to_csv(
        index=False,
        sep="\t"
    )
)

# %% Cell 59
metrics = pd.DataFrame({

    'Biến': [
        'Estimated Property Loss',
        'Estimated Contents Loss',
        'Total Casualties',
        'Response Time Min'
    ],

    'Skewness': [
        stats.skew(df_fire['Estimated Property Loss'].dropna()),
        stats.skew(df_fire['Estimated Contents Loss'].dropna()),
        stats.skew(df_fire['Total Casualties'].dropna()),
        stats.skew(df_fire['Response Time Min'].dropna())
    ],

    'Kurtosis': [
        stats.kurtosis(df_fire['Estimated Property Loss'].dropna()),
        stats.kurtosis(df_fire['Estimated Contents Loss'].dropna()),
        stats.kurtosis(df_fire['Total Casualties'].dropna()),
        stats.kurtosis(df_fire['Response Time Min'].dropna())
    ]
})

print(
    "%table " +
    metrics.to_csv(
        index=False,
        sep="\t"
    )
)

# %% Cell 60
fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)

sns.barplot(
    data=metrics,
    x='Biến',
    y='Skewness',
    ax=axes[0],
    color='cornflowerblue'
)

axes[0].set_title(
    'Độ lệch (Skewness)',
    fontweight='bold'
)

axes[0].tick_params(
    axis='x',
    rotation=20
)

axes[0].axhline(
    0,
    color='black',
    lw=0.8
)

sns.barplot(
    data=metrics,
    x='Biến',
    y='Kurtosis',
    ax=axes[1],
    color='salmon'
)

axes[1].set_title(
    'Độ nhọn vượt mức (Excess Kurtosis)',
    fontweight='bold'
)

axes[1].tick_params(
    axis='x',
    rotation=20
)

axes[1].axhline(
    0,
    color='black',
    lw=0.8
)

plt.tight_layout()

plt.show()

# %% Cell 64
n_fire = len(df_fire)
n_total = len(df)

snapshot = pd.DataFrame({
    'Chỉ số': [
        'Tổng sự cố (full dataset)',
        'Sự cố hỏa hoạn (df_fire)',
        'Tỷ lệ hỏa hoạn / tổng sự cố',
        'Có ghi Property Loss',
        'Có ghi Contents Loss',
        'Có Response Time hợp lệ',
        'Sự cố có thương vong (> 0)',
        'Median Property Loss ($)',
        'Median Contents Loss ($)',
        'Median Response Time (phút)',
    ],
    'Giá trị': [
        f'{n_total:,}',
        f'{n_fire:,}',
        f'{n_fire / n_total * 100:.2f}%',
        f"{df_fire['Estimated Property Loss'].notna().sum():,} ({df_fire['Estimated Property Loss'].notna().mean()*100:.1f}%)",
        f"{df_fire['Estimated Contents Loss'].notna().sum():,} ({df_fire['Estimated Contents Loss'].notna().mean()*100:.1f}%)",
        f"{df_fire['Response Time Min'].notna().sum():,} ({df_fire['Response Time Min'].notna().mean()*100:.1f}%)",
        f"{(df_fire['Total Casualties'] > 0).sum():,} ({(df_fire['Total Casualties'] > 0).mean()*100:.2f}%)",
        f"${df_fire['Estimated Property Loss'].median():,.0f}",
        f"${df_fire['Estimated Contents Loss'].median():,.0f}",
        f"{df_fire['Response Time Min'].median():.1f}",
    ],
})

print("%table " + snapshot.to_csv(index=False, sep="\t"))

# %% Cell 66
df_fire['Incident Date'] = pd.to_datetime(df_fire['Incident Date'], errors='coerce')

df_time = df_fire.dropna(subset=['Incident Date'])

yearly = (
    df_time.groupby(df_time['Incident Date'].dt.year)
    .size()
    .reset_index(name='Incidents')
)
yearly.columns = ['Year', 'Incidents']

monthly = (
    df_time.groupby(df_time['Incident Date'].dt.month)
    .size()
    .reindex(range(1, 13), fill_value=0)
)

df_alarm = df_fire.dropna(subset=['Alarm DtTm'])

hourly = (
    df_alarm.groupby(df_alarm['Alarm DtTm'].dt.hour)
    .size()
    .reindex(range(24), fill_value=0)
)


fig, ax = plt.subplots(figsize=(8, 4))

sns.lineplot(
    data=yearly,
    x='Year',
    y='Incidents',
    marker='o',
    color='crimson',
    ax=ax
)

ax.set_title('Sự cố hỏa hoạn theo năm', fontweight='bold')
ax.set_xlabel('Năm')
ax.set_ylabel('Số sự cố')

plt.tight_layout()
plt.show()


month_labels = [
    'Jan','Feb','Mar','Apr','May','Jun',
    'Jul','Aug','Sep','Oct','Nov','Dec'
]

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Theo tháng
sns.barplot(
    x=month_labels,
    y=monthly.values,
    ax=axes[0],
    color='darkorange'
)

axes[0].set_title('Sự cố hỏa hoạn theo tháng', fontweight='bold')
axes[0].set_xlabel('Tháng')
axes[0].set_ylabel('Số sự cố')
axes[0].tick_params(axis='x', rotation=45)

# Theo giờ
sns.barplot(
    x=list(range(24)),
    y=hourly.values,
    ax=axes[1],
    color='steelblue'
)

axes[1].set_title('Sự cố theo giờ báo động', fontweight='bold')
axes[1].set_xlabel('Giờ (0–23)')
axes[1].set_ylabel('Số sự cố')

plt.tight_layout()
plt.show()

# %% Cell 67
summary_time = pd.DataFrame({
    'Chỉ số': [
        'Năm đầu tiên',
        'Năm cuối cùng',
        'Tháng cao điểm',
        'Giờ cao điểm'
    ],
    'Giá trị': [
        yearly['Year'].min(),
        yearly['Year'].max(),
        f"{month_labels[monthly.values.argmax()]} ({monthly.max():,})",
        f"{hourly.idxmax()} ({hourly.max():,})"
    ]
})

print(
    "%table " +
    summary_time.to_csv(
        index=False,
        sep="\t"
    )
)

# %% Cell 69
top_n = 12


hood_counts = (
    df_fire['neighborhood_district']
    .value_counts()
    .head(top_n)
    .sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(10, 6))

sns.barplot(
    x=hood_counts.values,
    y=hood_counts.index,
    palette='Reds_r',
    ax=ax
)

ax.set_title(
    f'Top {top_n} khu vực theo số sự cố hỏa hoạn',
    fontweight='bold',
    fontsize=14
)
ax.set_xlabel('Số sự cố')
ax.set_ylabel('Khu vực')

ax.set_xlim(0, hood_counts.max() * 1.1)

for i, v in enumerate(hood_counts.values):
    ax.text(
        v + hood_counts.max() * 0.01,
        i,
        f'{v:,}',
        va='center',
        fontsize=9
    )

sns.despine(ax=ax)

plt.tight_layout()
plt.show()


hood_loss = (
    df_fire.groupby('neighborhood_district')['Estimated Property Loss']
    .agg(['count', 'median', 'sum'])
    .rename(columns={
        'count': 'n_with_loss',
        'median': 'median_loss',
        'sum': 'total_loss'
    })
    .query('n_with_loss >= 50')
    .sort_values('total_loss', ascending=False)
    .head(top_n)
)

fig, ax = plt.subplots(figsize=(10, 6))

sns.barplot(
    x=hood_loss['total_loss'],
    y=hood_loss.index,
    palette='Oranges_r',
    ax=ax
)

ax.set_title(
    f'Top {top_n} khu vực theo tổng thiệt hại tài sản',
    fontweight='bold',
    fontsize=14
)
ax.set_xlabel('Tổng thiệt hại tài sản (USD)')
ax.set_ylabel('Khu vực')

ax.set_xlim(0, hood_loss['total_loss'].max() * 1.1)

for i, v in enumerate(hood_loss['total_loss']):
    ax.text(
        v + hood_loss['total_loss'].max() * 0.01,
        i,
        f'${v:,.0f}',
        va='center',
        fontsize=9
    )

sns.despine(ax=ax)

plt.tight_layout()
plt.show()

# %% Cell 70
hood_loss_table = hood_loss.reset_index()

print(
    "%table " +
    hood_loss_table.to_csv(
        index=False,
        sep="\t"
    )
)

# %% Cell 72
import re
import numpy as np

top_n = 12

def extract_property_code(x):

    if pd.isna(x):
        return np.nan

    x = str(x).strip()

    # 429, 429-, 429 Multifamily...
    m = re.match(r'^(\d{3,4})', x)

    if m:

        code = m.group(1)

        # xử lý case 4191-, 0000...
        if len(code) > 3:
            code = code[:3]

        return code

    # mã đặc biệt
    m = re.match(r'^(NNN|UUU)', x)

    if m:
        return m.group(1)

    return np.nan


df_fire['property_code'] = (
    df_fire['Property Use']
    .apply(extract_property_code)
)


def build_canonical_label(group):

    labels = group.dropna().astype(str)

    if len(labels) == 0:
        return np.nan

    code = group.name

    # lấy mô tả dài nhất
    desc = max(labels, key=len)

    # bỏ phần mã phía trước
    desc = re.sub(
        r'^\d{3,4}\s*-?\s*',
        '',
        desc
    )

    return f'{code} - {desc}'


canonical_map = (
    df_fire
    .dropna(subset=['property_code'])
    .groupby('property_code')['Property Use']
    .apply(build_canonical_label)
    .to_dict()
)

df_fire['Property Use Clean'] = (
    df_fire['property_code']
    .map(canonical_map)
)

df_fire['Property Use Clean'] = (
    df_fire['Property Use Clean']
    .fillna(df_fire['Property Use'])
)


prop_counts = (
    df_fire['Property Use Clean']
    .value_counts()
    .head(top_n)
    .reset_index()
)

prop_counts.columns = [
    'Property Use',
    'Count'
]

prop_counts['Share (%)'] = (
    prop_counts['Count']
    / n_fire
    * 100
).round(2)

# %% Cell 73
fig, ax = plt.subplots(
    figsize=(10, 6)
)

plot_data = (
    prop_counts
    .sort_values(
        'Count',
        ascending=False
    )
)

sns.barplot(
    x='Count',
    y='Property Use',
    data=plot_data,
    palette='YlOrBr_r',
    ax=ax
)

ax.set_title(
    f'Top {top_n} loại cơ sở theo số sự cố',
    fontweight='bold',
    fontsize=14
)

ax.set_xlabel(
    'Số sự cố'
)

ax.set_ylabel(
    'Property Use'
)

ax.set_xlim(
    0,
    plot_data['Count'].max() * 1.1
)

for i, v in enumerate(
    plot_data['Count']
):

    ax.text(
        v + plot_data['Count'].max() * 0.01,
        i,
        f'{v:,}',
        va='center',
        fontsize=9
    )

sns.despine(ax=ax)

plt.tight_layout()

plt.show()

# %% Cell 74
prop_loss = (
    df_fire
    .groupby(
        'Property Use Clean',
        dropna=False
    )
    .agg(
        incidents=('Incident Number', 'count'),
        median_property_loss=(
            'Estimated Property Loss',
            'median'
        ),
        median_contents_loss=(
            'Estimated Contents Loss',
            'median'
        ),
        total_casualties=(
            'Total Casualties',
            'sum'
        )
    )
    .query('incidents >= 100')
)

fig, ax = plt.subplots(
    figsize=(10, 6)
)

plot_loss = (
    prop_loss
    .sort_values(
        'total_casualties',
        ascending=False
    )
    .head(top_n)
)

sns.barplot(
    x=plot_loss['total_casualties'],
    y=plot_loss.index,
    palette='Reds_r',
    ax=ax
)

ax.set_title(
    f'Top {top_n} loại cơ sở theo tổng thương vong',
    fontweight='bold',
    fontsize=14
)

ax.set_xlabel(
    'Tổng thương vong'
)

ax.set_ylabel(
    'Property Use'
)

ax.set_xlim(
    0,
    plot_loss['total_casualties'].max() * 1.1
)

for i, v in enumerate(
    plot_loss['total_casualties']
):

    ax.text(
        v + plot_loss['total_casualties'].max() * 0.01,
        i,
        f'{v:,}',
        va='center',
        fontsize=9
    )

sns.despine(ax=ax)

plt.tight_layout()

plt.show()

# %% Cell 75
print(
    "%table " +
    prop_counts.to_csv(
        index=False,
        sep="\t"
    )
)

# %% Cell 76
prop_loss_table = (
    prop_loss
    .sort_values(
        'incidents',
        ascending=False
    )
    .head(top_n)
    .reset_index()
)

print(
    "%table " +
    prop_loss_table.to_csv(
        index=False,
        sep="\t"
    )
)

# %% Cell 78
num_cols = [
    'Estimated Property Loss',
    'Estimated Contents Loss',
    'Total Casualties',
    'Response Time Min',
    'Suppression Units',
    'EMS Units',
    'Number of Alarms',
    'Civilian Fatalities',
    'Civilian Injuries',
]

corr = df_fire[num_cols].corr(numeric_only=True)

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt='.2f',
    cmap='RdBu_r',
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
)
plt.title('Ma trận tương quan — biến số chính', fontweight='bold')
plt.tight_layout()
plt.show()

# %% Cell 80
extreme_cols = [
    'Incident Number', 'Incident Date', 'Address', 'neighborhood_district',
    'Situation_Code', 'Situation_Desc',
    'Estimated Property Loss', 'Estimated Contents Loss',
    'Total Casualties', 'Response Time Min',
]

print("%table")

top10 = df_fire.nlargest(10, 'Estimated Property Loss')[extreme_cols]

print("\t".join(top10.columns))
for _, row in top10.iterrows():
    print("\t".join(map(str, row.values)))

# %% Cell 81
insights = pd.DataFrame({
    'Insight': [
        'Property Loss > $1M',
        'Contents Loss > $500K',
        'Response Time > 30 min',
        'Response Time > P99',
        'Total Casualties >= 3',
        'Civilian Fatalities > 0',
    ],
    'Incident Count': [
        (df_fire['Estimated Property Loss'] > 1_000_000).sum(),
        (df_fire['Estimated Contents Loss'] > 500_000).sum(),
        (df_fire['Response Time Min'] > 30).sum(),
        (df_fire['Response Time Min'] > df_fire['Response Time Min'].quantile(0.99)).sum(),
        (df_fire['Total Casualties'] >= 3).sum(),
        (df_fire['Civilian Fatalities'] > 0).sum(),
    ],
})

insights['Percentage (%)'] = (
    insights['Incident Count'] / len(df_fire) * 100
).round(3)

print("%table")

print("\t".join(insights.columns))

for _, row in insights.iterrows():
    print("\t".join(map(str, row.values)))

# %% Cell 82
fig, ax = plt.subplots(figsize=(8,5))

sns.barplot(
    data=insights,
    x='Percentage (%)',
    y='Insight',
    palette='Reds_r',
    ax=ax
)

ax.set_title(
    'Tỷ lệ các sự kiện cực đoan',
    fontweight='bold'
)

ax.set_xlabel('Percentage (%)')
ax.set_ylabel('')

for i, v in enumerate(insights['Percentage (%)']):

    ax.text(
        v + insights['Percentage (%)'].max()*0.01,
        i,
        f'{v:.3f}%',
        va='center'
    )

sns.despine()

plt.tight_layout()
plt.show()
