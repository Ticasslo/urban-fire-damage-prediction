# Code cells from notebooks/NFIRS_Notebook.json, NFIRS 2013 (Huỳnh Ngọc Thắng).
# Copied cell by cell; only the "%pyspark" line at the top of each cell is removed.
# Markdown, %sh and %angular cells are left out. Some cells use Zeppelin's z object.

# %% Cell 0
import warnings
import os
import numpy as np
import pandas as pd
from scipy import stats
import gdown
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import networkx as nx
import json

# ML
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              precision_recall_fscore_support, confusion_matrix,
                              classification_report)

# Regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib

warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')

# %% Cell 1
TMP = r'C:\FireDataset\NFIRS'
os.makedirs(TMP, exist_ok=True)

files = {
    'basic' : '1qO6h0Ize--4dRFJmitP_a4CiKdUah_m8',
    'fire'  : '1Fpcf4RUrgqrs02b4gKd2zaIKeHOCRUCq',
    'lookup': '1tXT_Og7pNMSJvm5gUsO5ta-umivV-XZ9',
}

for name, fid in files.items():
    out = os.path.join(TMP, f'{name}.csv')
    # Chỉ download nếu chưa có, tránh download lại
    if not os.path.exists(out):
        gdown.download(f'https://drive.google.com/uc?id={fid}', out, quiet=False)

basic  = pd.read_csv(os.path.join(TMP, 'basic.csv'),  low_memory=False, encoding='latin-1')
fire   = pd.read_csv(os.path.join(TMP, 'fire.csv'),   low_memory=False, encoding='latin-1')
lookup = pd.read_csv(os.path.join(TMP, 'lookup.csv'), encoding='latin-1', dtype=str)

# %% Cell 3
basic['INC_TYPE'] = pd.to_numeric(basic['INC_TYPE'], errors='coerce')

top10 = (basic['INC_TYPE']
         .value_counts()
         .head(10)
         .reset_index())
top10.columns = ['INC_TYPE', 'count']
top10 = top10.sort_values('count', ascending=False)

print("Top 10 loại sự cố trong basicincident.csv")
print("%table\n" + top10.to_csv(index=False, sep="\t"))

top10['INC_TYPE'] = top10['INC_TYPE'].astype(str)

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=top10, x='INC_TYPE', y='count',
            order=top10['INC_TYPE'],
            palette='Reds_r', ax=ax)
ax.set_title('Top 10 loại sự cố (INC_TYPE) trong basicincident.csv', fontsize=14)
ax.set_xlabel('Mã loại sự cố')
ax.set_ylabel('Số lượng')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
for bar in ax.patches:
    h = bar.get_height()
    if not np.isnan(h) and h > 0:
        ax.text(bar.get_x() + bar.get_width()/2, h + 1000,
                f'{int(h):,}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(TMP, '01_INC_TYPE_Top10.png'), dpi=150)
plt.show()

# %% Cell 5
basic_filtered = basic[basic['INC_TYPE'].between(111, 123)].copy()

print(f"Trước filter : {basic.shape[0]:>10,} rows")
print(f"Sau filter   : {basic_filtered.shape[0]:>10,} rows")
print(f"Đã loại bỏ  : {basic.shape[0] - basic_filtered.shape[0]:>10,} rows ({(1 - basic_filtered.shape[0]/basic.shape[0])*100:.1f}%)")

inc_pct_df = pd.DataFrame({
    'INC_TYPE': basic_filtered['INC_TYPE'].value_counts().sort_index().index,
    'count':    basic_filtered['INC_TYPE'].value_counts().sort_index().values,
})
inc_pct_df['pct'] = (inc_pct_df['count'] / basic_filtered.shape[0] * 100).round(1)

print("Phân phối INC_TYPE sau filter")
print("%table\n" + inc_pct_df.to_csv(index=False, sep="\t"))

fig = plt.figure(figsize=(14, 10))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 1, 2)

labels = ['Trước filter', 'Sau filter']
values = [basic.shape[0], basic_filtered.shape[0]]
sns.barplot(x=labels, y=values, palette=['#d9534f', '#5cb85c'], ax=ax1)
ax1.set_title('Số lượng sự cố trước và sau filter', fontsize=12)
ax1.set_ylabel('Số lượng')
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
for bar in ax1.patches:
    h = bar.get_height()
    if not np.isnan(h) and h > 0:
        ax1.text(bar.get_x() + bar.get_width()/2, h + 5000,
                f'{int(h):,}', ha='center', va='bottom', fontsize=11, fontweight='bold')

kept    = basic_filtered.shape[0]
removed = basic.shape[0] - kept
ax2.pie([kept, removed],
        labels=[f'Giữ lại\n{kept:,}', f'Loại bỏ\n{removed:,}'],
        colors=['#5cb85c', '#d9534f'],
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 11})
ax2.set_title('Tỷ lệ giữ lại sau filter', fontsize=12)

inc_sorted = inc_pct_df.sort_values('count', ascending=False)
inc_sorted['INC_TYPE'] = inc_sorted['INC_TYPE'].astype(str)
sns.barplot(data=inc_sorted, x='INC_TYPE', y='count', palette='Greens_r', ax=ax3)
ax3.set_title('Số lượng từng INC_TYPE sau filter (111–123)', fontsize=12)
ax3.set_xlabel('Mã INC_TYPE')
ax3.set_ylabel('Số lượng')
ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
for bar in ax3.patches:
    h = bar.get_height()
    if not np.isnan(h) and h > 0:
        ax3.text(bar.get_x() + bar.get_width()/2, h + 500,
                f'{int(h):,}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(TMP, '02_Filter_Before_After.png'), dpi=150)
plt.show()

# %% Cell 7
join_keys = ['STATE', 'FDID', 'INC_DATE', 'INC_NO', 'EXP_NO']
print(f"Duplicate keys in basic_filtered : {basic_filtered.duplicated(join_keys).sum()}")
print(f"Duplicate keys in fire           : {fire.duplicated(join_keys).sum()}")
fire = fire.drop_duplicates(subset=join_keys)

# LEFT JOIN
df = basic_filtered.merge(fire, on=join_keys, how='left', suffixes=('_basic', '_fire'))
df = df.drop(columns=['VERSION_fire'], errors='ignore')

print(f"\nbasic_filtered : {basic_filtered.shape[0]:>10,} rows | {basic_filtered.shape[1]} cols")
print(f"fire           : {fire.shape[0]:>10,} rows | {fire.shape[1]} cols")
print(f"df sau join    : {df.shape[0]:>10,} rows | {df.shape[1]} cols")

# Decode bằng codelookup
lookup['fieldid']    = lookup['fieldid'].str.strip()
lookup['code_value'] = lookup['code_value'].str.strip()
lookup['code_descr'] = lookup['code_descr'].str.strip()
lookup_clean = lookup[lookup['code_value'].notna()]
lookup_clean = lookup_clean[lookup_clean['code_value'] != '']

def decode(df, field):
    mapping = (lookup_clean[lookup_clean['fieldid'] == field]
               .drop_duplicates(subset='code_value', keep='first')
               .set_index('code_value')['code_descr'])

    # Tạo mapping bổ sung: thêm bản zero-padded 2 chữ số cho các key numeric
    extra = {}
    for k, v in mapping.items():
        if k.isdigit():
            extra[str(int(k))] = v          # "00" -> cũng map từ "0"
            extra[k.zfill(2)] = v            # "0"  -> cũng map từ "00"
    mapping_full = {**extra, **mapping.to_dict()}

    def safe_convert(x):
        if pd.isna(x):
            return None
        try:
            return str(int(float(x)))
        except (ValueError, TypeError):
            pass
        return str(x).strip()

    df[field + '_TEXT'] = df[field].apply(safe_convert).map(mapping_full)
    return df

# Thực hiện decode tất cả các file có trong codelookup
# Lấy danh sách fields từ codelookup
# Chỉ decode những field nào có trong dataset đã join
fields_in_lookup = lookup_clean['fieldid'].unique()
fields_in_df     = set(df.columns)

decoded, skipped = [], []
for field in fields_in_lookup:
    if field in fields_in_df:
        df = decode(df, field)
        decoded.append(field)
    else:
        skipped.append(field)

print(f"\nĐã decode : {len(decoded)} fields")
print("Các field đã decode:")
for f in sorted(decoded):
    print(f"  {f:<20} -> {f}_TEXT")
print(f"Bỏ qua   : {len(skipped)} fields (không có trong df)")
print(f"df cuối  : {df.shape[0]:>10,} rows | {df.shape[1]} cols")

print(f"\nDanh sách toàn bộ {df.shape[1]} cột trong df sau JOIN + decode")
for i, col in enumerate(sorted(df.columns), 1):
    print(f"  {i:>3}. {col}")

# Sửa FLAME_SPRD_TEXT: field này trong df là Y/N, không phải mã 1-5 như trong codelookup
df = df.drop(columns=['FLAME_SPRD_TEXT'], errors='ignore')
df['FLAME_SPRD_TEXT'] = df['FLAME_SPRD'].map({'Y': 'Yes - lan khỏi vị trí ban đầu', 'N': 'No - giới hạn tại vị trí ban đầu'})
print(f"\nFLAME_SPRD_TEXT sau khi sửa:")
print(df['FLAME_SPRD_TEXT'].value_counts(dropna=False))

# %% Cell 10
# Chọn các cột theo đề tài
cols_keep = [
    # Định danh & thời gian
    'STATE', 'INC_DATE', 'INC_TYPE',
    # Thời gian phản ứng
    'ALARM', 'ARRIVAL',
    # Thiệt hại & thương vong
    'PROP_LOSS', 'CONT_LOSS', 'FF_DEATH', 'OTH_DEATH', 'FF_INJ', 'OTH_INJ',
    # Nguyên nhân
    'CAUSE_IGN', 'HEAT_SOURC', 'FIRST_IGN',
    # Cấu trúc & lan rộng
    'STRUC_TYPE', 'PROP_USE', 'FIRE_SPRD', 'AREA_ORIG',
    # Yếu tố con người
    'HUM_FAC_1', 'HUM_FAC_2', 'HUM_FAC_3',
    # Hệ thống bảo vệ
    'DETECTOR', 'DET_EFFECT', 'AES_PRES', 'AES_OPER',
    # Cột TEXT tương ứng
    'STATE_TEXT', 'INC_TYPE_TEXT',
    'CAUSE_IGN_TEXT', 'HEAT_SOURC_TEXT', 'FIRST_IGN_TEXT',
    'STRUC_TYPE_TEXT', 'PROP_USE_TEXT', 'FIRE_SPRD_TEXT', 'AREA_ORIG_TEXT',
    'HUM_FAC_1_TEXT', 'HUM_FAC_2_TEXT', 'HUM_FAC_3_TEXT',
    'DETECTOR_TEXT', 'DET_EFFECT_TEXT', 'AES_PRES_TEXT', 'AES_OPER_TEXT',
]

df_sel = df[cols_keep].copy()
print(f"df sau chọn cột: {df_sel.shape[0]:,} rows | {df_sel.shape[1]} cols")

# Missing analysis trên cột đã chọn
missing = df_sel.isnull().sum()
missing_pct = (missing / len(df_sel) * 100).round(2)
missing_df = pd.DataFrame({
    'column': missing.index,
    'missing_count': missing.values,
    'missing_pct': missing_pct.values
}).sort_values('missing_pct', ascending=False).reset_index(drop=True)

print(f"Tổng số cột đã chọn : {df_sel.shape[1]}")
print(f"Số cột có missing   : {(missing_df['missing_count'] > 0).sum()}")
print(f"Số cột không missing: {(missing_df['missing_count'] == 0).sum()}")
print("Chi tiết missing values")
print("%table\n" + missing_df.to_csv(index=False, sep="\t"))

# Visualization — chỉ vẽ cột gốc (không _TEXT) cho gọn
cols_raw = [c for c in missing_df['column'] if not c.endswith('_TEXT')]
missing_raw = missing_df[missing_df['column'].isin(cols_raw)].sort_values('missing_pct', ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#d9534f' if v > 50 else '#f0ad4e' if v > 20 else '#5cb85c'
          for v in missing_raw['missing_pct']]
sns.barplot(data=missing_raw, y='column', x='missing_pct', palette=colors, ax=ax)
ax.set_title('Tỷ lệ Missing Values — các cột theo đề tài', fontsize=13)
ax.set_xlabel('% Missing')
ax.set_ylabel('Cột')
ax.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='Ngưỡng 50%')
ax.legend()
for i, v in enumerate(missing_raw['missing_pct']):
    ax.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(TMP, '03_Missing_Selected.png'), dpi=150)
plt.show()

# %% Cell 12
# Noise: PROP_LOSS / CONT_LOSS
print("PROP_LOSS / CONT_LOSS")
print(df[['PROP_LOSS', 'CONT_LOSS']].describe())
print(f"PROP_LOSS âm : {(df['PROP_LOSS'] < 0).sum()}")
print(f"CONT_LOSS âm : {(df['CONT_LOSS'] < 0).sum()}")

# Noise: Thương vong
print("\nThương vong: FF_DEATH / OTH_DEATH / FF_INJ / OTH_INJ")
casualty_cols = ['FF_DEATH', 'OTH_DEATH', 'FF_INJ', 'OTH_INJ']
print(df[casualty_cols].describe())
for c in casualty_cols:
    print(f"{c} âm : {(df[c] < 0).sum()}")
    
# Noise: ALARM / ARRIVAL (response time)
def parse_nfirs_datetime(series):
    s = series.astype('Int64').astype(str)
    date_part = s.str[:-4]   # MDDYYYY hoặc MMDDYYYY
    time_part = s.str[-4:]   # HHMM
    full = date_part + time_part.str.zfill(4)
    full = full.str.zfill(12)  # đảm bảo MMDDYYYYHHMM
    return pd.to_datetime(full, format='%m%d%Y%H%M', errors='coerce')

alarm_dt = parse_nfirs_datetime(df['ALARM'])
arrival_dt = parse_nfirs_datetime(df['ARRIVAL'])
response_min = (arrival_dt - alarm_dt).dt.total_seconds() / 60

print(response_min.describe())
print(f"Response time âm (ARRIVAL < ALARM) : {(response_min < 0).sum()}")
print(f"Response time = 0                   : {(response_min == 0).sum()}")
print(f"Response time > 60 phút             : {(response_min > 60).sum()}")
print(f"Không convert được (NaT)            : {alarm_dt.isna().sum()} (ALARM), {arrival_dt.isna().sum()} (ARRIVAL)")
print(f"Response time > 1440 phút (1 ngày)  : {(response_min > 1440).sum()}")

# %% Cell 14
# Convert INC_DATE (MMDDYYYY int) -> datetime
print("INC_DATE trước convert (5 mẫu):")
print(df['INC_DATE'].head().tolist())

df['INC_DATE'] = pd.to_datetime(df['INC_DATE'].astype(str).str.zfill(8),
                                 format='%m%d%Y', errors='coerce')

print("\nINC_DATE sau convert (5 mẫu):")
print(df['INC_DATE'].head().tolist())
print(f"Số giá trị NaT sau convert : {df['INC_DATE'].isna().sum()}")
print(f"Khoảng thời gian           : {df['INC_DATE'].min()} -> {df['INC_DATE'].max()}")

# TOTAL_LOSS
df['TOTAL_LOSS'] = df[['PROP_LOSS', 'CONT_LOSS']].sum(axis=1, min_count=1)
print("\nTOTAL_LOSS")
print(df['TOTAL_LOSS'].describe())
print(f"Số dòng TOTAL_LOSS = NaN (cả PROP_LOSS và CONT_LOSS đều NaN) : {df['TOTAL_LOSS'].isna().sum()}")

# TOTAL_CASUALTIES
casualty_cols = ['FF_DEATH', 'OTH_DEATH', 'FF_INJ', 'OTH_INJ']
print(f"\nMissing trước fillna(0): {df[casualty_cols].isna().sum().to_dict()}")
df[casualty_cols] = df[casualty_cols].fillna(0)
df['TOTAL_CASUALTIES'] = df[casualty_cols].sum(axis=1)

print("\nTOTAL_CASUALTIES")
print(df['TOTAL_CASUALTIES'].describe())
print(f"Số dòng TOTAL_CASUALTIES > 0 : {(df['TOTAL_CASUALTIES'] > 0).sum()} ({(df['TOTAL_CASUALTIES'] > 0).mean()*100:.2f}%)")

# RESPONSE_TIME (phút) = ARRIVAL - ALARM, xử lý noise
# Lưu ý: response_min được tính từ cell trước (Noise check ALARM/ARRIVAL), 
df['RESPONSE_TIME'] = response_min
invalid_mask = (df['RESPONSE_TIME'] < 0) | (df['RESPONSE_TIME'] > 1440)
print(f"\nRESPONSE_TIME: set NaN cho {invalid_mask.sum()} giá trị bất thường (<0 hoặc >1440 phút)")
df.loc[invalid_mask, 'RESPONSE_TIME'] = np.nan
print(df['RESPONSE_TIME'].describe())
print(f"Missing sau xử lý: {df['RESPONSE_TIME'].isna().sum()} ({df['RESPONSE_TIME'].isna().mean()*100:.2f}%)")

# %% Cell 15
cols_clean = [
    # Định danh & thời gian
    'STATE', 'INC_DATE', 'INC_TYPE',
    # Thời gian phản ứng
    'RESPONSE_TIME',
    # Thiệt hại & thương vong (đã fillna(0) ở trên)
    'PROP_LOSS', 'CONT_LOSS', 'FF_DEATH', 'OTH_DEATH', 'FF_INJ', 'OTH_INJ',
    # Biến tổng hợp
    'TOTAL_LOSS', 'TOTAL_CASUALTIES',
    # Nguyên nhân
    'CAUSE_IGN', 'HEAT_SOURC', 'FIRST_IGN',
    # Cấu trúc & lan rộng
    'STRUC_TYPE', 'PROP_USE', 'FIRE_SPRD', 'AREA_ORIG',
    # Hệ thống bảo vệ
    'DETECTOR', 'AES_PRES',
    # Cột TEXT tương ứng
    'STATE_TEXT', 'INC_TYPE_TEXT',
    'CAUSE_IGN_TEXT', 'HEAT_SOURC_TEXT', 'FIRST_IGN_TEXT',
    'STRUC_TYPE_TEXT', 'PROP_USE_TEXT', 'FIRE_SPRD_TEXT', 'AREA_ORIG_TEXT',
    'DETECTOR_TEXT', 'AES_PRES_TEXT',
]

df_clean = df[cols_clean].copy()

print(f"df_clean : {df_clean.shape[0]:,} rows | {df_clean.shape[1]} cols")
miss_clean = pd.DataFrame({
    'STT': range(1, len(df_clean.columns)+1),
    'column': df_clean.columns,
    'missing_count': [df_clean[c].isna().sum() for c in df_clean.columns],
    'missing_pct': [(df_clean[c].isna().sum()/len(df_clean)*100).round(1) for c in df_clean.columns]
})
print("Danh sách cột df_clean")
print("%table\n" + miss_clean.to_csv(index=False, sep="\t"))

# %% Cell 16
n_total = len(df_clean)
n_missing = df_clean['TOTAL_LOSS'].isna().sum()
n_valid = df_clean['TOTAL_LOSS'].notna().sum()
print(f"Tổng số dòng        : {n_total:,}")
print(f"TOTAL_LOSS missing  : {n_missing:,} ({n_missing/n_total*100:.1f}%)")
print(f"TOTAL_LOSS hợp lệ   : {n_valid:,} ({n_valid/n_total*100:.1f}%)")

# %% Cell 18
cols_stat = ['TOTAL_LOSS', 'PROP_LOSS', 'CONT_LOSS', 'TOTAL_CASUALTIES']
cols_log   = ['TOTAL_LOSS', 'PROP_LOSS', 'CONT_LOSS']

rows_common, rows_moment = [], []

for col in cols_stat:
    s = df_clean[col].dropna()
    n = len(s)
    q1, q2, q3 = s.quantile([0.25, 0.50, 0.75])
    mu, sigma = s.mean(), s.std()

    rows_common.append({
        'Biến'   : col,
        'Count'  : n,
        'Min'    : s.min(),
        'Max'    : s.max(),
        'Mean'   : round(mu, 2),
        'Median' : round(s.median(), 2),
        'Std'    : round(sigma, 2),
        'Q1'     : round(q1, 2),
        'Q2'     : round(q2, 2),
        'Q3'     : round(q3, 2),
        'IQR'    : round(q3 - q1, 2),
    })

    skew       = stats.skew(s)
    kurt       = stats.kurtosis(s)
    hyper_skew = ((s - mu)**5).mean() / sigma**5 if sigma > 0 else np.nan
    hyper_tail = ((s - mu)**6).mean() / sigma**6 if sigma > 0 else np.nan

    rows_moment.append({
        'Biến'                     : col,
        'Mean (μ)'                 : round(mu, 4),
        'Variance (σ²)'            : round(s.var(), 4),
        'Skewness (μ₃/σ³)'         : round(skew, 4),
        'Kurtosis (μ₄/σ⁴)'         : round(kurt, 4),
        'Hyperskewness (μ₅/σ⁵)'    : round(hyper_skew, 4),
        'Hypertailedness (μ₆/σ⁶)'  : round(hyper_tail, 4),
    })

df_common = pd.DataFrame(rows_common)
df_moment = pd.DataFrame(rows_moment)

print("  COMMON STATISTICS")
print(df_common.to_string(index=False))
print()
print("  MOMENT STATISTICS")
print(df_moment.to_string(index=False))

COLORS = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
HDR_COMMON = '#2C3E50'
HDR_MOMENT = '#1A5276'

# ====== ẢNH 1: 4 BOXPLOT DẠNG 2 HÀNG x 2 CỘT ======
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Boxplots — TOTAL_LOSS, PROP_LOSS, CONT_LOSS, TOTAL_CASUALTIES',
             fontsize=14, fontweight='bold', y=0.995)

axes_flat = axes.flatten()

for i, col in enumerate(cols_stat):
    s     = df_clean[col].dropna()
    color = COLORS[i]

    if col in cols_log:
        plot_data = np.log1p(s)
        xlabel    = f'log({col} + 1)'
    else:
        plot_data = s
        xlabel    = col

    ax = axes_flat[i]
    ax.boxplot(plot_data, vert=False, patch_artist=True, widths=0.5,
               boxprops=dict(facecolor=color, alpha=0.55),
               medianprops=dict(color='red', linewidth=2),
               whiskerprops=dict(linewidth=1.5),
               capprops=dict(linewidth=1.5),
               flierprops=dict(marker='o', markersize=2, alpha=0.25, color='gray'))
    ax.axvline(plot_data.mean(),   color='navy',   linestyle='--', lw=1.4,
               label=f'Mean={plot_data.mean():.2f}')
    ax.axvline(plot_data.median(), color='orange', linestyle='-',  lw=1.4,
               label=f'Median={plot_data.median():.2f}')
    ax.set_title(f'Boxplot — {col}', fontsize=11, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_yticks([])
    ax.legend(fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(TMP, '04_Boxplots_All.png'), dpi=150)
plt.show()

# ====== ẢNH 2: BẢNG COMMON STATISTICS GỘP 4 BIẾN ======
fig, ax = plt.subplots(figsize=(13, 1.2 + 0.6 * len(cols_stat)))
ax.axis('off')

common_cols = ['Biến', 'Count', 'Min', 'Max', 'Mean', 'Median', 'Std', 'Q1', 'Q2', 'Q3', 'IQR']
table_data = []
for rc in rows_common:
    table_data.append([f"{rc[c]:,}" if c != 'Biến' else rc[c] for c in common_cols])

tbl = ax.table(cellText=table_data,
               colLabels=common_cols,
               cellLoc='center', loc='center',
               colWidths=[0.16, 0.09, 0.08, 0.13, 0.09, 0.09, 0.1, 0.07, 0.07, 0.07, 0.07])
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.8)
for (r, c_), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor(HDR_COMMON)
        cell.set_text_props(color='white', fontweight='bold')
    elif r % 2 == 0:
        cell.set_facecolor('#EBF5FB')
    cell.set_edgecolor('#BDC3C7')

ax.set_title('Common Statistics — Tổng hợp', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(TMP, '04_Common_Statistics_Table.png'), dpi=150)
plt.show()

# ====== ẢNH 3: BẢNG MOMENT STATISTICS GỘP 4 BIẾN (bỏ Hyperskewness, Hypertailedness) ======
fig, ax = plt.subplots(figsize=(10, 1.2 + 0.6 * len(cols_stat)))
ax.axis('off')

moment_cols = ['Biến', 'Mean (μ)', 'Variance (σ²)', 'Skewness (μ₃/σ³)', 'Kurtosis (μ₄/σ⁴)']
table_data = []
for rm in rows_moment:
    table_data.append([f"{rm[c]:,}" if c != 'Biến' else rm[c] for c in moment_cols])

tbl2 = ax.table(cellText=table_data,
               colLabels=moment_cols,
               cellLoc='center', loc='center',
               colWidths=[0.22, 0.18, 0.2, 0.2, 0.2])
tbl2.auto_set_font_size(False)
tbl2.set_fontsize(10)
tbl2.scale(1, 1.8)
for (r, c_), cell in tbl2.get_celld().items():
    if r == 0:
        cell.set_facecolor(HDR_MOMENT)
        cell.set_text_props(color='white', fontweight='bold')
    elif r % 2 == 0:
        cell.set_facecolor('#EAFAF1')
    cell.set_edgecolor('#BDC3C7')

ax.set_title('Moment Statistics — Tổng hợp', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(TMP, '04_Moment_Statistics_Table.png'), dpi=150)
plt.show()

# %% Cell 20
tl = df_clean['TOTAL_LOSS'].dropna()
n  = len(tl)

n_bins     = int(np.ceil(1 + np.log2(n)))
tl_log     = np.log1p(tl)
n_bins_log = int(np.ceil(1 + np.log2(len(tl_log))))

print(f"Số bins theo Sturge (gốc) : {n_bins}")
print(f"Số bins theo Sturge (log) : {n_bins_log}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f'Phân phối TOTAL_LOSS — NFIRS 2013  (n = {n:,})',
             fontsize=14, fontweight='bold')

# Plot 1: Histogram thang gốc
ax1 = axes[0]
tl_clip = tl[tl <= tl.quantile(0.99)]
ax1.hist(tl_clip, bins=n_bins, color='#E74C3C', edgecolor='white', linewidth=0.4)
ax1.axvline(tl.mean(),   color='navy',   linestyle='--', lw=1.8,
            label=f'Mean = {tl.mean():,.0f}')
ax1.axvline(tl.median(), color='orange', linestyle='-',  lw=1.8,
            label=f'Median = {tl.median():,.0f}')
ax1.set_title(f'Histogram TOTAL_LOSS (Sturge, bins={n_bins})\n[clip 99th percentile]', fontsize=11)
ax1.set_xlabel('TOTAL_LOSS ($)')
ax1.set_ylabel('Số vụ')
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax1.legend(fontsize=10)

# Plot 2: Histogram log(TOTAL_LOSS + 1)
ax2 = axes[1]
ax2.hist(tl_log, bins=n_bins_log, color='#3498DB', edgecolor='white', linewidth=0.4)
ax2.axvline(tl_log.mean(),   color='navy',   linestyle='--', lw=1.8,
            label=f'Mean = {tl_log.mean():.2f}')
ax2.axvline(tl_log.median(), color='orange', linestyle='-',  lw=1.8,
            label=f'Median = {tl_log.median():.2f}')
ax2.set_title(f'Histogram log(TOTAL_LOSS + 1) (Sturge, bins={n_bins_log})', fontsize=11)
ax2.set_xlabel('log(TOTAL_LOSS + 1)')
ax2.set_ylabel('Số vụ')
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(TMP, '05_TOTAL_LOSS_Histogram.png'), dpi=150)
plt.show()

print(f"Mean log(TOTAL_LOSS+1)   : {np.log1p(tl).mean():.4f}")
print(f"Median log(TOTAL_LOSS+1) : {np.log1p(tl).median():.4f}")

# %% Cell 23
tl_all    = df_clean['TOTAL_LOSS'].dropna()
tl_nonzero = tl_all[tl_all > 0]

q1 = tl_nonzero.quantile(0.25)
q3 = tl_nonzero.quantile(0.75)

print(f"Tính phân vị chỉ trên {len(tl_nonzero):,} vụ có TOTAL_LOSS > 0:")
print(f"  Q1 (25th) = {q1:,.0f}")
print(f"  Q3 (75th) = {q3:,.0f}")
print()
print(f"Phân bố theo ngưỡng (trên {len(tl_all):,} vụ có dữ liệu):")
print(f"  TOTAL_LOSS = 0              : {(tl_all == 0).sum():>8,}  ({(tl_all == 0).sum()/len(tl_all)*100:.1f}%)")
print(f"  0 < TOTAL_LOSS <= Q1        : {((tl_all > 0) & (tl_all <= q1)).sum():>8,}  ({((tl_all > 0) & (tl_all <= q1)).sum()/len(tl_all)*100:.1f}%)")
print(f"  Q1 < TOTAL_LOSS <= Q3       : {((tl_all > q1) & (tl_all <= q3)).sum():>8,}  ({((tl_all > q1) & (tl_all <= q3)).sum()/len(tl_all)*100:.1f}%)")
print(f"  TOTAL_LOSS > Q3             : {(tl_all > q3).sum():>8,}  ({(tl_all > q3).sum()/len(tl_all)*100:.1f}%)")

def assign_severity(x):
    if pd.isna(x):   return np.nan
    elif x == 0:     return 'No Loss'
    elif x <= q1:    return 'Minor'
    elif x <= q3:    return 'Moderate'
    else:            return 'Major'

df_clean['SEVERITY'] = df_clean['TOTAL_LOSS'].apply(assign_severity)

sev_order  = ['No Loss', 'Minor', 'Moderate', 'Major']
sev_colors = ['#95A5A6', '#2ECC71', '#F39C12', '#E74C3C']

sev_counts = df_clean['SEVERITY'].value_counts()
nan_count  = df_clean['SEVERITY'].isna().sum()

print(f"\nPhân phối SEVERITY:")
for s in sev_order:
    v = sev_counts.get(s, 0)
    print(f"  {s:<12} : {v:>8,}  ({v/len(df_clean)*100:.1f}%)")
print(f"  {'Unknown':<12} : {nan_count:>8,}  ({nan_count/len(df_clean)*100:.1f}%)")

# Figure
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Phân phối SEVERITY — NFIRS 2013', fontsize=14, fontweight='bold')

# Plot 1: Bar count
ax1 = axes[0]
vals = [sev_counts.get(s, 0) for s in sev_order]
bars = ax1.bar(sev_order, vals, color=sev_colors, edgecolor='white', linewidth=0.5)
ax1.set_title('Số vụ theo SEVERITY', fontsize=11)
ax1.set_xlabel('SEVERITY')
ax1.set_ylabel('Số vụ')
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
for bar, v in zip(bars, vals):
    ax1.text(bar.get_x() + bar.get_width()/2, v + 500,
             f'{v:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 2: Histogram log(TOTAL_LOSS+1) có đánh dấu ngưỡng
ax2 = axes[1]
tl_log = np.log1p(tl_nonzero)
n_bins = int(np.ceil(1 + np.log2(len(tl_log))))
ax2.hist(tl_log, bins=n_bins, color='#BDC3C7', edgecolor='white', linewidth=0.3)
ax2.axvline(np.log1p(q1), color='#2ECC71', linestyle='--', lw=2,
            label=f'Q1 = {q1:,.0f} → log={np.log1p(q1):.2f}')
ax2.axvline(np.log1p(q3), color='#E74C3C', linestyle='--', lw=2,
            label=f'Q3 = {q3:,.0f} → log={np.log1p(q3):.2f}')
ax2.set_title('log(TOTAL_LOSS+1) với ngưỡng SEVERITY\n[chỉ vụ có TOTAL_LOSS > 0]', fontsize=11)
ax2.set_xlabel('log(TOTAL_LOSS + 1)')
ax2.set_ylabel('Số vụ')
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(TMP, '06_SEVERITY_Distribution.png'), dpi=150)
plt.show()

print(f"\ndf_clean sau khi thêm SEVERITY: {df_clean.shape[0]:,} rows | {df_clean.shape[1]} cols")
print(df_clean['SEVERITY'].value_counts(dropna=False))

# %% Cell 26
df_sev = df_clean[df_clean['SEVERITY'].notna()].copy()
SEV_ORDER  = ['No Loss', 'Minor', 'Moderate', 'Major']
SEV_COLORS = ['#95A5A6', '#2ECC71', '#F39C12', '#E74C3C']

counts = df_sev['SEVERITY'].value_counts().reindex(SEV_ORDER)
pcts   = (counts / counts.sum() * 100).round(2)

summary = pd.DataFrame({
    'SEVERITY' : SEV_ORDER,
    'Count'    : [counts[s] for s in SEV_ORDER],
    'Pct (%)'  : [pcts[s]   for s in SEV_ORDER],
})
print("Phân phối SEVERITY (đã loại Unknown):")
print(f"Tổng số vụ : {counts.sum():,}")
print()
print(summary.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle('Phân phối SEVERITY — NFIRS 2013', fontsize=14, fontweight='bold')

# Bar chart
ax1 = axes[0]
bars = ax1.bar(SEV_ORDER, [counts[s] for s in SEV_ORDER],
               color=SEV_COLORS, edgecolor='white', linewidth=0.5)
ax1.set_title('Số vụ theo SEVERITY', fontsize=12)
ax1.set_xlabel('SEVERITY')
ax1.set_ylabel('Số vụ')
ax1.set_ylim(0, counts.max() * 1.18)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
for bar, s in zip(bars, SEV_ORDER):
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 800,
             f'{counts[s]:,}\n({pcts[s]}%)',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# Pie chart
ax2 = axes[1]
ax2.pie([counts[s] for s in SEV_ORDER],
        labels=[f'{s}\n{counts[s]:,}' for s in SEV_ORDER],
        colors=SEV_COLORS,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 10})
ax2.set_title('Tỷ lệ theo SEVERITY', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(TMP, '07_SEVERITY_Count.png'), dpi=150)
plt.show()

# %% Cell 28
agg_cas = (df_sev.groupby('SEVERITY')['TOTAL_CASUALTIES']
           .agg(['count', 'mean', 'median'])
           .reindex(SEV_ORDER)
           .rename(columns={'count': 'Count', 'mean': 'Mean', 'median': 'Median'}))
agg_cas['Mean']   = agg_cas['Mean'].round(4)
agg_cas['Median'] = agg_cas['Median'].round(4)

print("Mean / Median TOTAL_CASUALTIES theo SEVERITY:")
print(agg_cas.to_string())

rate_cas = (df_sev.groupby('SEVERITY').apply(
    lambda g: (g['TOTAL_CASUALTIES'] > 0).mean() * 100
).reindex(SEV_ORDER).round(3))
print("\n% vụ có TOTAL_CASUALTIES > 0 theo SEVERITY:")
print(rate_cas.to_string())

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('TOTAL_CASUALTIES theo SEVERITY', fontsize=14, fontweight='bold')

# Plot 1: Mean
ax1 = axes[0]
bars1 = ax1.bar(SEV_ORDER, agg_cas['Mean'], color=SEV_COLORS, edgecolor='white')
ax1.set_title('Mean TOTAL_CASUALTIES theo SEVERITY', fontsize=11)
ax1.set_ylabel('Mean TOTAL_CASUALTIES')
ax1.set_ylim(0, agg_cas['Mean'].max() * 1.3 if agg_cas['Mean'].max() > 0 else 0.01)
for bar, v in zip(bars1, agg_cas['Mean']):
    ax1.text(bar.get_x() + bar.get_width()/2, v + agg_cas['Mean'].max()*0.02, f'{v:.4f}',
             ha='center', va='bottom', fontsize=9)

# Plot 2: Median
ax2 = axes[1]
bars2 = ax2.bar(SEV_ORDER, agg_cas['Median'], color=SEV_COLORS, edgecolor='white')
ax2.set_title('Median TOTAL_CASUALTIES theo SEVERITY', fontsize=11)
ax2.set_ylabel('Median TOTAL_CASUALTIES')
ax2.set_ylim(0, max(agg_cas['Median'].max(), 0.001) * 1.3)
for bar, v in zip(bars2, agg_cas['Median']):
    ax2.text(bar.get_x() + bar.get_width()/2, v + 0.0005, f'{v:.4f}',
             ha='center', va='bottom', fontsize=9)

# Plot 3: % vụ có thương vong
ax3 = axes[2]
bars3 = ax3.bar(SEV_ORDER, rate_cas, color=SEV_COLORS, edgecolor='white')
ax3.set_title('% vụ có thương vong (TOTAL_CASUALTIES > 0)', fontsize=11)
ax3.set_ylabel('% vụ')
ax3.set_ylim(0, rate_cas.max() * 1.3)
for bar, v in zip(bars3, rate_cas):
    ax3.text(bar.get_x() + bar.get_width()/2, v + rate_cas.max()*0.02, f'{v:.2f}%',
             ha='center', va='bottom', fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(TMP, '09_Casualties_by_SEVERITY.png'), dpi=150)
plt.show()

# %% Cell 30
# 3.4 Boxplot TOTAL_LOSS theo SEVERITY
box_stats = (df_sev.groupby('SEVERITY')['TOTAL_LOSS']
             .agg(['count', 'min', 'median', 'max'])
             .reindex(SEV_ORDER))
box_stats.columns = ['Count', 'Min', 'Median', 'Max']
print("Tóm tắt TOTAL_LOSS theo SEVERITY (phục vụ boxplot):")
print(box_stats.to_string())

q99 = df_sev['TOTAL_LOSS'].quantile(0.99)

data_loss = [df_sev.loc[df_sev['SEVERITY'] == s, 'TOTAL_LOSS'].values for s in SEV_ORDER]
data_log  = [np.log1p(df_sev.loc[df_sev['SEVERITY'] == s, 'TOTAL_LOSS']).values for s in SEV_ORDER]

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle('Boxplot TOTAL_LOSS theo SEVERITY — NFIRS 2013', fontsize=14, fontweight='bold')

for ax, datasets, title, xlabel, xlim in [
    (axes[0], data_log,  'Boxplot log(TOTAL_LOSS + 1) theo SEVERITY', 'log(TOTAL_LOSS + 1)', None),
    (axes[1], data_loss, f'Boxplot TOTAL_LOSS theo SEVERITY [clip X tại 99th = {q99:,.0f}]',
     'TOTAL_LOSS ($)', (0, q99 * 1.05)),
]:
    bp = ax.boxplot(datasets, vert=False, labels=SEV_ORDER, patch_artist=True, widths=0.5,
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker='o', markersize=2, alpha=0.25, color='gray'))
    for patch, color in zip(bp['boxes'], SEV_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('SEVERITY')
    if xlim:
        ax.set_xlim(*xlim)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{int(v):,}'))

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(TMP, '10_TOTAL_LOSS_Boxplot_by_SEVERITY.png'), dpi=150)
plt.show()

# %% Cell 33
# 4.1 Định danh — INC_TYPE (stacked bar % SEVERITY)
def sev_crosstab_pct(df, col, top_n=None):
    sub = df[[col, 'SEVERITY']].dropna(subset=[col])
    ct  = pd.crosstab(sub[col], sub['SEVERITY']).reindex(columns=SEV_ORDER, fill_value=0)
    if top_n:
        keep = sub[col].value_counts().head(top_n).index
        ct   = ct.loc[ct.index.isin(keep)]
    ct  = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]
    pct = ct.div(ct.sum(axis=1), axis=0) * 100
    return ct, pct.round(2)

def plot_sev_stack_h(ax, pct_df):
    """Stacked bar ngang — sort Major cao → thấp (trên xuống dưới)."""
    pct_df = pct_df.sort_values('Major', ascending=True)
    n      = len(pct_df)
    y      = np.arange(n)
    left   = np.zeros(n)

    for sev, color in zip(SEV_ORDER, SEV_COLORS):
        vals = pct_df[sev].values
        ax.barh(y, vals, left=left, height=0.72, label=sev, color=color,
                edgecolor='white', linewidth=0.9, alpha=0.93)
        for i, v in enumerate(vals):
            if v >= 4.5:
                if sev in ('Moderate', 'Major') and v >= 7:
                    txt_color = 'white'
                elif sev == 'No Loss' and v >= 45:
                    txt_color = 'white'
                else:
                    txt_color = '#1a1a1a'
                ax.text(left[i] + v / 2, y[i], f'{v:.1f}%',
                        ha='center', va='center', fontsize=11,
                        color=txt_color, fontweight='normal')
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(pct_df.index, fontsize=11, fontweight='normal')
    ax.set_xlim(0, 102)
    ax.set_xlabel('Tỷ lệ trong từng loại sự cố (%)', fontsize=13, labelpad=8, fontweight='normal')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{int(v)}%'))
    ax.xaxis.grid(True, linestyle='--', alpha=0.35, color='#aaaaaa')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=12)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('normal')

ct_inc, pct_inc = sev_crosstab_pct(df_sev, 'INC_TYPE_TEXT')
print(f"INC_TYPE × SEVERITY (n = {ct_inc.sum().sum():,})")
print("\nCount:")
print(ct_inc.to_string())
print("\n% theo hàng:")
print(pct_inc.to_string())

fig, ax = plt.subplots(figsize=(13, 10))
fig.patch.set_facecolor('white')
fig.suptitle('Phân bố SEVERITY theo INC_TYPE — NFIRS 2013',
             fontsize=17, fontweight='bold', y=0.98)

plot_sev_stack_h(ax, pct_inc)
leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                ncol=4, fontsize=12, frameon=False)
for t in leg.get_texts():
    t.set_fontweight('normal')

plt.subplots_adjust(left=0.38, bottom=0.10, top=0.93, right=0.97)
plt.savefig(os.path.join(TMP, '11_INC_TYPE_by_SEVERITY.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% Cell 35
# 4.2 Nguyên nhân — CAUSE_IGN (stacked bar + heatmap, Top 10)
df_cause = df_sev[df_sev['CAUSE_IGN_TEXT'].notna()].copy()
n_cause  = len(df_cause)
print(f"Số vụ có CAUSE_IGN: {n_cause:,} / {len(df_sev):,} ({n_cause / len(df_sev) * 100:.1f}%)")

ct_cause, pct_cause = sev_crosstab_pct(df_cause, 'CAUSE_IGN_TEXT', top_n=10)
print(f"\nCAUSE_IGN Top 10 × SEVERITY (n = {ct_cause.sum().sum():,})")
print("\nCount:")
print(ct_cause.to_string())
print("\n% theo hàng:")
print(pct_cause.to_string())

fig, axes = plt.subplots(2, 1, figsize=(13, 14))
fig.patch.set_facecolor('white')
fig.suptitle('Phân bố SEVERITY theo CAUSE_IGN — NFIRS 2013',
             fontsize=17, fontweight='bold', y=0.99)

# Plot 1: stacked bar ngang
plot_sev_stack_h(axes[0], pct_cause)
axes[0].set_title('Stacked bar — % SEVERITY theo nguyên nhân (Top 10)',
                  fontsize=14, fontweight='bold', pad=10)
axes[0].set_xlabel('Tỷ lệ trong từng nguyên nhân (%)', fontsize=13, labelpad=8)

# Plot 2: heatmap (sort Major cao → thấp)
hm_cause = pct_cause.sort_values('Major', ascending=False)
sns.heatmap(hm_cause, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1],
            linewidths=0.6, linecolor='white',
            cbar_kws={'label': '% vụ trong nhóm'},
            annot_kws={'size': 11})
axes[1].set_title('Heatmap — CAUSE_IGN × SEVERITY (%)', fontsize=14, fontweight='bold', pad=10)
axes[1].set_xlabel('SEVERITY', fontsize=13)
axes[1].set_ylabel('')
axes[1].tick_params(axis='y', labelsize=10)
axes[1].tick_params(axis='x', labelsize=12)

handles, labels = axes[0].get_legend_handles_labels()
leg = fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.50),
                 ncol=4, fontsize=12, frameon=True, edgecolor='#cccccc',
                 title='SEVERITY', title_fontsize=12)
for t in leg.get_texts():
    t.set_fontweight('normal')

plt.subplots_adjust(left=0.42, bottom=0.08, top=0.95, right=0.97, hspace=0.62)
plt.savefig(os.path.join(TMP, '12_CAUSE_IGN_by_SEVERITY.png'), dpi=150,
            bbox_inches='tight', bbox_extra_artists=[leg])
plt.show()

# %% Cell 37
# 4.4 Lan rộng — FIRE_SPRD (stacked bar + heatmap, Top 10)
df_fs = df_sev[df_sev['FIRE_SPRD_TEXT'].notna()].copy()
n_fs  = len(df_fs)
print(f"Số vụ có FIRE_SPRD: {n_fs:,} / {len(df_sev):,} ({n_fs / len(df_sev) * 100:.1f}%)")

ct_fs, pct_fs = sev_crosstab_pct(df_fs, 'FIRE_SPRD_TEXT', top_n=10)
print(f"\nFIRE_SPRD Top 10 × SEVERITY (n = {ct_fs.sum().sum():,})")
print("\nCount:")
print(ct_fs.to_string())
print("\n% theo hàng:")
print(pct_fs.to_string())

fig, axes = plt.subplots(2, 1, figsize=(13, 14))
fig.patch.set_facecolor('white')
fig.suptitle('Phân bố SEVERITY theo FIRE_SPRD — NFIRS 2013',
             fontsize=17, fontweight='bold', y=0.99)

plot_sev_stack_h(axes[0], pct_fs)
axes[0].set_title('Stacked bar — % SEVERITY theo mức lan rộng (Top 10)',
                  fontsize=14, fontweight='bold', pad=10)
axes[0].set_xlabel('Tỷ lệ trong từng mức lan rộng (%)', fontsize=13, labelpad=8)

hm_fs = pct_fs.sort_values('Major', ascending=False)
sns.heatmap(hm_fs, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1],
            linewidths=0.6, linecolor='white',
            cbar_kws={'label': '% vụ trong nhóm'},
            annot_kws={'size': 11})
axes[1].set_title('Heatmap — FIRE_SPRD × SEVERITY (%)', fontsize=14, fontweight='bold', pad=10)
axes[1].set_xlabel('SEVERITY', fontsize=13)
axes[1].set_ylabel('')
axes[1].tick_params(axis='y', labelsize=10)
axes[1].tick_params(axis='x', labelsize=12)

handles, labels = axes[0].get_legend_handles_labels()
leg = fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.50),
                 ncol=4, fontsize=12, frameon=True, edgecolor='#cccccc',
                 title='SEVERITY', title_fontsize=12)
for t in leg.get_texts():
    t.set_fontweight('normal')

plt.subplots_adjust(left=0.42, bottom=0.08, top=0.95, right=0.97, hspace=0.62)
plt.savefig(os.path.join(TMP, '14_FIRE_SPRD_by_SEVERITY.png'), dpi=150,
            bbox_inches='tight', bbox_extra_artists=[leg])
plt.show()

# %% Cell 39
# 4.5 Network — CAUSE_IGN → FIRE_SPRD → SEVERITY (3 tầng, data-driven layout)
NET_COLS = ['CAUSE_IGN_TEXT', 'FIRE_SPRD_TEXT', 'SEVERITY']
df_net   = df_sev.dropna(subset=NET_COLS).copy()
n_net    = len(df_net)
print(f"Số vụ đủ 3 trường: {n_net:,} / {len(df_sev):,} ({n_net / len(df_sev) * 100:.1f}%)")

def _short(s, n=34):
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + '…'

MIN_EDGE = max(150, int(n_net * 0.001))

df_n    = df_net.copy()
n_cause = df_n['CAUSE_IGN_TEXT'].nunique()
n_fs    = df_n['FIRE_SPRD_TEXT'].nunique()

print(f"\nToàn bộ category: {n_cause} CAUSE_IGN × {n_fs} FIRE_SPRD × 4 SEVERITY → {len(df_n):,} vụ")
print(f"Cạnh tối thiểu: {MIN_EDGE:,}")

def _nid(layer, val):
    return f'L{layer}|{val}'

G = nx.DiGraph()
pairs = [
    ('CAUSE_IGN_TEXT', 'FIRE_SPRD_TEXT', 0, 1),
    ('FIRE_SPRD_TEXT', 'SEVERITY',       1, 2),
]
for c1, c2, l1, l2 in pairs:
    agg = (df_n.groupby([c1, c2]).size()
           .reset_index(name='weight')
           .sort_values('weight', ascending=False))
    agg = agg[agg['weight'] >= MIN_EDGE]
    print(f"\n{c1} → {c2}: {len(agg)} cạnh (>= {MIN_EDGE:,})")
    for _, row in agg.head(6).iterrows():
        print(f"  {int(row['weight']):>6,}  {_short(row[c1])} → {_short(row[c2])}")
    if len(agg) > 6:
        print(f"  ... (+{len(agg) - 6} cạnh)")
    for _, row in agg.iterrows():
        G.add_edge(_nid(l1, row[c1]), _nid(l2, row[c2]), weight=int(row['weight']))

print(f"\nTổng nút: {G.number_of_nodes()}  |  Tổng cạnh: {G.number_of_edges()}")

node_flow = {}
for n in G.nodes():
    out_w = sum(d['weight'] for _, _, d in G.out_edges(n, data=True))
    in_w  = sum(d['weight'] for _, _, d in G.in_edges(n, data=True))
    node_flow[n] = out_w + in_w
max_flow = max(node_flow.values()) if node_flow else 1

X_LAYER = [0, 9, 18]
max_nodes = max(len([n for n in G.nodes if n.startswith(f'L{l}|')]) for l in range(3)) or 1
y_step  = 2.0
pos = {}
for layer in range(3):
    nodes = [n for n in G.nodes if n.startswith(f'L{layer}|')]
    nodes.sort(key=lambda n: -node_flow.get(n, 0))
    for i, n in enumerate(nodes):
        pos[n] = (X_LAYER[layer], (len(nodes) - 1 - i) * y_step)

SEV_COLORS  = {'No Loss': '#95A5A6', 'Minor': '#2ECC71',
               'Moderate': '#F39C12', 'Major': '#E74C3C'}
LAYER_FILL  = {0: '#AED6F1', 1: '#FAD7A0'}
LAYER_EDGE  = {0: '#2471A3', 1: '#D68910', 2: '#555555'}
CAUSE_PAL   = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD',
               '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF',
               '#393B79', '#637939', '#8C6D31', '#843C39', '#7B4173',
               '#3182BD', '#E6550D', '#31A354', '#756BB1', '#636363']

cause_nodes = sorted([n for n in G.nodes if n.startswith('L0|')],
                     key=lambda n: -node_flow.get(n, 0))
cause_color = {n: CAUSE_PAL[i % len(CAUSE_PAL)] for i, n in enumerate(cause_nodes)}

fig_h = max(10, (max_nodes - 1) * y_step * 0.55 + 2.5)
fig, ax = plt.subplots(figsize=(20, fig_h))
fig.patch.set_facecolor('white')
ax.set_facecolor('#FAFAFA')

edges = list(G.edges(data=True))
max_w = max(d['weight'] for _, _, d in edges) if edges else 1

def _edge_color(u, v):
    if u.startswith('L0|'):
        return cause_color.get(u, '#5D6D7E')
    if v.startswith('L2|'):
        return SEV_COLORS.get(v.split('|', 1)[1], '#5D6D7E')
    return '#5D6D7E'

for idx, (u, v, d) in enumerate(edges):
    w = d['weight']
    rad = 0.05 + (idx % 4) * 0.04
    nx.draw_networkx_edges(
        G, pos, edgelist=[(u, v)],
        width=2.0 + 10.0 * (w / max_w) ** 0.5,
        alpha=0.42 + 0.45 * (w / max_w),
        edge_color=_edge_color(u, v), arrows=True, arrowsize=20, arrowstyle='-|>',
        connectionstyle=f'arc3,rad={rad}', ax=ax)

node_colors, node_sizes, node_ec = [], [], []
for n in G.nodes():
    layer = int(n[1])
    label = n.split('|', 1)[1]
    if layer == 2:
        node_colors.append(SEV_COLORS.get(label, '#E74C3C'))
    elif layer == 0:
        node_colors.append(cause_color.get(n, LAYER_FILL[0]))
    else:
        node_colors.append(LAYER_FILL[layer])
    node_sizes.append(1400 + 600 * node_flow.get(n, 0) / max_flow)
    node_ec.append(LAYER_EDGE[layer])

nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                       edgecolors=node_ec, linewidths=2.0, ax=ax)

for n, (x, y) in pos.items():
    layer = int(n[1])
    raw   = n.split('|', 1)[1]
    if layer == 0:
        ax.text(x - 0.42, y, raw, fontsize=9.5, ha='right', va='center',
                color='#1a1a1a', clip_on=False)
    elif layer == 2:
        ax.text(x + 0.42, y, raw, fontsize=10.5, ha='left', va='center',
                color='#1a1a1a', clip_on=False)
    else:
        ax.text(x, y + 0.45, _short(raw, 44), fontsize=10, ha='center', va='bottom',
                color='#1a1a1a', clip_on=False)

y_vals = [p[1] for p in pos.values()] if pos else [0]
ax.set_xlim(-0.5, 19.0)
ax.set_ylim(min(y_vals) - 0.8, max(y_vals) + 1.2)
ax.set_aspect('auto')
ax.axis('off')

fig.subplots_adjust(top=0.78, bottom=0.08, left=0.19, right=0.91)
fig.suptitle('Network — CAUSE_IGN → FIRE_SPRD → SEVERITY',
             fontsize=15, fontweight='bold', y=0.96)
fig.text(0.5, 0.915,
         f'(3 tầng | {n_cause} CAUSE × {n_fs} FIRE_SPRD — toàn bộ | n = {len(df_n):,}, cạnh ≥ {MIN_EDGE:,})',
         ha='center', va='top', fontsize=11, color='#444444')

bbox = ax.get_position()
for x_frac, title in zip(
        [bbox.x0 + 0.05 * bbox.width, bbox.x0 + 0.50 * bbox.width, bbox.x0 + 0.95 * bbox.width],
        ['CAUSE_IGN', 'FIRE_SPRD', 'SEVERITY']):
    fig.text(x_frac, bbox.y1 + 0.008, title, fontsize=13, fontweight='bold',
             ha='center', va='bottom', color='#2C3E50')
fig.text(0.5, 0.02,
         'Cạnh trái: màu theo CAUSE_IGN | Cạnh phải: màu theo SEVERITY | Độ dày ∝ số vụ',
         ha='center', fontsize=10, color='#666666')
plt.savefig(os.path.join(TMP, '15_Network_CAUSE_FIRE_SEVERITY.png'), dpi=150,
            bbox_inches='tight', pad_inches=0.4)
plt.show()

# %% Cell 42
df_ci = df_sev[df_sev['CAUSE_IGN_TEXT'].notna()].copy()
n_ci  = len(df_ci)
print(f"Số vụ có CAUSE_IGN: {n_ci:,} / {len(df_sev):,} ({n_ci / len(df_sev) * 100:.1f}%)")

stat_ci = scoped_stats(df_ci, 'CAUSE_IGN_TEXT', top_n=None)
print(f"\nScoped Statistics — CAUSE_IGN (toàn bộ {len(stat_ci)} nhóm)")
print("(Count | Mean/Median TOTAL_LOSS $ | Mean/Median TOTAL_CASUALTIES người/vụ)")
print(stat_ci.to_string())
print(f"\nTổng Count: {stat_ci['Count'].sum():,}")

def plot_scoped_figure_v2(df, stat_df, col, col_name, suptitle, filepath, box_palette='Greens'):
    order    = stat_df.sort_values('Mean_Loss', ascending=True).index.tolist()
    labels   = [_short_lab(c, 32) for c in order]
    data_log = [np.log1p(df.loc[df[col] == c, 'TOTAL_LOSS'].values) for c in order]

    fig, ax = plt.subplots(figsize=(26, max(8, len(order) * 0.6)))
    fig.patch.set_facecolor('white')

    bp = ax.boxplot(data_log, vert=False, labels=labels, patch_artist=True, widths=0.55,
                    medianprops=dict(color='#C0392B', linewidth=2),
                    whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2),
                    flierprops=dict(marker='o', markersize=2, alpha=0.2, color='gray'))
    pal = sns.color_palette(box_palette, n_colors=len(order) + 2)[2:]
    for patch, color in zip(bp['boxes'], pal):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xlabel('log(TOTAL_LOSS + 1)', fontsize=12)
    ax.set_title('Boxplot TOTAL_LOSS theo nhóm (thang log)', fontsize=13, fontweight='bold', pad=10)
    ax.grid(axis='x', linestyle='--', alpha=0.35)
    ax.tick_params(axis='y', labelsize=9)

    fig.suptitle(suptitle, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.show()

plot_scoped_figure_v2(df_ci, stat_ci, 'CAUSE_IGN_TEXT', 'CAUSE_IGN_TEXT',
                      'Scoped Statistics — CAUSE_IGN',
                      os.path.join(TMP, '17_CAUSE_IGN_Scoped_Stats.png'),
                      box_palette='Greens')

# %% Cell 44
# 5.3 Scoped Statistics — FIRE_SPRD (chỉ boxplot)
df_fs2 = df_sev[df_sev['FIRE_SPRD_TEXT'].notna()].copy()
n_fs2  = len(df_fs2)
print(f"Số vụ có FIRE_SPRD: {n_fs2:,} / {len(df_sev):,} ({n_fs2 / len(df_sev) * 100:.1f}%)")

stat_fs = scoped_stats(df_fs2, 'FIRE_SPRD_TEXT', top_n=None)
print(f"\nScoped Statistics — FIRE_SPRD (toàn bộ {len(stat_fs)} nhóm)")
print("(Count | Mean/Median TOTAL_LOSS $ | Mean/Median TOTAL_CASUALTIES người/vụ)")
print(stat_fs.to_string())
print(f"\nTổng Count: {stat_fs['Count'].sum():,}")

plot_scoped_figure_v2(df_fs2, stat_fs, 'FIRE_SPRD_TEXT', 'FIRE_SPRD_TEXT',
                      'Scoped Statistics — FIRE_SPRD (Full)',
                      os.path.join(TMP, '18_FIRE_SPRD_Scoped_Stats.png'),
                      box_palette='Oranges')

# %% Cell 47
corr_cols = ['TOTAL_LOSS', 'PROP_LOSS', 'CONT_LOSS',
              'FF_DEATH', 'OTH_DEATH', 'FF_INJ', 'OTH_INJ', 'TOTAL_CASUALTIES',
              'RESPONSE_TIME']
df_corr = df_sev[corr_cols].dropna()
print(f"Số vụ dùng cho correlation: {len(df_corr):,} / {len(df_sev):,} ({len(df_corr)/len(df_sev)*100:.1f}%)")

spearman_corr = df_corr.corr(method='spearman')

print("Spearman Correlation Matrix")
print("%table\n" + spearman_corr.round(4).reset_index().rename(columns={'index': 'variable'}).to_csv(index=False, sep="\t"))

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(spearman_corr, dtype=bool), k=1)
sns.heatmap(spearman_corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            vmin=-1, vmax=1, square=True, linewidths=0.5, linecolor='white',
            cbar_kws={'shrink': 0.8}, annot_kws={'size': 9}, ax=ax)
ax.set_title('Spearman Correlation Matrix', fontsize=13, fontweight='bold')
ax.tick_params(axis='x', labelsize=9, rotation=45)
ax.tick_params(axis='y', labelsize=9, rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(TMP, '19_Correlation_Heatmap.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% Cell 50
# Bước 1: Lọc bỏ SEVERITY = Unknown
df_ml = df_sev.copy()
print(f"df_sev (đã loại Unknown): {len(df_ml):,} rows")

# Bước 2: Chọn features
feature_cols_cat = ['CAUSE_IGN', 'HEAT_SOURC', 'PROP_USE', 'STRUC_TYPE',
                     'FIRE_SPRD', 'DETECTOR', 'AES_PRES', 'INC_TYPE', 'STATE']
feature_cols_num = ['RESPONSE_TIME']
print(f"\nFeatures categorical: {feature_cols_cat}")
print(f"Features numeric    : {feature_cols_num}")

print(f"\nMissing trước khi xử lý (categorical):")
for c in feature_cols_cat:
    n_miss = df_ml[c].isna().sum()
    print(f"  {c:<12}: {n_miss:>7,} ({n_miss/len(df_ml)*100:.1f}%)")

print(f"\nMissing trước khi xử lý (numeric):")
for c in feature_cols_num:
    n_miss = df_ml[c].isna().sum()
    print(f"  {c:<12}: {n_miss:>7,} ({n_miss/len(df_ml)*100:.1f}%)")

# Bước 3: Điền missing categorical bằng "Not Reported"
for c in feature_cols_cat:
    df_ml[c] = df_ml[c].astype(str)
    df_ml[c] = df_ml[c].replace('nan', 'Not Reported')
print(f"\nSau khi điền 'Not Reported', missing còn lại (categorical):")
for c in feature_cols_cat:
    print(f"  {c:<12}: {df_ml[c].isna().sum()}")

# Bước 3b: Điền missing numeric bằng median
for c in feature_cols_num:
    median_val = df_ml[c].median()
    n_miss = df_ml[c].isna().sum()
    df_ml[c] = df_ml[c].fillna(median_val)
    print(f"\n{c}: điền {n_miss:,} giá trị missing bằng median = {median_val:.2f}")

# Bước 4: Encode categorical bằng LabelEncoder
encoders = {}
for c in feature_cols_cat:
    le = LabelEncoder()
    df_ml[c + '_ENC'] = le.fit_transform(df_ml[c])
    encoders[c] = le
encoded_cols = [c + '_ENC' for c in feature_cols_cat]
print(f"\nĐã encode {len(encoded_cols)} features categorical:")
for c in feature_cols_cat:
    print(f"  {c:<12}: {len(encoders[c].classes_)} categories")

# Tổng hợp danh sách cột feature cuối cùng
all_feature_cols = encoded_cols + feature_cols_num
print(f"\nTổng số features đưa vào model: {len(all_feature_cols)}")
print(f"  -> {all_feature_cols}")

# Bước 5: Tạo target
df_ml['LOG_TOTAL_LOSS'] = np.log1p(df_ml['TOTAL_LOSS'])
print(f"\nTarget Regression: LOG_TOTAL_LOSS")
print(df_ml['LOG_TOTAL_LOSS'].describe())
print(f"\nTarget Classification: SEVERITY")
print(df_ml['SEVERITY'].value_counts())
print(f"\nTỷ lệ % :")
print((df_ml['SEVERITY'].value_counts(normalize=True) * 100).round(2))

# Bước 6: Train/Test split
X = df_ml[all_feature_cols]
y_reg = df_ml['LOG_TOTAL_LOSS']
y_clf = df_ml['SEVERITY']
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf,
    test_size=0.2, random_state=42, stratify=y_clf
)
print(f"\nTrain/Test split")
print(f"X_train: {X_train.shape[0]:,} rows | X_test: {X_test.shape[0]:,} rows")
print(f"\nPhân phối SEVERITY trong train:")
print(y_clf_train.value_counts(normalize=True).round(4) * 100)
print(f"\nPhân phối SEVERITY trong test:")
print(y_clf_test.value_counts(normalize=True).round(4) * 100)

# Visualization cho 7.1
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Số categories mỗi feature (chỉ categorical)
n_cats = {c: len(encoders[c].classes_) for c in feature_cols_cat}
n_cats_sorted = dict(sorted(n_cats.items(), key=lambda x: -x[1]))
ax1 = axes[0]
bars = ax1.bar(n_cats_sorted.keys(), n_cats_sorted.values(),
               color='#3498DB', edgecolor='white')
ax1.set_title('Số categories mỗi feature categorical sau encode', fontsize=12, fontweight='bold')
ax1.set_ylabel('Số categories')
ax1.tick_params(axis='x', rotation=45)
for bar, v in zip(bars, n_cats_sorted.values()):
    ax1.text(bar.get_x() + bar.get_width()/2, v + 2, f'{v}',
             ha='center', va='bottom', fontsize=9, fontweight='bold')
ax1.set_yscale('log')

# Plot 2: Phân phối SEVERITY train vs test
ax2 = axes[1]
train_pct = y_clf_train.value_counts(normalize=True).reindex(SEV_ORDER) * 100
test_pct  = y_clf_test.value_counts(normalize=True).reindex(SEV_ORDER) * 100
x = np.arange(len(SEV_ORDER))
width = 0.35
ax2.bar(x - width/2, train_pct, width, label=f'Train (n={len(y_clf_train):,})',
        color='#5DADE2', edgecolor='white')
ax2.bar(x + width/2, test_pct, width, label=f'Test (n={len(y_clf_test):,})',
        color='#F5B041', edgecolor='white')
ax2.set_title('Phân phối SEVERITY: Train vs Test', fontsize=12, fontweight='bold')
ax2.set_ylabel('Tỷ lệ (%)')
ax2.set_xticks(x)
ax2.set_xticklabels(SEV_ORDER)
ax2.legend()
for i, (tr, te) in enumerate(zip(train_pct, test_pct)):
    ax2.text(i - width/2, tr + 0.5, f'{tr:.1f}%', ha='center', fontsize=8)
    ax2.text(i + width/2, te + 0.5, f'{te:.1f}%', ha='center', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(TMP, '21_ML_Data_Prep.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% Cell 52
# Model 1: Logistic Regression
lr_clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr_clf.fit(X_train, y_clf_train)

# Model 2: Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=15,
                                 class_weight='balanced',
                                 random_state=42, n_jobs=-1)
rf_clf.fit(X_train, y_clf_train)

print("Đã train xong 2 mô hình:")
print("  1. Logistic Regression (class_weight='balanced')")
print("  2. Random Forest Classifier (n_estimators=100, max_depth=15, class_weight='balanced')")

# Predict trên test
pred_clf = {
    'Logistic Regression': lr_clf.predict(X_test),
    'Random Forest':       rf_clf.predict(X_test),
}

# %% Cell 53
# Metrics so sánh
rows = []
for model_name, y_pred in pred_clf.items():
    acc      = accuracy_score(y_clf_test, y_pred)
    bal_acc  = balanced_accuracy_score(y_clf_test, y_pred)
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
print("So sánh metrics: Logistic Regression vs Random Forest")
print(metrics_clf_df.to_string(index=False))

# Classification report chi tiết theo từng class
for model_name, y_pred in pred_clf.items():
    print(f"\nClassification Report — {model_name}")
    print(classification_report(y_clf_test, y_pred, labels=SEV_ORDER, digits=3))

# %% Cell 54
SEV_ORDER = ['No Loss', 'Minor', 'Moderate', 'Major']

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
plt.savefig(os.path.join(TMP, '22a_ConfusionMatrix.png'), dpi=150, bbox_inches='tight')
plt.show()

# ===== Feature Importance =====
importances = pd.Series(rf_clf.feature_importances_, index=all_feature_cols)
importances = importances.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(importances.index, importances.values, color='#9b59b6')
ax.set_title('Feature Importance — Random Forest Classifier (SEVERITY)',
              fontsize=12, fontweight='bold')
ax.set_xlabel('Importance')
for i, v in enumerate(importances.values):
    ax.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(TMP, '22b_FeatureImportance.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% Cell 57
# Chỉ train trên rows có TOTAL_LOSS > 0
mask_loss = df_ml['TOTAL_LOSS'] > 0
df_loss = df_ml[mask_loss]
print(f"Rows dùng train regression: {len(df_loss):,} ({mask_loss.mean()*100:.1f}%)")

X_loss = df_loss[all_feature_cols]
y_loss = df_loss['LOG_TOTAL_LOSS']
X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
    X_loss, y_loss, test_size=0.2, random_state=42
)

# Model 1: Linear Regression
lr_reg = LinearRegression()
lr_reg.fit(X_tr_r, y_tr_r)

# Model 2: Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=15,
                                random_state=42, n_jobs=-1)
rf_reg.fit(X_tr_r, y_tr_r)

print("Đã train xong 2 mô hình (chỉ trên rows có TOTAL_LOSS > 0):")
print("  1. Linear Regression")
print("  2. Random Forest Regressor (n_estimators=100, max_depth=15)")

# Predict
pred = {
    'Linear Regression': {
        'train': lr_reg.predict(X_tr_r),
        'test':  lr_reg.predict(X_te_r),
    },
    'Random Forest': {
        'train': rf_reg.predict(X_tr_r),
        'test':  rf_reg.predict(X_te_r),
    },
}

# %% Cell 58
rows = []
for model_name, p in pred.items():
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
print("So sánh metrics: Linear Regression vs Random Forest")
print("%table\n" + metrics_df.to_csv(index=False, sep="\t"))

# %% Cell 59
# Scatter Actual vs Predicted (test set, thang log)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, (model_name, p) in zip(axes, pred.items()):
    ax.scatter(y_te_r, p['test'], s=5, alpha=0.1, color='#3498DB')
    lims = [0, y_te_r.max()]
    ax.plot(lims, lims, 'r--', lw=1.5, label='Đường lý tưởng (y=x)')
    ax.set_title(f'{model_name}\nActual vs Predicted (test set)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Actual log(TOTAL_LOSS+1)')
    ax.set_ylabel('Predicted log(TOTAL_LOSS+1)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(TMP, '22_Regression_Actual_vs_Predicted.png'), dpi=150, bbox_inches='tight')
plt.show()

# Feature Importance (Random Forest)
importances = pd.Series(rf_reg.feature_importances_, index=all_feature_cols).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(importances.index, importances.values, color='#27AE60', edgecolor='white')
ax.set_title('Feature Importance — Random Forest Regressor (TOTAL_LOSS)', fontsize=12, fontweight='bold')
ax.set_xlabel('Importance')
for bar, v in zip(bars, importances.values):
    ax.text(v + 0.002, bar.get_y() + bar.get_height()/2, f'{v:.3f}',
            va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(TMP, '23_Regression_Feature_Importance.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% Cell 61
MODEL_DIR = os.path.join(TMP, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(lr_reg,  os.path.join(MODEL_DIR, 'lr_reg.pkl'))
joblib.dump(rf_reg,  os.path.join(MODEL_DIR, 'rf_reg.pkl'))
joblib.dump(lr_clf,  os.path.join(MODEL_DIR, 'lr_clf.pkl'))
joblib.dump(rf_clf,  os.path.join(MODEL_DIR, 'rf_clf.pkl'))
joblib.dump(encoders,         os.path.join(MODEL_DIR, 'encoders.pkl'))
joblib.dump(code_lookup,      os.path.join(MODEL_DIR, 'code_lookup.pkl'))
joblib.dump(all_feature_cols, os.path.join(MODEL_DIR, 'all_feature_cols.pkl'))
joblib.dump(text_options if 'text_options' in dir() else None, os.path.join(MODEL_DIR, 'text_options.pkl'))

print("Đã lưu model vào:", MODEL_DIR)
print(os.listdir(MODEL_DIR))

# %% Cell 63
MODEL_DIR = os.path.join(TMP, 'models')

lr_reg  = joblib.load(os.path.join(MODEL_DIR, 'lr_reg.pkl'))
rf_reg  = joblib.load(os.path.join(MODEL_DIR, 'rf_reg.pkl'))
lr_clf  = joblib.load(os.path.join(MODEL_DIR, 'lr_clf.pkl'))
rf_clf  = joblib.load(os.path.join(MODEL_DIR, 'rf_clf.pkl'))
encoders         = joblib.load(os.path.join(MODEL_DIR, 'encoders.pkl'))
code_lookup       = joblib.load(os.path.join(MODEL_DIR, 'code_lookup.pkl'))
all_feature_cols  = joblib.load(os.path.join(MODEL_DIR, 'all_feature_cols.pkl'))
text_options      = joblib.load(os.path.join(MODEL_DIR, 'text_options.pkl'))

feature_cols_cat = [c for c in text_options.keys()]
print("Đã load model. Features:", all_feature_cols)

# %% Cell 64
default_inputs = {
    c: ('Not Reported' if 'Not Reported' in text_options[c] else text_options[c][0])
    for c in feature_cols_cat
}
# RESPONSE_TIME: dùng giá trị mặc định cố định (median đã lưu sẵn, hoặc hardcode)
default_inputs['RESPONSE_TIME'] = 6.0  # median RESPONSE_TIME đã biết từ phần khảo sát (Chương 1)
print(f"RESPONSE_TIME mặc định: {default_inputs['RESPONSE_TIME']}")

z.z.angularBind("feature_cols_js", feature_cols_cat)
z.z.angularBind("text_options_js", {c: text_options[c] for c in feature_cols_cat})
z.z.angularBind("selected_inputs", default_inputs)
z.z.angularBind("predict_result", "Chưa có kết quả")
z.z.angularBind("selected_json", json.dumps(default_inputs))

# %% Cell 65
selected_raw = z.z.angular("selected_json")
selected = json.loads(selected_raw)
print("DEBUG selected:", selected)

X_new_dict = {
    c + '_ENC': code_lookup[c][str(selected[c]).strip()]
    for c in feature_cols_cat
}
X_new_dict['RESPONSE_TIME'] = float(selected['RESPONSE_TIME'])
X_new = pd.DataFrame([X_new_dict])[all_feature_cols]

# Classifier trước
pred_clf_lr = lr_clf.predict(X_new)[0]
pred_clf_rf = rf_clf.predict(X_new)[0]
proba_lr = dict(zip(lr_clf.classes_, lr_clf.predict_proba(X_new)[0].round(3)))
proba_rf = dict(zip(rf_clf.classes_, rf_clf.predict_proba(X_new)[0].round(3)))

# Regression sau (two-stage)
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
KẾT QUẢ DỰ ĐOÁN
{'─' * 40}

⚠️  MỨC ĐỘ NGHIÊM TRỌNG (SEVERITY)
  • Linear Regression  : {pred_clf_lr}
  • Random Forest      : {pred_clf_rf}

📊 XÁC SUẤT - Linear Regression
{fmt_proba(proba_lr)}

📊 XÁC SUẤT - Random Forest
{fmt_proba(proba_rf)}

💰 TỔNG THIỆT HẠI (TOTAL LOSS)
  • Linear Regression  : ${pred_usd_lr:>12,.0f}
  • Random Forest      : ${pred_usd_rf:>12,.0f}
  ⚠️ Chỉ dự đoán khi SEVERITY ≠ No Loss, trường hợp No Loss mặc định $0
""".strip()

z.z.angularBind("predict_result", result)
print(result)
