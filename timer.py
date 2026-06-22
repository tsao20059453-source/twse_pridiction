import time, warnings
warnings.filterwarnings('ignore')
from Pipeline_clean import *

config = replace(CONFIG, stock_no='ALL', diagnostic_scope='common', display_mode='research')
print('開始計時...')
t0 = time.time()
stock_list = fetch_twse_top_turnover_stocks(limit=30)
t1 = time.time()
print(f'抓取 Top 30 清單：{t1-t0:.1f}s')
symbols = stock_list['symbol'].astype(str).tolist()
combined = fetch_and_prepare_multi_stock_data_with_config(symbols=symbols, base_config=config)
t2 = time.time()
print(f'抓取與特徵工程：{t2-t1:.1f}s')
pipeline, feature_cols, metrics = train_model(data=combined, target_col='target_safety_10d_cs_rank', config=config)
t3 = time.time()
print(f'模型訓練：{t3-t2:.1f}s')
print(f'總時間：{t3-t0:.1f}s')
