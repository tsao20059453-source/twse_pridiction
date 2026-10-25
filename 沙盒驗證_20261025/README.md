# 沙盒重跑驗證（2026-10-25）

目的：確認 `桌面\專題` 的程式碼就是 9/21 論文最終版。

- 環境：Claude 雲端沙盒，Python 3.13，套件版本與 requirements.txt 相同（scikit-learn 1.8.0、pandas 3.0.3、numpy 2.4.5）。
- 資料：`data\raw` 快取（TWSE 網站在沙盒中無法連線，全部使用快取），股票池使用 `output\cache\twse_top_turnover_30_latest.csv`（9/21 凍結）。
- 程式內「今天」以 time-machine 固定為 2026-09-21 11:15，與論文最終版執行時間點相同；程式碼本身未做任何修改。

## 結果
- Mode 3（Top 30 投組回測）：總報酬 120.62%、CAGR 59.45%、MDD −10.64%、Sharpe 2.64、Calmar 5.59，與論文第八章完全相同。
- 逐股績效表 `backtest_batch_TWSE_20260921_112414_perf.csv` 與原檔 `backtest_batch_TWSE_20260921_112049_perf.csv` 逐格比對：570 格 0 格不同。
- Mode 1（2330 個股診斷）：safety Spearman 0.2377、return Spearman 0.0911，與論文 IC ≈ 0.24／0.09 一致。

檔案：`run_mode3_backtest_top30.log`、`run_mode1_2330.log`（已移除連不上 TWSE 的重複警告行）。
