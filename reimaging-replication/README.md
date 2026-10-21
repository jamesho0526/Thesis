# (Re-)Imag(in)ing Price Trends 复现

本仓库聚焦 Jiang、Kelly 与 Xiu 的《(Re-)Imag(in)ing Price Trends》I5/R5 简化复现，以及将 S&P 500 模型冻结后迁移到期货的延伸实验。I5/R5 使用 32×15 像素图像；Windows + AMD GPU 通过 DirectML 加速训练。

## 论文精要
- **核心理念**：把过去的价格轨迹渲染为图像（包含 OHLC、均线与成交量条），用 CNN 自由学习最具预测力的形态，而非预设动量、反转等因子。
- **数据与训练**：基于 1993–2019 年 CRSP 日度数据制作 5/20/60 日窗图像，预测未来 20/60 日收益，使用交叉熵损失、Adam（lr=1e-5）、批归一化、Dropout 与早停。
- **主要发现**：
  - CNN 在所有窗口上都能把个股方向预测准确率提升到约 53%，显著优于动量、月度或周度反转等传统指标。
  - 以 CNN 预测值构建的高减低（H–L）组合 Sharpe 可达 2.4（月度）及 7.2（周度），同时换手率与短期反转相当。
  - 所获形态具有跨频率与跨市场的可迁移性：用日度学到的模式预测月度/季度数据依然有效，美国模型迁移到 26 个海外市场也能带来 Sharpe 提升 0.4。

## 仓库结构
- `paper_text.txt`、`Xiu-Re-Imagining-Price-Trends.pdf`：论文全文，方便撰写或查阅。
- `src/`：可复用模块
  - `models.py`：I5 两个 ConvBlock（64 → 128）与初始化。
  - `data_utils.py`：内存映射图像数据集、时间切片、指标持久化。
  - `i5_r5.py`：按论文 I5/R5 规则生成 32×15 OHLCV 图像与 5 日标签。
  - `device.py`：选择 DirectML GPU1（RX 9070 XT）。
  - `training.py`、`visualization.py`：训练循环与图表输出。
- `scripts/`：S&P 建置/训练、期货建置、S&P→期货推论与资料清洗脚本。
- `data/paper_replication/`：論文復現資料，例如固定 2000 S&P 名單與篩選後資料。
- `data/extensions/`：期貨延伸資料與推論輸入。
- `experiments/paper_replication/`：Table IV 復現模型、checkpoint 與結果。
- `experiments/extensions/`：期貨模型與 S&P→期貨推論結果。

## 环境准备
推荐使用 Python ≥ 3.9；本專案以 `uv` 管理虛擬環境與依賴：

```powershell
uv venv
.\.venv\Scripts\activate
uv pip install torch pandas numpy matplotlib pyarrow
```

腳本預設使用 DirectML GPU1（RX 9070 XT）；可用 `$env:DIRECTML_DEVICE_ID="0"` 切換到內顯。Feather 文件讀寫依賴 `pyarrow`。

## 数据说明
1. **S&P 500 简化复现数据**：`data\paper_replication\sp500\filtered_1day`
2. **期货延伸数据**：`data\extensions\futures\futures_full_1day`
3. **中间 I5/R5 影像**：分别位于 `data\paper_replication` 与 `data\extensions\futures`

## 复现流程
### 1. 快速冒烟（可选）
任一训练脚本添加 `--quick-test` 会限制样本与轮数，方便检验依赖是否正确。

### 2. 簡化論文復現：S&P 500 I5/R5

這條流程只使用現有 `data\paper_replication\sp500\filtered_1day` 中有
1993–2019 資料的股票；它是簡化復現，不做論文 CRSP 的動態市值排名。

```powershell
uv run python .\scripts\build_i5_r5.py `
  --source sp500 `
  --source-dir ..\data\paper_replication\sp500\filtered_1day `
  --output-dir ..\data\paper_replication\sp500\images_i5_r5

uv run python .\scripts\train_i5_r5.py `
  --data-root ..\data\paper_replication\sp500\images_i5_r5 `
  --output-dir .\experiments\paper_replication\sp500_i5_r5 `
  --epochs 15 --batch-size 128 --seeds 11,22,33,44,55
```

模型使用 1993–2000 訓練/驗證，2001–2019 做固定模型的樣本外測試。
訓練時會顯示 batch、epoch、loss、accuracy 與 checkpoint 更新狀態。

### 3. 期貨延伸：I5/R5

期貨資料完全放在 `extensions`，不列入論文復現結果：

```powershell
uv run python .\scripts\build_i5_r5.py `
  --source futures `
  --source-dir ..\data\extensions\futures\futures_full_1day `
  --output-dir ..\data\extensions\futures\images_i5_r5

uv run python .\scripts\train_i5_r5.py `
  --data-root ..\data\extensions\futures\images_i5_r5 `
  --output-dir .\experiments\extensions\futures_i5_r5 `
  --epochs 15 --batch-size 128 --seeds 11,22,33,44,55
```

期貨資料的上市年份不同，腳本會只輸出實際存在且落在
1993-01-01 至 2019-12-31 的樣本；沒有足夠 5+5 交易日資料的商品會跳過。

### 4. S&P 模型直接预测期货

```powershell
uv run python .\scripts\predict_futures_with_sp500.py `
  --data-root ..\data\extensions\futures\images_i5_r5 `
  --checkpoint-dir .\experiments\paper_replication\sp500_i5_r5 `
  --output-dir .\experiments\extensions\futures_with_sp500_model
```

這是 frozen transfer inference，不使用期貨資料更新權重。結果包含方向準確率、
每商品報酬與按日期等權 portfolio 指標。

## 下一步可以做什么？
1. 继续对 5 日/60 日窗口与不同持有期的配置运行脚本，检验多尺度可迁移性。
2. 结合传统动量、反转、流动性指标做线性或树模型，评估 CNN 信号的增量信息含量。
3. 把十等分回测扩展到交易成本、换手约束与价值加权组合，进一步靠近实盘假设。
