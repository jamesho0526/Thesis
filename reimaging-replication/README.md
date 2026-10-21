# (Re-)Imag(in)ing Price Trends 复现

本仓库复现了 Jiang、Kelly 与 Xiu（2020）提出的《(Re-)Imag(in)ing Price Trends》一文：利用机器学习从价格图像中自动学习最能预测未来收益的模式，而不是手工设定动量或反转等先验信号。I5/R5 使用 64×15 像素图像；旧的 20 日资料仍使用 64×60。Windows + AMD GPU 可通过 DirectML 加速训练。

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
  - `models.py`：TradingCNN（3 个 ConvBlock + 全连接）与初始化。
  - `data_utils.py`：内存映射图像数据集、时间切片、指标持久化。
  - `i5_r5.py`：按论文 I5/R5 规则生成 64×15 OHLCV 图像与 5 日标签。
  - `cn_data.py`：从中国市场原始日度数据构建图像与标签。
  - `training.py`、`backtest.py`、`visualization.py`：训练循环、十等分回测、图表输出。
- `scripts/`：命令行脚本，涵盖数据准备、训练、评估、跨市场迁移等流程。
- `data/paper_replication/`：論文復現資料，例如固定 2000 S&P 名單與篩選後資料。
- `data/extensions/`：延伸實作資料，例如中國市場、Yahoo robustness 與 futures。
- `experiments/paper_replication/`：Table IV 復現模型、checkpoint 與結果。
- `experiments/extensions/`：中國市場、跨市場與其他延伸模型結果。

## 环境准备
推荐使用 Python ≥ 3.9；本專案以 `uv` 管理虛擬環境與依賴：

```powershell
uv venv
.\.venv\Scripts\activate
uv pip install torch pandas numpy matplotlib pyarrow
```

脚本默认使用 CPU，但若检测到 GPU 会自动切换到 CUDA。Feather 文件读写依赖 `pyarrow`。

## 数据说明
1. **作者提供的美股图像 (`img_data/monthly_20d/…`)**  
   需向原作者申请，目录里包含逐年 `*_images.dat` 与对应的 `*_labels_w_delay.feather`（20 日窗口示例）。将其存放在仓库并把路径传给脚本的 `--data-root` 参数即可。
2. **中国市场原始数据**  
   `scripts/cn_prepare_data.py` 期望读取类似 CSMAR `TRD_DalyrYYYY.xlsx` 的日度文件。脚本会：
   - 汇总 Excel/CSV 为 `cn_tabular.csv`；
   - 计算复权价、20 日均线、未来收益；
   - 渲染 `window_days × 3` 像素宽的 OHLC+成交量图像，并保存为 `.dat` + `.feather`。

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
  --epochs 15 --batch-size 64
```

模型使用 1993–2000 訓練/驗證，2001–2019 做固定模型的樣本外測試。
訓練時會顯示 batch、epoch、loss、accuracy 與 checkpoint 更新狀態。

### 3. 原作者影像資料訓練（可選）
```bash
python scripts/train_author.py \
  --data-root data/paper_replication/author_images/monthly_20d \
  --output-dir experiments/paper_replication/table_iv_i5_r5 \
  --epochs 15 --batch-size 128
```
脚本会使用 1993–2000 年数据训练，并在 2001–2019 年样本上报告测试精度；`experiments/author_full/` 将包含最优权重、训练日志与损失曲线。

### 4. 回测美股 H–L 组合
```bash
python scripts/eval_author.py \
  --data-root data/paper_replication/author_images/monthly_20d \
  --checkpoint experiments/paper_replication/table_iv_i5_r5/best_author_model.pth \
  --output-dir experiments/paper_replication/table_iv_i5_r5/eval
```
输出包括逐股票预测 (`predictions_2001_2019.csv`)、分类指标 (`metrics_*.json`)、十等分高减低序列与累计收益图。

### 5. 期貨延伸：I5/R5

期貨資料完全放在 `extensions`，不列入論文復現結果：

```powershell
uv run python .\scripts\build_i5_r5.py `
  --source futures `
  --source-dir ..\data\extensions\futures\futures_full_1day `
  --output-dir ..\data\extensions\futures\images_i5_r5

uv run python .\scripts\train_i5_r5.py `
  --data-root ..\data\extensions\futures\images_i5_r5 `
  --output-dir .\experiments\extensions\futures_i5_r5 `
  --epochs 15 --batch-size 64
```

期貨資料的上市年份不同，腳本會只輸出實際存在且落在
1993-01-01 至 2019-12-31 的樣本；沒有足夠 5+5 交易日資料的商品會跳過。

### 6. 构建并训练中国市场模型
```bash
# 生成 20 日窗图像与标签
python scripts/cn_prepare_data.py \
  --data-root /path/to/TRD_Dalyr_excels \
  --image-output-dir data/extensions/china_images \
  --prefix cn_20d --window 20 --future 20

# 训练 CNN
python scripts/cn_train.py \
  --data-root data/extensions/china_images \
  --glob-pattern 'cn_*_images.dat' \
  --label-suffix '_labels.feather' \
  --output-dir experiments/extensions/china \
  --epochs 25
```

### 5. 跨市场迁移实验
```bash
python scripts/cross_market_eval.py \
  --cn-data-root data/extensions/china_images \
  --author-checkpoint experiments/paper_replication/table_iv_i5_r5/best_author_model.pth \
  --cn-checkpoint experiments/extensions/china/best_cn_model.pth \
  --output-dir experiments/extensions/cross_market
```
脚本会分别把美股训练模型与中国本地模型应用到相同的 CN 图像上，比较分类表现与 H–L 组合。

## 关键结果

| 实验 | 分类准确率 | 上涨样本准确率 | 下跌样本准确率 | 平均月度 H–L spread | 累计 spread（样本期） | 数据来源 |
| --- | --- | --- | --- | --- | --- | --- |
| 美股 20d→20d（作者数据） | 0.523 | 0.516 | 0.532 | 1.56% | ≈3.55（2001–2019） | CRSP (作者图像) |
| 美股模型迁移至中国 | 0.499 | 0.670 | 0.315 | 0.007% | 0.016 | CSMAR 20 日窗 |
| 中国本地训练模型 | 0.589 | 0.660 | 0.589 | 7.48% | 17.05 | 同上 |

> 注：spread 为各期高组与低组平均收益差；美股回测为月度截面，迁移实验样本长度与中国模型一致（228 期）。

与论文主结论一致，我们复现到的美股测试准确率约 52.6%，高减低月度累计收益保持稳健增长。而在跨市场实验中，美股模型在中国市场几乎无增益（与论文强调的“需考虑市场结构差异”相符），但本地训练可以把准确率提升到 58.9%，高减低平均差达到 7.5%。

## 下一步可以做什么？
1. 继续对 5 日/60 日窗口与不同持有期的配置运行脚本，检验多尺度可迁移性。
2. 结合传统动量、反转、流动性指标做线性或树模型，评估 CNN 信号的增量信息含量。
3. 把十等分回测扩展到交易成本、换手约束与价值加权组合，进一步靠近实盘假设。
