\# Advanced Alpha Pipeline



高级 A 股预测与回测系统，基于 \*\*沪深 A 股 + 创业板\*\*，支持 \*\*多数据源（TuShare + AKShare）\*\* 和 \*\*多模型集成（LightGBM / XGBoost / CatBoost / PyTorch MLP / 阿里云子模型）\*\*。  



\## ✨ 功能亮点

\- \*\*因子工程\*\*：行业/概念强度、中性化、IC/ICIR 指标

\- \*\*回测执行\*\*：T+0、2日、3日、5日持有，支持止盈止损

\- \*\*阈值搜索\*\*：自动搜索 prob 阈值 + TOPK 以优化夏普率

\- \*\*模型融合\*\*：LightGBM、XGBoost、CatBoost、MLP + 阿里云远程子模型

\- \*\*前台看板\*\*：基于 Streamlit，可视化净值曲线、热力图、预测结果、行业热点等

\- \*\*结果输出\*\*：预测 CSV、日志 CSV、报表 HTML、图片（曲线/单股跟踪）



\## 📦 安装依赖

在本地运行前，先安装所需依赖：

```bash

pip install -r requirements.txt



