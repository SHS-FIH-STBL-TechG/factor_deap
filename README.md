# 因子挖掘项目说明（factor_deap）

> 使用遗传算法（DEAP）+ 多大模型（DeepSeek / 通义千问）自动生成并筛选日频量化因子，用于单标的日 K 数据的因子研究。

---

## 1. 项目简介

本项目的目标是：

1. **读取日 K 线 CSV 数据**（单标的），字段包括：  
   `交易日期, 开盘点位, 最高点位, 最低点位, 收盘价, 涨跌, 涨跌幅(%), 开始日累计涨跌, 开始日累计涨跌幅, 成交量(万股), 成交额(万元), 持仓量`
2. **用大模型自动设计因子**（DeepSeek + 通义千问），将因子公式保存为可复用的 JSON 文件。
3. **对 AI 因子做系统化打分**：  
   - 穷举 2~5 个因子的组合；  
   - 计算：  
     - 全样本 Pearson 相关系数 `corr_full`  
     - 滑窗 IC 序列（252 日滚动）  
     - `|IC|` 均值、`IC` 均值、`IC` 标准差  
   - 按条件筛选：`|IC|均值 > 0.1` 或 `corr_full > 0.1`，若无则输出 `|IC|均值` 最大的组合。
4. 后续可以在此基础上继续接入**回测框架**和**Markdown 报告生成**。

---

## 2. 主要功能模块

### 2.1 GA 因子选择（早期版本）

- 文件：`src/factor_ga.py`
- 作用：使用 DEAP 做简单的遗传算法搜索，在一批候选因子中寻找与 `未来1日涨跌幅` 相关性较高的组合。
- 后续主力将逐步转移到 **AI 自动生成因子 + 穷举 IC 扫描** 这条线。

### 2.2 AI 因子生成模块

- 文件：`src/ai_factor_ideation.py`
- 依赖：`src/ai_clients/` 下的各个大模型客户端：
  - `deepseek_client.py`（使用 `deepseek-reasoner` 推理模型）
  - `qwen_client.py`（使用 `qwen-max` 推理模型）
  - （可选）`gemini_client.py`（当前可以不启用）
- 核心逻辑：
  1. 读取已有的 AI 因子定义（`factors_generated.json`）。
  2. 给每个大模型下发统一 Prompt，让其在**已有因子名称之外**继续设计新的因子。
  3. 模型必须返回严格的 **JSON 数组**，每个元素包含：
     - `name`：因子名称，必须以 `因子_` 开头；
     - `description`：因子含义说明；
     - `code`：一行 pandas 代码，如 `df["因子_xxx"] = ...`。
  4. 本地对这些因子做一次“自检”（在虚拟 df 上执行一遍），无法执行或不产生同名列的因子会被删除。
  5. 合并新旧因子后写回：`src/factors_generated.json`。

### 2.3 AI 因子落地与使用

- 文件：`src/factors_generated.py`
- 作用：
  - 从 `factors_generated.json` 中读取最新的因子定义；
  - 提供：
    - `load_generated_factors()`：返回因子列表（`[{name, description, code}, ...]`）；
    - `add_generated_factors(df)`：在实际的行情 `DataFrame` 上执行所有因子代码，生成对应列。

### 2.4 AI 因子组合 IC 扫描模块

- 文件：`src/ai_factor_ic_scan.py`
- 输入：
  - `data/` 目录下的所有 `*.csv`（单标的日 K 线数据）；
  - `src/factors_generated.json` 中定义的 AI 因子。
- 核心流程：
  1. 对每个 CSV：
     - 读入数据，转换数值列；
     - 生成目标列：`未来1日涨跌幅 = "涨跌幅(%)".shift(-1)`；
     - 生成手工基础因子（振幅、成交额变化率等）；
     - 调用 `add_generated_factors(df)` 加上 AI 因子。
  2. 从 JSON 因子列表中筛选出**在 df.columns 中真实存在的 AI 因子**（自动过滤代码执行失败的因子）。
  3. 穷举 2~K 个因子组合：
     - `K` 当前设定为最大 `5`（`MAX_COMBO_K`），可配置；
     - 每个标的最多评估 `MAX_COMBOS_PER_SYMBOL` 个组合（缺省 `100000`，可配置）。
  4. 对每个组合：
     - 对因子做 z-score 标准化；
     - 取多因子简单平均作为“组合因子”；
     - 计算：
       - 全样本 Pearson 相关：`corr_full`；
       - 252 日滑窗 IC 序列：`IC_t = corr(组合因子, 未来1日涨跌幅)`（在每个窗口上）；
       - `|IC|均值`、`IC均值`、`IC标准差`。
  5. 结果筛选逻辑：
     - 条件：`|IC|均值 > ic_threshold` **或** `corr_full > corr_threshold`（默认阈值都是 `0.1`）；
     - 若有满足条件的组合：
       - 按 `|IC|均值` 从大到小排序，日志中展示 Top N（默认前 10 个）；
     - 若一个也没有：
       - 输出 `|IC|均值` 最大的组合（作为当前最优候选）。
  6. 所有 `(标的, 因子组合)` 的打分结果会缓存到：
     - `results/factor_combo_cache.json`  
       下次遇到同一个组合就不再重复计算。

### 2.5 日志模块

- 文件：`src/logging_utils.py`
- 特点：
  - 所有模块通过 `get_logger(name)` 获取 logger；
  - 所有日志写入统一目录：`logs/`；
  - 日志文件名格式：`log_YYMMDDHHMMSS.log`  
    例如：`log_251128155137.log`；
  - 每一行日志格式：
    ```text
    时间 [级别] [logger名] [函数名] 消息
    ```
  - 同时输出到控制台 + 日志文件。

---

## 3. 目录结构说明

```text
factor_deap/
├─ data/                         # 放 K 线 CSV 的目录（你自己的数据）
│   └─ 000001.SH-行情统计-20251117.csv
├─ results/
│   └─ factor_combo_cache.json   # 因子组合 IC 结果缓存
├─ logs/
│   └─ log_YYMMDDHHMMSS.log      # 每次运行生成一个新日志文件
├─ src/
│   ├─ run_all.py                # 一键运行：先 AI 生成因子，再扫描组合
│   ├─ factor_ga.py              # GA 版因子选择（早期版本，可选用）
│   ├─ ai_factor_ideation.py     # AI 因子生成主脚本
│   ├─ ai_factor_ic_scan.py      # AI 因子组合 IC 扫描主脚本
│   ├─ factors_generated.json    # AI 生成因子定义（name / description / code）
│   ├─ factors_generated.py      # 负责加载/执行 JSON 因子
│   ├─ logging_utils.py          # 统一日志配置
│   └─ ai_clients/
│        ├─ deepseek_client.py   # 封装 DeepSeek 推理模型调用（deepseek-reasoner）
│        ├─ qwen_client.py       # 封装通义千问推理模型调用（qwen-max）
│        └─ gemini_client.py     # 封装 Gemini（当前可选，不一定启用）
└─ .venv/                        # Python 虚拟环境（PyCharm 自动管理）
```



---

## 4. 环境准备

### 4.1 Python 版本

- 推荐使用 **Python 3.11**。
- 不建议用 3.14（部分三方库兼容性问题多）。

### 4.2 创建虚拟环境（PyCharm 中操作）

1. 打开 PyCharm（中文版）。
2. `File` → `Open...` → 打开项目根目录 `factor_deap`。
3. 进入：  
   `File` → `Settings` → `项目: factor_deap` → `Python 解释器`。
4. 右上角齿轮图标 → `添加...`：
   - 选择：`Existing environment`（已有解释器） 或 `Virtualenv`；
   - 使用你安装的 `Python 3.11` 可执行文件；
   - 虚拟环境路径可以设为：`F:\work\factor_deap\.venv`。

### 4.3 安装依赖

在 PyCharm 右下角或 Terminal 中，激活虚拟环境后执行（示例）：

```bash
pip install pandas numpy deap requests matplotlib
```

后续如有新增库，再按实际需要补。

---

## 5. API Key 配置

### 5.1 DeepSeek

- 环境变量：`DEEPSEEK_API_KEY`
- 使用的模型：`deepseek-reasoner`

Windows PowerShell 临时设置示例：

```powershell
$env:DEEPSEEK_API_KEY = "你的_deepseek_api_key"
```

### 5.2 通义千问（DashScope）

- 环境变量：
  - 优先读取 `DASHSCOPE_API_KEY`；
  - 若没有，则尝试读取 `QWEN_API_KEY`。
- 使用的模型：`qwen-max`

PowerShell 临时设置示例：

```powershell
$env:DASHSCOPE_API_KEY = "你的_dashscope_api_key"
```

> 也可以在系统“环境变量”里直接配置为永久变量。

---

## 6. 数据格式要求（CSV）

- 放在项目根目录下的 `data/` 目录；
- 每个文件代表**单一标的**的日 K 线数据；
- 必须包含以下列（列名需要一致）：

```text
交易日期, 开盘点位, 最高点位, 最低点位, 收盘价,
涨跌, 涨跌幅(%), 开始日累计涨跌, 开始日累计涨跌幅,
成交量(万股), 成交额(万元), 持仓量
```

- 日期列：`交易日期` 会被自动解析为 `datetime64[ns]`；
- 数字列中若有 `,` 或 `%` 或制表符，程序会自动清理；
- 目标变量：  
  - `未来1日涨跌幅 = 当前行 "涨跌幅(%)" 向下平移一行`（`shift(-1)`）。

---

## 7. 使用方式

### 7.1 一键在线运行（有网）

当你希望**一次性完成：AI 生成因子 + 穷举组合 IC**，可以运行：

```bash
python src/run_all.py
```

流程：

1. Step 1：调用 DeepSeek + 通义千问，生成 / 更新因子，写入 `factors_generated.json`；
2. Step 2：对 `data/` 下的所有 CSV：
   - 落地所有 AI 因子；
   - 穷举组合并计算 `corr_full` + 滑窗 `IC`;
   - 将结果写入缓存 `results/factor_combo_cache.json`；
3. 同时日志写入 `logs/log_YYMMDDHHMMSS.log`。

---

### 7.2 拆步运行：白天有网生成因子，晚上离线计算 IC

**白天（有网：AI 生成因子）：**

```bash
python src/ai_factor_ideation.py
```

- 只做因子设计，不做组合扫描；
- 会更新 `src/factors_generated.json`。

**晚上（断网：只做组合扫描）：**

```bash
python src/ai_factor_ic_scan.py
```

- 不调用任何 API；
- 使用当前已有的 `factors_generated.json` + `data/*.csv` 做穷举计算；
- 更新 `results/factor_combo_cache.json`。

> 以后如果增加“回测 + 报告生成”脚本（例如 `generate_reports.py`），可以再作为第三步单独跑。

---

## 8. 结果指标说明

在日志中会看到类似：

```text
#7 因子组合: ['因子_实体中心偏移', '因子_成交额强度', '因子_累计收益动量', '因子_累计收益稳定性']
   | corr_full=-0.0359 | |IC|均值=0.1070 | IC均值=-0.1006 | IC std=0.0796
```

含义如下：

- **`corr_full`**：  
  - 全样本 Pearson 相关系数；  
  - 对“组合因子 vs 未来1日涨跌幅”在整个样本期上算一个相关值；  
  - 接近 0 表示长期线性关系不强。

- **`IC_t`（信息系数）**：  
  - 用 252 日滚动窗口，在每个窗口上计算：  
    `IC_t = corr(组合因子, 未来1日涨跌幅)`；  
  - 得到一串时间序列。

- **`|IC|均值`**：  
  - `mean(|IC_t|)`  
  - 表示“不同时间段上，该组合因子与收益的相关性强度平均值”（不看方向，只看强弱）；  
  - 你目前的筛选门槛是 `> 0.1`。

- **`IC均值`**：  
  - `mean(IC_t)`  
  - 表示方向性的平均：  
    - 正值：组合因子高 → 未来涨的预期更大（顺势因子）；  
    - 负值：组合因子高 → 未来跌的预期更大（反向 / 反转因子）。

- **`IC std`**：  
  - `std(IC_t)`  
  - 表示不同时间窗口之间，IC 的波动程度；  
  - 越小表示不同时间段表现越稳定。

---

## 9. 常见问题 / 调参建议

1. **“跑得太慢 / 等不动”**  
   - 当前组合枚举是：
     - 最大因子个数：`MAX_COMBO_K = 5`；
     - 每个标的最多评估组合数：`MAX_COMBOS_PER_SYMBOL`（默认 100000，可调小到 20000 ~ 50000）。
   - 如果 AI 因子数量继续增加，可以适当调小这两个参数。

2. **“某些因子总是算不出来”**  
   - 生成因子时会先做一次“虚拟 df 自检”，明显执行失败的因子会被过滤；
   - 真正执行时若出现错误，会在日志中打印：
     ```text
     WARNING: 计算因子 因子_xxx 失败: 具体错误信息
     ```
   - 可以据此手工检查因子定义，或让 AI 重新生成。

3. **“想只用 DeepSeek 或只用通义千问”**  
   - 在 `ai_factor_ideation.py` 中，有一段模型列表：
     ```python
     models_in_order = [
         ("Gemini", "gemini"),
         ("DeepSeek", "deepseek"),
         ("Qwen", "qwen"),
     ]
     ```
   - 可以按需注释掉不想使用的模型。

4. **“想改滑窗长度 / IC 阈值 / 相关阈值”**  
   - 在 `ai_factor_ic_scan.py` → `scan_ai_factor_combinations_for_df` 被调用时：  
     - `window=252` → 滑窗长度；  
     - `corr_threshold=0.1`；  
     - `ic_threshold=0.1`；  
   - 可以按需要修改。

---

## 10. 后续可以扩展的方向

- 加一个 `generate_reports.py`：
  - 读取 `factor_combo_cache.json`；
  - 对每个标的 / 每个 Top 因子组合：
    - 用图表展示 IC 序列、收益分布；
    - 调用大模型写一份 Markdown 报告（因子含义 + 市场风格 + 注意事项）；
  - 输出到 `reports/xxx.md`。
- 引入简单回测框架：
  - 根据因子组合构建打分；
  - 按分位数分组回测（高分组 / 低分组）；
  - 计算收益曲线、夏普、最大回撤等指标。

---
### （新增）：
## IC 报告生成模块（ic_report.py）

`ic_report.py` 用来基于已穷举好的因子组合，生成偏向研究报告风格的 **IC 分析报告**（Markdown + 图表）。

它会：

- 从 `results/factor_combo_cache.json.gz` 中读取每个标的的因子组合评估结果；
- 重新加载对应的 CSV 数据，复算组合因子与目标（`未来1日涨跌幅`）的：
  - 全样本 Pearson 相关系数 `corr_full`
  - 滑动窗口 IC 序列（默认窗口 252 日）
  - IC 绝对值均值 `ic_abs_mean`、IC 均值 `ic_mean`、IC 标准差 `ic_std`
- 生成包含图表与文字说明的 Markdown 报告。

---

### 1. 前置条件

运行 `ic_report.py` 之前需要：

1. 已经跑过一次穷举扫描（`run_all.py` 的 Step 2）并生成缓存：  
   - `results/factor_combo_cache.json.gz`
2. `data/` 目录下存在对应标的的 CSV 文件（例如：`data/000001.SH-行情统计-20251117.csv`），格式与主流程一致。
3. 已经有 `factors_generated.json` / `factors_generated.py`，确保可以复现 AI 因子。

---

### 2. 基本用法

在项目根目录下（包含 `src/`、`data/`）运行：

```bash
# Windows 示例
F:\work\factor_deap\.venv\Scripts\python.exe F:\work\factor_deap\src\ic_report.py
