# src/factor_ga.py
from logging_utils import get_logger
from pathlib import Path

# 自动生成因子模块：如果不存在，用一个空实现兜底
try:
    from factors_generated import add_generated_factors
except ImportError:
    def add_generated_factors(df):
        logger.warning("WARNING: 未找到 factors_generated.py，跳过自动生成因子。")
        return df
import os
import random
from typing import List

import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from ai_clients.qwen_client import ask_qwen
from pathlib import Path
from datetime import datetime

logger = get_logger("factor_ga")

# ========== 1. 读取 K 线 CSV，并构造因子 ==========
def load_kline_data() -> pd.DataFrame:
    """
    从 data/kline.csv 读取 K 线数据：
    1. 自动处理 utf-8 / gbk 编码
    2. 把数字列里的逗号、百分号等去掉，强制转成 float
    3. 构造简单因子 + 未来1日涨跌幅
    """
    # 1. 拼出 CSV 路径：项目根目录 / data / kline.csv
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "data", "000001.SH-行情统计-20251117.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到 CSV 文件：{csv_path}")

    # 2. 先尝试 utf-8，不行再用 gbk（国内软件常用）
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="gbk")

    # 3. 检查需要的列是否都有
    expected_cols = [
        "交易日期", "开盘点位", "最高点位", "最低点位",
        "收盘价", "涨跌", "涨跌幅(%)", "开始日累计涨跌",
        "开始日累计涨跌幅", "成交量(万股)", "成交额(万元)", "持仓量",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少列：{missing}\n当前列名：{df.columns.tolist()}")

    # 4. 把“数字应该是数字”的那些列，全部强制转成 float
    numeric_cols = [
        "开盘点位", "最高点位", "最低点位",
        "收盘价", "涨跌", "涨跌幅(%)", "开始日累计涨跌",
        "开始日累计涨跌幅", "成交量(万股)", "成交额(万元)", "持仓量",
    ]

    for col in numeric_cols:
        # 先全部当成字符串处理：去掉逗号、百分号、空格等
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)   # 去千分位逗号：1,234.56 -> 1234.56
            .str.replace("%", "", regex=False)   # 去百分号：3.45% -> 3.45
            .str.replace("\t", "", regex=False)  # 去制表符
            .str.strip()                         # 去前后空格
        )
        # 再转成数值，转不了的变成 NaN
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5. 交易日期转成真正的日期类型，方便排序
    df["交易日期"] = pd.to_datetime(df["交易日期"], errors="coerce")

    # 6. 按交易日期排序
    df = df.sort_values("交易日期").reset_index(drop=True)

    # 7. 目标：未来1日涨跌幅（直接用“涨跌幅(%)”向后平移一行）
    df["未来1日涨跌幅"] = df["涨跌幅(%)"].shift(-1)

    # 8. 构造几个简单因子（这里用的都是刚刚已经转成 float 的列）
    # 振幅：(最高 - 最低) / 收盘价
    df["因子_振幅"] = (df["最高点位"] - df["最低点位"]) / df["收盘价"]

    # K 线实体占收盘价比例：|收盘 - 开盘| / 收盘价
    df["因子_实体占振幅"] = (df["收盘价"] - df["开盘点位"]).abs() / df["收盘价"]

    # 成交额变化率
    df["因子_成交额变化率"] = df["成交额(万元)"].pct_change(fill_method=None)

    # 成交量变化率
    df["因子_成交量变化率"] = df["成交量(万股)"].pct_change(fill_method=None)

    # 当日涨跌幅作为一个简单动量因子
    df["因子_当日涨跌幅"] = df["涨跌幅(%)"]

    # 9. 调用自动生成的因子（如果 factors_generated.py 里有的话）
    df = add_generated_factors(df)

    # 10. 只删掉“未来1日涨跌幅”为 NaN 的行（主要是最后一行）
    before_len = len(df)
    df = df.dropna(subset=["未来1日涨跌幅"]).reset_index(drop=True)
    after_len = len(df)

    logger.info(f"原始行数: {before_len}, 删除 NaN 后行数: {after_len}")

    # （可选）打印一下前几行和字段类型，方便你确认
    logger.info("=== 数据预览（前 5 行） ===")
    logger.info(df[["交易日期", "开盘点位", "最高点位", "最低点位", "收盘价"]].head())
    logger.info("\n=== 各列的数据类型 ===")
    logger.info(df.dtypes)

    return df



data = load_kline_data()
FACTOR_COLS: List[str] = [c for c in data.columns if c.startswith("因子_")]
TARGET_COL = "未来1日涨跌幅"
NUM_FACTORS = len(FACTOR_COLS)

print("共加载数据行数：", len(data))
print("候选因子列表：", FACTOR_COLS)


# ========== 2. 配置 DEAP 遗传算法 ==========
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # 最大化
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# 个体：长度 = 因子个数，每一位 0/1，1 表示选这个因子
toolbox.register("attr_bit", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat,
                 creator.Individual, toolbox.attr_bit, n=NUM_FACTORS)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def build_combo_factor(individual, df: pd.DataFrame, factor_cols: List[str]):
    """
    从 GA 个体 + 数据构造“综合因子”：
    1. 找出被选中的因子列
    2. 对这些因子做 z-score 标准化
    3. 对行方向等权平均，得到 combo 序列

    返回值：
    - combo: pd.Series 或 None
    - selected_cols: 选中的因子名列表
    """
    selected_cols = [c for bit, c in zip(individual, factor_cols) if bit == 1]
    if not selected_cols:
        return None, []

    X = df[selected_cols].astype(float)

    # z-score 标准化
    X = (X - X.mean()) / X.std(ddof=0)

    combo = X.mean(axis=1)
    if combo.isna().all():
        return None, selected_cols

    return combo, selected_cols

def evaluate_factor_set(
    individual,
    df: pd.DataFrame,
    factor_cols: List[str],
    target_col: str,
    window: int | None = 252,  # None = 用全样本 corr, 否则用滑窗 |IC| 均值
):
    """
    适应度函数：
    - 先构造综合因子 combo
    - 如果 window is None：返回全样本 Pearson 相关系数
    - 如果 window 是整数：返回滑窗 |IC| 均值（类似时间序列 IC 的 |IC| 平均）
    """
    combo, selected_cols = build_combo_factor(individual, df, factor_cols)
    if combo is None or not selected_cols:
        return -1e6,

    y = df[target_col].astype(float)

    # 模式 1：全样本相关系数（老逻辑）
    if window is None:
        corr_full = combo.corr(y)
        if pd.isna(corr_full):
            return -1e6,
        return corr_full,

    # 模式 2：滑窗 |IC| 均值
    ic_series = combo.rolling(window).corr(y).dropna()
    if ic_series.empty:
        return -1e6,

    score_ic_abs_mean = ic_series.abs().mean()
    if pd.isna(score_ic_abs_mean):
        return -1e6,

    return score_ic_abs_mean,



toolbox.register("evaluate", evaluate_factor_set,
                 df=data, factor_cols=FACTOR_COLS, target_col=TARGET_COL, window=252)
toolbox.register("mate", tools.cxTwoPoint)                        # 交叉
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)          # 变异
toolbox.register("select", tools.selTournament, tournsize=3)      # 选择


def run_ga():
    """
    运行遗传算法，返回最优个体和对应的因子列表。
    """
    pop = toolbox.population(n=50)  # 种群大小
    hof = tools.HallOfFame(1)       # 保存历史最优解

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    # 使用 DEAP 自带的简单 GA 算法
    pop, logbook = algorithms.eaSimple(
        population=pop,
        toolbox=toolbox,
        cxpb=0.5,     # 交叉概率
        mutpb=0.2,    # 变异概率
        ngen=30,      # 迭代代数
        stats=stats,
        halloffame=hof,
        verbose=True,
    )

    best_ind = hof[0]
    selected_factors = [c for bit, c in zip(best_ind, FACTOR_COLS) if bit == 1]

    print("\n===== GA 搜索结果 =====")
    print("最佳个体（0/1）：", list(best_ind))
    print("被选中的因子：", selected_factors)
    print("适应度（相关系数）：", best_ind.fitness.values[0])

    return best_ind, selected_factors

def run_backtest(best_ind, selected_factors, df):
    """用最佳因子组合做一个简单多空/空仓策略回测。"""
    if not selected_factors:
        logger.warning("没有选中任何因子，无法回测。")
        return None, None

    logger.info("开始回测，使用因子：%s", selected_factors)

    # 1. 计算组合因子（和适应度里一样的标准化 + 等权）
    X = df[selected_factors].astype(float)
    X = (X - X.mean()) / X.std(ddof=0)
    combo = X.mean(axis=1)

    bt = df.copy()
    bt["combo"] = combo

    # 2. 使用“前一日因子”做当日持仓，避免未来函数
    # future_ret 已经是“未来1日涨跌幅”，我们当成百分比，用/100 换成收益率
    future_ret = bt["未来1日涨跌幅"] / 100.0
    signal = (bt["combo"] > 0).astype(int)       # 当日信号
    position = signal.shift(1).fillna(0)         # 当日持仓 = 昨日信号

    strat_ret = position * future_ret            # 策略日收益
    bench_ret = future_ret                       # 基准 = 买入持有

    bt["strategy_ret"] = strat_ret
    bt["bench_ret"] = bench_ret
    bt["strategy_equity"] = (1 + strat_ret).cumprod()
    bt["bench_equity"] = (1 + bench_ret).cumprod()

    # 3. 统计指标
    num_days = len(bt)
    start_date = bt["交易日期"].iloc[0].strftime("%Y-%m-%d")
    end_date = bt["交易日期"].iloc[-1].strftime("%Y-%m-%d")

    strat_total = bt["strategy_equity"].iloc[-1] - 1.0
    bench_total = bt["bench_equity"].iloc[-1] - 1.0

    # 年化收益（近似）：(最终净值)^(252/天数)-1
    strat_ann = bt["strategy_equity"].iloc[-1] ** (252.0 / num_days) - 1.0
    bench_ann = bt["bench_equity"].iloc[-1] ** (252.0 / num_days) - 1.0

    strat_vol = strat_ret.std() * (252.0 ** 0.5)
    bench_vol = bench_ret.std() * (252.0 ** 0.5)

    strat_sharpe = strat_ret.mean() / strat_ret.std() * (252.0 ** 0.5) if strat_ret.std() > 0 else 0.0

    # 最大回撤
    equity = bt["strategy_equity"]
    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1.0
    max_dd = drawdown.min()

    stats = {
        "start_date": start_date,
        "end_date": end_date,
        "num_days": int(num_days),
        "selected_factors": selected_factors,
        "strategy_total_return": float(strat_total),
        "bench_total_return": float(bench_total),
        "strategy_annual_return": float(strat_ann),
        "bench_annual_return": float(bench_ann),
        "strategy_vol": float(strat_vol),
        "bench_vol": float(bench_vol),
        "strategy_sharpe": float(strat_sharpe),
        "max_drawdown": float(max_dd),
    }

    logger.info("回测完成：策略总收益=%.4f, 年化=%.4f, 夏普=%.4f, 最大回撤=%.4f",
                strat_total, strat_ann, strat_sharpe, max_dd)

    return stats, bt

def generate_markdown_report(stats):
    """调用千问生成 Markdown 回测报告，并写入 reports/ 目录。"""
    if stats is None:
        logger.warning("没有回测结果，跳过报告生成。")
        return

    import json
    stats_json = json.dumps(stats, ensure_ascii=False, indent=2)

    prompt = f"""
你是一名专业量化研究员，现在给你一份单标的日频策略的回测摘要，请你用 Markdown 写一份研究报告。

【回测统计数据（JSON）】：
{stats_json}

要求：
1. 报告结构包含：策略简介、因子组合解释（基于 selected_factors 字段）、回测结果概览、风险分析、可能存在的问题（例如过拟合、样本偏差）、后续改进方向。
2. 重点用通俗但专业的语言解释这些数字代表什么含义，比如：总收益、年化收益、夏普、最大回撤分别说明了什么。
3. 使用 Markdown 标题和列表，例如 #、##、- 等。
4. 不要在报告中重复贴出原始 JSON，只在文字中引用指标。
"""

    try:
        md_text = ask_qwen(prompt)
    except Exception as e:
        logger.error("生成 Markdown 报告时调用 Qwen 失败：%s", e)
        return

    project_root = Path(__file__).resolve().parent.parent
    report_dir = project_root / "reports"
    report_dir.mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"backtest_report_{ts}.md"
    report_path.write_text(md_text, encoding="utf-8")

    logger.info("已生成回测 Markdown 报告：%s", report_path)

def explain_best_individual_with_qwen(best_ind, factor_cols):
    """
    用通义千问(Qwen) 帮忙用中文解释 GA 找到的最佳因子组合。
    """
    selected = [c for bit, c in zip(best_ind, factor_cols) if bit == 1]
    if not selected:
        print("没有选中任何因子，无法解释。")
        return

    prompt = f"""
我做了一个基于遗传算法的因子挖掘实验。

数据是单个标的的日K线，字段包括：
交易日期,开盘点位,最高点位,最低点位,收盘价,涨跌,涨跌幅(%),
开始日累计涨跌,开始日累计涨跌幅,成交量(万股),成交额(万元),持仓量。

我根据这些字段构造了几个简单因子，例如：
- 因子_振幅 = (最高点位 - 最低点位) / 收盘价
- 因子_实体占振幅 = |收盘价 - 开盘点位| / 收盘价
- 因子_成交额变化率 = 成交额(万元) 的日度百分比变化
- 因子_成交量变化率 = 成交量(万股) 的日度百分比变化
- 因子_当日涨跌幅 = 当天的涨跌幅(%)

然后用遗传算法在这些因子里选择子集，让“因子组合对未来1日涨跌幅的相关系数”最大。

遗传算法最终选出的因子列名如下：
{selected}

请你：
1. 基于这些因子名称和定义，解释一下每个因子大概在捕捉什么市场信息；
2. 综合这些因子，推测这个组合整体偏向于哪类风格（比如波动性、短期动量、量价配合等）；
3. 给出一些后续可以改进的建议，比如可以增加哪些类型的因子，或者如何进一步做严谨回测。

回答请使用中文，条理清晰，尽量分点列出。
"""

    try:
        answer = ask_qwen(prompt)
        print("\n===== AI 对最佳因子组合的解释（来自通义千问） =====")
        print(answer)
    except Exception as e:
        print("调用 Qwen 解释失败：", e)
        print("请检查 DASHSCOPE_API_KEY 是否设置正确，或网络是否可访问阿里云。")

if __name__ == "__main__":
    best_ind, selected_factors = run_ga()

    # 1）AI 解释最佳因子组合
    explain_best_individual_with_qwen(best_ind, FACTOR_COLS)

    # 2）回测 + Markdown 报告
    stats, bt_df = run_backtest(best_ind, selected_factors, data)
    generate_markdown_report(stats)


