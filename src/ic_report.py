# src/ic_report.py
"""
基于 factor_combo_cache.json.gz 的 IC 报告生成脚本（增强版）：

功能：
- 读取缓存中的组合结果（results/factor_combo_cache.json.gz）
- 支持两种模式：
  1) 自动模式（默认）：
     - 对每个标的，选出 |IC|均值最高的 TOP_N_COMBOS 组，生成报告
  2) 手动指定模式：
     - 命令行参数指定标的 (--symbols)
     - 命令行参数指定因子组合 (--combo "因子A,因子B,...")
     - 没指定 symbols 就对缓存中的所有标的使用指定组合

- 对每组因子组合：
    * 重新加载该标的 CSV + 因子（调用 ai_factor_ic_scan.load_kline_data_for_file）
    * 计算组合因子时间序列
    * 计算滑窗 IC 序列（默认 252 日）
    * 画出：
        - IC 时间序列图
        - IC 分布直方图
    * 生成 Markdown 报告（results/reports/xxx__report.md）
"""

import json
import gzip
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
import matplotlib

matplotlib.use("Agg")  # 无界面环境下绘图
import matplotlib.pyplot as plt
import warnings

from logging_utils import get_logger

# 复用 ai_factor_ic_scan 里的工具
from ai_factor_ic_scan import (
    project_root,
    cache_path,
    load_kline_data_for_file,
)

logger = get_logger("ic_report")

TARGET_COL = "未来1日涨跌幅"
ROLLING_WINDOW = 252   # 滑窗长度
TOP_N_COMBOS = 3       # 自动模式下：每个标的取前 N 个组合


# ===== Matplotlib 字体 & 警告设置（解决中文缺字刷屏） =====

# 优先尝试中文字体（Windows 上一般有 SimHei 或 Microsoft YaHei）
plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
# 允许坐标轴负号正常显示
plt.rcParams["axes.unicode_minus"] = False

# 静音「Glyph XXX missing from font(s) ...」这种 UserWarning，避免刷屏
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Glyph .* missing from font\\(s\\) .*",
)


# ===== 工具函数 =====

def load_cache_raw() -> Dict[str, Dict[str, Any]]:
    """
    直接从 gzip JSON 读取缓存。
    结构：
    {
      "symbol1": {
        "因子A|因子B|因子C": { ...指标... },
        ...
      },
      ...
    }
    """
    p = cache_path()
    if not p.exists():
        logger.warning("未找到缓存文件：%s", p)
        return {}
    try:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        logger.warning("缓存文件 %s 内容不是 dict，忽略。", p)
        return {}
    except Exception as e:
        logger.warning("读取缓存文件 %s 失败：%s", p, e)
        return {}


def compute_combo_and_ic_series(
    df: pd.DataFrame,
    factor_names: List[str],
    window: int = ROLLING_WINDOW,
) -> Tuple[pd.Series, pd.Series]:
    """
    给定 df 和若干因子列名：
    - 做 z-score 标准化
    - 组合因子 = 各因子标准化后简单平均
    - 计算滑窗 IC 序列（Pearson 相关系数）

    返回：
    - combo: 组合因子时间序列
    - ic_series: 滑窗 IC 序列
    """
    X = df[factor_names].astype(float)
    X = (X - X.mean()) / X.std(ddof=0)

    combo = X.mean(axis=1)
    y = df[TARGET_COL].astype(float)

    ic_series = combo.rolling(window).corr(y).dropna()
    return combo, ic_series


def ensure_reports_dir() -> Path:
    root = project_root()
    reports_dir = root / "results" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def sanitize_filename(text: str) -> str:
    """
    把中文/特殊字符的组合名转成一个比较安全的文件名。
    简单处理：空格和 '|' 换成 '_'，其它不做太激进的过滤。
    """
    return (
        text.replace(" ", "_")
        .replace("|", "_")
        .replace(":", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def fmt_float(x: Optional[float]) -> str:
    """
    把可能为 None 的数值格式化成字符串：
    - 数值：保留 4 位小数
    - None：返回 "NaN"
    """
    if x is None or pd.isna(x):
        return "NaN"
    return f"{float(x):.4f}"


def generate_single_report(
    symbol: str,
    factor_names: List[str],
    cache_res: Optional[Dict[str, Any]] = None,
):
    """
    对单个标的 + 一组因子名生成报告：
    - 重新加载 df
    - 计算组合因子 & 滑窗 IC
    - 画图 + 写 markdown

    cache_res:
        如果这个组合在缓存中有记录，可以传对应的 dict，用来做指标对比；
        如果没有（或者我们不关心），可以传 None。
    """
    root = project_root()
    data_dir = root / "data"
    csv_path = data_dir / f"{symbol}.csv"
    if not csv_path.exists():
        logger.warning("标的 %s 的 CSV 文件未找到：%s，跳过该组合。", symbol, csv_path)
        return

    df = load_kline_data_for_file(csv_path)

    # 确保所有因子都在 df 里
    missing = [f for f in factor_names if f not in df.columns]
    if missing:
        logger.warning(
            "标的 %s 缺少以下因子列，无法生成报告：%s",
            symbol, missing,
        )
        return

    combo_series, ic_series = compute_combo_and_ic_series(df, factor_names, window=ROLLING_WINDOW)

    if ic_series.empty:
        logger.warning("标的 %s 组合 %s 的 IC 序列为空，跳过绘图和报告。", symbol, factor_names)
        return

    # 重新计算统计量（与缓存中的做个对照）
    corr_full = combo_series.corr(df[TARGET_COL].astype(float))
    ic_abs_mean = float(ic_series.abs().mean())
    ic_mean = float(ic_series.mean())
    ic_std = float(ic_series.std())
    n_ic = int(ic_series.shape[0])

    reports_dir = ensure_reports_dir()

    # 生成文件名前缀
    factor_str = "_".join(factor_names)
    base_name = f"{symbol}__{factor_str}"
    base_name = sanitize_filename(base_name)

    # 1) IC 时间序列图
    ts_png = reports_dir / f"{base_name}__ic_ts.png"
    plt.figure(figsize=(10, 4))
    ic_series.plot()
    plt.title(f"{symbol} - IC时间序列\n{' + '.join(factor_names)}")
    plt.xlabel("时间")
    plt.ylabel("IC")
    plt.tight_layout()
    plt.savefig(ts_png)
    plt.close()

    # 2) IC 分布直方图
    hist_png = reports_dir / f"{base_name}__ic_hist.png"
    plt.figure(figsize=(6, 4))
    plt.hist(ic_series, bins=30)
    plt.title(f"{symbol} - IC分布直方图\n{' + '.join(factor_names)}")
    plt.xlabel("IC")
    plt.ylabel("频数")
    plt.tight_layout()
    plt.savefig(hist_png)
    plt.close()

    # 3) Markdown 报告
    md_path = reports_dir / f"{base_name}__report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# 标的 {symbol} 因子组合报告\n\n")
        f.write(f"- 因子组合：`{', '.join(factor_names)}`\n")
        f.write(f"- 滑窗长度：{ROLLING_WINDOW}\n")
        f.write(f"- IC 样本数量：{n_ic}\n")
        f.write(f"- 全样本 Pearson 相关系数：{corr_full:.4f}\n")
        f.write(f"- |IC| 均值：{ic_abs_mean:.4f}\n")
        f.write(f"- IC 均值：{ic_mean:.4f}\n")
        f.write(f"- IC 标准差：{ic_std:.4f}\n\n")

        if cache_res is not None:
            cache_corr = cache_res.get("corr_full")
            cache_ic_abs_mean = cache_res.get("ic_abs_mean")
            cache_ic_mean = cache_res.get("ic_mean")
            cache_ic_std = cache_res.get("ic_std")

            f.write("## 与缓存中指标的对比\n\n")
            f.write("| 指标 | 重新计算 | 缓存记录 |\n")
            f.write("|------|----------|----------|\n")
            f.write(f"| corr_full | {corr_full:.4f} | {fmt_float(cache_corr)} |\n")
            f.write(f"| |IC|均值 | {ic_abs_mean:.4f} | {fmt_float(cache_ic_abs_mean)} |\n")
            f.write(f"| IC均值 | {ic_mean:.4f} | {fmt_float(cache_ic_mean)} |\n")
            f.write(f"| IC std | {ic_std:.4f} | {fmt_float(cache_ic_std)} |\n\n")

        f.write("## IC 时间序列\n\n")
        f.write(f"![IC 时间序列]({ts_png.name})\n\n")

        f.write("## IC 分布直方图\n\n")
        f.write(f"![IC 分布直方图]({hist_png.name})\n\n")

    logger.info("已生成报告：%s", md_path)


def generate_report_auto_for_symbol(
    symbol: str,
    combos_dict: Dict[str, Dict[str, Any]],
):
    """
    自动模式：对单个标的，
    - 从 combos_dict 中挑选 ic_abs_mean 最大的 TOP_N_COMBOS 个组合，
    - 分别生成报告。
    """
    # 过滤掉没有 ic_abs_mean 的组合
    records = []
    for key, res in combos_dict.items():
        ic_abs = res.get("ic_abs_mean")
        if ic_abs is None:
            continue
        records.append((key, res))

    if not records:
        logger.info("标的 %s 缓存中没有有效的 ic_abs_mean，跳过。", symbol)
        return

    # 按 |IC|均值 降序
    records.sort(key=lambda x: x[1].get("ic_abs_mean", -1e9), reverse=True)
    top_records = records[:TOP_N_COMBOS]

    logger.info("标的 %s 自动模式：选取 TOP %d 组合生成报告。", symbol, len(top_records))

    for rank, (combo_key, res) in enumerate(top_records, start=1):
        factor_names = res.get("factors") or combo_key.split("|")
        logger.info(
            "标的 %s 自动组合 #%d：因子组合=%s",
            symbol, rank, factor_names,
        )
        generate_single_report(symbol, factor_names, cache_res=res)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="基于 factor_combo_cache.json.gz 生成 IC 报告",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="",
        help="逗号分隔的标的名列表，例如：000001.SH-行情统计-20251117,000300.SH-xxx。"
             "如果不指定，则对缓存中的所有标的生成。",
    )
    parser.add_argument(
        "--combo",
        action="append",
        default=[],
        help="指定一组因子组合，格式为：\"因子A,因子B,因子C\"。"
             "可以多次使用 --combo 传入多组组合。",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cache = load_cache_raw()
    if not cache:
        logger.error("缓存为空，可能还没有运行过 ai_factor_ic_scan。")
        return

    # 目标 symbol 列表
    if args.symbols.strip():
        target_symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        logger.info("只对指定标的生成报告：%s", target_symbols)
    else:
        target_symbols = list(cache.keys())
        logger.info("未指定 symbols，将对缓存中的所有标的生成报告：%s", target_symbols)

    # 用户指定的因子组合列表
    user_combos: List[List[str]] = []
    for combo_str in args.combo:
        names = [x.strip() for x in combo_str.split(",") if x.strip()]
        if len(names) >= 1:
            user_combos.append(names)

    if user_combos:
        logger.info("使用手动指定模式，因子组合如下：")
        for combo in user_combos:
            logger.info("  - %s", combo)
    else:
        logger.info("未指定因子组合，将使用自动模式（每个标的取 TOP_%d 组合）。", TOP_N_COMBOS)

    for symbol in target_symbols:
        combos_dict = cache.get(symbol, {})
        if not combos_dict:
            logger.warning("缓存中未找到标的 %s 的组合记录，跳过。", symbol)
            continue

        logger.info("=== 开始生成标的 %s 的报告 ===", symbol)

        if user_combos:
            # 手动指定模式：对每个组合都生成报告
            for factor_names in user_combos:
                # 如果缓存中有这个组合，就拿出来做对比；否则 cache_res = None
                key = "|".join(factor_names)
                cache_res = combos_dict.get(key)
                generate_single_report(symbol, factor_names, cache_res=cache_res)
        else:
            # 自动模式
            generate_report_auto_for_symbol(symbol, combos_dict)


if __name__ == "__main__":
    main()
