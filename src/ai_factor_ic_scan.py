# src/ai_factor_ic_scan.py
"""
对 AI 生成的因子做全量组合扫描：
- data/ 目录下每个 csv 单独处理
- 只使用 AI 自动生成的因子（factors_generated.GENERATED_FACTORS）
- 穷举 2~K 个因子的所有组合（K 可配置，默认 5）
- 对每个组合：
    * 算全样本 Pearson 相关系数 corr_full
    * 算滑窗 IC 序列，取 |IC| 均值 ic_abs_mean
- 条件：ic_abs_mean > ic_threshold 或 corr_full > corr_threshold
- 如果没有任何组合满足，输出 ic_abs_mean 最大的组合
- 每个 (csv, 因子组合) 的结果会缓存到 results/factor_combo_cache.json.gz
  下次运行遇到相同组合就不再重复计算。
"""

import itertools
import json
import gzip
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

# 日志：优先使用你项目里的 logging_utils
try:
    from logging_utils import get_logger
except ImportError:  # 兜底
    import logging

    def get_logger(name: str = "ai_factor_ic_scan") -> logging.Logger:
        logger = logging.getLogger(name)
        if logger.handlers:
            return logger
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        h = logging.StreamHandler()
        h.setFormatter(fmt)
        logger.addHandler(h)
        return logger

# 从自动因子模块里拿 AI 因子定义 + 执行函数
try:
    from factors_generated import GENERATED_FACTORS, add_generated_factors
except ImportError:
    GENERATED_FACTORS = []

    def add_generated_factors(df: pd.DataFrame) -> pd.DataFrame:
        print("WARNING: 未找到 factors_generated.py，跳过 AI 因子。")
        return df

logger = get_logger("ai_factor_ic_scan")

TARGET_COL = "未来1日涨跌幅"

# 控制组合枚举规模，避免 26 因子全组合爆炸
MAX_COMBO_K = 5               # 每组最多因子个数
MAX_COMBOS_PER_SYMBOL = 100000  # 每个标的最多评估多少个组合


# ===== 工具函数 =====

def project_root() -> Path:
    """项目根目录：即包含 src/ 和 data/ 的那一层"""
    return Path(__file__).resolve().parent.parent


def cache_path() -> Path:
    """
    组合结果缓存文件路径（使用 gzip 压缩）
    文件名：factor_combo_cache.json.gz
    """
    root = project_root()
    result_dir = root / "results"
    result_dir.mkdir(exist_ok=True)
    return result_dir / "factor_combo_cache.json.gz"


def load_cache() -> Dict[str, Dict[str, Any]]:
    """
    从 gzip JSON 加载缓存：
    结构：
    {
      "symbol1": {
        "因子A|因子B|因子C": {
          "factors": [...],
          "corr_full": 0.123,
          "ic_abs_mean": 0.15,
          "ic_mean": 0.08,
          "ic_std": 0.02
        },
        ...
      },
      "symbol2": { ... }
    }
    """
    p = cache_path()
    if not p.exists():
        return {}
    try:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        else:
            logger.warning("缓存文件 %s 内容不是 dict，忽略。", p)
            return {}
    except Exception as e:
        logger.warning("读取缓存文件 %s 失败，将重新生成。详情: %s", p, e)
        return {}


def save_cache(cache: Dict[str, Dict[str, Any]]) -> None:
    """把缓存写回 gzip JSON"""
    p = cache_path()
    try:
        with gzip.open(p, "wt", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
        logger.info("缓存已写入：%s", p)
    except Exception as e:
        logger.error("写入缓存文件 %s 失败：%s", p, e)


def load_kline_data_for_file(csv_path: Path) -> pd.DataFrame:
    """
    针对单个 CSV：
    - 读取 K 线数据
    - 转换数字列
    - 生成目标列 未来1日涨跌幅
    - 生成手工基础因子
    - 调用 add_generated_factors 加上 AI 因子
    """
    logger.info("读取 CSV：%s", csv_path.name)

    # 1. 读 CSV，尝试 utf-8 -> gbk
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="gbk")

    expected_cols = [
        "交易日期", "开盘点位", "最高点位", "最低点位",
        "收盘价", "涨跌", "涨跌幅(%)", "开始日累计涨跌",
        "开始日累计涨跌幅", "成交量(万股)", "成交额(万元)", "持仓量",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name} 缺少列: {missing}, 当前列: {df.columns.tolist()}")

    # 2. 数字列转换
    numeric_cols = [
        "开盘点位", "最高点位", "最低点位",
        "收盘价", "涨跌", "涨跌幅(%)", "开始日累计涨跌",
        "开始日累计涨跌幅", "成交量(万股)", "成交额(万元)", "持仓量",
    ]
    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.replace("\t", "", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. 日期 & 排序
    df["交易日期"] = pd.to_datetime(df["交易日期"], errors="coerce")
    df = df.sort_values("交易日期").reset_index(drop=True)

    # 4. 目标：未来1日涨跌幅（使用当前“涨跌幅(%)”平移一行）
    df[TARGET_COL] = df["涨跌幅(%)"].shift(-1)

    # 5. 手工基础因子（和 factor_ga.py 里的保持一致）
    df["因子_振幅"] = (df["最高点位"] - df["最低点位"]) / df["收盘价"]
    df["因子_实体占振幅"] = (df["收盘价"] - df["开盘点位"]).abs() / df["收盘价"]
    df["因子_成交额变化率"] = df["成交额(万元)"].pct_change()
    df["因子_成交量变化率"] = df["成交量(万股)"].pct_change()
    df["因子_当日涨跌幅"] = df["涨跌幅(%)"]

    # 6. AI 自动生成因子
    df = add_generated_factors(df)

    # 7. 只删掉目标缺失（最后一行）
    before_len = len(df)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    after_len = len(df)
    logger.info("行数: 原始=%d, 删除 NaN 后=%d", before_len, after_len)

    return df


def get_ai_factor_names(df: pd.DataFrame) -> List[str]:
    """
    从 GENERATED_FACTORS 中：
    - 找出真正存在于 df.columns 的 AI 因子列名
    - 同时打印出哪些定义了但没算出来（方便排查）
    """
    names: List[str] = []
    missing: List[str] = []

    for fac in GENERATED_FACTORS:
        name = fac.get("name")
        if not isinstance(name, str):
            continue
        if name in df.columns:
            names.append(name)
        else:
            missing.append(name)

    names = sorted(set(names))

    logger.info(
        "AI 因子定义总数: %d, 在当前 df 中实际存在的: %d, 计算失败/未生成的: %d",
        len(GENERATED_FACTORS),
        len(names),
        len(missing),
    )
    if missing:
        logger.info("以下 AI 因子在 df.columns 中没有找到（可能计算失败或代码有误）：%s", missing)

    return names


def evaluate_combo(
    df: pd.DataFrame,
    factor_names: List[str],
    window: int = 252,
) -> Dict[str, Any]:
    """
    对给定的一组因子名称，计算：
    - 全样本 Pearson 相关系数 corr_full
    - 滑窗 IC 序列 ic_series
    - 滑窗 |IC| 均值 ic_abs_mean
    """
    if len(factor_names) < 2:
        raise ValueError("因子组合长度必须 >= 2")

    X = df[factor_names].astype(float)

    # z-score 标准化
    X = (X - X.mean()) / X.std(ddof=0)

    combo = X.mean(axis=1)
    if combo.isna().all():
        return {
            "factors": factor_names,
            "corr_full": None,
            "ic_abs_mean": None,
            "ic_mean": None,
            "ic_std": None,
        }

    y = df[TARGET_COL].astype(float)

    corr_full = combo.corr(y)
    ic_series = combo.rolling(window).corr(y).dropna()

    if ic_series.empty:
        ic_abs_mean = None
        ic_mean = None
        ic_std = None
    else:
        ic_abs_mean = float(ic_series.abs().mean())
        ic_mean = float(ic_series.mean())
        ic_std = float(ic_series.std())

    return {
        "factors": factor_names,
        "corr_full": float(corr_full) if not pd.isna(corr_full) else None,
        "ic_abs_mean": ic_abs_mean,
        "ic_mean": ic_mean,
        "ic_std": ic_std,
    }


def scan_ai_factor_combinations_for_df(
    df: pd.DataFrame,
    symbol_name: str,
    cache: Dict[str, Dict[str, Any]],
    window: int = 252,
    corr_threshold: float = 0.1,
    ic_threshold: float = 0.1,
):
    """
    针对一个 df（一个 csv），只用 AI 因子：
    - 穷举所有 2~K 个因子组合（K = MAX_COMBO_K；如果因子数量不足，就到 len(ai_factors)）
    - 计算全样本 Pearson 和滑窗 |IC| 均值
    - 条件：ic_abs_mean > ic_threshold 或 corr_full > corr_threshold
    - 如果一个也没有，就输出 ic_abs_mean 最大的那个组合

    组合结果缓存到 cache[symbol_name] 下，key 为 "因子A|因子B|..."。
    """
    ai_factors = get_ai_factor_names(df)
    logger.info("标的 %s 的 AI 因子数量：%d", symbol_name, len(ai_factors))

    if len(ai_factors) < 2:
        logger.warning("标的 %s 的 AI 因子数量不足 2 个，跳过。", symbol_name)
        return

    max_k = min(MAX_COMBO_K, len(ai_factors))
    logger.info("组合因子个数范围：2 ~ %d", max_k)

    symbol_cache = cache.get(symbol_name, {})
    cache_changed = False

    passed: List[Dict[str, Any]] = []
    best_overall: Dict[str, Any] | None = None

    combos_evaluated = 0
    stop_early = False

    for k in range(2, max_k + 1):
        if stop_early:
            break
        logger.info("  枚举 %d 个因子的组合...", k)
        for combo_names in itertools.combinations(ai_factors, k):
            combo_list = list(combo_names)
            key = "|".join(combo_list)  # 组合的唯一标识

            if key in symbol_cache:
                res = symbol_cache[key]
            else:
                res = evaluate_combo(df, combo_list, window=window)
                symbol_cache[key] = res
                cache_changed = True

            combos_evaluated += 1
            if combos_evaluated >= MAX_COMBOS_PER_SYMBOL:
                logger.warning(
                    "标的 %s 已评估组合数量达到上限 %d，提前停止枚举。",
                    symbol_name,
                    MAX_COMBOS_PER_SYMBOL,
                )
                stop_early = True
                break

            ic_abs = res["ic_abs_mean"]
            corr_full = res["corr_full"]

            # 更新 best_overall（用 ic_abs_mean 作为主指标）
            if ic_abs is not None:
                if best_overall is None or ic_abs > best_overall.get("ic_abs_mean", -1e9):
                    best_overall = res

            # 过滤 NaN 的情况
            if ic_abs is None and corr_full is None:
                continue

            # 条件：|IC| 均值 > ic_threshold 或 Pearson > corr_threshold
            cond_ic = (ic_abs is not None and ic_abs > ic_threshold)
            cond_corr = (corr_full is not None and corr_full > corr_threshold)
            if cond_ic or cond_corr:
                passed.append(res)

    # 更新 cache
    cache[symbol_name] = symbol_cache
    if cache_changed:
        logger.info("标的 %s 的组合结果已更新缓存。", symbol_name)

    logger.info("=== 标的 %s 穷举结果（共评估组合 %d 个） ===", symbol_name, combos_evaluated)

    if passed:
        passed_sorted = sorted(
            passed,
            key=lambda x: (x["ic_abs_mean"] or -1e9),
            reverse=True,
        )
        top_n = min(10, len(passed_sorted))
        logger.info(
            "找到满足条件（|IC|均值>%.3f 或 corr>%.3f）的组合数量：%d，展示前 %d 个：",
            ic_threshold, corr_threshold, len(passed_sorted), top_n,
        )

        for i, res in enumerate(passed_sorted[:top_n], start=1):
            logger.info(
                "  #%d 因子组合: %s | corr_full=%s | |IC|均值=%s | IC均值=%s | IC std=%s",
                i,
                res["factors"],
                f"{res['corr_full']:.4f}" if res["corr_full"] is not None else "NaN",
                f"{res['ic_abs_mean']:.4f}" if res["ic_abs_mean"] is not None else "NaN",
                f"{res['ic_mean']:.4f}" if res["ic_mean"] is not None else "NaN",
                f"{res['ic_std']:.4f}" if res["ic_std"] is not None else "NaN",
            )
    else:
        logger.warning(
            "标的 %s：没有任何组合满足条件（|IC|均值>%.3f 或 corr>%.3f），输出全局最佳组合。",
            symbol_name, ic_threshold, corr_threshold,
        )
        if best_overall is not None:
            res = best_overall
            logger.info(
                "  最佳组合: %s | corr_full=%s | |IC|均值=%s | IC均值=%s | IC std=%s",
                res["factors"],
                f"{res['corr_full']:.4f}" if res["corr_full"] is not None else "NaN",
                f"{res['ic_abs_mean']:.4f}" if res["ic_abs_mean"] is not None else "NaN",
                f"{res['ic_mean']:.4f}" if res["ic_mean"] is not None else "NaN",
                f"{res['ic_std']:.4f}" if res["ic_std"] is not None else "NaN",
            )
        else:
            logger.warning("  连 best_overall 也没有，可能数据或因子有问题。")


def main():
    root = project_root()
    data_dir = root / "data"
    csv_files = sorted(data_dir.glob("*.csv"))

    if not csv_files:
        logger.error("data 目录下没有找到任何 csv 文件。")
        return

    logger.info("在 data/ 下找到 %d 个 CSV：", len(csv_files))
    for p in csv_files:
        logger.info("  - %s", p.name)

    cache = load_cache()
    cache_before = json.dumps(cache, ensure_ascii=False, sort_keys=True)

    for csv_path in csv_files:
        symbol = csv_path.stem  # 用文件名（不含后缀）作为标的名字
        logger.info("\n############################")
        logger.info("开始处理标的：%s", symbol)
        try:
            df = load_kline_data_for_file(csv_path)
        except Exception as e:
            logger.error("加载 %s 时出错：%s", csv_path.name, e)
            continue

        scan_ai_factor_combinations_for_df(
            df,
            symbol_name=symbol,
            cache=cache,
            window=252,
            corr_threshold=0.1,
            ic_threshold=0.1,
        )

    # 判断缓存是否有变化，有则写盘
    cache_after = json.dumps(cache, ensure_ascii=False, sort_keys=True)
    if cache_after != cache_before:
        save_cache(cache)
    else:
        logger.info("本次运行未产生新的组合结果，缓存无需更新。")


if __name__ == "__main__":
    main()
