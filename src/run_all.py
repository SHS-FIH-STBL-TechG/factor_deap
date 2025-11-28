# src/run_all.py
"""
一键跑全流程：

1. 调用 ai_factor_ideation.py 里的 main():
   - 让 AI 在已有因子基础上脑暴新的因子
   - 更新 src/factors_generated.py

2. 调用 ai_factor_ic_scan.py 里的 main():
   - 对 data/ 目录下所有 CSV
   - 使用 AI 生成的因子做 2~10 因子的全量组合扫描
   - 计算每个组合的：
       * 全样本 Pearson 相关系数
       * 滑窗 |IC| 均值
   - 输出满足条件（|IC|>0.1 或 corr>0.1）的组合
     若没有则输出 |IC| 均值最大的组合
   - 并把 (csv, 因子组合) 的结果缓存起来，下次不再重复计算
"""

from logging_utils import get_logger  # 你已经有这个模块
from ai_factor_ideation import main as ideation_main
from ai_factor_ic_scan import main as ic_scan_main


def main():
    logger = get_logger("run_all")

    logger.info("===== Step 1: 调用 AI 生成 / 更新因子 =====")
    try:
        ideation_main()
        logger.info("Step 1 完成：AI 已更新 factors_generated.py。")
    except Exception as e:
        logger.error("Step 1 失败：AI 生成因子时出错：%s", e)
        # 这里你可以选择是直接退出，还是继续用旧的因子库
        # 我这里选择继续用旧因子库：
        logger.warning("继续使用现有的 AI 因子库。")

    logger.info("===== Step 2: 对所有 CSV 扫描 AI 因子组合 =====")
    try:
        ic_scan_main()
        logger.info("Step 2 完成：所有 CSV 的 AI 因子组合扫描结束。")
    except Exception as e:
        logger.error("Step 2 失败：扫描 AI 因子组合时出错：%s", e)
        return

    logger.info("===== 全部流程完成 =====")


if __name__ == "__main__":
    main()
