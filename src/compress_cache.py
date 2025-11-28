# src/compress_cache.py
import gzip
from pathlib import Path

def main():
    # 项目根目录 = 当前文件的父目录的父目录
    root = Path(__file__).resolve().parent.parent
    src = root / "results" / "factor_combo_cache.json"
    dst = root / "results" / "factor_combo_cache.json.gz"

    if not src.exists():
        print("找不到源文件：", src)
        return

    data = src.read_bytes()
    with gzip.open(dst, "wb") as f:
        f.write(data)

    size_mb = dst.stat().st_size / 1024 / 1024
    print("压缩完成：", dst)
    print(f"压缩后文件大小：{size_mb:.2f} MB")

if __name__ == "__main__":
    main()
