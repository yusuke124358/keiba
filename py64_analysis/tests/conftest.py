"""
pytest 共通設定

src/ レイアウトで pip install -e . なしでもテスト実行可能にする
"""
import sys
from pathlib import Path

# プロジェクトの src ディレクトリを Python パスに追加
# これにより `import keiba` が解決できる
project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "src"

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# py32_fetcher のパスも追加（パーサーテスト用）
py32_path = project_root.parent / "py32_fetcher"
if py32_path.exists() and str(py32_path.parent) not in sys.path:
    sys.path.insert(0, str(py32_path.parent))



