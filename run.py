"""本地开发入口：python -I run.py

显式把仓库根目录与可选的 .pylibs（无 pip/venv 隔离环境下的依赖目录）
加入 sys.path，兼容 -I 同时隐含 -P（脚本目录不入路径）的解释器行为。
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _extra in (_ROOT, os.path.join(_ROOT, ".pylibs")):
    if os.path.isdir(_extra) and _extra not in sys.path:
        sys.path.insert(0, _extra)

from app.web import create_app  # noqa: E402

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)
