#!/usr/bin/env python3
"""
Whisper Transcriber 启动脚本
可以从任何目录运行此脚本来启动应用程序
"""

import os
import sys
from pathlib import Path

# 获取脚本所在目录
script_dir = Path(__file__).parent.absolute()

# 添加src目录到Python路径
src_dir = script_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# 切换工作目录到项目根目录
os.chdir(script_dir)

# 导入并运行主程序
if __name__ == "__main__":
    try:
        from main import main
        main()
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保已安装所有依赖包：pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        sys.exit(1) 