"""修复 openai-clip 在 Python 3.12+ 上的兼容问题。

openai-clip 的 clip.py 用了 `from pkg_resources import packaging`，
而 pkg_resources 在新版 setuptools 中已被移除，导致 `import clip` 报错。
setup.bat 装完依赖后会自动运行本脚本打补丁。
"""

import sys
import site
import os

TARGET_TAIL = os.path.join("clip", "clip.py")
OLD = "from pkg_resources import packaging"
NEW = "import packaging.version"


def find_clip():
    for base in site.getsitepackages() + [site.getusersitepackages()]:
        path = os.path.join(base, TARGET_TAIL)
        if os.path.isfile(path):
            return path
    return None


def main():
    path = find_clip()
    if not path:
        print("未找到 clip 包，跳过补丁（如果之后报错，请先安装 openai-clip）")
        return
    with open(path, encoding="utf-8") as f:
        content = f.read()
    if OLD not in content:
        print(f"clip 已打过补丁或不需要: {path}")
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(content.replace(OLD, NEW))
    print(f"clip 补丁完成: {path}")


if __name__ == "__main__":
    main()
