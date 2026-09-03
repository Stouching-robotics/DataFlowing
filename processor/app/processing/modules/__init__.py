"""Built-in processing modules — auto-discovered; adding one file is enough.

每个 .py 文件 = 一个模块(内部用 @register 注册)。新增模块只需放一个文件,
无需改这里的 import 列表。文件名字典序加载,与画布顺序无关(catalog 会排序)。
"""

import importlib
import pkgutil

for _m in pkgutil.iter_modules(__path__):
    if _m.name.startswith("_"):
        continue
    importlib.import_module(f"{__name__}.{_m.name}")
