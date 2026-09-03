@echo off
rem ============================================================
rem  !!! 编码警告: 本文件必须保持 GBK(ANSI) 编码 + CRLF 行尾 !!!
rem  勿用记事本 / VSCode 另存（默认存成 UTF-8 会让 cmd 乱码并
rem  静默失败）；乱码后请从 GitLab 重新下载原文件。
rem ============================================================
rem ============================================================
rem  DAQ 数据采集系统 —— Windows 一键部署脚本
rem
rem  用法:
rem    start.bat               部署(按需) + 启动主程序
rem    start.bat reinstall     删除 venv 强制重装（出问题首选）
rem    start.bat extras        追加安装 mediapipe / pyrealsense2
rem    start.bat extras-torch  追加安装 CPU 版 torch
rem    start.bat help          打开 使用说明.md
rem
rem  依赖安装顺序: 离线 wheels\ 包 → 阿里云镜像 → 清华镜像 → 官方源
rem  错误码 A-G 对应 使用说明.md「常见异常与解决方案」章节
rem ============================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "MODE=run"
set "FORCE=0"
set "MAIN_ARGS="
if /i "%~1"=="reinstall"    goto :sub_reinstall
if /i "%~1"=="extras"       goto :sub_extras
if /i "%~1"=="extras-torch" goto :sub_extras_torch
if /i "%~1"=="help"         goto :show_help
if /i "%~1"=="guide"        goto :show_help
set "MAIN_ARGS=%*"
goto :banner

:sub_reinstall
set "FORCE=1"
goto :banner
:sub_extras
set "MODE=extras"
goto :banner
:sub_extras_torch
set "MODE=extras-torch"
goto :banner

:banner
echo.
echo  ============================================================
echo     DAQ 数据采集系统 —— 一键部署
echo  ============================================================
echo.

rem ── 错误 G: 解压层次自检 ──
if not exist "main.py"          goto :errG
if not exist "requirements.txt" goto :errG

rem ────────────────────────────────────────────────────────────
rem  [1/6] 定位 Python（版本需 >= 3.10，推荐 3.12）
rem ────────────────────────────────────────────────────────────
set "PY="
set "VPY=venv\Scripts\python.exe"
if not exist "%VPY%" goto :find_py
set "PY=%VPY%"
goto :have_python

:find_py
echo  [1/6] 检查 Python 环境 ...
call :try_py py -3.12
if defined PY goto :have_python
call :try_py py -3
if defined PY goto :have_python
call :try_py python
if defined PY goto :have_python

rem ── 未检测到 → 自动下载并静默安装 Python 3.12 ──
echo  [1/6] 未检测到 Python，自动下载安装 Python 3.12（约 25MB）...
set "PY_VER=3.12.10"
set "PY_EXE=%TEMP%\python-%PY_VER%-amd64.exe"
set "PY_URL=https://mirrors.aliyun.com/python-release/windows/python-%PY_VER%-amd64.exe"
set "PY_URL2=https://registry.npmmirror.com/-/binary/python/%PY_VER%/python-%PY_VER%-amd64.exe"
set "PY_URL3=https://mirrors.huaweicloud.com/python/%PY_VER%/python-%PY_VER%-amd64.exe"
if exist "wheels\python-%PY_VER%-amd64.exe" goto :py_copy_local
curl -L -o "%PY_EXE%" "%PY_URL%" --connect-timeout 20 --retry 2 --silent --show-error
if exist "%PY_EXE%" goto :py_run_installer
curl -L -o "%PY_EXE%" "%PY_URL2%" --connect-timeout 20 --retry 2 --silent --show-error
if exist "%PY_EXE%" goto :py_run_installer
curl -L -o "%PY_EXE%" "%PY_URL3%" --connect-timeout 20 --retry 2 --silent --show-error
if exist "%PY_EXE%" goto :py_run_installer
goto :errA
:py_copy_local
echo  [1/6] 使用 wheels\ 目录内的 Python 安装包 ...
copy /y "wheels\python-%PY_VER%-amd64.exe" "%PY_EXE%" >nul
:py_run_installer
echo  [1/6] 静默安装 Python 中（请勿关闭窗口，约 1 分钟）...
"%PY_EXE%" /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_test=0 Include_launcher=1
del "%PY_EXE%" >nul 2>&1
call :try_py py -3.12
if defined PY goto :have_python
goto :errA

:have_python
echo  [1/6] 使用 Python: %PY%

rem ────────────────────────────────────────────────────────────
rem  [2/6] 虚拟环境 venv
rem ────────────────────────────────────────────────────────────
if not exist "%VPY%" goto :make_venv
if not "%FORCE%"=="1" goto :deps_check
echo  [2/6] reinstall: 删除旧 venv ...
rmdir /s /q "venv" 2>nul
if exist "%VPY%" goto :errC2
:make_venv
echo  [2/6] 创建虚拟环境 venv（首次约 1 分钟）...
rem 注意: PY 是命令（py -3.12 / python），不能加引号
%PY% -m venv "venv"
if not exist "%VPY%" goto :errC

rem ────────────────────────────────────────────────────────────
rem  [3/6] 安装依赖（requirements.txt 有变化或首次运行时）
rem ────────────────────────────────────────────────────────────
:deps_check
rem 用 requirements.txt 的 修改时间|大小 做签名（内容变了才重装）
for %%I in (requirements.txt) do set "SIG=%%~tI;%%~zI"
if "%FORCE%"=="1" goto :install_deps
if not exist "venv\.deps-ok" goto :install_deps
set /p STAMP=<"venv\.deps-ok"
if "%STAMP%"=="%SIG%" goto :after_deps

:install_deps
echo  [3/6] 安装依赖（首次约 3-10 分钟，之后启动秒开）...
"%VPY%" -m pip install --upgrade pip >nul 2>&1
call :pip_install_req
if errorlevel 1 goto :errD
> "venv\.deps-ok" echo %SIG%

rem ────────────────────────────────────────────────────────────
rem  [4/6] 依赖冒烟自检（导入完整主程序链）
rem ────────────────────────────────────────────────────────────
:after_deps
echo  [4/6] 依赖自检 ...
"%VPY%" -c "import main" >nul 2>&1
if errorlevel 1 goto :errE
echo  [4/6] 依赖自检通过

rem ────────────────────────────────────────────────────────────
rem  [5/6] 可选功能（extras / extras-torch 子命令）
rem ────────────────────────────────────────────────────────────
if "%MODE%"=="extras"      goto :install_extras
if "%MODE%"=="extras-torch" goto :install_torch
goto :launch

:install_extras
if exist "venv\.extras-ok" goto :extras_done
echo  [5/6] 安装可选功能: mediapipe(裸手3D关键点) + pyrealsense2(D435/D405) ...
call :pip_pkg "mediapipe"
if errorlevel 1 echo   [警告] mediapipe 安装失败（主程序不受影响，详见使用说明.md）
call :pip_pkg "pyrealsense2"
if errorlevel 1 echo   [警告] pyrealsense2 安装失败（主程序不受影响，详见使用说明.md）
> "venv\.extras-ok" echo done
:extras_done
echo  [5/6] 可选功能安装完成。双击 start.bat 启动主程序。
pause
exit /b 0

:install_torch
if exist "venv\.torch-ok" goto :torch_done
echo  [5/6] 安装可选功能: torch CPU 版（手部关键点 RTMPose 用）...
call :pip_torch
if errorlevel 1 echo   [警告] torch 安装失败（主程序不受影响，GPU 版安装见使用说明.md）
> "venv\.torch-ok" echo done
:torch_done
echo  [5/6] 可选功能安装完成。双击 start.bat 启动主程序。
pause
exit /b 0

rem ────────────────────────────────────────────────────────────
rem  [6/6] 启动主程序
rem ────────────────────────────────────────────────────────────
:launch
echo  [6/6] 启动主程序 ...
echo.
echo  【操作指引】
echo    · 设备面板: 相机插入后约 2 秒自动出现，点击即可预览
echo    · 网格布局: 拖动分割条调整画面大小与位置
echo    · 录制: 每路相机独立的 开始/停止 按钮；正常停止=保存，异常停止=丢弃
echo    · 任务: 选择任务后开始录制；左侧可查看录制历史与回放
echo    · 上传: 录制完成后可上传服务器（配置见 data\server_config.example.json）
echo    · 语言: 设置页可切换中英文界面
echo    · 完整说明: 双击 start.bat help 或查看 使用说明.md
echo.
set "QT_QPA_PLATFORM_PLUGIN_PATH=%~dp0venv\Lib\site-packages\PyQt5\Qt5\plugins\platforms"
rem 标记本次为 start.bat 启动 → 主程序弹出使用步骤窗口（可勾选不再显示）
set "DAQ_SHOW_GUIDE=1"
"%VPY%" main.py %MAIN_ARGS%
set "EXITCODE=%errorlevel%"
if "%EXITCODE%"=="0" exit /b 0
goto :errF

rem ────────────────────────────────────────────────────────────
rem  子程序: 校验并记录可用 Python 解释器
rem  参数: 解释器命令（如 "py -3.12" / "python"）
rem ────────────────────────────────────────────────────────────
:try_py
%* -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if errorlevel 1 exit /b 1
set "PY=%*"
exit /b 0

rem ── wheels\ 目录内是否有 .whl 文件 ──
:wheels_exists
dir /b "wheels\*.whl" >nul 2>&1
exit /b %errorlevel%

rem ── 安装 requirements.txt（离线优先 → 阿里云 → 清华 → 官方）──
:pip_install_req
call :wheels_exists
if errorlevel 1 goto :req_online
echo  [3/6] 检测到 wheels\ 离线包，优先离线安装 ...
"%VPY%" -m pip install --no-index --find-links "wheels" -r requirements.txt
if not errorlevel 1 exit /b 0
echo  [3/6] 离线包安装失败，转在线安装 ...
:req_online
"%VPY%" -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
if not errorlevel 1 exit /b 0
"%VPY%" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if not errorlevel 1 exit /b 0
"%VPY%" -m pip install -r requirements.txt
exit /b %errorlevel%

rem ── 安装单个包（离线优先 → 阿里云 → 清华 → 官方）──
:pip_pkg
call :wheels_exists
if errorlevel 1 goto :pkg_online
"%VPY%" -m pip install --no-index --find-links "wheels" "%~1"
if not errorlevel 1 exit /b 0
echo  [5/6] 离线包缺失或失败，转在线安装 ...
:pkg_online
"%VPY%" -m pip install "%~1" -i https://mirrors.aliyun.com/pypi/simple/
if not errorlevel 1 exit /b 0
"%VPY%" -m pip install "%~1" -i https://pypi.tuna.tsinghua.edu.cn/simple
if not errorlevel 1 exit /b 0
"%VPY%" -m pip install "%~1"
exit /b %errorlevel%

rem ── 安装 CPU 版 torch（离线优先 → 阿里云 CPU 源 → 官方 CPU 源）──
:pip_torch
call :wheels_exists
if errorlevel 1 goto :torch_online
"%VPY%" -m pip install --no-index --find-links "wheels" torch
if not errorlevel 1 exit /b 0
echo  [5/6] 离线包缺失或失败，转在线安装 ...
:torch_online
"%VPY%" -m pip install torch --index-url https://mirrors.aliyun.com/pytorch-wheels/cpu/
if not errorlevel 1 exit /b 0
"%VPY%" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
exit /b %errorlevel%

rem ────────────────────────────────────────────────────────────
rem  帮助 / 使用说明
rem ────────────────────────────────────────────────────────────
:show_help
echo.
echo  【常用命令】
echo    start.bat               部署并启动（默认）
echo    start.bat reinstall     删除 venv 重装（出问题首选）
echo    start.bat extras        追加安装 mediapipe / pyrealsense2
echo    start.bat extras-torch  追加安装 CPU 版 torch
echo    start.bat help          打开本文档
echo.
echo    English guide: 使用说明_EN.md
echo.
if exist "使用说明.md" start "" "使用说明.md"
if not exist "使用说明.md" echo  [警告] 未找到 使用说明.md，请从 GitLab 重新下载完整代码
pause
exit /b 0

rem ────────────────────────────────────────────────────────────
rem  异常处理（错误码 A-G，与 使用说明.md 对应）
rem ────────────────────────────────────────────────────────────
:errA
echo.
echo  [错误 A] 未能找到或自动安装 Python 3.12
echo  ------------------------------------------------------------
echo   0. 已有 Python 但版本低于 3.10？按下面步骤安装 3.12 即可
echo   1. 离线环境: 将 python-3.12.10-amd64.exe 放入本目录 wheels\ 后重试
echo      （由管理员用 scripts\pack_wheels.py 生成，见使用说明.md）
echo   2. 手动安装: 即将打开官网下载页，请下载 Python 3.12.x 64 位
echo      安装时务必勾选 "Add python.exe to PATH"
echo   3. 已安装仍报错: 电脑可能装有 Microsoft Store 版 Python 干扰，
echo      请在 设置-应用 中卸载后安装官网版
echo.
start https://www.python.org/downloads/
pause
exit /b 1

:errC
echo.
echo  [错误 C] 虚拟环境创建失败
echo  ------------------------------------------------------------
echo   1. 磁盘空间不足: 清理磁盘后重试（需要约 2GB 空闲）
echo   2. 杀毒软件拦截: 将本目录加入白名单后双击 start.bat reinstall
echo   3. 路径过长: 把整个项目文件夹移到短路径（如 C:\DAQ_sdk）后重试
echo   4. 路径含特殊字符: 换一个纯英文/数字的目录重试
echo.
pause
exit /b 1

:errC2
echo.
echo  [错误 C2] 旧 venv 删除失败（文件被占用）
echo  ------------------------------------------------------------
echo   请先关闭正在运行的主程序窗口，再双击 start.bat reinstall
echo.
pause
exit /b 1

:errD
echo.
echo  [错误 D] 依赖下载/安装失败
echo  ------------------------------------------------------------
echo   1. 网络问题: 检查网络后重新双击 start.bat（已下载部分会缓存，不重复下载）
echo   2. 公司网络限制/代理: 请管理员放行 pypi 镜像，或改用离线包交付
echo      （管理员运行 scripts\pack_wheels.py 生成 wheels\，见使用说明.md）
echo   3. 杀毒软件/防火墙拦截 pip: 加入白名单后重试
echo   4. 多次失败: 双击 start.bat reinstall 重装
echo.
pause
exit /b 1

:errE
echo.
echo  [错误 E] 依赖自检失败（依赖已安装但程序无法导入）
echo  ------------------------------------------------------------
echo   1. 杀毒软件隔离了 venv 文件: 从隔离区恢复并加入白名单
echo   2. 依赖版本冲突: 双击 start.bat reinstall 重装
echo   3. 查看具体原因: 在 cmd 中运行
echo        venv\Scripts\python.exe -c "import main"
echo.
pause
exit /b 1

:errF
echo.
echo  [错误 F] 主程序启动后异常退出
echo  ------------------------------------------------------------
echo   1. 显卡驱动过旧: 更新显卡驱动后重试
echo   2. 远程桌面/虚拟机环境: 请在本地实机运行
echo   3. Qt 平台插件错误: 双击 start.bat reinstall 重装
echo   4. 摄像头无画面: Windows 设置 - 隐私和安全性 - 相机，
echo      允许应用访问相机后重启主程序
echo   5. 查看具体错误: 在 cmd 中运行
echo        venv\Scripts\python.exe main.py
echo.
pause
exit /b 1

:errG
echo.
echo  [错误 G] 未找到 main.py —— 解压层次不对
echo  ------------------------------------------------------------
echo   请保持文件夹结构完整: start.bat 与 main.py 必须在同一目录。
echo   部分解压工具会多套一层文件夹，请进入内层目录再双击 start.bat。
echo.
pause
exit /b 1
