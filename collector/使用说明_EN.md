# DAQ Data Collection System · User Guide

> Operation guide and troubleshooting for end users.
> Error codes (A–G) printed by `start.bat` correspond one-to-one with the codes in this document —
> look up the code on this page when an error occurs.

> 中文版见 [使用说明.md](使用说明.md)。

---

## 1. One-Click Deployment (First Use)

1. Download the code archive from GitLab and extract it to any folder (a short pure-English path is recommended, e.g. `D:\collector`).
   ⚠ After extraction, make sure `start.bat` and `main.py` are **in the same folder** (some extractors add an extra folder level — enter the inner folder).
2. Double-click **`start.bat`**. The script automatically:
   - checks for Python → downloads and installs Python 3.12 if missing (~25 MB, silent install)
   - creates the virtual environment `venv/`
   - installs dependencies (offline `wheels/` package first → Aliyun mirror → Tsinghua mirror → official source)
   - self-checks dependencies → launches the main program
3. The first deployment takes about **3–10 minutes** (depending on network speed); afterwards every double-click starts in seconds — nothing is reinstalled.

> When dependencies are updated, the script detects changes in `requirements.txt` and installs the missing parts automatically — no manual steps needed.

---

## 2. Common Commands

| Command | Purpose |
|---|---|
| Double-click `start.bat` | Deploy (on demand) + launch the main program |
| `start.bat reinstall` | Delete the venv and force a full reinstall (**first choice when something breaks**) |
| `start.bat extras` | Additionally install mediapipe (bare-hand 3D keypoints) + pyrealsense2 (D435/D405 cameras) |
| `start.bat extras-torch` | Additionally install CPU-only torch (RTMPose hand-keypoints backend) |
| `start.bat help` | Open the user guide document |

Linux equivalent: `./start.sh`, same commands.

---

## 3. Main Window Operation Guide

> The main program pops up a "Usage Guide" window at startup (the complete steps from the server
> address to recording, playback and upload); you can check "don't show again" and reopen it
> anytime via **Help → Usage Guide**. The in-window guide matches this document; the window takes precedence.

- **Device panel**: plug in a camera (UVC camera / Intel RealSense D435, D405 / stereo camera S80C, shown as FaysSense S80M in the panel) and it appears within ~2 seconds; click to preview. Power on a haptic glove and enable PC Bluetooth and it appears in the "🧤 Gloves" group automatically; clicking it connects automatically (left/right hand recognized by the broadcast name L/R).
- **Grid layout**: drag the edges and splitter bars to resize and rearrange each view.
- **Recording**: each camera has its own "Start / Stop" buttons.
  - **Normal stop** = save this recording; **abnormal stop** = discard this recording (the elapsed time is shown above the buttons).
  - **Video format**: adaptive by default — the machine's encoding performance is probed when recording starts; H.265/HEVC (CRF 30, about 1/10 the size of H.264) is preferred and H.264 is used automatically on slower machines; HEVC videos skip the upload recompression (faster, only one generation of quality loss). **HEVC videos won't open in the built-in Windows player**: use the free VLC player, or install "HEVC Video Extensions" from the Microsoft Store; alternatively set `RECORD_VIDEO_ENCODER` to `"x264"` in `config/settings.py` so that new recordings are H.264 (existing files are unchanged). Playback inside the main app always works.
- **Tasks**: choose a task at the top before recording; the left side shows the recording history for playback.
- **Upload**: recordings can be uploaded to the server (server address etc. are configured in `data/server_config.example.json`; copy it and rename the copy to `data/server_config.json` after editing).
- **Language**: switch between the Chinese and English UI on the settings page.
- **Stereo camera (S80C) depth** (v1.0.11): when the stereo camera is on, a third tile next to the left/right views shows a real-time depth heatmap; recording also saves a depth heatmap video (`depth/stereo_depth/stereo_depth.mp4`) and raw depth data (`depth/stereo_depth/000001.png …`, PNG 16-bit grayscale, uint16 millimeters; D435/D405 depth data uses the same format).
- **Advanced features** (run `start.bat extras` first):
  - D435/D405 depth camera RGB+depth dual-stream recording
  - hand 3D keypoint processing (`tools/hand_3d_d435/` etc.)

---

## 4. Common Problems and Solutions

### 4.1 Error Code Reference (matches start.bat output)

| Code | Symptom | Solution |
|---|---|---|
| **A** | Python not found / auto-install failed | ① Offline environment: put `python-3.12.10-amd64.exe` into `wheels/` and retry; ② Manual install: download 64-bit Python 3.12.x from python.org and make sure to check **Add python.exe to PATH**; ③ Python already installed but the error persists: a Microsoft Store version may interfere — uninstall it in Settings-Apps and install the python.org version; ④ Existing Python older than 3.10 also requires step ② |
| **C** | Failed to create the virtual environment | ① Low disk space (about 2 GB free needed); ② Antivirus blocking → whitelist the folder then `start.bat reinstall`; ③ Path too long → move the project to a short path like `C:\collector`; ④ Special characters in the path → move to a pure-English folder |
| **C2** | Failed to delete the old venv | The main program is still running — close its window first, then `start.bat reinstall` |
| **D** | Dependency download/install failed | ① Network down → fix the network and rerun (downloaded parts are cached, no re-download); ② Corporate network restrictions → ask the administrator to allow PyPI mirrors, or use the offline delivery (Section 6); ③ Antivirus/firewall blocking pip → whitelist; ④ Repeated failures → `start.bat reinstall` |
| **E** | Dependency self-check failed | ① Antivirus quarantined venv files → restore them from quarantine + whitelist; ② `start.bat reinstall`; ③ Run `venv\Scripts\python.exe -c "import main"` in cmd to see the actual error |
| **F** | Main program exited abnormally after launch | ① Outdated GPU driver → update it; ② Remote desktop / VM → run on a local physical machine; ③ Qt platform plugin error → `start.bat reinstall`; ④ No camera image → Windows "Settings-Privacy & security-Camera": allow apps to access the camera; ⑤ Run `venv\Scripts\python.exe main.py` in cmd to see the actual error |
| **G** | "main.py not found" | Wrong extraction level: enter the inner folder where `start.bat` and `main.py` are side by side, then double-click |

### 4.2 Camera-Related Problems

| Symptom | Solution |
|---|---|
| Camera not shown in the device panel | ① Windows "Settings-Privacy & security-Camera": allow desktop apps to access the camera; ② Device Manager: check the driver is OK; ③ Close other apps that are using the camera (WeChat, Tencent Meeting, etc.); ④ Try another USB port (prefer direct motherboard / 3.0 ports) |
| Image shown but laggy / low FPS | Use a USB 3.0 port; avoid USB hubs or extension cables; close other camera apps |
| RealSense camera not recognized | Run `start.bat extras` first, then replug the device |
| Recorded files corrupt / dropped frames | Don't plug or unplug devices while recording; wait a moment after a normal stop before exiting |
| Stereo camera (S80C) and RealSense cannot be enabled together | Device limitation: close the RealSense device first, then enable the stereo camera |
| No device in the "Gloves" group | Make sure the glove is powered on and PC Bluetooth is enabled; wait for the automatic panel refresh (a few seconds); if still missing, power-cycle the glove and retry |

### 4.3 Upload-Related Problems

| Symptom | Solution |
|---|---|
| Upload failed / timed out | Check network and server connectivity; confirm the address in `data/server_config.json` is correct (first use: copy `server_config.example.json` and rename it) |
| Upload says success but nothing appears on the server | Contact the administrator to check the server import logs (batch import failures are silently dropped), and provide the session name and task name |
| Large file upload is slow | Normal — recording files are large; please be patient |

### 4.4 Other

| Symptom | Solution |
|---|---|
| Double-clicking `start.bat` blocked by Windows (SmartScreen blue warning) | This is the security mark browsers add to downloaded files, a normal prompt: click "More info" → "Run anyway"; or check "Unblock" in the file properties |
| Garbled Chinese in the console window | `start.bat` is GBK-encoded: don't re-save it with Notepad / VSCode (they save as UTF-8 by default, which makes cmd misparse the script and fail silently); if garbled, re-download the original file from GitLab |
| Antivirus flags / blocks the installer | One-click deployment downloads and silently installs Python — this is normal behavior; whitelist the project folder |
| Low disk space warning | Clean up the disk; recording data grows with use (see Section 5) |
| Recorded video won't open in the Windows player | Videos are H.265/HEVC and the built-in Windows player has no decoder. **First choice: install the free VLC player** (playback inside the main app also works); or install "HEVC Video Extensions" from the Microsoft Store; or set `RECORD_VIDEO_ENCODER` to `"x264"` in `config/settings.py` and record again (only affects future recordings) |
| "Encoder may be falling behind, consider lowering resolution/streams" after recording | The machine's encoding performance is insufficient: record fewer streams or a lower resolution, or set `RECORD_VIDEO_ENCODER` to `"x264"` in `config/settings.py` (fastest fallback encoder) |

---

## 5. Data Storage and Backup

- **Recordings**: `data/recordings/<task name>/<task name>_NNNNNN/` (an MP4 video per device + parquet time-series data + metadata JSON)
- **Recording history & upload queue**: `data/pipeline.db` (SQLite)
- **Backup**: copy the whole `data/` folder.

---

## 6. Offline Delivery (intranet machines, administrator operation)

On an internet-connected machine, generate the offline package (installing `uv` first is recommended;
on a Linux dev machine cross-platform dependency resolution needs it; without uv, run directly on a Windows machine):

```bash
python scripts/pack_wheels.py            # core dependencies only
python scripts/pack_wheels.py --extras   # + mediapipe / pyrealsense2
python scripts/pack_wheels.py --torch    # + CPU-only torch
```

The output is a `wheels/` folder (dependency wheels + the Python installer).
**Copy the whole `wheels/` into the customer's project root** — their `start.bat` detects it
automatically and installs fully offline, with no internet needed at all.

---

## 7. Support

For problems not covered in this document, please send
**the error code / symptom screenshot + the diagnostic information required in Section 4.1**
(e.g. the full output of running `venv\Scripts\python.exe main.py` in cmd) to the development team.
