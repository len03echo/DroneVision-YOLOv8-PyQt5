# DroneVision 打包说明

## 打包前检查清单

- [ ] 已激活 `yolov8_pyqt5` Conda 环境
- [ ] `models/drone_yolov8n_best.pt` 存在
- [ ] `models/yolov8n.pt` 存在
- [ ] 运行 `python drone_vision_gui.py` 确认 GUI 正常启动

## 打包命令

**方式一（已激活环境）：**
```bat
cd /d <项目根目录>
pyinstaller DroneVision.spec --noconfirm --clean
```

**方式二（使用脚本，自动激活环境）：**
```
双击 packaging\build_conda.bat
```

**方式三（已激活环境，使用脚本）：**
```
双击 packaging\build.bat
```

## 输出结构

```
dist\DroneVision\
├── DroneVision.exe        ← 主程序
├── _internal\             ← PyInstaller 依赖库（勿删除）
│   └── models\
│       ├── drone_yolov8n_best.pt
│       └── yolov8n.pt
└── ...
```

## 运行时路径说明

| 资源 | 源码运行路径 | EXE 运行路径 |
|------|-------------|-------------|
| 模型权重 | `models/` | `dist\DroneVision\_internal\models\` |
| 检测结果 | `outputs/` | `%USERPROFILE%\Documents\DroneVision\outputs\` |
| 对号图标（临时） | 项目根目录 | `%TEMP%\_check_icon.png` |

## 常见问题

**Q：打包后双击 EXE 无反应或闪退**  
A：在 `DroneVision.spec` 中将 `console=False` 改为 `console=True`，重新打包后运行，查看控制台错误信息。

**Q：提示找不到模型文件**  
A：确认 `dist\DroneVision\_internal\models\` 目录中存在 `drone_yolov8n_best.pt`。`.spec` 中已配置自动复制。

## 命令行自检（不打开 GUI）

```bat
dist\DroneVision\DroneVision.exe --smoke-test --smoke-output=build\exe_smoke.json
```

返回码为 0 且 JSON 中 `success` 为 `true`，表示打包后的程序已实际完成模型加载和一次推理。

**Q：打包产物体积过大（>3GB）**  
A：PyTorch + CUDA 本身约 2-3GB，属正常现象。可尝试在 `.spec` 的 `excludes` 中追加不需要的模块，或改用 CPU-only 版 PyTorch 重新打包演示版本。

**Q：启动时出现 "Failed to execute script"**  
A：临时改为 `console=True` 查看具体报错；常见原因是某个隐式导入未收集，在 `hiddenimports` 中追加后重新打包。

**Q：如何制作桌面图标**  
A：准备 256×256 的 `.ico` 文件，放至 `packaging/icon.ico`，取消 `.spec` 中 `icon=` 行的注释后重新打包。
