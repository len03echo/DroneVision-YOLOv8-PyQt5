# -*- mode: python ; coding: utf-8 -*-
# DroneVision.spec  (v3 - 使用 PyInstaller 官方社区 hooks)
# PyInstaller 打包配置文件
# 生成方式：onedir（程序目录模式），无控制台窗口
# 运行方式：dist\DroneVision\DroneVision.exe
#
# v3 removes the local torch/ultralytics hook overrides. The maintained hooks
# bundled with pyinstaller-hooks-contrib collect their binaries/data and set
# module_collection_mode='pyz+py', which is required by these packages.

from pathlib import Path

# ── 项目根目录 ──────────────────────────────────────────────────
PROJECT_DIR = Path(SPECPATH)

# ── 数据文件 ────────────────────────────────────────────────────
datas = [
    (str(PROJECT_DIR / 'models' / 'drone_yolov8n_best.pt'), 'models'),
    (str(PROJECT_DIR / 'models' / 'yolov8n.pt'),            'models'),
]

# torch/torchvision binaries are collected by their standard hooks.
binaries = []

# ── 隐式导入 ────────────────────────────────────────────────────
hiddenimports = []

# ── 排除不必要的大型模块 ────────────────────────────────────────
excludes = [
    'tensorflow',
    'keras',
    'PySide2',
    'PySide6',
    'PyQt6',
]

# ── Analysis ─────────────────────────────────────────────────────
a = Analysis(
    [str(PROJECT_DIR / 'drone_vision_gui.py')],
    pathex=[str(PROJECT_DIR)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    # Do not add packaging/hooks here. Those legacy hooks shadowed the
    # maintained hook-torch.py/hook-ultralytics.py from hooks-contrib and
    # omitted their required module_collection_mode = 'pyz+py'.
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DroneVision',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,              # 正式发布：无控制台
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon=str(PROJECT_DIR / 'packaging' / 'icon.ico'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='DroneVision',
)
