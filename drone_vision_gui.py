"""
drone_vision_gui.py
DroneVision · 无人机智能目标检测系统
入口：直接运行本文件即可启动 GUI。
"""

import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QSize, QRect, QRectF, QPointF
)
from PyQt5.QtGui import (
    QImage, QPixmap, QFont, QColor, QPainter, QIcon, QCursor,
    QBrush, QPen, QPolygonF
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QSlider, QSpinBox, QDoubleSpinBox, QFileDialog, QMessageBox,
    QHBoxLayout, QVBoxLayout, QGridLayout, QFrame, QSizePolicy,
    QStackedWidget, QScrollArea, QCheckBox, QGroupBox, QSplitter,
    QProgressBar, QToolTip, QStyle
)

from drone_detector import DroneDetector

# ======================================================================
# 全局资源路径（兼容 PyInstaller 打包）
# ======================================================================
_BASE_DIR = Path(__file__).parent


def _res(rel: str) -> str:
    """返回资源文件绝对路径字符串，兼容 PyInstaller _MEIPASS。"""
    try:
        # noinspection PyUnresolvedReferences
        base = Path(sys._MEIPASS)
    except AttributeError:
        base = _BASE_DIR
    return str(base / rel)


def _output_dir(sub: str) -> Path:
    """
    返回检测结果输出目录并确保其存在。
    - 源码运行：项目根目录 / outputs / sub
    - PyInstaller EXE：用户文档目录 / DroneVision / outputs / sub
    sub 可为 'images' | 'videos' | 'snapshots'
    """
    if getattr(sys, "frozen", False):
        # 打包后写入用户文档目录，避免向只读程序目录写入
        base = Path.home() / "Documents" / "DroneVision" / "outputs"
    else:
        base = _BASE_DIR / "outputs"
    target = base / sub
    target.mkdir(parents=True, exist_ok=True)
    return target


def _ts_filename(prefix: str, ext: str) -> str:
    """生成带时间戳的默认文件名，格式：prefix_YYYYMMDD_HHMMSS.ext"""
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}{ext}"


# ======================================================================
# 全局 QSS 样式表
# ======================================================================
QSS = """
/* ---- 全局 ---- */
QWidget {
    background-color: #F3F8FF;
    color: #243B53;
    font-family: "Microsoft YaHei UI", "Microsoft YaHei", "SimHei", sans-serif;
    font-size: 14px;
}
QToolTip {
    background-color: #243B53;
    color: #FFFFFF;
    border: none;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 13px;
}

/* ---- 左侧导航面板 ---- */
#NavPanel {
    background-color: #E6F1FF;
    border-right: 1px solid #D8E5F2;
}
#BrandLabel {
    color: #3B82F6;
    font-size: 18px;
    font-weight: bold;
    letter-spacing: 1px;
}
#DevLabel {
    color: #68839F;
    font-size: 12px;
}

/* ---- 导航分组标题 ---- */
#NavSectionLabel {
    color: #68839F;
    font-size: 12px;
    letter-spacing: 1px;
    padding-left: 8px;
    padding-top: 8px;
}

/* ---- 导航切换按钮（模式选择） ---- */
#NavModeBtn {
    background-color: transparent;
    color: #243B53;
    border: none;
    border-radius: 8px;
    padding: 9px 12px;
    text-align: left;
    font-size: 14px;
}
#NavModeBtn:hover {
    background-color: #DCEEFF;
    color: #3B82F6;
}
#NavModeBtn[active="true"] {
    background-color: #DCEEFF;
    color: #3B82F6;
    font-weight: bold;
}

/* ---- 操作按钮 ---- */
#ActionBtn {
    background-color: #3B82F6;
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    padding: 9px 12px;
    font-size: 14px;
    font-weight: bold;
}
#ActionBtn:hover {
    background-color: #2563EB;
}
#ActionBtn:pressed {
    background-color: #1D4ED8;
}
#ActionBtn:disabled {
    background-color: #CBD5E1;
    color: #94A3B8;
}
#ActionBtnSecondary {
    background-color: #DCEEFF;
    color: #3B82F6;
    border: 1px solid #BFDBFE;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 14px;
}
#ActionBtnSecondary:hover {
    background-color: #BFDBFE;
}
#ActionBtnSecondary:disabled {
    background-color: #F1F5F9;
    color: #CBD5E1;
    border-color: #E2E8F0;
}

/* ---- 卡片容器 ---- */
#Card {
    background-color: #FFFFFF;
    border: 1px solid #D8E5F2;
    border-radius: 12px;
}

/* ---- 视频/图片显示区 ---- */
#DisplayLabel {
    background-color: #EBF4FF;
    border: 2px dashed #BFDBFE;
    border-radius: 10px;
    color: #68839F;
    font-size: 15px;
}

/* ---- 信息栏 ---- */
#InfoBar {
    background-color: #FFFFFF;
    border: 1px solid #D8E5F2;
    border-radius: 8px;
}
#InfoKey {
    color: #68839F;
    font-size: 13px;
}
#InfoVal {
    color: #243B53;
    font-size: 13px;
    font-weight: bold;
}

/* ---- 右侧参数面板 ---- */
#ParamPanel {
    background-color: #FFFFFF;
    border: 1px solid #D8E5F2;
    border-radius: 12px;
}
#ParamTitle {
    color: #243B53;
    font-size: 15px;
    font-weight: bold;
}
#ParamLabel {
    color: #68839F;
    font-size: 13px;
}
#ParamGroupBox {
    border: 1px solid #D8E5F2;
    border-radius: 8px;
    margin-top: 6px;
    padding: 6px;
}
#ParamGroupBox::title {
    color: #68839F;
    font-size: 12px;
    subcontrol-origin: margin;
    left: 8px;
}

/* ---- Slider ---- */
QSlider::groove:horizontal {
    height: 4px;
    background: #DCEEFF;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #3B82F6;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::sub-page:horizontal {
    background: #3B82F6;
    border-radius: 2px;
}

/* ---- SpinBox ---- */
QDoubleSpinBox, QSpinBox {
    background-color: #F3F8FF;
    border: 1px solid #D8E5F2;
    border-radius: 6px;
    padding: 4px 6px;
    color: #243B53;
}
QDoubleSpinBox:focus, QSpinBox:focus {
    border-color: #3B82F6;
}

/* ---- CheckBox ---- */
QCheckBox {
    spacing: 6px;
    color: #243B53;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1.5px solid #BFDBFE;
    border-radius: 4px;
    background: #F3F8FF;
}
QCheckBox::indicator:hover {
    border-color: #3B82F6;
    background: #EBF4FF;
}
QCheckBox::indicator:checked {
    background: #3B82F6;
    border-color: #3B82F6;
    image: none;
}

/* ---- 状态标签 ---- */
#StatusLabel {
    color: #68839F;
    font-size: 13px;
    padding: 4px 8px;
}
#ModelStatusReady {
    color: #16A34A;
    font-weight: bold;
    font-size: 13px;
}
#ModelStatusNotReady {
    color: #DC2626;
    font-size: 13px;
}

/* ---- 分割线 ---- */
#HSep {
    background-color: #D8E5F2;
    max-height: 1px;
    min-height: 1px;
}
"""


# ======================================================================
# 工作线程：图片检测
# ======================================================================
class ImageDetectThread(QThread):
    result_ready = pyqtSignal(np.ndarray, int, float)  # 结果图, 数量, 耗时ms
    error_signal = pyqtSignal(str)

    def __init__(self, detector: DroneDetector, frame: np.ndarray):
        super().__init__()
        self.detector = detector
        self.frame = frame

    def run(self):
        try:
            result, count, elapsed = self.detector.detect(self.frame)
            self.result_ready.emit(result, count, elapsed)
        except Exception as e:
            self.error_signal.emit(str(e))


# ======================================================================
# 工作线程：视频 / 摄像头逐帧检测
# ======================================================================
class VideoDetectThread(QThread):
    frame_ready  = pyqtSignal(np.ndarray, int, float, float)  # 帧图, 数量, 耗时, fps
    progress     = pyqtSignal(int, int)     # 当前帧, 总帧数
    finished     = pyqtSignal(str)          # 完成消息
    error_signal = pyqtSignal(str)

    def __init__(self, detector: DroneDetector, source,
                 save_path: Optional[str] = None, total_frames: int = 0):
        """
        source: 视频文件路径(str) 或摄像头索引(int)
        save_path: 导出视频保存路径，None 表示不保存
        """
        super().__init__()
        self.detector = detector
        self.source = source
        self.save_path = save_path
        self.total_frames = total_frames
        self._running = True
        self._paused = False
        self._writer: Optional[cv2.VideoWriter] = None

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def stop(self):
        self._running = False
        self._paused = False

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.error_signal.emit(f"无法打开视频源：{self.source}")
            return

        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 初始化视频写入器
        if self.save_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self.save_path, fourcc, fps_src, (w, h))
            if not self._writer.isOpened():
                self.error_signal.emit(f"无法创建输出视频：{self.save_path}")
                self._writer = None

        frame_idx = 0
        fps_smooth = 0.0
        t_prev = time.perf_counter()

        while self._running:
            if self._paused:
                time.sleep(0.05)
                continue

            ret, frame = cap.read()
            if not ret:
                break  # 视频结束

            result, count, elapsed_ms = self.detector.detect(frame)

            # 计算实时 FPS（滑动平均）
            t_now = time.perf_counter()
            interval = t_now - t_prev
            t_prev = t_now
            instant_fps = 1.0 / interval if interval > 0 else 0.0
            fps_smooth = 0.7 * fps_smooth + 0.3 * instant_fps

            if self._writer:
                self._writer.write(result)

            frame_idx += 1
            self.frame_ready.emit(result, count, elapsed_ms, fps_smooth)
            if self.total_frames > 0:
                self.progress.emit(frame_idx, self.total_frames)

        cap.release()
        if self._writer:
            self._writer.release()
            self.finished.emit(f"视频已保存：{self.save_path}")
        else:
            # 区分自然结束（_running=True）与用户停止（_running=False）
            if self._running:
                self.finished.emit("检测完成")
            else:
                self.finished.emit("检测已停止")


# ======================================================================
# 通用：将 cv2 BGR ndarray 转为 QPixmap，保持宽高比
# ======================================================================
def ndarray_to_pixmap(frame: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(img)


def fit_pixmap(pixmap: QPixmap, target: QSize) -> QPixmap:
    """缩放 pixmap 到 target 内，保持宽高比。"""
    return pixmap.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)


# ======================================================================
# 信息项小组件（Key + Value 一行）
# ======================================================================
class InfoItem(QWidget):
    def __init__(self, key: str, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(6)
        self._key = QLabel(key)
        self._key.setObjectName("InfoKey")
        self._val = QLabel("—")
        self._val.setObjectName("InfoVal")
        lay.addWidget(self._key)
        lay.addStretch()
        lay.addWidget(self._val)

    def set_value(self, v: str):
        self._val.setText(v)


# ======================================================================
# 右侧参数面板
# ======================================================================
class ParamPanel(QWidget):
    param_changed = pyqtSignal()  # 任意参数变化时通知

    def __init__(self, detector: DroneDetector, parent=None):
        super().__init__(parent)
        self.setObjectName("ParamPanel")
        self.detector = detector
        self.setFixedWidth(270)
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(10)

        # 标题
        title = QLabel("⚙  检测参数")
        title.setObjectName("ParamTitle")
        lay.addWidget(title)

        sep = QFrame(); sep.setObjectName("HSep"); lay.addWidget(sep)

        # ---- 模型状态 ----
        model_group = QGroupBox("模型")
        model_group.setObjectName("ParamGroupBox")
        mg_lay = QVBoxLayout(model_group)
        mg_lay.setSpacing(6)

        self.lbl_model_name = QLabel("未加载")
        self.lbl_model_name.setObjectName("ModelStatusNotReady")
        self.lbl_model_name.setWordWrap(True)
        mg_lay.addWidget(self.lbl_model_name)

        self.btn_load_model = QPushButton("选择权重文件…")
        self.btn_load_model.setObjectName("ActionBtnSecondary")
        self.btn_load_model.setToolTip("选择 .pt 权重文件（训练完成后加载 best.pt）")
        self.btn_load_model.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_load_model.clicked.connect(self._on_load_model)
        mg_lay.addWidget(self.btn_load_model)

        lay.addWidget(model_group)

        # ---- 推理参数 ----
        infer_group = QGroupBox("推理参数")
        infer_group.setObjectName("ParamGroupBox")
        ig_lay = QGridLayout(infer_group)
        ig_lay.setSpacing(8)

        # Conf
        lbl_conf = QLabel("置信度阈值 Conf")
        lbl_conf.setObjectName("ParamLabel")
        lbl_conf.setToolTip("置信度低于此值的检测框将被过滤")
        ig_lay.addWidget(lbl_conf, 0, 0)
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.01, 0.99)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setDecimals(2)
        self.spin_conf.setValue(0.25)
        self.spin_conf.setToolTip("建议范围：0.10 ~ 0.50")
        self.spin_conf.valueChanged.connect(self._apply_params)
        ig_lay.addWidget(self.spin_conf, 0, 1)

        # IoU
        lbl_iou = QLabel("IoU 阈值")
        lbl_iou.setObjectName("ParamLabel")
        lbl_iou.setToolTip("NMS 去重阈值，值越小框越少")
        ig_lay.addWidget(lbl_iou, 1, 0)
        self.spin_iou = QDoubleSpinBox()
        self.spin_iou.setRange(0.1, 0.9)
        self.spin_iou.setSingleStep(0.05)
        self.spin_iou.setDecimals(2)
        self.spin_iou.setValue(0.45)
        self.spin_iou.setToolTip("建议范围：0.30 ~ 0.60")
        self.spin_iou.valueChanged.connect(self._apply_params)
        ig_lay.addWidget(self.spin_iou, 1, 1)

        # ImgSz
        lbl_imgsz = QLabel("输入尺寸 ImgSz")
        lbl_imgsz.setObjectName("ParamLabel")
        lbl_imgsz.setToolTip("推理时将图像缩放到此尺寸（像素）")
        ig_lay.addWidget(lbl_imgsz, 2, 0)
        self.spin_imgsz = QSpinBox()
        self.spin_imgsz.setRange(320, 1280)
        self.spin_imgsz.setSingleStep(32)
        self.spin_imgsz.setValue(640)
        self.spin_imgsz.setToolTip("建议 640；显存不足可改为 320")
        self.spin_imgsz.valueChanged.connect(self._apply_params)
        ig_lay.addWidget(self.spin_imgsz, 2, 1)

        lay.addWidget(infer_group)

        # ---- 显示选项 ----
        disp_group = QGroupBox("显示选项")
        disp_group.setObjectName("ParamGroupBox")
        dg_lay = QVBoxLayout(disp_group)
        dg_lay.setSpacing(6)

        self.chk_show_label = QCheckBox("显示类别名称")
        self.chk_show_label.setChecked(True)
        self.chk_show_label.setToolTip("是否在检测框上显示 drone 标签")
        self.chk_show_label.stateChanged.connect(self._apply_params)
        dg_lay.addWidget(self.chk_show_label)

        self.chk_show_conf = QCheckBox("显示置信度数值")
        self.chk_show_conf.setChecked(True)
        self.chk_show_conf.setToolTip("是否在检测框上显示置信度百分比")
        self.chk_show_conf.stateChanged.connect(self._apply_params)
        dg_lay.addWidget(self.chk_show_conf)

        lay.addWidget(disp_group)

        # 恢复默认按钮
        self.btn_reset = QPushButton("恢复默认参数")
        self.btn_reset.setObjectName("ActionBtnSecondary")
        self.btn_reset.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_reset.clicked.connect(self._reset_params)
        lay.addWidget(self.btn_reset)

        lay.addStretch()

        # 底部项目标识
        dev_lbl = QLabel("DroneVision · YOLOv8")
        dev_lbl.setObjectName("DevLabel")
        dev_lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(dev_lbl)

    def _on_load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 YOLOv8 权重文件", str(_BASE_DIR / "models"), "权重文件 (*.pt)"
        )
        if not path:
            return
        ok, msg = self.detector.load_model(path)
        if ok:
            name = Path(path).name
            self.lbl_model_name.setText(f"✔ {name}")
            self.lbl_model_name.setObjectName("ModelStatusReady")
        else:
            self.lbl_model_name.setText("✘ 加载失败")
            self.lbl_model_name.setObjectName("ModelStatusNotReady")
            QMessageBox.warning(self, "模型加载失败", msg)
        # 刷新样式
        self.lbl_model_name.style().unpolish(self.lbl_model_name)
        self.lbl_model_name.style().polish(self.lbl_model_name)
        self.param_changed.emit()

    def _apply_params(self):
        self.detector.conf       = self.spin_conf.value()
        self.detector.iou        = self.spin_iou.value()
        self.detector.imgsz      = self.spin_imgsz.value()
        self.detector.show_label = self.chk_show_label.isChecked()
        self.detector.show_conf  = self.chk_show_conf.isChecked()
        self.param_changed.emit()

    def _reset_params(self):
        self.spin_conf.setValue(0.25)
        self.spin_iou.setValue(0.45)
        self.spin_imgsz.setValue(640)
        self.chk_show_label.setChecked(True)
        self.chk_show_conf.setChecked(True)

    def try_load_default(self):
        """启动时尝试自动加载默认权重。模型不存在时更新标签提示用户手动选择。"""
        default = self.detector.get_default_weight()
        if default and Path(default).is_file():
            ok, msg = self.detector.load_model(default)
            if ok:
                self.lbl_model_name.setText(f"✔ {Path(default).name}")
                self.lbl_model_name.setObjectName("ModelStatusReady")
                self.lbl_model_name.style().unpolish(self.lbl_model_name)
                self.lbl_model_name.style().polish(self.lbl_model_name)
        else:
            # 无可用默认模型，提示用户手动选择
            self.lbl_model_name.setText("请选择模型权重文件")
            self.lbl_model_name.setObjectName("ModelStatusNotReady")


# ======================================================================
# 图片检测页
# ======================================================================
class ImagePage(QWidget):
    status_msg = pyqtSignal(str)

    def __init__(self, detector: DroneDetector, parent=None):
        super().__init__(parent)
        self.detector = detector
        self._orig_frame: Optional[np.ndarray] = None
        self._result_frame: Optional[np.ndarray] = None
        self._thread: Optional[ImageDetectThread] = None
        self._show_result = False
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        # ---- 图片显示区（左右对比） ----
        display_row = QHBoxLayout()
        display_row.setSpacing(8)

        # 原图
        left_card = QWidget(); left_card.setObjectName("Card")
        ll = QVBoxLayout(left_card); ll.setContentsMargins(8, 8, 8, 8)
        lbl_orig_title = QLabel("原始图像"); lbl_orig_title.setObjectName("ParamLabel")
        lbl_orig_title.setAlignment(Qt.AlignCenter)
        ll.addWidget(lbl_orig_title)
        self.lbl_orig = QLabel("请导入图片")
        self.lbl_orig.setObjectName("DisplayLabel")
        self.lbl_orig.setAlignment(Qt.AlignCenter)
        self.lbl_orig.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_orig.setMinimumSize(320, 240)
        ll.addWidget(self.lbl_orig)
        display_row.addWidget(left_card, 1)

        # 结果图
        right_card = QWidget(); right_card.setObjectName("Card")
        rl = QVBoxLayout(right_card); rl.setContentsMargins(8, 8, 8, 8)
        lbl_res_title = QLabel("检测结果"); lbl_res_title.setObjectName("ParamLabel")
        lbl_res_title.setAlignment(Qt.AlignCenter)
        rl.addWidget(lbl_res_title)
        self.lbl_result = QLabel("尚未检测")
        self.lbl_result.setObjectName("DisplayLabel")
        self.lbl_result.setAlignment(Qt.AlignCenter)
        self.lbl_result.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_result.setMinimumSize(320, 240)
        rl.addWidget(self.lbl_result)
        display_row.addWidget(right_card, 1)

        root.addLayout(display_row, 1)

        # ---- 信息栏 ----
        info_bar = QWidget(); info_bar.setObjectName("InfoBar")
        info_lay = QHBoxLayout(info_bar); info_lay.setContentsMargins(8, 4, 8, 4)
        self.info_count  = InfoItem("检测目标")
        self.info_time   = InfoItem("推理耗时")
        self.info_status = InfoItem("状态")
        for w in [self.info_count, self.info_time, self.info_status]:
            info_lay.addWidget(w)
            info_lay.addStretch()
        root.addWidget(info_bar)

    # ---- 外部调用接口 ----
    def import_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", str(_BASE_DIR / "test_images"),
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        if not path:
            return
        frame = cv2.imread(path)
        if frame is None:
            QMessageBox.warning(self, "错误", f"无法读取图片：{path}")
            return
        self._orig_frame = frame
        self._result_frame = None
        self._show_pixmap(self.lbl_orig, frame)
        self.lbl_result.setText("尚未检测")
        self.lbl_result.setPixmap(QPixmap())
        self.info_count.set_value("—")
        self.info_time.set_value("—")
        self.info_status.set_value("已导入，等待检测")
        self.status_msg.emit(f"已导入：{Path(path).name}")

    def start_detect(self):
        if self._orig_frame is None:
            QMessageBox.information(self, "提示", "请先导入图片。")
            return
        if not self.detector.is_loaded:
            QMessageBox.warning(self, "模型未加载", "请先在右侧参数面板加载模型权重。")
            return
        if self._thread and self._thread.isRunning():
            return

        self.info_status.set_value("检测中…")
        self._thread = ImageDetectThread(self.detector, self._orig_frame)
        self._thread.result_ready.connect(self._on_result)
        self._thread.error_signal.connect(self._on_error)
        self._thread.start()

    def save_result(self):
        if self._result_frame is None:
            QMessageBox.information(self, "提示", "尚无检测结果可保存。")
            return
        default_path = str(_output_dir("images") / _ts_filename("result", ".jpg"))
        path, _ = QFileDialog.getSaveFileName(
            self, "保存检测结果", default_path,
            "图片文件 (*.jpg *.png)"
        )
        if not path:
            return
        ok = cv2.imwrite(path, self._result_frame)
        if ok:
            QMessageBox.information(self, "保存成功", f"结果已保存：{path}")
        else:
            QMessageBox.warning(self, "保存失败", f"无法写入文件：{path}")

    def _on_result(self, result: np.ndarray, count: int, elapsed_ms: float):
        self._result_frame = result
        self._show_pixmap(self.lbl_result, result)
        self.info_count.set_value(str(count))
        self.info_time.set_value(f"{elapsed_ms:.1f} ms")
        self.info_status.set_value("检测完成")
        self.status_msg.emit(f"检测完成，发现 {count} 个目标，耗时 {elapsed_ms:.1f} ms")

    def _on_error(self, msg: str):
        self.info_status.set_value("推理出错")
        QMessageBox.warning(self, "检测错误", msg)

    def _show_pixmap(self, label: QLabel, frame: np.ndarray):
        px = ndarray_to_pixmap(frame)
        label.setPixmap(fit_pixmap(px, label.size()))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._orig_frame is not None:
            self._show_pixmap(self.lbl_orig, self._orig_frame)
        if self._result_frame is not None:
            self._show_pixmap(self.lbl_result, self._result_frame)


# ======================================================================
# 视频检测页（复用于摄像头检测，通过 mode 区分）
# ======================================================================
class VideoPage(QWidget):
    status_msg     = pyqtSignal(str)
    detect_finished = pyqtSignal()   # 检测自然结束或出错时通知 MainWindow 恢复导航栏

    MODE_VIDEO  = "video"
    MODE_CAMERA = "camera"

    def __init__(self, detector: DroneDetector, mode: str = MODE_VIDEO, parent=None):
        super().__init__(parent)
        self.detector = detector
        self.mode = mode
        self._thread: Optional[VideoDetectThread] = None
        self._last_frame: Optional[np.ndarray] = None
        self._video_path: Optional[str] = None
        self._total_frames: int = 0
        # 用于过滤旧线程延迟信号的版本号
        self._session_id: int = 0
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        # ---- 视频显示区 ----
        card = QWidget(); card.setObjectName("Card")
        cl = QVBoxLayout(card); cl.setContentsMargins(8, 8, 8, 8)

        placeholder = "请导入视频文件" if self.mode == self.MODE_VIDEO else "点击【开始检测】打开摄像头"
        self.lbl_video = QLabel(placeholder)
        self.lbl_video.setObjectName("DisplayLabel")
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_video.setMinimumSize(480, 320)
        cl.addWidget(self.lbl_video)

        # 进度条（仅视频模式显示）
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setStyleSheet(
            "QProgressBar { background:#DCEEFF; border-radius:2px; }"
            "QProgressBar::chunk { background:#3B82F6; border-radius:2px; }"
        )
        self.progress_bar.setVisible(self.mode == self.MODE_VIDEO)
        cl.addWidget(self.progress_bar)

        root.addWidget(card, 1)

        # ---- 信息栏 ----
        info_bar = QWidget(); info_bar.setObjectName("InfoBar")
        info_lay = QHBoxLayout(info_bar); info_lay.setContentsMargins(8, 4, 8, 4)
        self.info_count  = InfoItem("检测目标")
        self.info_time   = InfoItem("推理耗时")
        self.info_fps    = InfoItem("实时 FPS")
        self.info_status = InfoItem("状态")
        for w in [self.info_count, self.info_time, self.info_fps, self.info_status]:
            info_lay.addWidget(w)
            info_lay.addStretch()
        root.addWidget(info_bar)

    # ---- 外部接口 ----
    def import_video(self):
        """仅视频模式使用。"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", str(_BASE_DIR / "test_videos"),
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv)"
        )
        if not path:
            return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QMessageBox.warning(self, "错误", f"无法读取视频：{path}")
            return
        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 预览第一帧
        ret, frame = cap.read()
        cap.release()
        if ret:
            self._last_frame = frame
            self._show_pixmap(frame)
        self._video_path = path
        self.info_status.set_value("已导入，等待检测")
        self.status_msg.emit(f"已导入：{Path(path).name}  共 {self._total_frames} 帧")

    def start_detect(self, save_path: Optional[str] = None):
        # 若有旧线程仍在运行则拦截（防止重复点击）
        if self._thread and self._thread.isRunning():
            return
        if not self.detector.is_loaded:
            QMessageBox.warning(self, "模型未加载", "请先在右侧参数面板加载模型权重。")
            return

        if self.mode == self.MODE_VIDEO:
            if not self._video_path:
                QMessageBox.information(self, "提示", "请先导入视频文件。")
                return
            source = self._video_path
        else:
            source = 0  # 默认摄像头

        # 递增会话号，旧线程的延迟信号将被忽略
        self._session_id += 1
        current_session = self._session_id

        # 重置统计信息（新一轮开始）
        self.progress_bar.setValue(0)
        self.info_count.set_value("0")
        self.info_time.set_value("—")
        self.info_fps.set_value("—")
        self.info_status.set_value("检测中…")
        self.status_msg.emit("检测已开始")

        # 释放旧线程对象（已确认不在运行）
        self._thread = None

        thread = VideoDetectThread(
            self.detector, source, save_path, self._total_frames
        )
        # 用 lambda 捕获 current_session，过滤过时信号
        thread.frame_ready.connect(
            lambda f, c, e, fps, sid=current_session: self._on_frame(f, c, e, fps, sid)
        )
        thread.progress.connect(
            lambda cur, tot, sid=current_session: self._on_progress(cur, tot, sid)
        )
        thread.finished.connect(
            lambda msg, sid=current_session: self._on_finished(msg, sid)
        )
        thread.error_signal.connect(
            lambda msg, sid=current_session: self._on_error(msg, sid)
        )
        self._thread = thread
        self._thread.start()

    def pause_detect(self):
        if self._thread:
            self._thread.pause()
            self.info_status.set_value("已暂停")

    def resume_detect(self):
        if self._thread:
            self._thread.resume()
            self.info_status.set_value("检测中…")

    def stop_detect(self):
        if self._thread:
            self._thread.stop()
            self._thread.wait(3000)
            self._thread = None
        # 递增会话号，丢弃任何残留信号
        self._session_id += 1
        self.info_status.set_value("已停止")
        self.status_msg.emit("检测已停止")

    def export_video(self):
        """弹出保存对话框后重新运行视频检测并导出。"""
        if self.mode != self.MODE_VIDEO or not self._video_path:
            QMessageBox.information(self, "提示", "请先导入视频文件。")
            return
        if self._thread and self._thread.isRunning():
            QMessageBox.information(self, "提示", "请先停止当前检测后再导出。")
            return
        default_path = str(_output_dir("videos") / _ts_filename("video", ".mp4"))
        path, _ = QFileDialog.getSaveFileName(
            self, "保存检测视频", default_path,
            "视频文件 (*.mp4)"
        )
        if not path:
            return
        self.start_detect(save_path=path)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def _on_frame(self, frame: np.ndarray, count: int, elapsed_ms: float, fps: float,
                  sid: int = 0):
        # 忽略过时会话的信号
        if sid != self._session_id:
            return
        self._last_frame = frame
        self._show_pixmap(frame)
        self.info_count.set_value(str(count))
        self.info_time.set_value(f"{elapsed_ms:.1f} ms")
        self.info_fps.set_value(f"{fps:.1f}")

    def _on_progress(self, cur: int, total: int, sid: int = 0):
        if sid != self._session_id:
            return
        pct = int(cur / total * 100) if total > 0 else 0
        self.progress_bar.setValue(pct)

    def _on_finished(self, msg: str, sid: int = 0):
        if sid != self._session_id:
            return
        # 自然结束：清理线程引用，恢复导航栏
        self._thread = None
        self._session_id += 1   # 防止后续重复触发
        self.info_status.set_value("完成")
        self.status_msg.emit(msg)
        if self.progress_bar.isVisible():
            self.progress_bar.setValue(100)
        # 通知 MainWindow 恢复导航栏按钮状态
        self.detect_finished.emit()

    def _on_error(self, msg: str, sid: int = 0):
        if sid != self._session_id:
            return
        self._thread = None
        self._session_id += 1
        self.info_status.set_value("出错")
        self.status_msg.emit(f"错误：{msg}")
        self.detect_finished.emit()
        QMessageBox.warning(self, "检测错误", msg)

    def _show_pixmap(self, frame: np.ndarray):
        px = ndarray_to_pixmap(frame)
        self.lbl_video.setPixmap(fit_pixmap(px, self.lbl_video.size()))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._last_frame is not None:
            self._show_pixmap(self._last_frame)


# ======================================================================
# 左侧导航栏
# ======================================================================
class NavPanel(QWidget):
    mode_changed    = pyqtSignal(int)   # 0=图片, 1=视频, 2=摄像头
    action_import   = pyqtSignal()
    action_start    = pyqtSignal()
    action_pause    = pyqtSignal()
    action_resume   = pyqtSignal()
    action_stop     = pyqtSignal()
    action_save     = pyqtSignal()
    action_export   = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("NavPanel")
        self.setFixedWidth(210)
        self._current_mode = 0
        self._paused = False
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 16, 12, 16)
        lay.setSpacing(4)

        # 品牌标识
        brand = QLabel("DroneVision")
        brand.setObjectName("BrandLabel")
        brand.setAlignment(Qt.AlignCenter)
        lay.addWidget(brand)

        sub = QLabel("len")
        sub.setObjectName("DevLabel")
        sub.setAlignment(Qt.AlignCenter)
        lay.addWidget(sub)

        lay.addSpacing(10)
        sep = QFrame(); sep.setObjectName("HSep"); lay.addWidget(sep)
        lay.addSpacing(6)

        # 模式选择
        mode_lbl = QLabel("检测模式")
        mode_lbl.setObjectName("NavSectionLabel")
        lay.addWidget(mode_lbl)

        self._mode_btns = []
        for i, (icon_text, label) in enumerate([
            ("🖼", "图片检测"),
            ("🎬", "视频检测"),
            ("📷", "摄像头检测"),
        ]):
            btn = QPushButton(f"  {icon_text}  {label}")
            btn.setObjectName("NavModeBtn")
            btn.setCheckable(False)
            btn.setCursor(QCursor(Qt.PointingHandCursor))
            btn.clicked.connect(lambda checked, idx=i: self._on_mode(idx))
            lay.addWidget(btn)
            self._mode_btns.append(btn)

        lay.addSpacing(10)
        sep2 = QFrame(); sep2.setObjectName("HSep"); lay.addWidget(sep2)
        lay.addSpacing(6)

        # 操作按钮
        op_lbl = QLabel("操作")
        op_lbl.setObjectName("NavSectionLabel")
        lay.addWidget(op_lbl)

        self.btn_import = self._make_btn("  📂  导入图片/视频", primary=False)
        self.btn_import.clicked.connect(self.action_import)
        lay.addWidget(self.btn_import)

        self.btn_start = self._make_btn("  ▶  开始检测", primary=True)
        self.btn_start.clicked.connect(self.action_start)
        lay.addWidget(self.btn_start)

        self.btn_pause = self._make_btn("  ⏸  暂停", primary=False)
        self.btn_pause.clicked.connect(self._toggle_pause)
        self.btn_pause.setVisible(False)
        lay.addWidget(self.btn_pause)

        self.btn_stop = self._make_btn("  ⏹  停止检测", primary=False)
        self.btn_stop.clicked.connect(self.action_stop)
        self.btn_stop.setVisible(False)
        lay.addWidget(self.btn_stop)

        lay.addSpacing(6)
        sep3 = QFrame(); sep3.setObjectName("HSep"); lay.addWidget(sep3)
        lay.addSpacing(6)

        save_lbl = QLabel("结果")
        save_lbl.setObjectName("NavSectionLabel")
        lay.addWidget(save_lbl)

        self.btn_save = self._make_btn("  💾  保存当前结果", primary=False)
        self.btn_save.clicked.connect(self.action_save)
        lay.addWidget(self.btn_save)

        self.btn_export = self._make_btn("  📤  导出检测视频", primary=False)
        self.btn_export.clicked.connect(self.action_export)
        self.btn_export.setVisible(False)
        lay.addWidget(self.btn_export)

        lay.addStretch()

        # 底部
        footer = QLabel("DroneVision · YOLOv8")
        footer.setObjectName("DevLabel")
        footer.setAlignment(Qt.AlignCenter)
        footer.setWordWrap(True)
        lay.addWidget(footer)

        # 初始化选中状态
        self._set_active_mode(0)

    def _make_btn(self, text: str, primary: bool) -> QPushButton:
        btn = QPushButton(text)
        btn.setObjectName("ActionBtn" if primary else "ActionBtnSecondary")
        btn.setCursor(QCursor(Qt.PointingHandCursor))
        return btn

    def _on_mode(self, idx: int):
        self._set_active_mode(idx)
        self.mode_changed.emit(idx)

    def _set_active_mode(self, idx: int):
        self._current_mode = idx
        for i, btn in enumerate(self._mode_btns):
            btn.setProperty("active", "true" if i == idx else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        # 摄像头模式不显示导入按钮
        is_camera = (idx == 2)
        is_video  = (idx == 1)
        self.btn_import.setVisible(not is_camera)
        self.btn_export.setVisible(is_video)

    def _toggle_pause(self):
        if not self._paused:
            self._paused = True
            self.btn_pause.setText("  ▶  继续")
            self.action_pause.emit()
        else:
            self._paused = False
            self.btn_pause.setText("  ⏸  暂停")
            self.action_resume.emit()

    def set_running(self, running: bool):
        """检测运行中时切换按钮状态。"""
        self.btn_start.setVisible(not running)
        self.btn_pause.setVisible(running and self._current_mode != 0)
        self.btn_stop.setVisible(running and self._current_mode != 0)
        if not running:
            self._paused = False
            self.btn_pause.setText("  ⏸  暂停")


# ======================================================================
# 主窗口
# ======================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DroneVision · 无人机智能目标检测系统")
        self.setMinimumSize(960, 640)

        # 居中并自适应屏幕
        screen = QApplication.primaryScreen().availableGeometry()
        w = min(1250, screen.width() - 80)
        h = min(780, screen.height() - 80)
        self.resize(w, h)
        self.move(
            screen.x() + (screen.width() - w) // 2,
            screen.y() + (screen.height() - h) // 2
        )

        self.detector = DroneDetector()
        self._build_ui()
        self._connect_signals()

        # 尝试自动加载默认权重
        self.param_panel.try_load_default()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_lay = QHBoxLayout(central)
        main_lay.setContentsMargins(0, 0, 0, 0)
        main_lay.setSpacing(0)

        # ---- 左侧导航 ----
        self.nav = NavPanel()
        main_lay.addWidget(self.nav)

        # ---- 中间内容区 ----
        content_wrap = QWidget()
        content_wrap.setObjectName("Card")  # 统一背景
        content_wrap.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        cw_lay = QVBoxLayout(content_wrap)
        cw_lay.setContentsMargins(12, 10, 12, 8)
        cw_lay.setSpacing(8)

        # 顶部标题栏
        top_bar = QHBoxLayout()
        self.lbl_page_title = QLabel("图片检测")
        self.lbl_page_title.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #243B53;"
        )
        top_bar.addWidget(self.lbl_page_title)
        top_bar.addStretch()
        self.lbl_model_badge = QLabel("模型未加载")
        self.lbl_model_badge.setObjectName("ModelStatusNotReady")
        top_bar.addWidget(self.lbl_model_badge)
        cw_lay.addLayout(top_bar)

        sep = QFrame(); sep.setObjectName("HSep")
        cw_lay.addWidget(sep)

        # 页面栈
        self.stack = QStackedWidget()
        self.page_image  = ImagePage(self.detector)
        self.page_video  = VideoPage(self.detector, VideoPage.MODE_VIDEO)
        self.page_camera = VideoPage(self.detector, VideoPage.MODE_CAMERA)
        self.stack.addWidget(self.page_image)
        self.stack.addWidget(self.page_video)
        self.stack.addWidget(self.page_camera)
        cw_lay.addWidget(self.stack, 1)

        # 底部状态栏
        self.status_bar_lbl = QLabel("就绪")
        self.status_bar_lbl.setObjectName("StatusLabel")
        cw_lay.addWidget(self.status_bar_lbl)

        main_lay.addWidget(content_wrap, 1)

        # ---- 右侧参数面板 ----
        self.param_panel = ParamPanel(self.detector)
        main_lay.addWidget(self.param_panel)

    def _connect_signals(self):
        nav = self.nav

        nav.mode_changed.connect(self._on_mode_changed)
        nav.action_import.connect(self._on_import)
        nav.action_start.connect(self._on_start)
        nav.action_pause.connect(self._on_pause)
        nav.action_resume.connect(self._on_resume)
        nav.action_stop.connect(self._on_stop)
        nav.action_save.connect(self._on_save)
        nav.action_export.connect(self._on_export)

        # 页面状态消息
        self.page_image.status_msg.connect(self._set_status)
        self.page_video.status_msg.connect(self._set_status)
        self.page_camera.status_msg.connect(self._set_status)

        # 视频/摄像头检测自然结束或出错时，恢复导航栏
        self.page_video.detect_finished.connect(lambda: self.nav.set_running(False))
        self.page_camera.detect_finished.connect(lambda: self.nav.set_running(False))

        # 参数变化时刷新模型徽章
        self.param_panel.param_changed.connect(self._refresh_model_badge)

    def _current_video_page(self) -> VideoPage:
        idx = self.stack.currentIndex()
        return self.page_video if idx == 1 else self.page_camera

    def _on_mode_changed(self, idx: int):
        titles = ["图片检测", "视频检测", "摄像头实时检测"]
        self.lbl_page_title.setText(titles[idx])
        self.stack.setCurrentIndex(idx)
        # 停止旧页面的检测线程
        if idx != 1 and self.page_video.is_running():
            self.page_video.stop_detect()
        if idx != 2 and self.page_camera.is_running():
            self.page_camera.stop_detect()
        self.nav.set_running(False)

    def _on_import(self):
        idx = self.stack.currentIndex()
        if idx == 0:
            self.page_image.import_image()
        elif idx == 1:
            self.page_video.import_video()

    def _on_start(self):
        idx = self.stack.currentIndex()
        if idx == 0:
            self.page_image.start_detect()
        else:
            page = self._current_video_page()
            # 记录启动前状态，start_detect 内部验证失败时不会改变 is_running
            page.start_detect()
            # 只有线程确实启动才切换导航栏状态
            if page.is_running():
                self.nav.set_running(True)

    def _on_pause(self):
        self._current_video_page().pause_detect()

    def _on_resume(self):
        self._current_video_page().resume_detect()

    def _on_stop(self):
        self._current_video_page().stop_detect()
        self.nav.set_running(False)

    def _on_save(self):
        idx = self.stack.currentIndex()
        if idx == 0:
            self.page_image.save_result()
        else:
            # 视频/摄像头：保存当前帧截图
            page = self._current_video_page()
            if page._last_frame is None:
                QMessageBox.information(self, "提示", "尚无画面可保存。")
                return
            default_path = str(_output_dir("snapshots") / _ts_filename("snapshot", ".jpg"))
            path, _ = QFileDialog.getSaveFileName(
                self, "保存当前帧", default_path,
                "图片文件 (*.jpg *.png)"
            )
            if path:
                ok = cv2.imwrite(path, page._last_frame)
                if ok:
                    QMessageBox.information(self, "已保存", f"当前帧已保存：{path}")
                else:
                    QMessageBox.warning(self, "保存失败", f"无法写入文件：{path}")

    def _on_export(self):
        self.page_video.export_video()

    def _set_status(self, msg: str):
        self.status_bar_lbl.setText(msg)

    def _refresh_model_badge(self):
        if self.detector.is_loaded:
            name = Path(self.detector.model_path).name
            self.lbl_model_badge.setText(f"✔ {name}")
            self.lbl_model_badge.setObjectName("ModelStatusReady")
        else:
            self.lbl_model_badge.setText("模型未加载")
            self.lbl_model_badge.setObjectName("ModelStatusNotReady")
        self.lbl_model_badge.style().unpolish(self.lbl_model_badge)
        self.lbl_model_badge.style().polish(self.lbl_model_badge)

    def closeEvent(self, event):
        """关闭窗口时安全停止所有线程并释放资源。"""
        for page in [self.page_video, self.page_camera]:
            if page.is_running():
                page.stop_detect()
        event.accept()


# ======================================================================
# 程序入口
# ======================================================================

def _make_check_icon() -> str:
    """
    用 QPainter 在内存中绘制蓝底白色对号图标（16×16），
    保存为可写的临时路径，返回可嵌入 QSS image:url() 的路径。
    兼容 PyInstaller 打包：打包后 _BASE_DIR 只读，改写入系统临时目录。
    """
    import tempfile, os

    size = 16
    px = QPixmap(size, size)
    px.fill(Qt.transparent)

    painter = QPainter(px)
    painter.setRenderHint(QPainter.Antialiasing, True)

    # 蓝色圆角背景
    painter.setBrush(QBrush(QColor("#3B82F6")))
    painter.setPen(Qt.NoPen)
    painter.drawRoundedRect(QRectF(0, 0, size, size), 3, 3)

    # 白色对号（折线：左下→中→右上）
    pen = QPen(QColor("#FFFFFF"))
    pen.setWidth(2)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    check = QPolygonF([
        QPointF(3.0,  8.0),
        QPointF(6.5, 11.5),
        QPointF(13.0, 4.0),
    ])
    painter.drawPolyline(check)
    painter.end()

    # 写入可写目录：运行时用项目根目录，打包后用系统临时目录
    if getattr(sys, "frozen", False):
        # PyInstaller 打包后 _MEIPASS 只读，使用系统临时目录
        write_dir = Path(tempfile.gettempdir())
    else:
        write_dir = _BASE_DIR

    icon_path = write_dir / "_check_icon.png"
    px.save(str(icon_path), "PNG")
    return str(icon_path).replace("\\", "/")   # QSS url() 需要正斜杠


def main():
    if "--smoke-test" in sys.argv:
        sys.exit(_run_smoke_test())

    # 高 DPI 适配
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)

    # 生成对号图标并注入 QSS
    check_icon = _make_check_icon()
    dynamic_qss = QSS + f"""
QCheckBox::indicator:checked {{
    background: #3B82F6;
    border-color: #3B82F6;
    image: url("{check_icon}");
}}
"""
    app.setStyleSheet(dynamic_qss)

    # 全局字体
    font = QFont("Microsoft YaHei UI", 10)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


def _run_smoke_test() -> int:
    """Headless EXE verification for model loading and one real inference."""
    import json
    import tempfile
    import traceback

    def _arg_value(prefix: str) -> str:
        return next((arg[len(prefix):] for arg in sys.argv if arg.startswith(prefix)), "")

    output_arg = _arg_value("--smoke-output=")
    output_path = (
        Path(output_arg)
        if output_arg
        else Path(tempfile.gettempdir()) / "dronevision_smoke.json"
    )
    image_arg = _arg_value("--smoke-image=")
    report = {
        "success": False,
        "frozen": bool(getattr(sys, "frozen", False)),
        "executable": sys.executable,
    }

    try:
        import torch

        detector = DroneDetector()
        weight_path = detector.get_default_weight()
        report.update({
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "weight_path": weight_path,
        })
        if torch.cuda.is_available():
            report["cuda_device"] = torch.cuda.get_device_name(0)

        ok, message = detector.load_model(weight_path)
        report["load_message"] = message
        if not ok:
            raise RuntimeError(message)

        if image_arg:
            frame = cv2.imread(image_arg)
            if frame is None:
                raise RuntimeError(f"无法读取自检图片：{image_arg}")
            report["input"] = image_arg
        else:
            frame = np.zeros((640, 640, 3), dtype=np.uint8)
            report["input"] = "synthetic-black-640x640"

        _, count, elapsed_ms = detector.detect(frame)
        if detector.last_error:
            raise RuntimeError(detector.last_error)
        report.update({
            "success": True,
            "model_task": getattr(detector.model, "task", None),
            "input_shape": list(frame.shape),
            "detection_count": count,
            "elapsed_ms": round(elapsed_ms, 3),
        })
    except Exception as exc:
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0 if report["success"] else 1


if __name__ == "__main__":
    main()
