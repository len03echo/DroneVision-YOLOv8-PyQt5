"""
drone_detector.py
DroneVision —— YOLO 模型加载与推理模块
与 GUI 解耦，方便单独测试和后续打包。
"""

import time
import traceback
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np


# 默认权重路径（相对于本文件所在的项目根目录）
_SCRIPT_DIR = Path(__file__).parent
# 优先加载自训练的无人机专用模型
DEFAULT_WEIGHT = _SCRIPT_DIR / "models" / "drone_yolov8n_best.pt"
# 原始 COCO 预训练权重（仅供训练时迁移学习使用，不作为检测替代）
_COCO_WEIGHT   = _SCRIPT_DIR / "models" / "yolov8n.pt"


def _write_error_log(stage: str, details: str) -> str:
    """Persist a full traceback somewhere writable in source and frozen runs."""
    if getattr(__import__("sys"), "frozen", False):
        log_dir = Path.home() / "Documents" / "DroneVision" / "logs"
    else:
        log_dir = _SCRIPT_DIR / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "dronevision_error.log"
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{'=' * 72}\n{timestamp} [{stage}]\n{details}\n")
    return str(log_path)


class DroneDetector:
    """
    封装 YOLOv8 模型的加载与推理。
    全部推理在调用线程执行；GUI 通过 QThread 调用，不会阻塞主线程。
    """

    def __init__(self) -> None:
        self.model = None           # ultralytics.YOLO 实例
        self.model_path: str = ""   # 当前加载的权重路径
        self.is_loaded: bool = False
        self.last_error: str = ""

        # 推理参数（可由 GUI 运行时修改）
        self.conf: float = 0.25
        self.iou: float = 0.45
        self.imgsz: int = 640
        self.show_label: bool = True
        self.show_conf: bool = True

        # 检测框颜色 BGR
        self._box_color = (59, 130, 246)   # #3B82F6 主强调蓝
        self._text_bg   = (59, 130, 246)

    # ------------------------------------------------------------------
    # 模型管理
    # ------------------------------------------------------------------

    def load_model(self, weight_path: str) -> Tuple[bool, str]:
        """
        加载指定权重文件。
        返回 (success, message)。
        不抛出异常，所有错误以字符串形式返回。
        """
        path = Path(weight_path)
        if not path.exists():
            return False, f"权重文件不存在：{weight_path}"
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(path))
            self.model_path = str(path)
            self.is_loaded = True
            self.last_error = ""
            return True, f"模型加载成功：{path.name}"
        except Exception as e:
            self.model = None
            self.is_loaded = False
            self.last_error = traceback.format_exc()
            log_path = _write_error_log("load_model", self.last_error)
            return False, f"模型加载失败：{e}\n详细日志：{log_path}"

    def get_default_weight(self) -> str:
        """
        返回优先使用的默认权重路径。
        只返回自训练的无人机专用模型；若不存在则返回空字符串，
        由 GUI 提示用户手动选择，不以 COCO 通用模型冒充专用模型。
        """
        if DEFAULT_WEIGHT.exists():
            return str(DEFAULT_WEIGHT)
        return ""   # 空字符串表示无可用默认模型

    # ------------------------------------------------------------------
    # 核心推理
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, int, float]:
        """
        对单帧 BGR 图像执行推理，返回：
          (绘制了检测框的结果图, 检测目标数量, 推理耗时ms)
        未加载模型时直接返回原图。
        """
        if not self.is_loaded or self.model is None:
            return frame.copy(), 0, 0.0

        t0 = time.perf_counter()
        try:
            results = self.model(
                frame,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                verbose=False,
            )
            self.last_error = ""
        except Exception:
            self.last_error = traceback.format_exc()
            _write_error_log("detect", self.last_error)
            return frame.copy(), 0, 0.0
        elapsed_ms = (time.perf_counter() - t0) * 1000

        result_frame = frame.copy()
        count = 0
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                count = len(boxes)
                result_frame = self._draw_boxes(result_frame, boxes)

        return result_frame, count, elapsed_ms

    def _draw_boxes(self, frame: np.ndarray, boxes) -> np.ndarray:
        """在 frame 上绘制检测框、类别名和置信度。
        字体大小与检测框笔画粗细根据图像短边分辨率自适应，
        确保高分辨率图像缩小显示后文字仍清晰可读。
        """
        names = self.model.names if self.model else {0: "drone"}
        h_img, w_img = frame.shape[:2]

        # ---- 根据图像短边自适应字体与线宽 ----
        # 640p → scale≈1.0，1080p → ≈1.7，4K → ≈3.4
        short_side = min(h_img, w_img)
        scale_factor = short_side / 640.0

        font_scale  = max(0.65, min(1.6, 0.65 * scale_factor))  # 0.65 ~ 1.6
        font_thick  = max(1, round(1.5 * scale_factor))          # 字体笔画
        box_thick   = max(2, round(2.5 * scale_factor))          # 检测框线宽
        pad_x       = max(5, round(6 * scale_factor))            # 标签水平内边距
        pad_y       = max(4, round(5 * scale_factor))            # 标签垂直内边距

        font = cv2.FONT_HERSHEY_SIMPLEX

        for box in boxes:
            # 坐标
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id   = int(box.cls[0])
            conf_val = float(box.conf[0])
            label    = names.get(cls_id, str(cls_id))

            # 绘制检测框（自适应线宽）
            cv2.rectangle(frame, (x1, y1), (x2, y2), self._box_color, box_thick)

            # 组织标签文字
            if self.show_label and self.show_conf:
                text = f"{label} {conf_val:.2f}"
            elif self.show_label:
                text = label
            elif self.show_conf:
                text = f"{conf_val:.2f}"
            else:
                continue

            # 计算文字尺寸（含自适应字体参数）
            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, font_thick)

            # 标签背景高度
            tag_h = th + baseline + pad_y * 2

            # 优先放在框上方；若上方空间不足则放在框内顶部
            if y1 - tag_h >= 0:
                tag_y1 = y1 - tag_h
                tag_y2 = y1
                text_y = y1 - baseline - pad_y
            else:
                tag_y1 = y1
                tag_y2 = y1 + tag_h
                text_y = y1 + th + pad_y

            tag_x2 = min(x1 + tw + pad_x * 2, w_img)

            # 绘制标签背景（实心矩形）
            cv2.rectangle(
                frame,
                (x1, tag_y1),
                (tag_x2, tag_y2),
                self._box_color, -1
            )

            # 绘制白色文字，LINE_AA 抗锯齿
            cv2.putText(
                frame, text,
                (x1 + pad_x, text_y),
                font, font_scale,
                (255, 255, 255), font_thick, cv2.LINE_AA
            )

        return frame
