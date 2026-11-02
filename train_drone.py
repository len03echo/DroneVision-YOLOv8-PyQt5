"""
DroneVision - YOLOv8 无人机目标检测训练脚本
课程设计：无人机目标检测（DroneVision）

使用方法：
    直接在 PyCharm 中运行本文件，或命令行：
        python train_drone.py
    覆盖参数示例：
        python train_drone.py --epochs 50 --batch 4
    恢复中断训练：
        python train_drone.py --resume
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# ==============================================================
# 训练参数配置区 —— 在 PyCharm 中直接修改此处的默认值
# ==============================================================
DEFAULT_EPOCHS   = 80        # 训练轮数
DEFAULT_IMGSZ    = 640       # 输入图像尺寸
DEFAULT_BATCH    = 8         # 批大小（显存不足时改为 4）
DEFAULT_DEVICE   = "0"       # GPU 编号，多卡可写 "0,1"；CPU 调试填 "cpu"
DEFAULT_WORKERS  = 0         # Windows 下建议保持 0
DEFAULT_MODEL    = str(Path(__file__).parent / "models" / "yolov8n.pt")  # 预训练权重（已整理至models/）
DEFAULT_PROJECT  = "runs/train"   # 训练结果根目录
DEFAULT_NAME     = "drone_yolov8n"  # 本次实验子目录名
DEFAULT_RESUME   = False     # 是否恢复上次中断的训练
# ==============================================================


def parse_args():
    parser = argparse.ArgumentParser(description="DroneVision YOLOv8 训练脚本")
    parser.add_argument("--epochs",   type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--imgsz",    type=int,   default=DEFAULT_IMGSZ)
    parser.add_argument("--batch",    type=int,   default=DEFAULT_BATCH)
    parser.add_argument("--device",   type=str,   default=DEFAULT_DEVICE)
    parser.add_argument("--workers",  type=int,   default=DEFAULT_WORKERS)
    parser.add_argument("--model",    type=str,   default=DEFAULT_MODEL)
    parser.add_argument("--project",  type=str,   default=DEFAULT_PROJECT)
    parser.add_argument("--name",     type=str,   default=DEFAULT_NAME)
    parser.add_argument("--resume",   action="store_true", default=DEFAULT_RESUME,
                        help="恢复上次中断的训练（自动定位 last.pt）")
    return parser.parse_args()


def resolve_project(project: str) -> str:
    """将 project 路径解析为项目根目录下的绝对路径，避免受 PyCharm 工作目录影响。"""
    p = Path(project)
    if not p.is_absolute():
        p = Path(__file__).parent / p
    return str(p.resolve())


def check_gpu(device: str) -> None:
    """检查 CUDA 可用性，device='cpu' 时跳过检查。"""
    if device.lower() == "cpu":
        print("[警告] 当前配置为 CPU 训练，速度极慢，建议仅用于调试。")
        confirm = input("确认使用 CPU 继续？(y/N): ").strip().lower()
        if confirm != "y":
            print("已取消。请检查 CUDA 环境后重新运行。")
            sys.exit(0)
        return

    try:
        import torch
        if not torch.cuda.is_available():
            print("[错误] CUDA 不可用，但 device 配置为 GPU。")
            print("  请检查：NVIDIA 驱动、CUDA Toolkit、PyTorch GPU 版本是否匹配。")
            print("  如需强制使用 CPU（不推荐），请将 DEFAULT_DEVICE 改为 'cpu' 后重新运行。")
            sys.exit(1)
        gpu_idx  = int(device.split(",")[0])
        gpu_name = torch.cuda.get_device_name(gpu_idx)
        total_mem = torch.cuda.get_device_properties(gpu_idx).total_memory / 1024 ** 3
        print(f"[GPU] {gpu_name}  显存: {total_mem:.1f} GB")
    except ImportError:
        print("[错误] 无法导入 torch，请确认已在 yolov8_pyqt5 环境中运行。")
        sys.exit(1)


def find_last_checkpoint(project: str, name: str) -> Optional[Path]:
    """定位上次训练保存的 last.pt。"""
    last_pt = Path(project) / name / "weights" / "last.pt"
    if last_pt.exists():
        return last_pt
    return None


def train(args) -> Path:
    """执行训练，返回 best.pt 路径。"""
    from ultralytics import YOLO

    # 定位 data.yaml（相对于本脚本所在目录）
    script_dir = Path(__file__).parent
    data_yaml  = script_dir / "dataset" / "data.yaml"

    if not data_yaml.exists():
        print(f"[错误] 找不到数据集配置文件: {data_yaml}")
        sys.exit(1)

    # 处理恢复训练
    if args.resume:
        last_pt = find_last_checkpoint(args.project, args.name)
        if last_pt is None:
            print(f"[错误] 未找到可恢复的 last.pt，路径: {args.project}/{args.name}/weights/last.pt")
            print("  请确认之前已有训练记录，或去掉 --resume 参数重新开始。")
            sys.exit(1)
        print(f"[恢复训练] 从 {last_pt} 继续...")
        model = YOLO(str(last_pt))
        resume_flag = True
    else:
        print(f"[新建训练] 加载预训练权重: {args.model}")
        model = YOLO(args.model)
        resume_flag = False

    print(f"\n{'='*50}")
    print(f"  数据集  : {data_yaml}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  ImgSz   : {args.imgsz}")
    print(f"  Batch   : {args.batch}")
    print(f"  Device  : {args.device}")
    print(f"  输出目录: {args.project}/{args.name}")
    print(f"{'='*50}\n")

    # 开始训练
    results = model.train(
        data     = str(data_yaml),
        epochs   = args.epochs,
        imgsz    = args.imgsz,
        batch    = args.batch,
        device   = args.device,
        workers  = args.workers,
        project  = args.project,
        name     = args.name,
        resume   = resume_flag,
        exist_ok = True,       # 同名目录继续写入，不自动编号
        pretrained = True,
        # 训练曲线与指标自动保存至 project/name/
    )

    best_pt = Path(args.project) / args.name / "weights" / "best.pt"
    return best_pt


def evaluate(best_pt: Path, args) -> None:
    """使用 best.pt 对 test 集进行评估。"""
    from ultralytics import YOLO

    if not best_pt.exists():
        print(f"[警告] 找不到 best.pt: {best_pt}，跳过评估。")
        return

    script_dir = Path(__file__).parent
    data_yaml  = script_dir / "dataset" / "data.yaml"

    print(f"\n{'='*50}")
    print(f"  评估阶段：使用 best.pt 对 test 集推理")
    print(f"{'='*50}\n")

    model = YOLO(str(best_pt))
    metrics = model.val(
        data    = str(data_yaml),
        split   = "test",       # 使用 data.yaml 中的 test 路径
        imgsz   = args.imgsz,
        batch   = args.batch,
        device  = args.device,
        workers = args.workers,
        project = args.project,
        name    = args.name + "_eval",
        exist_ok = True,
    )

    # 输出核心指标
    print("\n========== 评估结果 ==========")
    print(f"  Precision  : {metrics.box.mp:.4f}")
    print(f"  Recall     : {metrics.box.mr:.4f}")
    print(f"  mAP50      : {metrics.box.map50:.4f}")
    print(f"  mAP50-95   : {metrics.box.map:.4f}")
    print("==============================\n")


if __name__ == "__main__":
    args = parse_args()

    # 1. GPU 检查
    check_gpu(args.device)

    # 将输出路径统一解析为绝对路径，避免 PyCharm 工作目录变化导致路径错误
    args.project = resolve_project(args.project)

    # 2. 训练
    best_pt = train(args)

    # 3. 评估
    evaluate(best_pt, args)

    print("[完成] 训练与评估已结束。")
    print(f"  权重文件 : {args.project}/{args.name}/weights/best.pt")
    print(f"  训练曲线 : {args.project}/{args.name}/results.png")
