#!/usr/bin/env python3
import os
import argparse
import torch

try:
    import pynvml
except ImportError:
    pynvml = None


def bytes_to_mb(x):
    return int(x / 1024 / 1024)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-used-mb", type=int, default=500, help="显存占用小于等于该值认为空闲")
    parser.add_argument("--max-util", type=int, default=10, help="GPU 利用率小于等于该值认为空闲")
    args = parser.parse_args()

    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>"))
    print("torch.cuda.is_available() =", torch.cuda.is_available())

    if not torch.cuda.is_available():
        return

    if pynvml is None:
        print("\n缺少 pynvml，请先运行：pip install nvidia-ml-py3")
        return

    pynvml.nvmlInit()

    free_ids = []

    print()
    print("基于 PyTorch cuda 编号的 GPU 空闲情况:")
    print("-" * 110)
    print(
        f"{'torch id':<10} "
        f"{'name':<30} "
        f"{'used/total(MB)':<18} "
        f"{'free(MB)':<10} "
        f"{'util(%)':<8} "
        f"{'status'}"
    )
    print("-" * 110)

    for torch_id in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(torch_id)

        # 关键：NVML handle 按 PyTorch 的 cuda 编号取
        handle = pynvml.nvmlDeviceGetHandleByIndex(torch_id)

        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)

        used_mb = bytes_to_mb(mem.used)
        total_mb = bytes_to_mb(mem.total)
        free_mb = bytes_to_mb(mem.free)
        gpu_util = util.gpu

        is_free = used_mb <= args.max_used_mb and gpu_util <= args.max_util

        if is_free:
            status = "FREE"
            free_ids.append(torch_id)
        else:
            status = "BUSY"

        print(
            f"cuda:{torch_id:<5} "
            f"{name[:30]:<30} "
            f"{used_mb}/{total_mb:<16} "
            f"{free_mb:<10} "
            f"{gpu_util:<8} "
            f"{status}"
        )

    print("-" * 110)

    print("空闲 PyTorch CUDA 编号:", " ".join(map(str, free_ids)) if free_ids else "None")

    if free_ids:
        print()
        print("train.py 可用参数:")
        print("--gpus " + " ".join(map(str, free_ids)))

    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()