import platform

import torch


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    device = get_device()

    print(f"Python platform: {platform.platform()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Selected device: {device}")

    x = torch.randn(3, 3, device=device)
    y = x @ x.T

    print(f"Tensor device: {y.device}")
    print("Device test passed ✓")


if __name__ == "__main__":
    main()
