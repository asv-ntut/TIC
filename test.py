import torch
import torchvision

print("✅ torch version:", torch.__version__)
print("✅ torchvision version:", torchvision.__version__)
print("🧠 CUDA version:", torch.version.cuda)
print("💻 CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("🚀 GPU name:", torch.cuda.get_device_name(0))
else:
    print("⚠️ 目前使用的是 CPU")
