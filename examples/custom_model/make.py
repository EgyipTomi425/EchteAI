"""Build the custom test network, fill its BN statistics on 256 ImageNetV2 images (no weight training) and save
it with torch.save, as input for 43_static_analysis.py --module and 44_validate_static.py --module."""
import sys, torch
from pathlib import Path
from PIL import Image
from torchvision import transforms as T
sys.path.insert(0, str(Path(__file__).parent))
from mynet import MyNet
torch.manual_seed(1)
m = MyNet()
for mod in m.modules():
    if isinstance(mod, torch.nn.BatchNorm2d):
        mod.momentum = None                                  # cumulative running statistics
        torch.nn.init.uniform_(mod.weight, 0.5, 1.5); torch.nn.init.normal_(mod.bias, 0, 0.2)
tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
root = Path(__file__).resolve().parents[2] / "data" / "imagenetv2"
files = sorted(root.rglob("*.jpeg"))[::-1][:256]             # BN statistics from real images
m.train()
with torch.no_grad():
    for i in range(0, len(files), 32):
        m(torch.stack([tf(Image.open(p).convert("RGB")) for p in files[i:i + 32]]))
m.eval()
torch.save(m, Path(__file__).parent / "my_net.pt")
print("saved", sum(p.numel() for p in m.parameters()) / 1e6, "M params")
