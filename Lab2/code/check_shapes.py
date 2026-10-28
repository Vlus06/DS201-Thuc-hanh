import torch
import torch.nn as nn

from models import (BasicBlock, GoogLeNetMini, InceptionBlock, LeNet5,
                    ResNet18, layer_table, n_params)


def main():
    lenet = LeNet5(3, 64, 21)
    print("LeNet 64x64:", lenet.n_flat, f"{n_params(lenet):,}")
    for r in layer_table(lenet, 64):
        print(f"  {r[0]:<8}{r[1]:<12}{r[2]:>10,}")
    m32 = LeNet5(3, 32, 21)
    print("LeNet 32x32:", m32.n_flat, f"{n_params(m32):,}",
          "conv", n_params(m32.c1) + n_params(m32.c3))

    b1 = InceptionBlock(1, 16, 16, 24, 4, 8, 8)
    b2 = InceptionBlock(56, 32, 32, 48, 8, 16, 16)
    print("Inception 1:", b1(torch.rand(1, 1, 28, 28)).shape[1], f"{n_params(b1):,}")
    print("Inception 2:", b2(torch.rand(1, 56, 14, 14)).shape[1], f"{n_params(b2):,}")

    g = GoogLeNetMini(21)
    x = g.stem(torch.rand(2, 3, 64, 64))
    print("GoogLeNet stem:", tuple(x.shape), f"{n_params(g.stem):,}")
    for i, (blk, pool) in enumerate(zip(g.blocks, g.pools), 1):
        x = blk(x)
        print(f"  block {i}: out {tuple(x.shape)}  params {n_params(blk):,}")
        x = pool(x)
    print("GoogLeNet total:", f"{n_params(g):,}", "logits", tuple(g(torch.rand(2, 3, 64, 64)).shape))

    r18 = ResNet18(21)
    print("ResNet18:", f"{n_params(r18):,}")
    z = r18.stem(torch.rand(1, 3, 96, 96))
    print("  stem", tuple(z.shape), f"{n_params(r18.stem):,}")
    for name in ("layer1", "layer2", "layer3", "layer4"):
        layer = getattr(r18, name)
        z = layer(z)
        print(f"  {name}", tuple(z.shape), f"{n_params(layer):,}")
    for size in (64, 96, 112, 224):
        z = r18.stem(torch.rand(1, 3, size, size))
        for name in ("layer1", "layer2", "layer3", "layer4"):
            z = getattr(r18, name)(z)
        print("  input", size, "-> layer4", tuple(z.shape[1:]))
    try:
        from torchvision.models import resnet18 as tv
        t = tv(weights=None)
        t.fc = nn.Linear(512, 21)
        print("torchvision resnet18, 21 lop:", f"{n_params(t):,}")
    except ImportError:
        pass
    n_proj = [n for n, m in r18.named_modules()
              if isinstance(m, BasicBlock) and not isinstance(m.shortcut, nn.Identity)]
    print("block co projection:", n_proj)

    # câu hỏi kiểm tra nhanh
    c = nn.Conv2d(3, 16, 5, 1, 2)
    print("Conv2d(3,16,5,1,2):", tuple(c(torch.rand(32, 3, 64, 64)).shape), n_params(c))
    print("filter 3x3 tren 64 channel:", 3 * 3 * 64, "trong so +1 bias")
    print("MLP 512x512:", 512 * 512 * 3 * 512 * 512 * 16)
    print("Linear(2048,21):", n_params(nn.Linear(2048, 21)))
    print("Flatten head vs GAP head: ", 7 * 7 * 512 * 21 + 21, 512 * 21 + 21)


if __name__ == "__main__":
    main()
