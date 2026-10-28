import torch
import torch.nn as nn


def n_params(module, trainable_only=False):
    ps = module.parameters()
    if trainable_only:
        ps = (p for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in ps)


class LeNet5(nn.Module):
    def __init__(self, in_ch=3, img=64, num_classes=21, pool="avg"):
        super().__init__()
        Pool = nn.AvgPool2d if pool == "avg" else nn.MaxPool2d
        self.c1 = nn.Conv2d(in_ch, 6, kernel_size=5)
        self.s2 = Pool(2, 2)
        self.c3 = nn.Conv2d(6, 16, kernel_size=5)
        self.s4 = Pool(2, 2)
        self.n_flat = self._flat_dim(in_ch, img)
        self.f5 = nn.Linear(self.n_flat, 120)
        self.f6 = nn.Linear(120, 84)
        self.out = nn.Linear(84, num_classes)

    def _flat_dim(self, in_ch, img):
        with torch.no_grad():
            z = torch.zeros(1, in_ch, img, img)
            z = self.s4(self.c3(self.s2(self.c1(z))))
        return int(z.flatten(1).shape[1])

    def forward(self, x):
        x = self.s2(torch.relu(self.c1(x)))
        x = self.s4(torch.relu(self.c3(x)))
        x = x.flatten(1)
        x = torch.relu(self.f5(x))
        x = torch.relu(self.f6(x))
        return self.out(x)


def layer_table(model, img, in_ch=3):
    x = torch.zeros(1, in_ch, img, img)
    rows = []
    for name, layer in [("C1", model.c1), ("S2", model.s2),
                        ("C3", model.c3), ("S4", model.s4)]:
        x = layer(x)
        rows.append((name, "x".join(map(str, tuple(x.shape)[1:])), n_params(layer)))
    x = x.flatten(1)
    rows.append(("flatten", str(x.shape[1]), 0))
    for name, layer in [("F5", model.f5), ("F6", model.f6), ("OUT", model.out)]:
        x = layer(x)
        rows.append((name, str(x.shape[1]), n_params(layer)))
    return rows


def conv(in_ch, out_ch, k, bn, **kw):
    layers = [nn.Conv2d(in_ch, out_ch, k, bias=not bn, **kw)]
    if bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class InceptionBlock(nn.Module):
    def __init__(self, in_ch, f1, f3r, f3, f5r, f5, fp, bn=False):
        super().__init__()
        self.b1 = conv(in_ch, f1, 1, bn)
        self.b2 = nn.Sequential(conv(in_ch, f3r, 1, bn),
                                conv(f3r, f3, 3, bn, padding=1))
        self.b3 = nn.Sequential(conv(in_ch, f5r, 1, bn),
                                conv(f5r, f5, 5, bn, padding=2))
        self.b4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1),
                                conv(in_ch, fp, 1, bn))
        self.out_ch = f1 + f3 + f5 + fp

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


GOOGLENET_CFG = [
    ((32, 32, 48, 8, 16, 16), False),
    ((64, 48, 96, 12, 32, 32), True),
    ((96, 64, 128, 16, 48, 48), True),
    ((128, 96, 160, 24, 64, 64), False),
]


class GoogLeNetMini(nn.Module):
    def __init__(self, num_classes=21, cfg=GOOGLENET_CFG, bn=True, dropout=0.4):
        super().__init__()
        self.stem = nn.Sequential(
            conv(3, 32, 3, bn, padding=1),
            nn.MaxPool2d(2, 2),
            conv(32, 64, 3, bn, padding=1))
        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = 64
        for args, pool in cfg:
            blk = InceptionBlock(ch, *args, bn=bn)
            self.blocks.append(blk)
            self.pools.append(nn.MaxPool2d(2, 2) if pool else nn.Identity())
            ch = blk.out_ch
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(ch, num_classes)

    def forward(self, x):
        x = self.stem(x)
        for blk, pool in zip(self.blocks, self.pools):
            x = pool(blk(x))
        return self.fc(self.drop(self.gap(x).flatten(1)))


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.shortcut(x))


class ResNet18(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1))
        self.layer1 = self._stage(64, 64, 1)
        self.layer2 = self._stage(64, 128, 2)
        self.layer3 = self._stage(128, 256, 2)
        self.layer4 = self._stage(256, 512, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    @staticmethod
    def _stage(in_ch, out_ch, stride):
        return nn.Sequential(BasicBlock(in_ch, out_ch, stride),
                             BasicBlock(out_ch, out_ch, 1))

    def forward(self, x):
        x = self.stem(x)
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            x = layer(x)
        return self.fc(self.pool(x).flatten(1))


class PretrainedResnet(nn.Module):
    def __init__(self, num_classes=21, freeze_backbone=False):
        super().__init__()
        from transformers import ResNetForImageClassification
        base = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
        self.resnet = base.resnet
        self.classifier = nn.Linear(2048, num_classes, bias=True)
        if freeze_backbone:
            for p in self.resnet.parameters():
                p.requires_grad = False

    def forward(self, images):
        features = self.resnet(images).pooler_output
        features = features.squeeze(-1).squeeze(-1)
        return self.classifier(features)
