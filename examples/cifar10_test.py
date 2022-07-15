#    Copyright 2022 nandnor93 (https://github.com/nandnor93)

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from argparse import ArgumentParser
import sys

sys.path.insert(0, "..")

from torch_tbhook import TensorBoardHook

import os
import torch
import torch.nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models, utils

try:
    from tqdm import tqdm
except ImportError:
    def tdqm(i, *args, **kwargs):
        return i


class ConvBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, bias=True, batchnorm=False, relu=True, pool_size=None):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=bias)
        self.bn = torch.nn.BatchNorm2d(out_ch) if batchnorm else None
        self.pool = torch.nn.MaxPool2d(pool_size, pool_size) if pool_size else None
        self.relu = torch.nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.pool:
            x = self.pool(x)
        if self.relu:
            x = self.relu(x)
        return x


class FCBlock(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, relu=True, batchnorm=False):
        super().__init__()
        self.fc = torch.nn.Linear(in_features, out_features, bias=bias)
        self.bn = torch.nn.BatchNorm1d(out_features) if batchnorm else None
        self.relu = torch.nn.ReLU() if relu else None

    def forward(self, x):
        x = self.fc(x)
        if self.bn:
            x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x


class SmallVGG(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = ConvBlock(3, 64, bias=False, batchnorm=True)

        self.conv2 = ConvBlock(64, 128, bias=False, batchnorm=True, pool_size=2)
        # 16x16

        self.conv3_1 = ConvBlock(128, 256, bias=False, batchnorm=True)
        self.conv3_2 = ConvBlock(256, 256, bias=False, batchnorm=True, pool_size=2)
        # 8x8

        self.conv4_1 = ConvBlock(256, 512, bias=False, batchnorm=True)
        self.conv4_2 = ConvBlock(512, 512, bias=False, batchnorm=True, pool_size=2)
        # 4x4
        
        self.flatten = torch.nn.Flatten()

        self.fc6 = FCBlock(512*4*4, 1024, bias=False, batchnorm=True)
        self.fc7 = FCBlock(1024, 1024)
        self.fc8 = FCBlock(1024, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.flatten(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        return x


def train(model, device, optim, loader, criteria):
    model.train()
   
    for img, lbl in tqdm(loader, desc="Train"):
        optim.zero_grad()
        
        img = img.to(device)
        lbl = lbl.to(device)
        out = model(img)
        
        loss = criteria(out, lbl)
        loss.backward()
        optim.step()


def test(model, device, loader):
    model.eval()
    
    total = 0
    correct = 0
        
    for img, lbl in tqdm(loader, desc="Test"):
        img = img.to(device)
        lbl = lbl.to(device)
        with torch.no_grad():
            out = model(img)
            correct += (out.argmax(dim=1) == lbl).sum().item()
            total += out.shape[0]
    
    return correct / total


def main():
    parser = ArgumentParser(description="An example for TebsorBoardHook utility class.")
    parser.add_argument("--dataset-root", "-r", default="./", help="root directory for CIFAR-10 dataset.")
    parser.add_argument("--batch-size", "-b", type=int, default=256, help="batch size.")
    parser.add_argument("--epochs", "-e", type=int, default=10, help="number of epochs.")
    args = parser.parse_args()

    summary_writer = SummaryWriter()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = SmallVGG().to(device)
    summary_writer.add_graph(model, torch.rand((1, 3, 32, 32)).to(device))

    #################
    hooks = {}
    for name_block, block in model.named_children():  # "conv1", "conv2", ...
        for name_module, module in block.named_children():  # "conv", "bn", "relu"
            name = "%s/%s" % (name_block, name_module)
            hook = TensorBoardHook(summary_writer, name, module)
            hooks[name] = hook, module
    #################

    dataset_train = datasets.CIFAR10(
        root=args.dataset_root,
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )
    dataset_test = datasets.CIFAR10(
        root=args.dataset_root,
        train=False,
        transform=transforms.ToTensor(),
        download=True
    )

    dl_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True
    )
    dl_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False
    )

    optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    criteria = torch.nn.CrossEntropyLoss()

    for epoch in tqdm(range(args.epochs), desc="Training"):
        for hook, module in hooks.values():
            hook.register_forward(epoch, image=True, histogram=True)

        train(model, device, optim, dl_train, criteria)
        acc_test = test(model, device, dl_test)
        
        print("Epoch %d: %5.2f %%" % (epoch, acc_test * 100))
        summary_writer.add_scalar("Test/Accuracy", acc_test, global_step=epoch)
        
        for name_block, block in model.named_children():
            for name_param, param in block.named_parameters():
                name = "%s/%s" % (name_block, name_param)
                summary_writer.add_histogram(name, param, global_step=epoch)

if __name__ == "__main__":
    main()
