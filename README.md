# torch_tensorboard_hooks
PyTorch utility classes for exporting intermediate tensors in a Module to TensorBoard.

## Overview

The utility class `TensorBoardHook` helps add a forward hook to an nn.Module instance so that the intermediate tensors during the forward path.  
The `nn.Module.register_forward_hook()` method allows us to add a "hook" that receives the in-situ inputs and outputs of the module.  
`TensorBoardHook` registers itself as a hook to an `nn.Module` with proper names and arguments.


## Installation

```
git clone https://github.com/nandnor93/torch_tensorboard_hooks.git
pip install ./torch_tensorboard_hooks
```

## How to use

First, instantiate a model (an `nn.Module` subclass) as usual.

```python:
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(...)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(...)
        # ........
    def forward(self, x):
        return # ........

model = MyModel()
```

Initialize a `SummaryWriter` instance.

```python:
writer = torch.utils.tensorboard.SummaryWriter()
```

Then, obtain `TensorBoardHook` instances.

```python:
hooks = {
    "conv1", TensorBoardHook(writer, "conv1", model.conv1)
    "relu1", TensorBoardHook(writer, "relu1", model.relu1)
    "conv2", TensorBoardHook(writer, "conv2", model.conv2)
}
```

During the training epoch loop, kick the hook before the first batch flows into the model.

```python:
for epoch in range(EPOCH):
    model.train()
    optim.zero_grad()
    
    hooks["conv1"].register_forward(global_step=epoch)
    hooks["relu"].register_forward(global_step=epoch)
    hooks["conv2"].register_forward(global_step=epoch)
    
    for image, label in data_loader:
        out = model(image)
        loss = criteria(out, label)
        loss.backward()
    # ......
```

INFO: you may want to use `nn.Module.named_children()` or `nn.Module.named_modules()` method.

