import setuptools

setuptools.setup(
    name="torch_tbhook",
    version="0.1.0",
    description="A PyTorch utility for exporting intermediate tensors within a forward path.",
    author="nandnor93",
    url="https://github.com/nandnor93/torch_tensorboard_hooks",
    install_requires=[
        "torch>=1.1.0",
        "torchvision",
        "tensorboard"
    ],
    packages=[
        "torch_tbhook"
    ]
)