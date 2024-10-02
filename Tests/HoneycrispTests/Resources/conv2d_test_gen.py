"""
Generates the JSON data for conv2d.json.gz

Will print the data to stdout; you must gzip it yourself.
"""

import json

import torch
import torch.nn as nn
import torch.nn.functional as F

items = []
for image_size in [(12, 11)]:
    for kernel_size in [(1, 1), (3, 2), (3, 3)]:
        for in_channels in [3, 4]:
            for out_channels in [3, 4]:
                for groups in [1, 2, 3]:
                    if in_channels % groups or out_channels % groups:
                        continue
                    for strides in [(1, 1), (2, 3)]:
                        for dilation in [(1, 1), (3, 4)]:
                            for padding in [(0, 0), (2, 3)]:
                                kernel = torch.arange(
                                    out_channels
                                    * in_channels
                                    // groups
                                    * kernel_size[0]
                                    * kernel_size[1]
                                ).reshape(out_channels, in_channels // groups, *kernel_size)
                                image = -torch.arange(
                                    in_channels * image_size[0] * image_size[1]
                                ).view(1, in_channels, *image_size)
                                image_param = nn.Parameter(image.float())
                                kernel_param = nn.Parameter(kernel.float())
                                try:
                                    out_tensor = F.conv2d(
                                        image_param,
                                        kernel_param,
                                        stride=strides,
                                        padding=padding,
                                        dilation=dilation,
                                        groups=groups,
                                    )
                                    out_tensor.backward(
                                        (torch.arange(out_tensor.numel()) * 4)
                                        .float()
                                        .reshape(out_tensor.shape)
                                    )
                                    output = out_tensor.int().flatten().tolist()
                                    image_grad = image_param.grad.int().flatten().tolist()
                                    kernel_grad = kernel_param.grad.int().flatten().tolist()
                                except:
                                    continue
                                items.append(
                                    dict(
                                        conv=dict(
                                            kernelSize=(*kernel_size, out_channels),
                                            imageSize=(*image_size, in_channels),
                                            stride=strides,
                                            dilation=dilation,
                                            padding=padding,
                                            groups=groups,
                                        ),
                                        outShape=out_tensor.shape,
                                        output=output,
                                        imageGrad=image_grad,
                                        kernelGrad=kernel_grad,
                                    )
                                )
print(json.dumps(items))
