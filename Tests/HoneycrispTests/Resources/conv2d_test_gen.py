"""
Generates the JSON data for conv2d.json.gz

Will print the data to stdout; you must gzip it yourself.
"""

import json

import torch
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
                                try:
                                    out_tensor = F.conv2d(
                                        image.float(),
                                        kernel.float(),
                                        stride=strides,
                                        padding=padding,
                                        dilation=dilation,
                                        groups=groups,
                                    )
                                    output = out_tensor.int().flatten().tolist()
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
                                    )
                                )
print(json.dumps(items))
