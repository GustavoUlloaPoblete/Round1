#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 11:21:13 2025

@author: gustavo
"""

# from monai.visualize import plot_network
# import torch
# from monai.networks.nets import UNet
# C=3

# model = UNet(
#     spatial_dims=2,
#     in_channels=C,
#     out_channels=2,
#     channels=(64, 128, 256, 512, 1024),
#     strides=(2, 2, 2, 2),#no hay maxpooling
#     norm=("batch", {"affine": True}),
#     num_res_units=0
# ).to("cuda")

# x = torch.randn(1,1,96,96,96)
# plot_network(model, x, filename="unet_monai.png")


# # tx = torch.randn(1,3,160,192)
# # plot_network(model, tx, filename="unet_monai.png")




import torch
from monai.networks.nets import UNet
from torchviz import make_dot

model = UNet(spatial_dims=3, in_channels=1, out_channels=2,
             channels=(16,32,64,128), strides=(2,2,2), num_res_units=0)

x = torch.randn(1,1,96,96,96)
out = model(x)

dot = make_dot(out, params=dict(list(model.named_parameters())))
dot.render("unet_graph", format="png")
