#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 14:07:51 2024

@author: gustavo
"""

import torch
from torch import nn
import cv2
import numpy as np
# from torch.nn.modules.loss import _Loss
# import lovasz_losses as L
# from skimage import morphology

eps = 1e-7

# P y G tienen 2 mapas de características (N,2,H,W), uno para la clase background y uno para la clase foreground
def Matriz_confusion_2MC(P, G, batch_loss=True):# batch=True para cada ejemplo del batch por separado
    # print(f"batch_loss: {batch_loss}")
    N = len(P) # batch size
    if batch_loss:
        # print(f"batch_loss True: {batch_loss}")
        axis=1
    else:
        # print(f"batch_loss False: {batch_loss}")
        axis=None
    TP = torch.sum(torch.multiply(G[:,1], P[:,1]).view(N,-1), axis)
    FP = torch.sum(torch.multiply(G[:,0], P[:,1]).view(N,-1), axis)
    FN = torch.sum(torch.multiply(G[:,1], P[:,0]).view(N,-1), axis)
    TN = torch.sum(torch.multiply(G[:,0], P[:,0]).view(N,-1), axis)
    return TN, FP, FN, TP

def Dice_metric(P, G, batch_loss=True):# batch=True para cada ejemplo del batch por separado
    # print(f"batch_loss: {batch_loss}")
    N = len(P) # batch size
    if batch_loss:
        # print(f"batch_loss True: {batch_loss}")
        axis=1
    else:
        # print(f"batch_loss False: {batch_loss}")
        axis=None
    numerador = 2*torch.sum(torch.multiply(G[:,1], P[:,1]).view(N,-1), axis) + eps
    denominador = torch.sum(G[:,1].view(N,-1), axis) + torch.sum(P[:,1].view(N,-1), axis) + eps
    dice = numerador/denominador
    return dice

def Generalized_dice_metric(P, G, batch_loss=True): # G.shape=P.shape=N2HW(img 2D) o N2HWD(img 3D)
    N = len(P) # batch size
    if batch_loss:
        axis=1
    else:
        axis=None
    G = G.view([N,2,-1])
    P = P.view([N,2,-1])
    
    w0 = ( 1/(torch.sum(G[:,0], axis)**2+eps)).view([N,1])
    w1 = ( 1/(torch.sum(G[:,1], axis)**2+eps)).view([N,1])
    numerador = 2*(torch.sum(w0*torch.multiply(G[:,0], P[:,0]), axis) + torch.sum(w1*torch.multiply(G[:,1], P[:,1]), axis))
    denominador = ( (torch.sum(w0*(G[:,0]), axis) + torch.sum(w0*(P[:,0]), axis)) + (torch.sum(w1*G[:,1], axis) + torch.sum(w1*P[:,1], axis)))
    dice = (numerador + eps)/(denominador + eps)
    return dice# size N

class BCE_loss(nn.Module):
    
    def __init__(self):
        super(BCE_loss, self).__init__()

    def forward(self, y_pred, y_true, batch_loss=True):
        # '''
        # El promedio de los promedios de las filas es igual al promedio de todo...,
        # por tanto, no es necesario generar una pérdida por cada ejemplo del batch
        # '''
        # print('BCE:')
        # N = len(y_true) # batch size
        Y0 = y_true[:,0]*torch.log(torch.clamp(y_pred[:,0], min=eps, max=1))  # FP
        Y1 = y_true[:,1]*torch.log(torch.clamp(y_pred[:,1], min=eps, max=1))  # FN
        loss = -Y0 - Y1
        # print(f'loss:{loss.shape}, {torch.mean(loss)}')
        # El promedio de los promedios de las filas es igual al promedio de todo...
        # loss = loss.view(N,-1)
        # BL_cardinalidad = loss.shape[1]
        # print(f'BL_cardinalidad:{BL_cardinalidad}')
        # print(f'loss:{loss.shape}, {torch.mean(loss)}')
        # loss = torch.sum(loss, axis = 1)/BL_cardinalidad
        # print(f'loss:{loss.shape}, {torch.mean(loss)}')
        return torch.mean(loss)

class Dice_loss(nn.Module):
    
    def __init__(self):
        super(Dice_loss, self).__init__()

    def forward(self, y_pred, y_true, batch_loss=True):# axis=1 para cada ejemplo del batch por separado
        # print('Dice_loss new:')
        # N, C, H, W = y_true.shape
        # tn, fp, fn, tp = Matriz_confusion_2MC(y_pred, y_true, batch_loss)
        dice = Dice_metric(y_pred, y_true, batch_loss)
        # print(f'dice:{dice} {dice.shape} {dice.dtype}')
        loss = 1-dice
        # print(f'loss:{loss} {loss.shape} {loss.dtype}')
        return torch.mean(loss)

class GDL(nn.Module):
    
    def __init__(self):
        super(GDL, self).__init__()

    def forward(self, y_pred, y_true, batch_loss=True):# axis=1 para cada ejemplo del batch por separado
        # print('GDL:')
        gdice = Generalized_dice_metric(y_pred, y_true)
        loss = 1-gdice
        # print(f'loss:{loss}')
        return torch.mean(loss)

class Boundary_loss(nn.Module):
    
    def __init__(self):
        super(Boundary_loss, self).__init__()
    
    def forward(self, y_pred, y_true, G_sdf, alpha, batch=True):
        # print('Boundary_loss:')
        # print(f'y_pred:{y_pred.shape}, y_true:{y_true.shape}, G_sdf:{G_sdf.shape}')
        N, C, H, W = y_true.shape
        # tn, fp, fn, tp = Matriz_confusion_2MC(y_pred, y_true, batch)
        # dice = Dice_metric(tn, fp, fn, tp)
        # dice_loss = 1-dice
        
        gdice = Generalized_dice_metric(y_pred, y_true)
        gdice_loss = 1-gdice
        
        producto = y_pred[:,1]*G_sdf
        if batch:
            BL_sumatoria = torch.sum(producto.view(N,-1), 1)
            BL_cardinalidad = np.prod(producto.shape[1:])
        else:
            BL_sumatoria = torch.sum(producto.view(N,-1))
            BL_cardinalidad = np.prod(producto.shape[:])
        B_loss = BL_sumatoria/BL_cardinalidad
        
        gdice_loss_cpu = gdice_loss.detach().cpu().numpy()
        B_loss_cpu = B_loss.cpu().detach().numpy()
        print(f'gdice_lc:{gdice_loss_cpu.shape} {np.around(gdice_loss_cpu[:],4)}')
        print(f'BL_lc:{B_loss_cpu.shape} {np.around(B_loss_cpu[:],4)}, mean:{np.around(B_loss_cpu[:].mean(),4)} std:{np.around(B_loss_cpu[:].std(),4)}')
        
        loss = alpha*gdice_loss + (1-alpha)*B_loss
        # print(f"loss: {loss} {loss.shape}, torch.mean(loss):{torch.mean(loss)} {loss.dtype}")
        return torch.mean(loss)

class MD_loss(nn.Module):
    '''
    - Pondera Y_MDF por parMD_pot
    '''
    def __init__(self):
        super(MD_loss, self).__init__()
        
    def forward(self, pred, Y, alpha, Y_MDF, parMD_weight, parMD_pot):
        # print('\nMD_loss_B_new:')
        N, C, H, W = Y.shape
        
        gdice = Generalized_dice_metric(pred, Y)
        gdice_loss = 1-gdice
        
        array_FP = Y[:,0]*pred[:,1]
        array_FN = Y[:,1]*pred[:,0]
        array_FPFN = array_FP+array_FN
        # print(f'array_FPFN:{array_FPFN.shape} {array_FPFN.get_device()}')
        producto = array_FPFN * parMD_weight*Y_MDF**parMD_pot
        # print(f'producto:{producto.shape} {producto.dtype} {torch.nanmean(producto)} suma:{torch.sum(producto,axis=(1,2))}')
        # print(f'ASSD:{ASSD.shape} {ASSD}')
        
        sumatoria = torch.sum(producto,axis=(1,2))
        # print(f'sumatoria:{sumatoria.shape} {sumatoria}')
        cardinalidad = np.prod(producto.shape[1:])
        MD_loss = sumatoria/cardinalidad
        
        loss = alpha*gdice_loss + (1-alpha)*MD_loss
        # loss = alpha*gdice_loss + (1-alpha)*(MD_loss + parMD_weight*MD_loss_term2)
        # loss = torch.nansum(alpha*gdice_loss + (1-alpha)*MD_loss)
        # print(f"loss: {loss} {loss.shape}, torch.mean(loss):{torch.mean(loss)} {loss.dtype}")
        return torch.mean(loss)


# =============================================================================
#     
# =============================================================================

class HD_loss(nn.Module):
    
    def __init__(self):
        super(HD_loss, self).__init__()
    
    def forward(self, y_pred, y_true, G_dtm, S_dtm, alpha, batch=True):
        # print('HD_loss:')
        N, C, H, W = y_true.shape
        # tn, fp, fn, tp = Matriz_confusion_2MC(y_pred, y_true, batch)
        # dice = Dice_metric(tn, fp, fn, tp)
        # dice_loss = 1-dice
        gdice = Generalized_dice_metric(y_pred, y_true)
        gdice_loss = 1-gdice
        # print(f'HD_loss con gdice_loss:{gdice_loss.shape} {gdice_loss}')
        
        alpha_HD = 2
        interior_sumatoria = torch.pow((y_pred[:,1]-y_true[:,1]), 2) * ( torch.pow(G_dtm,alpha_HD)+torch.pow(S_dtm, alpha_HD) )
        if batch:
            HD_cardinalidad = np.prod(interior_sumatoria.shape[1:])
            hd_loss = torch.sum(interior_sumatoria.view(N,-1), 1)/HD_cardinalidad
        else:
            HD_cardinalidad = np.prod(interior_sumatoria.shape[:])
            hd_loss = torch.sum(interior_sumatoria.view(N,-1))/HD_cardinalidad

        # gdice_loss_cpu = gdice_loss.detach().cpu().numpy()
        # hd_loss_cpu = hd_loss.cpu().detach().numpy()
        # print(f'gdice_lc:{gdice_loss_cpu.shape} {np.around(gdice_loss_cpu[:],4)}')
        # print(f'hd_loss:{hd_loss_cpu.shape} {np.around(hd_loss_cpu[:],4)}')
        
        loss = alpha*gdice_loss + (1-alpha)*hd_loss
        # print(f"loss: {loss} {loss.shape}, torch.mean(loss):{torch.mean(loss)} {loss.dtype}")
        return torch.mean(loss)

    
# =============================================================================
# ABL: Active Boundary Loss (2022)
# =============================================================================
# import torch
# import torch.nn as nn
import torch.nn.functional as F

# import numpy as np
from scipy.ndimage import distance_transform_edt as distance
# can find here: https://github.com/CoinCheung/pytorch-loss/blob/af876e43218694dc8599cc4711d9a5c5e043b1b2/label_smooth.py
# from .label_smooth import LabelSmoothSoftmaxCEV1 as LSSCE
from label_smooth import LabelSmoothSoftmaxCEV1 as LSSCE
from torchvision import transforms
from functools import partial
from operator import itemgetter
# Tools
def kl_div(a,b): # q,p
    return F.softmax(b, dim=1) * (F.log_softmax(b, dim=1) - F.log_softmax(a, dim=1))   

def one_hot2dist(seg):
    res = np.zeros_like(seg)
    for i in range(len(seg)):
        # posmask = seg[i].astype(np.bool)###
        posmask = seg[i].astype(np.bool_)
        if posmask.any():
            negmask = ~posmask
            res[i] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res

def class2one_hot(seg, C):
    seg = seg.unsqueeze(dim=0) if len(seg.shape) == 2 else seg
    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    return res

# Active Boundary Loss
class ABL(nn.Module):
    def __init__(self, isdetach=True, max_N_ratio = 1/100, ignore_label = 255, label_smoothing=0.2, weight = None, max_clip_dist = 20.):
        super(ABL, self).__init__()
        self.ignore_label = ignore_label
        self.label_smoothing = label_smoothing
        self.isdetach=isdetach
        self.max_N_ratio = max_N_ratio

        self.weight_func = lambda w, max_distance=max_clip_dist: torch.clamp(w, max=max_distance) / max_distance

        self.dist_map_transform = transforms.Compose([
            lambda img: img.unsqueeze(0),
            lambda nd: nd.type(torch.int64),
            partial(class2one_hot, C=1),
            itemgetter(0),
            lambda t: t.cpu().numpy(),
            one_hot2dist,
            lambda nd: torch.tensor(nd, dtype=torch.float32)
        ])

        if label_smoothing == 0:
            self.criterion = nn.CrossEntropyLoss(
                weight=weight,
                ignore_index=ignore_label,
                reduction='none'
            )
        else:
            self.criterion = LSSCE(
                reduction='none',
                ignore_index=ignore_label,
                lb_smooth = label_smoothing
            )

    def logits2boundary(self, logit):
        eps = 1e-5
        _, _, h, w = logit.shape
        max_N = (h*w) * self.max_N_ratio
        kl_ud = kl_div(logit[:, :, 1:, :], logit[:, :, :-1, :]).sum(1, keepdim=True)
        kl_lr = kl_div(logit[:, :, :, 1:], logit[:, :, :, :-1]).sum(1, keepdim=True)
        kl_ud = torch.nn.functional.pad(
            kl_ud, [0, 0, 0, 1, 0, 0, 0, 0], mode='constant', value=0)
        kl_lr = torch.nn.functional.pad(
            kl_lr, [0, 1, 0, 0, 0, 0, 0, 0], mode='constant', value=0)
        kl_combine = kl_lr+kl_ud
        while True: # avoid the case that full image is the same color
            kl_combine_bin = (kl_combine > eps).to(torch.float)
            if kl_combine_bin.sum() > max_N:
                eps *=1.2
            else:
                break
        #dilate
        dilate_weight = torch.ones((1,1,3,3)).cuda()
        edge2 = torch.nn.functional.conv2d(kl_combine_bin, dilate_weight, stride=1, padding=1)
        edge2 = edge2.squeeze(1)  # NCHW->NHW
        kl_combine_bin = (edge2 > 0)
        return kl_combine_bin

    def gt2boundary(self, gt, ignore_label=-1):  # gt NHW
        gt_ud = gt[:,1:,:]-gt[:,:-1,:]  # NHW
        gt_lr = gt[:,:,1:]-gt[:,:,:-1]
        gt_ud = torch.nn.functional.pad(gt_ud, [0,0,0,1,0,0], mode='constant', value=0) != 0 
        gt_lr = torch.nn.functional.pad(gt_lr, [0,1,0,0,0,0], mode='constant', value=0) != 0
        gt_combine = gt_lr+gt_ud
        del gt_lr
        del gt_ud
        
        # set 'ignore area' to all boundary
        gt_combine += (gt==ignore_label)
        
        return gt_combine > 0

    def get_direction_gt_predkl(self, pred_dist_map, pred_bound, logits):
        # NHW,NHW,NCHW
        eps = 1e-5
        # bound = torch.where(pred_bound)  # 3k
        bound = torch.nonzero(pred_bound*1)
        n,x,y = bound.T
        max_dis = 1e5

        logits = logits.permute(0,2,3,1) # NHWC

        pred_dist_map_d = torch.nn.functional.pad(pred_dist_map,(1,1,1,1,0,0),mode='constant', value=max_dis) # NH+2W+2

        logits_d = torch.nn.functional.pad(logits,(0,0,1,1,1,1,0,0),mode='constant') # N(H+2)(W+2)C
        logits_d[:,0,:,:] = logits_d[:,1,:,:] # N(H+2)(W+2)C
        logits_d[:,-1,:,:] = logits_d[:,-2,:,:] # N(H+2)(W+2)C
        logits_d[:,:,0,:] = logits_d[:,:,1,:] # N(H+2)(W+2)C
        logits_d[:,:,-1,:] = logits_d[:,:,-2,:] # N(H+2)(W+2)C
        
        """
        | 4| 0| 5|
        | 2| 8| 3|
        | 6| 1| 7|
        """
        x_range = [1, -1,  0, 0, -1,  1, -1,  1, 0]
        y_range = [0,  0, -1, 1,  1,  1, -1, -1, 0]
        dist_maps = torch.zeros((0,len(x))).cuda() # 8k
        kl_maps = torch.zeros((0,len(x))).cuda() # 8k

        kl_center = logits[(n,x,y)] # KC

        for dx, dy in zip(x_range, y_range):
            dist_now = pred_dist_map_d[(n,x+dx+1,y+dy+1)]
            dist_maps = torch.cat((dist_maps,dist_now.unsqueeze(0)),0)

            if dx != 0 or dy != 0:
                logits_now = logits_d[(n,x+dx+1,y+dy+1)]
                # kl_map_now = torch.kl_div((kl_center+eps).log(), logits_now+eps).sum(2)  # 8KC->8K
                if self.isdetach:
                    logits_now = logits_now.detach()
                kl_map_now = kl_div(kl_center, logits_now)
                
                kl_map_now = kl_map_now.sum(1)  # KC->K
                kl_maps = torch.cat((kl_maps,kl_map_now.unsqueeze(0)),0)
                torch.clamp(kl_maps, min=0.0, max=20.0)

        # direction_gt shound be Nk  (8k->K)
        direction_gt = torch.argmin(dist_maps, dim=0)
        # weight_ce = pred_dist_map[bound]
        weight_ce = pred_dist_map[(n,x,y)]
        # print(weight_ce)

        # delete if min is 8 (local position)
        direction_gt_idx = [direction_gt!=8]
        direction_gt = direction_gt[direction_gt_idx]


        kl_maps = torch.transpose(kl_maps,0,1)
        direction_pred = kl_maps[direction_gt_idx]
        weight_ce = weight_ce[direction_gt_idx]

        return direction_gt, direction_pred, weight_ce

    def get_dist_maps(self, target):
        target_detach = target.clone().detach()
        dist_maps = torch.cat([self.dist_map_transform(target_detach[i]) for i in range(target_detach.shape[0])])
        out = -dist_maps
        out = torch.where(out>0, out, torch.zeros_like(out))
        
        return out

    # logits=y_pred(NCHW); target=GTB(NHW), y_true(NCHW), donde C=2
    # logits in [0,1] (deducido por el ejemplo al final __main__)
    def forward(self, logits, target, y_true, alpha, wa_ABL):
        
        # =============================================================================
        #         Agregar Dice_loss y CE+IoU
        # =============================================================================
        
        # print(f"Shape of y_true: {y_true.shape} {y_true.dtype}")
        # print(f"Shape of logits: {logits.shape} {logits.dtype}")
        # print(f"Shape of target: {target.shape} {target.dtype}")
        # print()
        
        # # BCE_loss
        # Y0 = y_true[:,0]*torch.log(torch.clamp(logits[:,0], min=eps, max=1))  # FP
        # Y1 = y_true[:,1]*torch.log(torch.clamp(logits[:,1], min=eps, max=1))  # FN
        # bce_loss = torch.mean(-Y0 - Y1)
        # print(f'bce_loss:{bce_loss.shape}, {torch.mean(bce_loss)}')
        
        # # IoU_loss: lovasz_softmax
        # # labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
        # labels = y_true[:,1]
        # labels = labels.to(torch.int64)
        # loss_lovasz_softmax = torch.mean(L.lovasz_softmax(logits, labels, ignore=255))#No es necesario promediar, ya era escalar
        # print(f'loss_lovasz_softmax:{loss_lovasz_softmax.shape}, {torch.mean(loss_lovasz_softmax)}')
        
        # Dice_loss
        # tn, fp, fn, tp = Matriz_confusion_2MC(logits, y_true, True)
        # dice = Dice_metric(tn, fp, fn, tp)
        # dice_loss = torch.mean(1-dice)
        # print(f'dice_loss:{dice_loss.shape}, {torch.mean(dice_loss)}')
        # print()
        
        # GDL
        gdice = Generalized_dice_metric(logits, y_true)
        gdice_loss = 1-gdice
        # =============================================================================
        
        # eps = 1e-10
        ph, pw = logits.size(2), logits.size(3)
        h, w = target.size(1), target.size(2)

        if ph != h or pw != w:
            logits = F.interpolate(input=logits, size=(
                h, w), mode='bilinear', align_corners=True)

        gt_boundary = self.gt2boundary(target, ignore_label=self.ignore_label)

        dist_maps = self.get_dist_maps(gt_boundary).cuda() # <-- it will slow down the training, you can put it to dataloader.

        pred_boundary = self.logits2boundary(logits)
        if pred_boundary.sum() < 1: # avoid nan
            print('you should check in the outside. if None, skip this loss.!')
            # return None # you should check in the outside. if None, skip this loss.
            abl_loss = 0
        else:
            direction_gt, direction_pred, weight_ce = self.get_direction_gt_predkl(dist_maps, pred_boundary, logits) # NHW,NHW,NCHW
    
            # direction_pred [K,8], direction_gt [K]
            loss = self.criterion(direction_pred, direction_gt) # careful
            
            weight_ce = self.weight_func(weight_ce)
            abl_loss = (loss * weight_ce).mean()  # add distance weight # Es un escalar
            # print(f'abl_loss:{abl_loss.shape}, {torch.mean(abl_loss)}')
        
        # =============================================================================
        #   De acuerdo al paper: CE + IoU + wa*ABL
        # =============================================================================
        # total_loss = bce_loss + loss_lovasz_softmax + wa_ABL*abl_loss
        # print(f'bce_loss:{bce_loss}, loss_lovasz_softmax:{loss_lovasz_softmax}, dice_loss:{dice_loss}, abl_loss:{abl_loss}, total_loss:{total_loss}')
        
        # =============================================================================
        #   De acuerdo a Boundary loss: alpha*dice_loss + (1-alpha)*ABL
        # =============================================================================
        # total_loss = alpha*dice_loss + (1-alpha)*abl_loss
        
        gdice_loss_cpu = gdice_loss.detach().cpu().numpy()
        abl_loss_cpu = abl_loss.cpu().detach().numpy()
        print(f'gdice_lc:{gdice_loss_cpu.shape} {np.around(gdice_loss_cpu,4)}')
        print(f'abl_loss:{abl_loss_cpu.shape} {np.around(abl_loss_cpu,4)}')
        
        total_loss = alpha*gdice_loss + (1-alpha)*abl_loss
        # print(f"total_loss: {total_loss} {total_loss.shape}, torch.mean(total_loss):{torch.mean(total_loss)} {total_loss.dtype}")
        
        return torch.mean(total_loss)

# Ejemplo de los autores:
# if __name__ == '__main__':
#     from torch.backends import cudnn
#     import os
#     import random
#     cudnn.benchmark = False
#     cudnn.deterministic = True

#     seed = 0
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

#     random.seed(seed)
#     np.random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)

#     n,c,h,w = 1,2,100,100
#     gt = torch.zeros((n,h,w)).cuda()
#     gt[0,5] = 1
#     gt[0,50] = 1
#     logits = torch.randn((n,c,h,w)).cuda()

#     abl = ABL()
#     print(abl(logits, gt))


# =============================================================================
# BSL: Boundary Sensitive Loss (2023)
# =============================================================================
from torch.nn.modules.loss import _Loss

class BSLoss(_Loss):

    def __init__(self, *args, **kwargs):
        super(BSLoss, self).__init__()

    def forward(self, prediction, ground_truth, alpha):
        bs_loss = boundary_sensitive_loss(prediction, ground_truth, alpha)
        return bs_loss

class BSL_LC(_Loss):
    def __init__(self, *args, **kwargs):
        super(BSL_LC, self).__init__()

    def forward(self, prediction, ground_truth, alpha, beta):
        bs_loss = boundary_sensitive_loss(prediction, ground_truth, alpha)
        lc = location_constraint(prediction, ground_truth)
        bs_col = beta*bs_loss + (1-beta)*0.000001 * lc
        return bs_col

def location_constraint(prediction, label):
    N = prediction.shape[0]
    # pos_sum = torch.sum(label, 3)
    # pos_sum = torch.sum(pos_sum, 2)
    x = torch.abs((torch.sum(prediction, 2) - torch.sum(label, 2)))
    y = torch.abs((torch.sum(prediction, 3) - torch.sum(label, 3)))
    x = x.view(N, -1)
    y = y.view(N, -1)
    x = x.sum() / N
    y = y.sum() / N
    loss = (x + y) / 2
    return loss

#################################################################################
# new boundary dice loss 2: concern the inside/outside boundary of GT and Pred
# import warnings
def get_boundary(img):
    """
    Get dilated edge image of input image:
    Input: [1, 1, H, W] Tensor or [1, H, W]
    """
    # 1. Convert to numpy
    img = img.detach().cpu().squeeze(0).squeeze(0).numpy()

    # print(f'get_boundary img: {img.shape} {img.dtype} {np.sum(img)} {np.unique(img)}')
    # print(f'get_boundary img: {img.shape} {img.dtype} {np.sum(img)}')
    img = np.array(img * 255)
    img = img.astype('uint8')
    # with warnings.catch_warnings():
    #     try:
    #         img = img.astype('uint8')
    #         # answer = arr / 0
    #     except Warning as e:
    #         print('error found:', e)
    #         print(f'get_boundary img: {img.shape} {img.dtype} {np.sum(img)}')
    #         print(f'get_boundary img: {img.shape} {img.dtype} {np.sum(img)} {np.unique(img)}')


    # 2. Get edge image
    edge_img = cv2.Canny(img, 100, 200)
    _, edge_img = cv2.threshold(edge_img, 127, 255, cv2.THRESH_BINARY)

    # 3. Dilate the edge image
    kernel = np.ones((2, 2), np.uint8)
    edge_dialte_img = cv2.dilate(edge_img, kernel, iterations=2)  # adjust manually
    _, edge_dialte_img = cv2.threshold(edge_dialte_img, 127, 255, cv2.THRESH_BINARY)

    # 4. normalization [0, 255] -> [0, 1]
    edge_dialte_img = edge_dialte_img / 255

    return edge_dialte_img

# def boundary_sensitive_loss(prediction, label, alpha, eps=1e-5):
def boundary_sensitive_loss(prediction, label, alpha, eps=1e-5):
    """
    New boundary dice loss 2: concern the inside/outside boundary of GT and prediction
    Input:
        prediction: [0, 1]  N, C, H, W
        label: {0, 1}  N, 1, H, W
        label_edge: {0, 1}
    """
    # print(f'prediction:{prediction.shape} {prediction.dtype} {prediction.get_device()}, label:{label.shape} {label.dtype} {label.get_device()}')
    # prediction = torch.unsqueeze(p[:,1],1)
    # label = torch.unsqueeze(l[:,1],1)
    # print(f'prediction:{prediction.shape} {prediction.dtype}, label:{label.shape} {label.dtype}')
    
    N, C, H, W = prediction.size()
    w_edge = alpha
    w_true = 1 - alpha
    w_bk = w_true

    # ------------------------------------------------------------------------
    # 1. Extract edge images of predictions
    predict_edge = None
    gt_edge = None
    for bs_id in range(N):
        pred = prediction[bs_id, :, :, :]
        gt = label[bs_id, :, :, :]
        # print(f'gt:{gt.shape} {gt.dtype} {gt.get_device()}')
        pred_edge_img = get_boundary(pred)
        gt_edge_img = get_boundary(gt)
        # print(f'gt_edge_img:{gt_edge_img.shape} {gt_edge_img.dtype}')#' {gt_edge_img.get_device()}')
        pred_edge_img_tensor = torch.from_numpy(pred_edge_img).unsqueeze(0).unsqueeze(0)
        gt_edge_img_tensor = torch.from_numpy(gt_edge_img).unsqueeze(0).unsqueeze(0)
        # print(f'gt_edge_img_tensor:{gt_edge_img_tensor.shape} {gt_edge_img_tensor.dtype} {gt_edge_img_tensor.get_device()}')
              
        if bs_id == 0:
            predict_edge = pred_edge_img_tensor
            gt_edge = gt_edge_img_tensor
        else:
            predict_edge = torch.cat((predict_edge, pred_edge_img_tensor), dim=0)
            gt_edge = torch.cat((gt_edge, pred_edge_img_tensor), dim=0)
    
    # print(f'gt_edge:{gt_edge.shape} {gt_edge.dtype} {gt_edge.get_device()}')
          
    # ------------------------------------------------------------------------
    # 2. Flatten data [B, 1, H, W] -> [B, H x W], same as pred, label and edge
    prediction = prediction.contiguous().view(N, -1)
    label = label.contiguous().view(N, -1)
    label_edge = gt_edge.contiguous().view(N, -1).cuda()# Faltaba .cuda()
    predict_edge = predict_edge.contiguous().view(N, -1).cuda()

    # ------------------------------------------------------------------------
    # 3. TP, FP, FN TN
    TP = prediction * label
    FP = prediction - TP
    FN = label - TP

    # ------------------------------------------------------------------------
    # 4. Weight FN
    # print(f'prediction:{prediction.shape} {prediction.dtype} {prediction.get_device()}, label:{label.shape} {label.dtype} {label.get_device()}')
    # print(f'FN:{FN.shape} {FN.dtype} {FN.get_device()}, label_edge:{label_edge.shape} {label_edge.dtype} {label_edge.get_device()}')
    # label_edge.cuda()
    FN_in_boundary = FN * label_edge
    FN_out_boundary = FN * predict_edge

    FN_boundary_intersection = FN_in_boundary * FN_out_boundary
    FN_out_boundary = FN_out_boundary - FN_boundary_intersection

    FN_gt = FN - (FN_in_boundary + FN_out_boundary)

    FN = FN_in_boundary * w_edge + FN_out_boundary * w_edge + FN_gt * w_true

    # ------------------------------------------------------------------------
    # 5. Weight FP
    FP_in_boundary = FP * predict_edge
    FP_out_boundary = FP * label_edge

    FP_boundary_intersection = FP_in_boundary * FP_out_boundary
    FP_out_boundary = FP_out_boundary - FP_boundary_intersection

    FP_bk = FP - (FP_in_boundary + FP_out_boundary)

    FP = FP_in_boundary * w_edge + FP_out_boundary * w_edge + FP_bk * w_bk

    # ------------------------------------------------------------------------
    # 6. Loss
    loss = (2 * torch.sum(TP, dim=1) + eps) / (2 * torch.sum(TP, dim=1) + torch.sum(FP, dim=1) + torch.sum(FN, dim=1) + eps)

    loss = 1 - loss.sum() / N
    # print(f'loss:{loss.shape} {loss}')
    return loss


# =============================================================================
# CBL: Conditional Boundary Loss (2023)
# =============================================================================

def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
        param mask (numpy array, uint8): binary mask

        param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal

        return: boundary mask (numpy array)
    """    
    mask = mask.squeeze(dim=-1)
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask.cpu().numpy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode

# Clase para CBL
class NeighborExtractor5(nn.Module):
    def __init__(self, input_channel):
        super(NeighborExtractor5, self).__init__()
        same_class_neighbor = np.array([[1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1],
                                        [1, 1, 0, 1, 1],
                                        [1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1], ], dtype='float32')
        same_class_neighbor = same_class_neighbor.reshape((1, 1, 5, 5))
        same_class_neighbor = np.repeat(same_class_neighbor, input_channel, axis=0)
        self.same_class_extractor = nn.Conv2d(input_channel, input_channel, kernel_size=5, padding=2, bias=False, groups=input_channel)
        self.same_class_extractor.weight.data = torch.from_numpy(same_class_neighbor)

    def forward(self, feat):
        output = self.same_class_extractor(feat)
        return output

# CBL
class CBL(nn.Module):
    
    def __init__(self):
        super(CBL, self).__init__()
        self.output_er = None
        self.gt_boundary_seg = None

        # Se usa en el metodo er_loss
        # ===========================        

        self.er_input_ch = 128
        base_weight = np.array([[1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1],
                                [1, 1, 0, 1, 1],
                                [1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1], ], dtype='float32')

        base_weight = base_weight.reshape((1, 1, 5, 5))
        self.same_class_extractor_weight = np.repeat(base_weight, self.er_input_ch, axis=0)
        self.same_class_extractor_weight = torch.FloatTensor(self.same_class_extractor_weight)

        self.same_class_number_extractor_weight = base_weight
        self.same_class_number_extractor_weight = torch.FloatTensor(self.same_class_number_extractor_weight)
        # ===========================


    # LA2P&N
    def context_loss(self, er_input, seg_label, gt_boundary_seg, conv10, kernel_size=5):        
        seg_label = F.interpolate(seg_label.float(), size=er_input.shape[2:], mode='nearest').long()
        gt_boundary_seg = F.interpolate(gt_boundary_seg.unsqueeze(1).float(), size=er_input.shape[2:], mode='nearest').long()        
        context_loss_final = torch.tensor(0.0, device=er_input.device)
        context_loss = torch.tensor(0.0, device=er_input.device)
        # 获取参与运算的边缘像素mask gt_b (B,1,H,W)
        # get the ground truth mask of boundary pixels for CBL calculation
        gt_b = gt_boundary_seg
        # 将ignore的像素置零，现在gt_b里面的0表示不参与loss计算
        # set all ignored pixels as 0 in the boundary pixel mask
        gt_b[gt_b==255]=0
        seg_label_copy = seg_label.clone()
        seg_label_copy[seg_label_copy==255]=0
        gt_b = gt_b*seg_label_copy
        num_classes = 2        

        #seg_label_one_hot = F.one_hot(seg_label.squeeze(1), num_classes=num_classes)[:,:,:,0:num_classes]

        seg_label_one_hot = seg_label.clone()

        b,c,h,w = er_input.shape        
        scale_num = b
        context_loss_pi = None
        position_shift_list = None
        for i in range(b):
            cal_mask = (gt_b[i][1]>0).bool()
            if cal_mask.sum()<1:
                scale_num = scale_num-1
                continue
            
            position = torch.where(gt_b[i][1])
            position_mask = ((kernel_size//2)<=position[0]) * (position[0]<=(er_input.shape[-2]-1-(kernel_size//2))) * ((kernel_size//2)<=position[1]) * (position[1]<=(er_input.shape[-1]-1-(kernel_size//2)))
            position_selected = (position[0][position_mask], position[1][position_mask])
            position_shift_list = []
            for ki in range(kernel_size):
                for kj in range(kernel_size):
                    if ki==kj==(kernel_size//2):
                        continue
                    position_shift_list.append((position_selected[0]+ki-(kernel_size//2),position_selected[1]+kj-(kernel_size//2)))
            # context_loss_batchi = torch.zeros_like(er_input[i].permute(1,2,0)[position_selected][0])
            context_loss_pi = torch.tensor(0.0, device=er_input.device)
            for pi in range(len(position_shift_list)):
                boudary_simi = F.cosine_similarity(er_input[i].permute(1,2,0)[position_selected], er_input[i].permute(1,2,0)[position_shift_list[pi]], dim=1)
                boudary_simi_label = torch.sum(seg_label_one_hot[i].permute(1,2,0)[position_selected] * seg_label_one_hot[i].permute(1,2,0)[position_shift_list[pi]], dim=-1)
                context_loss_pi = context_loss_pi + F.mse_loss(boudary_simi, boudary_simi_label.float())            
            context_loss += (context_loss_pi / len(position_shift_list))
        
        context_loss = context_loss/scale_num        
        if torch.isnan(context_loss):            
            return context_loss_final
        else:
            return context_loss

    # LA2C
    def er_loss(self, er_input, seg_label, seg_logit, gt_boundary_seg, conv10):

        shown_class = [0,1]#list(seg_label.unique())
        num_classes = 2

        pred_label = seg_logit.max(dim=1)[1]
        pred_label_one_hot = F.one_hot(pred_label, num_classes=num_classes).permute(0,3,1,2)
        pred_label_one_hot = F.interpolate(pred_label_one_hot.float(), size=er_input.shape[2:], mode='nearest').long()

        seg_label = F.interpolate(seg_label.float(), size=er_input.shape[2:], mode='nearest').long()
        gt_boundary_seg = F.interpolate(gt_boundary_seg.unsqueeze(1).float(), size=er_input.shape[2:], mode='nearest').long()
        # 获取参与运算的边缘像素mask gt_b (B,1,H,W)
        # get the ground truth mask of boundary pixels for CBL calculation
        gt_b = gt_boundary_seg
        # 将ignore的像素置零，现在gt_b里面的0表示不参与loss计算
        # set all ignored pixels as 0 in the boundary pixel mask
        gt_b[gt_b==255]=0
        edge_mask = gt_b.squeeze(1)
        # 下面按照每个出现的类计算每个类的er loss
        # calculate er loss in a class-wise manner, only for those that have showned in this batch
        # 首先提取出每个类各自的boundary
        # get the boundary pixels of each class
        #seg_label_one_hot = F.one_hot(seg_label.squeeze(1), num_classes=num_classes)[:,:,:,0:num_classes].permute(0,3,1,2)
        seg_label_one_hot = seg_label.clone()
        if self.same_class_extractor_weight.device!=er_input.device:
            self.same_class_extractor_weight = self.same_class_extractor_weight.to(er_input.device)
            #print("er move:",self.same_class_extractor_weight.device)
        if self.same_class_number_extractor_weight.device!=er_input.device:
            self.same_class_number_extractor_weight = self.same_class_number_extractor_weight.to(er_input.device)
        #print(self.same_class_number_extractor_weight)
        same_class_extractor = NeighborExtractor5(self.er_input_ch)
        same_class_extractor.same_class_extractor.weight.data = self.same_class_extractor_weight
        same_class_number_extractor = NeighborExtractor5(1)
        same_class_number_extractor.same_class_extractor.weight.data = self.same_class_number_extractor_weight

        # try:
        #     shown_class.remove(torch.tensor(1))
        # except:
        #     pass
        # er_input = er_input.permute(0,2,3,1)
        neigh_classfication_loss_total = torch.tensor(0.0, device=er_input.device)
        close2neigh_loss_total = torch.tensor(0.0, device=er_input.device)
        cal_class_num = len(shown_class)#2
        for i in range(len(shown_class)):
            now_class_mask = seg_label_one_hot[:,shown_class[i],:,:]
            now_pred_class_mask = pred_label_one_hot[:,shown_class[i],:,:]
            # er_input 乘当前类的mask，就把所有不是当前类的像素置为0了
            # 得到的now_neighbor_feat是只有当前类的特征
            # get the pixel feature of only the current class            
            #xd = er_input*now_class_mask.unsqueeze(1)
            #print(xd.shape)
            now_neighbor_feat = same_class_extractor(er_input*now_class_mask.unsqueeze(1))
            now_correct_neighbor_feat = same_class_extractor(er_input*(now_class_mask*now_pred_class_mask).unsqueeze(1))
            # 下面是获得当前类的每个像素邻居中同属当前点的像素个数
            # count the number of each pixel's neighbor belongs to the same class
            now_class_num_in_neigh = same_class_number_extractor(now_class_mask.unsqueeze(1).float())
            now_correct_class_num_in_neigh = same_class_number_extractor((now_class_mask*now_pred_class_mask).unsqueeze(1).float())
            # 需要得到 可以参与er loss 计算的像素
            # 一个像素若要参与er loss计算，需要具备：
            # 1.邻居中具有同属当前类的像素
            # 2.当前像素要在边界上
            # 如果没有满足这些条件的像素 则直接跳过这个类的计算
            # get all the pixels for this loss
            # a pixels should satisfy the following conditions:
            # 1. some neighbor belongs to the same class with that of itself
            # 2. the current pixel should be one of the boundary pixel
            # if a pixel does not meet these conditions at the same time, it will be ignored
            pixel_cal_mask = (now_class_num_in_neigh.squeeze(1)>=1)*(edge_mask.bool()*now_class_mask.bool()).detach()
            pixel_mse_cal_mask = (now_correct_class_num_in_neigh.squeeze(1)>=1)*(edge_mask.bool()*now_class_mask.bool()*now_pred_class_mask.bool()).detach()
            if pixel_cal_mask.sum()<1 or pixel_mse_cal_mask.sum()<1:
                cal_class_num = cal_class_num - 1
                continue
            class_forward_feat = now_neighbor_feat/(now_class_num_in_neigh+1e-5)
            class_correct_forward_feat = now_correct_neighbor_feat/(now_correct_class_num_in_neigh+1e-5)

            # 选择出参与loss计算的像素的原始特征
            # get the original feature of those pixel included in the loss calculation
            # origin_pixel_feat = er_input.permute(0,2,3,1)[pixel_cal_mask].permute(1,0).unsqueeze(0).unsqueeze(-1)
            origin_mse_pixel_feat = er_input.permute(0,2,3,1)[pixel_mse_cal_mask].permute(1,0).unsqueeze(0).unsqueeze(-1)
            # 选择出参与loss计算的像素的邻居平均特征
            # get the avg feature of its neighbors
            neigh_pixel_feat = class_forward_feat.permute(0,2,3,1)[pixel_cal_mask].permute(1,0).unsqueeze(0).unsqueeze(-1)
            neigh_mse_pixel_feat = class_correct_forward_feat.permute(0,2,3,1)[pixel_mse_cal_mask].permute(1,0).unsqueeze(0).unsqueeze(-1)
            # 邻居平均特征也要能够正确分类，且用同样的分类器才行
            # make sure this averaged feature should also be correctly classified by the same classifier

            # weight & bias era conv10.weight.to(neigh_pixel_feat.dtype).detach(), conv10.bias.to(neigh_pixel_feat.dtype).detach() respectivamente
            weight_conv10_amplified = torch.cat([conv10.weight.to(neigh_pixel_feat.dtype), conv10.weight.to(neigh_pixel_feat.dtype)], dim=1)
            #weight_conv10_amplified = F.interpolate(conv10.weight.to(neigh_pixel_feat.dtype), size=neigh_mse_pixel_feat.shape[:2], mode='nearest').long()

            neigh_pixel_feat_prediction = F.conv2d(neigh_pixel_feat, weight=weight_conv10_amplified, bias=conv10.bias.to(neigh_pixel_feat.dtype))
            # 为邻居平均特征的分类loss产生GT，即当前类中像素的邻居（因为都是同属当前类的邻居）的标签也是当前类
            # 因此乘shown_class[i]
            # get the ground truth class for the avg feature
            gt_for_neigh_output = shown_class[i]*torch.ones((1,neigh_pixel_feat_prediction.shape[2],1)).to(er_input.device).long()
            neigh_classfication_loss = F.cross_entropy(neigh_pixel_feat_prediction, gt_for_neigh_output)
            # 当前点的像素 要向周围同类像素的平均特征靠近
            # the feature of the current pixel should be pushed close to the avg feature of its same-class neighbors
            # close2neigh_loss = F.mse_loss(origin_pixel_feat, neigh_pixel_feat.detach())
            neigh_mse_pixel_feat_prediction = F.conv2d(neigh_mse_pixel_feat, weight=weight_conv10_amplified, bias=conv10.bias.to(neigh_pixel_feat.dtype))
            gt_for_neigh_mse_output = shown_class[i]*torch.ones((1,neigh_mse_pixel_feat_prediction.shape[2],1)).to(er_input.device).long()
            neigh_classfication_loss = neigh_classfication_loss + F.cross_entropy(neigh_mse_pixel_feat_prediction, gt_for_neigh_mse_output)            

            close2neigh_loss = F.mse_loss(origin_mse_pixel_feat, neigh_mse_pixel_feat.detach())
            neigh_classfication_loss_total = neigh_classfication_loss_total + neigh_classfication_loss
            close2neigh_loss_total = close2neigh_loss_total + close2neigh_loss
        if cal_class_num==0:
            return neigh_classfication_loss_total, close2neigh_loss_total
        neigh_classfication_loss_total = neigh_classfication_loss_total / cal_class_num
        close2neigh_loss_total = close2neigh_loss_total / cal_class_num

        #                L^SCE_A2C                   L^pair_A2C
        return neigh_classfication_loss_total, close2neigh_loss_total        

    #                 pred,      Y,         CBL,             bottle_neck    
    def forward(self, seg_logit, seg_label, gt_boundary_seg, output_er, conv10, alpha, batch, gamma_CBL):        
        """Compute segmentation loss."""
        # print('CBL:')
        
        # print(f"Shape of seg_label: {seg_label.shape} {seg_label.dtype}")
        # print(f"Shape of seg_logit: {seg_logit.shape} {seg_logit.dtype}")
        # print(f"Shape of gt_boundary_seg: {gt_boundary_seg.shape} {gt_boundary_seg.dtype}")
        # print()
        
        # # BCE_loss
        # Y0 = seg_label[:,0]*torch.log(torch.clamp(seg_logit[:,0], min=eps, max=1))  # FP
        # Y1 = seg_label[:,1]*torch.log(torch.clamp(seg_logit[:,1], min=eps, max=1))  # FN
        # bce_loss = torch.mean(-Y0 - Y1)
        # print(f'bce_loss:{bce_loss.shape}, {torch.mean(bce_loss)}')
        
        # Dice_loss
        # tn, fp, fn, tp = Matriz_confusion_2MC(seg_logit, seg_label, True)
        # dice = Dice_metric(tn, fp, fn, tp)
        # dice_loss = torch.mean(1-dice)
        # print(f'dice_loss:{dice_loss.shape}, {torch.mean(dice_loss)}')
        # print()
        
        # GDL
        gdice = Generalized_dice_metric(seg_logit, seg_label)
        gdice_loss = torch.mean(1-gdice)
        
        loss = dict()
        loss['loss_context'] = self.context_loss(output_er, seg_label, gt_boundary_seg, conv10).requires_grad_(True)
        loss['loss_NCE'], loss['loss_CN'] = self.er_loss(output_er, seg_label, seg_logit, gt_boundary_seg, conv10)
        loss['loss_NCE'] = loss['loss_NCE'].requires_grad_(True)
        loss['loss_CN'] = loss['loss_CN'].requires_grad_(True)
        
        # Combine losses and calculate gradients
        # loss['loss_context']+ 0.2 * loss['loss_NCE'] + 2 * loss['loss_CN']# autores con gamma_CBL=2
        cbl_loss = 0.5 * loss['loss_context']+ 0.1 * loss['loss_NCE'] + 1 * loss['loss_CN'] # Está bién el orden
        # print(f'cbl_loss:{cbl_loss.shape} {cbl_loss}')
        
        # =============================================================================
        #   De acuerdo al paper: CE + gamma*CBL
        # =============================================================================
        # total_loss = bce_loss + gamma_CBL*cbl_loss
        # print(f'cbl_loss:{cbl_loss}, bce_loss:{bce_loss}, gamma_CBL:{gamma_CBL}, total_loss:{total_loss}')
        
        # =============================================================================
        #   De acuerdo a Boundary loss: alpha*gdice_loss + (1-alpha)*ABL
        # =============================================================================
        total_loss = alpha*gdice_loss + (1-alpha)*cbl_loss
        # print(f"dice_loss:{dice_loss.shape} {torch.mean(dice_loss)}, cbl_loss:{cbl_loss.shape} {cbl_loss}, total_loss: {total_loss.shape} {total_loss} torch.mean(total_loss):{torch.mean(total_loss)}")
        
        return torch.mean(total_loss)