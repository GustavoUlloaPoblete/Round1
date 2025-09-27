#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 22:44:43 2023

@author: gustavo
"""

import os, sys, numpy as np
import Modelos, LossesAndMetrics
import Biblioteca_General as bbg
from time import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
from torch.utils.data import DataLoader
from skimage.segmentation import find_boundaries

import ModuloA

from monai.networks.nets import BasicUNet, UNet, AttentionUnet, SwinUNETR
from torch.amp import autocast, GradScaler

# torch.manual_seed(18)

def Get_s_umbral(S_theta, umbral, G, paciente): # Creo función para poder paralelizar resultado de cada umbral, aunque terminé solo utilizando umbral 0.5
    S = S_theta > umbral
    tn, fp, fn, tp = bbg.Matriz_confusion(G, S)
    tn = round(tn); fp = round(fp); fn = round(fn); tp = round(tp)
    if tp+fp == 0 or tp+fn == 0:
        # continue
        return 'k0'
    # =============================================================================
    # Métricas de surface-distances
    # =============================================================================
    d_mb = bbg.Metricas_borde(G, S, ['hd', 'hd95', 'hd90', 'assd', 'assd95'])
    hd = round(d_mb['hd'], 8)
    hd95 = round(d_mb['hd95'], 8)
    hd90 = round(d_mb['hd90'], 8)
    assd = round(d_mb['assd'], 8)
    assd95 = round(d_mb['assd95'], 8)
    # =============================================================================
    # Métricas overlapping
    # =============================================================================
    if tp+fp == 0 or tp+fn == 0:
        # continue
        return 'k0'
    rvd = round(bbg.RVD(G, S), 8)
    TPR = round(bbg.Metrica_TPR(tn, fp, fn, tp), 8)
    PPV = round(bbg.Metrica_PPV(tn, fp, fn, tp), 8)
    F1 = round(bbg.Metrica_F1(tn, fp, fn, tp), 8)
    F2 = round(bbg.Metrica_F2(tn, fp, fn, tp), 8)
    AUC_ROC = round(bbg.Metrica_AUC_ROC(G.flatten(), S.flatten()), 8)
    AUC_PR = round(bbg.Metrica_AUC_PR(G.flatten(), S.flatten()), 8)
    
    s_umbral = ('paciente:'+paciente+' u:'+str(umbral).ljust(4, '0') +
                ' TN:'+str(round(tn, 2))+' FP:'+str(round(fp, 2)).rjust(5, '0') +
                ' FN:'+str(round(fn, 2)).rjust(5, '0')+' TP:'+str(round(tp, 2)).rjust(5, '0') +
                ' HD:'+format(hd, '0.8f').rjust(11, '0')+' HD95:'+format(hd95, '0.8f').rjust(11, '0')+' HD90:'+format(hd90, '0.8f').rjust(11, '0') +
                ' ASSD:'+format(assd, '0.8f')+' ASSD95:'+format(assd95, '0.8f')+' RVD:'+format(rvd, '0.8f') + ' AUC_ROC:'+format(AUC_ROC, '0.8f')+
                ' AUC_PR:'+format(AUC_PR, '0.8f') + ' TPR:'+format(TPR, '0.8f') + ' PPV:'+format(PPV, '0.8f') + 
                ' F1:'+format(F1, '0.8f') + ' F2:'+format(F2, '0.8f'))
    return s_umbral

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dir_images, lista, tipo='train', G_DTM=False, G_SDF=False, GTB=False, CBL=False, MDF=False, G_PDF=False, return_slide=False):
        self.data_dir = dir_images
        N_slides_axial = 160# Se puede deducir
        if tipo=='train':
            plantilla_target = '{}_{}_target.npy'
            plantilla_input = '{}_{}'
            lista_new = []
            for paciente in lista:
                for n in range(N_slides_axial):
                    nombre_target = plantilla_target.format(paciente, n)
                    nombre_input = plantilla_input.format(paciente, n)
                    y = np.load(os.path.join(self.data_dir,nombre_target))
                    if y.sum()>=umbral_vol_training:
                        lista_new.append(nombre_input)
            self.images = lista_new
        elif tipo=='eval':
            lista_new = []
            for idz in range(N_slides_axial):
                lista_new.append(lista[0]+'_'+str(idz))
                self.images = lista_new
        self.GTB = GTB
        self.G_SDF = G_SDF
        self.G_DTM = G_DTM
        self.CBL = CBL
        self.MDF = MDF
        self.G_PDF = G_PDF
        self.return_slide = return_slide
  
    # Defining the length of the dataset
    def __len__(self):
        return len(self.images)
    
    # Defining the method to get an item from the dataset
    def __getitem__(self, index):
        image_path = os.path.join(self.data_dir, self.images[index]+'.npy')
        image = np.load(image_path)# CHW (3,160,192)
        target_path = os.path.join(self.data_dir, self.images[index]+'_target.npy')
        target = np.load(target_path)# HW (160,192)
        if self.GTB:
            target_GTB = target.copy()
        if self.G_SDF:
            target_SDF = target.copy()
        if self.G_DTM:
            target_DTM = target.copy()
        if self.CBL:
            target_CBL = target.copy()
        # if self.G_PDF:# Eliminar PDF
        #     target_PDF = target.copy()
        
# =============================================================================
#         Tranformar target (HW->2HW)
# =============================================================================
        H, W = target.shape
        Y = np.zeros((2,H,W),dtype=np.float32)
        Y[0,:,:] = 1-target
        Y[1,:,:] = target
        target = Y
        image = torch.from_numpy(image).to(torch.float32)
        target = torch.from_numpy(target).to(torch.float32)
# =============================================================================        
        if self.GTB:
            target_GTB = find_boundaries(target_GTB, mode='inner')
            target_GTB = torch.from_numpy(target_GTB).to(torch.float32)
            return image, target, target_GTB
        elif self.G_SDF:
            target_SDF = bbg.SDF(target_SDF)
            return image, target, target_SDF
        elif self.G_DTM:
            target_DTM = bbg.DTM(target_DTM)
            return image, target, target_DTM
        elif self.CBL:
            target_CBL = LossesAndMetrics.mask_to_boundary(torch.from_numpy(target_CBL))            
            return image, target, target_CBL    
        elif self.MDF:
# =============================================================================
#           Cargar mdf desde dMDF generada de manera global
# =============================================================================
            slide=np.int16(self.images[index].split('_')[-1])
            paciente='_'.join(self.images[index].split('_')[:2])
            mdf = dMDF[paciente][slide]
            if np.isnan(np.mean(mdf)):# Tambien se pudo agregar en ModuloA.MDF
                mdf = np.zeros_like(mdf)
            mdf = torch.from_numpy(mdf).to(torch.float32)
# =============================================================================
            if self.return_slide:
                return image, target, mdf, paciente+'-'+str(slide)
            else:
                return image, target, mdf
        else:
            return image, target

def SCP():
    print('*'*10+'SCP()'+'*'*10)
    nombre_carpeta_ce_scp = 'scp_ce_'+d['corrida']
    path_carpeta_ce_scp = path_carpeta_principal+'/'+nombre_carpeta_ce_scp
    if nombre_carpeta_ce_scp not in os.listdir(path_carpeta_principal):
        os.mkdir(path_carpeta_ce_scp)
    for nombre_archivo in os.listdir(path_carpeta_ce):
        print(nombre_archivo)
        if '.txt' in nombre_archivo:
            os.system('cp '+path_carpeta_ce+'/'+nombre_archivo +
                      ' '+path_carpeta_ce_scp+'/'+nombre_archivo)
    llamada = 'scp -r '+path_carpeta_ce_scp + ' user@xxx.xxx.xx.x:'+path_carpeta_principal
    print(llamada)
    if os.uname()[1] != 'mineria' and os.uname()[1] != 'f15':
        # 'mineria' = central_server
        os.system(llamada)

def Get_parametros_entrada(argv):
    print(f'Dentro de Get_parametros_entrada: {argv}')
    d = {}
    for dato in argv[1:]:
        llave, valor = dato.split(':')
        d[llave] = valor
    return d
    
def Training():
    print('Training():')
    C, H, W = input_dim_2D
    print(f'input_dim_2D:{(input_dim_2D)}, H:{H}, W:{W}, C:{C}')
    if d['loss']=='CBL':# Unet con retornos extras además de predicción
        model = getattr(Modelos,'Unet_CBL')(C, 2).to(device)
    # elif d['red'] == 'BasicUnet_MONAI':
    #     model = BasicUNet(
    #         spatial_dims=2,
    #         in_channels=C,
    #         out_channels=2,
    #         features=(32,64,128,256,512,1024),
    #         act=("relu", {"inplace": True}),
    #         norm=("batch", {"affine": True}),
    #         bias=True, dropout=0.0)
    elif d['red'] == 'Unet_MONAI':
        print('Unet_MONAI Unet_MONAI')
        model = UNet(
            spatial_dims=2,
            in_channels=C,
            out_channels=2,
            channels=(64, 128, 256, 512, 1024),
            strides=(2, 2, 2, 2),#no hay maxpooling
            norm=("batch", {"affine": True}),
            num_res_units=1
        )
    elif d['red'] == 'Attention-Unet_MONAI':
        print('Attention-Unet_MONAI')
        model = AttentionUnet(
            spatial_dims=2,
            in_channels=C,
            out_channels=2,
            channels=(64, 128, 256, 512, 1024),
            strides=(2, 2, 2, 2),#no hay maxpooling
        )
    # elif d['red'] == 'SwinUNETR_MONAI':
    #     model = SwinUNETR(
    #     in_channels=C,
    #     out_channels=2,
    #     spatial_dims=2,            # 2 si trabajas en 2D
    #     patch_size=2,              # tamaño de patch interno del Swin
    #     depths=(2, 2, 2, 2),       # capas por nivel (ajusta VRAM/complejidad)
    #     num_heads=(3, 6, 12, 24),  # cabezas por nivel
    #     window_size=7,             # ventana de atención
    #     feature_size=48,           # base de canales (sube para +capacidad, +VRAM)
    #     norm_name="instance",
    #     use_checkpoint=True        # activa recompute para ahorrar VRAM ✔
    #     )#.to(device)
    else:
        model = getattr(Modelos,d['red'])(C, 2).to(device)
    model = model.to(device)
    # print(model)
        
    G_DTM = False; G_SDF = False; GTB=False; CBL=False; MDF=False
    G_PDF = False; return_slide = False
    if nombre_carpeta_ce not in os.listdir(path_carpeta_principal):
        os.mkdir(path_carpeta_ce)
    if d['loss'] == 'HD_loss':
        G_DTM = True
    elif d['loss'] == 'Boundary_loss':
        G_SDF = True
    elif d['loss'] == 'ABL':
        GTB = True
    elif d['loss'] == 'CBL':
        CBL = True
    # elif d['loss'] == 'MD_loss':
    elif 'MD_loss' in d['loss']:
        MDF = True
    
    print(f'\ndata_path:{data_path}')
    print(f'pacientes_train:{pacientes_train}')
    print(f'G_DTM:{G_DTM}, G_SDF:{G_SDF}, GTB:{GTB}, CBL:{CBL}, MDF:{MDF}, G_PDF:{G_PDF}')
    
    train_dataset = ImageDataset(data_path,pacientes_train,tipo='train',G_DTM=G_DTM,G_SDF=G_SDF,GTB=GTB,CBL=CBL,MDF=MDF,G_PDF=G_PDF, return_slide=return_slide)
    print(f'\ntrain_dataset.images: {len(train_dataset.images)}')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Utilizado para validación, testing y métricas
    val_dataset  = ImageDataset(data_path, pacientes_val, tipo='train', G_DTM=G_DTM, G_SDF=G_SDF, GTB=GTB, CBL=CBL,MDF=MDF,G_PDF=G_PDF, return_slide=return_slide)
    val_dataloader  = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False, drop_last=True)
    
    test_dataset  = ImageDataset(data_path, pacientes_test, tipo='train', G_DTM=G_DTM, G_SDF=G_SDF, GTB=GTB, CBL=CBL,MDF=MDF,G_PDF=G_PDF, return_slide=return_slide)
    test_dataloader  = DataLoader(test_dataset, batch_size=batch_size_val, shuffle=False, drop_last=True)
    
    print('\n'+d['DS']+':')
    print(f'train_dataset: {len(train_dataset)} // {batch_size} = {len(train_dataset)//batch_size}, {len(train_dataset)} % {batch_size} = {len(train_dataset)%batch_size}')

    d_dataloader = {}# Retorna todas las slides del paciente ordenadas
    for paciente in pacientes_val+pacientes_test:# Crear un dataloader para cada paciente con todas las slides
        if paciente not in d_dataloader:
            d_dataloader[paciente] = DataLoader(ImageDataset(data_path, [paciente], tipo='eval'), batch_size=batch_size_val, shuffle=False)
    
    d_epocas = {'loss_train':[],'dice_train':[],'loss_val':[],'dice_val':[],'loss_test':[],'dice_test':[],
                'HD_val':[],'HD95_val':[],'HD90_val':[],'ASSD_val':[],'ASSD95_val':[],'RVD_val':[],'F1_val':[],'TPR_val':[],'PPV_val':[],'AUC_ROC_val':[],'AUC_PR_val':[],'F2_val':[],
                'HD_test':[],'HD95_test':[],'HD90_test':[],'ASSD_test':[],'ASSD95_test':[],'RVD_test':[],'F1_test':[],'TPR_test':[],'PPV_test':[],'AUC_ROC_test':[],'AUC_PR_test':[],'F2_test':[]}
    
    print("Sumando parametros: ",sum(p.numel() for p in model.parameters()))
    total = 0
    for name, p in model.named_parameters():
        n = p.numel()
        # print(f"{name:60s} {n}")
        total += n
    print("TOTAL:", total)
    # return None

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)#, eps=1e-07)# eps como keras
    if d['mixed_precision']=="T":
        USE_AMP = True
    else:
        USE_AMP = False
    if USE_AMP:
        scaler = GradScaler()
    
    # scaler = torch.cuda.amp.GradScaler(enabled=use_amp) # mixed precision
    # scaler = GradScaler("cuda")  # reemplaza torch.cuda.amp.GradScaler(...)
    t_ini = time()
    freq_metricas = 1
    
    flag_es = True
    contador_epocas_patience = 0
    promedio_val_F1_maximo = 0
    
    for epoca in range(1, epocas+1):
        tiempo_epoca = time()
        print(f'\népoca {epoca}',' /', nombre_carpeta_principal,'/',nombre_carpeta_ce,'/ k:',k)
        if epoca==5: ###########
            break    ##############
        alpha = max(round(1 - (epoca-1)/epocas, 4), 0.01)# Como en los papers Boundary loss y ABL
        print('---alpha:',alpha)
        print('training...')
        tiempo_training = time()
        model.train()
        lista_loss_train = []
        lista_dice_train = []
        
        # for oo in range(1):
        #     tensores = next(iter(train_dataloader))
        for tensores in train_dataloader:
            # print("tensores!")
            if d['loss'] == 'HD_loss':
                X, Y, Y_DTM = tensores
                Y_DTM = Y_DTM.to(device)
            elif d['loss'] == 'Boundary_loss':
                X, Y, Y_SDF = tensores
                Y_SDF = Y_SDF.to(device)
            elif d['loss'] == 'ABL':
                X, Y, GTB = tensores
                GTB = GTB.to(device)
            elif d['loss'] == 'CBL':
                X, Y, CBL = tensores
                CBL = CBL.to(device)
            elif d['loss']=='MD_loss':
                X, Y, Y_MDF = tensores
                Y_MDF = Y_MDF.to(device)
            else: # Dice, FTL, ASD, BCE, ... que solo necesitan entrada X y target Y
                X, Y = tensores
            X, Y = X.to(device), Y.to(device)
            # X = X.to(device, non_blocking=True)
            # Y = Y.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device, enabled=USE_AMP):
                if 'MONAI' in d['red']:
                    logits = model(X)
                    pred = torch.softmax(logits, dim=1)
                elif d['loss'] == 'CBL':
                    pred, bottle_neck, last_conv = model(X)
                else:
                    pred = model(X)
                    
                if d['loss'] == 'HD_loss':# actualizar
                    pred_DTM = bbg.DTM_2D(pred[:,1].detach().cpu().numpy())# procesa cada una de las N=batch_size slides
                    pred_DTM = torch.from_numpy(pred_DTM); pred_DTM = pred_DTM.to(device)
                    loss = loss_function(pred, Y, Y_DTM, pred_DTM, alpha, batch_loss)# para HD_loss
                elif d['loss'] == 'Boundary_loss':
                    loss = loss_function(pred, Y, Y_SDF, alpha, batch_loss)# para Boundary_loss
                elif d['loss'] == 'ABL':
                    loss = loss_function(pred, GTB, Y, alpha, wa_ABL)# para ABL
                elif d['loss'] == 'BSLoss':
                    loss = loss_function(torch.unsqueeze(pred[:,1],1), torch.unsqueeze(Y[:,1],1), alpha_BS)# para BSLoss
                elif d['loss'] == 'BSL_LC':
                    loss = loss_function(torch.unsqueeze(pred[:,1],1), torch.unsqueeze(Y[:,1],1), alpha_BS, beta_BS_LC)# para BSL_LC
                elif d['loss'] == 'FT_loss':
                    loss = loss_function(pred, Y, alpha_TL, beta_TL, gamma, batch_loss)# para Focal tversky loss
                elif d['loss'] == 'CBL':
                    loss = loss_function(pred, Y, CBL, bottle_neck, last_conv, alpha, batch_loss, gamma_CBL)# para CBL
                
                elif d['loss']=='MD_loss':
                    loss = loss_function(pred, Y, alpha, Y_MDF, parMD_weight, parMD_pot)
                
                else: # Dice, FTL, ASD, BCE, GDL,...
                    loss = loss_function(pred, Y, batch_loss)
            
            # break
            if torch.isnan(loss):# Filtrar errores utilizando BSL_LC y CBL, sobretodo en DS WMH2017
                continue # No actualiza los pesos, tampoco guarda pérdida y métrica
            lista_loss_train.append(loss.item())
            dice = LossesAndMetrics.Dice_metric(pred, Y, batch_loss)
            # print(f'train dice:{dice} {dice.shape} {dice.dtype}')
            dice = torch.mean(dice)
            # print(f'train dice:{dice} {dice.shape} {dice.dtype}')
            lista_dice_train.append(torch.mean(dice).item())
            
            if USE_AMP:
                # Backpropagation with mixed precision
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Sin mixed precision
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            # # El "scaler" funciona igual en ambos casos (real o _NullScaler)
            # scaler.scale(loss).backward()
            # scaler.unscale_(optimizer)      # seguro también con _NullScaler
            # # opcional: torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            # scaler.step(optimizer)
            # scaler.update()
        
            # break
    
        d_epocas['loss_train'].append(np.around(np.nanmean(lista_loss_train),8))
        d_epocas['dice_train'].append(np.around(np.nanmean(lista_dice_train),8))
        print(f'loss_epoca_train: {d_epocas["loss_train"][-1]} - dice_epoca_train: {d_epocas["dice_train"][-1]}')
        print(f'tiempo_training:{round(time()-tiempo_training,2)}[sg]')
        # continue
        # return None
        
# =============================================================================
#       Validación
# =============================================================================
        print('validation...')
        tiempo_validation=time()
        model.eval()
        lista_loss_val = []
        lista_dice_val = []
        for tensores in val_dataloader:
            if d['loss'] == 'HD_loss':
                X, Y, Y_DTM = tensores
                Y_DTM = Y_DTM.to(device)
            elif d['loss'] == 'Boundary_loss':
                X, Y, Y_SDF = tensores
                Y_SDF = Y_SDF.to(device)
            elif d['loss'] == 'ABL':
                X, Y, GTB = tensores
                GTB = GTB.to(device)
            elif d['loss'] == 'CBL':
                X, Y, CBL = tensores
                CBL = CBL.to(device)
            elif d['loss']=='MD_loss':
                X, Y, Y_MDF = tensores
                Y_MDF = Y_MDF.to(device)
            else: # Dice, FTL, ASD, BCE, ... que solo necesitan entrada X y target Y
                X, Y = tensores
            X, Y = X.to(device), Y.to(device)
            
            # with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp): # mixed precision
            with autocast(device_type=device, enabled=USE_AMP):
                if 'MONAI' in d['red']:
                    logits = model(X)
                    pred = torch.softmax(logits, dim=1)
                elif d['loss'] == 'CBL':
                    pred, bottle_neck, last_conv = model(X)
                else:
                    pred = model(X)
                
                if d['loss'] == 'HD_loss':
                    pred_DTM = bbg.DTM_2D(pred[:,1].detach().cpu().numpy())# procesa cada una de las N=batch_size slides
                    pred_DTM = torch.from_numpy(pred_DTM); pred_DTM = pred_DTM.to(device)
                    loss = loss_function(pred, Y, Y_DTM, pred_DTM, alpha, batch_loss)# para HD_loss
                elif d['loss'] == 'Boundary_loss':
                    loss = loss_function(pred, Y, Y_SDF, alpha, batch_loss)# para Boundary_loss
                elif d['loss'] == 'ABL':
                    loss = loss_function(pred, GTB, Y, alpha, wa_ABL)# para ABL
                elif d['loss'] == 'BSLoss':
                    loss = loss_function(torch.unsqueeze(pred[:,1],1), torch.unsqueeze(Y[:,1],1), alpha_BS)# para BSLoss
                elif d['loss'] == 'BSL_LC':
                    loss = loss_function(torch.unsqueeze(pred[:,1],1), torch.unsqueeze(Y[:,1],1), alpha_BS, beta_BS_LC)# para BSL_LC
                elif d['loss'] == 'FT_loss':
                    loss = loss_function(pred, Y, alpha_TL, beta_TL, gamma, batch_loss)# para Focal tversky loss
                elif d['loss'] == 'CBL':
                    loss = loss_function(pred, Y, CBL, bottle_neck, last_conv, alpha, batch_loss, gamma_CBL)# para CBL
                
                elif d['loss']=='MD_loss':
                    loss = loss_function(pred, Y, alpha, Y_MDF, parMD_weight, parMD_pot)
                
                else: # Dice, FTL, ASD, BCE, GDL,...
                    loss = loss_function(pred, Y, batch_loss)
            
            lista_loss_val.append(loss.item())
            # break###
            
        # return None ###
        pacientes_pred_paciente = []
        for paciente in pacientes_val[:]:
            Y_paciente_pred = np.zeros((160,2,160,192),dtype=np.float32)
            Y_paciente = np.zeros((160,2,160,192),dtype=np.float32)
            ini = 0
            # fin = batch_size
            fin = batch_size_val
            for tensores in d_dataloader[paciente]:
                X, Y = tensores
                X = X.to(device)
                if 'MONAI' in d['red']:
                    logits = model(X)
                    pred = torch.softmax(logits, dim=1)
                elif d['loss'] == 'CBL':
                    pred, bottle_neck, last_conv = model(X)
                else:
                    pred = model(X)
                
                # Detectar si hay problemas con pred, ya que se cae aveces
                pred = pred.detach().cpu().numpy()
                Y = Y.detach().cpu().numpy()
                Y_paciente_pred[ini:fin] = pred
                Y_paciente[ini:fin] = Y
                ini = fin
                # fin += batch_size
                fin += batch_size_val
            Y_paciente_pred = Y_paciente_pred[:,1,:,:]# NCHW->NHW
            Y_paciente = Y_paciente[:,1,:,:]# NCHW->NHW
            Y_paciente_pred = np.transpose(Y_paciente_pred, axes=(1,2,0)).squeeze()# HWN. Reordenar slides en img 3D
            Y_paciente = np.transpose(Y_paciente, axes=(1,2,0)).squeeze()# HWN
            dice = bbg.Dice_metric(Y_paciente_pred, Y_paciente)
            lista_dice_val.append(dice)
            
            if epoca % freq_metricas == 0:# Guardar volúmenes para obtener métricas en paralelo
                pacientes_pred_paciente.append([Y_paciente_pred, Y_paciente, paciente])
        # =============================================================================
        #   Guardar métricas después de umbralizar salida softmax Y_paciente_pred
        # =============================================================================
        if epoca % freq_metricas == 0:
            # =============================================================================
            # Paralelizar 
            # =============================================================================
            resultados = []
            with ProcessPoolExecutor() as executor:
                functions = []
                for Y_paciente_pred, Y_paciente, paciente in pacientes_pred_paciente:
                    function = executor.submit(Get_s_umbral, Y_paciente_pred, 0.5, Y_paciente, paciente)
                    functions.append(function)
                for function in as_completed(functions):
                    resultados.append(function.result())
            dr_val = {}
            for r in resultados:
                for dato in r.split()[2:]:
                    llave,valor=dato.split(':')
                    if llave in ['ASSD','ASSD95','HD','HD95','HD90','RVD','F1','AUC_ROC','AUC_PR','TPR','PPV','F2']:
                        if llave not in dr_val:
                            dr_val[llave]=[]
                        dr_val[llave].append(float(valor))
            d_epocas['HD_val'].append(np.around(np.mean(dr_val['HD']),8))
            d_epocas['HD95_val'].append(np.around(np.mean(dr_val['HD95']),8))
            d_epocas['HD90_val'].append(np.around(np.mean(dr_val['HD90']),8))
            d_epocas['ASSD_val'].append(np.around(np.mean(dr_val['ASSD']),8))
            d_epocas['ASSD95_val'].append(np.around(np.mean(dr_val['ASSD95']),8))
            d_epocas['RVD_val'].append(np.around(np.mean(dr_val['RVD']),8))
            d_epocas['AUC_ROC_val'].append(np.around(np.mean(dr_val['AUC_ROC']),8))
            d_epocas['AUC_PR_val'].append(np.around(np.mean(dr_val['AUC_PR']),8))
            d_epocas['F1_val'].append(np.around(np.mean(dr_val['F1']),8))
            d_epocas['TPR_val'].append(np.around(np.mean(dr_val['TPR']),8))
            d_epocas['PPV_val'].append(np.around(np.mean(dr_val['PPV']),8))
            d_epocas['F2_val'].append(np.around(np.mean(dr_val['F2']),8))
        # =============================================================================
        d_epocas['loss_val'].append(np.around(np.nanmean(lista_loss_val),8))
        d_epocas['dice_val'].append(np.around(np.nanmean(lista_dice_val),8))
        print(f'loss_epoca_val: {d_epocas["loss_val"][-1]} - dice_epoca_val: {d_epocas["dice_val"][-1]}')
        print(f'tiempo_validation:{round(time()-tiempo_validation,2)}[sg]')
        
# =============================================================================
#       Testing
# =============================================================================
        print('testing...')
        tiempo_testing=time()
        # No importa el sufijo _val ya que las estructuras se reinician vacías
        # tiempo_validation=time()
        model.eval()
        lista_loss_val = []
        lista_dice_val = []
        # for tensores in val_dataloader:
        for tensores in test_dataloader:
            if d['loss'] == 'HD_loss':
                X, Y, Y_DTM = tensores
                Y_DTM = Y_DTM.to(device)
            elif d['loss'] == 'Boundary_loss':
                X, Y, Y_SDF = tensores
                Y_SDF = Y_SDF.to(device)
            elif d['loss'] == 'ABL':
                X, Y, GTB = tensores
                GTB = GTB.to(device)
            elif d['loss'] == 'CBL':
                X, Y, CBL = tensores
                CBL = CBL.to(device)
            elif d['loss']=='MD_loss':
                X, Y, Y_MDF = tensores
                Y_MDF = Y_MDF.to(device)
            else: # Dice, FTL, ASD, BCE, ... que solo necesitan entrada X y target Y
                X, Y = tensores
            X, Y = X.to(device), Y.to(device)
            
            # with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp): # mixed precision
            with autocast(device_type=device, enabled=USE_AMP):
                if 'MONAI' in d['red']:
                    logits = model(X)
                    pred = torch.softmax(logits, dim=1)
                elif d['loss'] == 'CBL':
                    pred, bottle_neck, last_conv = model(X)
                else:
                    pred = model(X)
                
                if d['loss'] == 'HD_loss':
                    pred_DTM = bbg.DTM_2D(pred[:,1].detach().cpu().numpy())# procesa cada una de las N=batch_size slides
                    pred_DTM = torch.from_numpy(pred_DTM); pred_DTM = pred_DTM.to(device)
                    loss = loss_function(pred, Y, Y_DTM, pred_DTM, alpha, batch_loss)# para HD_loss
                elif d['loss'] == 'Boundary_loss':
                    loss = loss_function(pred, Y, Y_SDF, alpha, batch_loss)# para Boundary_loss
                elif d['loss'] == 'ABL':
                    loss = loss_function(pred, GTB, Y, alpha, wa_ABL)# para ABL
                elif d['loss'] == 'BSLoss':
                    loss = loss_function(torch.unsqueeze(pred[:,1],1), torch.unsqueeze(Y[:,1],1), alpha_BS)# para BSLoss
                elif d['loss'] == 'BSL_LC':
                    loss = loss_function(torch.unsqueeze(pred[:,1],1), torch.unsqueeze(Y[:,1],1), alpha_BS, beta_BS_LC)# para BSL_LC
                elif d['loss'] == 'FT_loss':
                    loss = loss_function(pred, Y, alpha_TL, beta_TL, gamma, batch_loss)# para Focal tversky loss
                elif d['loss'] == 'CBL':
                    loss = loss_function(pred, Y, CBL, bottle_neck, last_conv, alpha, batch_loss, gamma_CBL)# para CBL
                
                elif d['loss']=='MD_loss':
                    loss = loss_function(pred, Y, alpha, Y_MDF, parMD_weight, parMD_pot)
                
                else: # Dice, FTL, ASD, BCE, GDL,...
                    loss = loss_function(pred, Y, batch_loss)

            
            lista_loss_val.append(loss.item())
            # break###
            
        # return None ###
        pacientes_pred_paciente = []
        # for paciente in pacientes_val[:]:
        for paciente in pacientes_test[:]:
            Y_paciente_pred = np.zeros((160,2,160,192),dtype=np.float32)
            Y_paciente = np.zeros((160,2,160,192),dtype=np.float32)
            ini = 0
            # fin = batch_size
            fin = batch_size_val
            for tensores in d_dataloader[paciente]:
                X, Y = tensores
                X = X.to(device)
                if 'MONAI' in d['red']:
                    logits = model(X)
                    pred = torch.softmax(logits, dim=1)
                elif d['loss'] == 'CBL':
                    pred, bottle_neck, last_conv = model(X)
                else:
                    pred = model(X)
                
                # Detectar si hay problemas con pred, ya que se cae aveces
                pred = pred.detach().cpu().numpy()
                Y = Y.detach().cpu().numpy()
                Y_paciente_pred[ini:fin] = pred
                Y_paciente[ini:fin] = Y
                ini = fin
                # fin += batch_size
                fin += batch_size_val
            Y_paciente_pred = Y_paciente_pred[:,1,:,:]# NCHW->NHW
            Y_paciente = Y_paciente[:,1,:,:]# NCHW->NHW
            Y_paciente_pred = np.transpose(Y_paciente_pred, axes=(1,2,0)).squeeze()# HWN. Reordenar slides en img 3D
            Y_paciente = np.transpose(Y_paciente, axes=(1,2,0)).squeeze()# HWN
            dice = bbg.Dice_metric(Y_paciente_pred, Y_paciente)
            lista_dice_val.append(dice)
            
            if epoca % freq_metricas == 0:# Guardar volúmenes para obtener métricas en paralelo
                pacientes_pred_paciente.append([Y_paciente_pred, Y_paciente, paciente])
        # =============================================================================
        #   Guardar métricas después de umbralizar salida softmax Y_paciente_pred
        # =============================================================================
        if epoca % freq_metricas == 0:
            # =============================================================================
            # Paralelizar 
            # =============================================================================
            resultados = []
            with ProcessPoolExecutor() as executor:
                functions = []
                for Y_paciente_pred, Y_paciente, paciente in pacientes_pred_paciente:
                    function = executor.submit(Get_s_umbral, Y_paciente_pred, 0.5, Y_paciente, paciente)
                    functions.append(function)
                for function in as_completed(functions):
                    resultados.append(function.result())
            dr_val = {}
            for r in resultados:
                for dato in r.split()[2:]:
                    llave,valor=dato.split(':')
                    if llave in ['ASSD','ASSD95','HD','HD95','HD90','RVD','F1','AUC_ROC','AUC_PR','TPR','PPV','F2']:
                        if llave not in dr_val:
                            dr_val[llave]=[]
                        dr_val[llave].append(float(valor))
            d_epocas['HD_test'].append(np.around(np.mean(dr_val['HD']),8))
            d_epocas['HD95_test'].append(np.around(np.mean(dr_val['HD95']),8))
            d_epocas['HD90_test'].append(np.around(np.mean(dr_val['HD90']),8))
            d_epocas['ASSD_test'].append(np.around(np.mean(dr_val['ASSD']),8))
            d_epocas['ASSD95_test'].append(np.around(np.mean(dr_val['ASSD95']),8))
            d_epocas['RVD_test'].append(np.around(np.mean(dr_val['RVD']),8))
            d_epocas['AUC_ROC_test'].append(np.around(np.mean(dr_val['AUC_ROC']),8))
            d_epocas['AUC_PR_test'].append(np.around(np.mean(dr_val['AUC_PR']),8))
            d_epocas['F1_test'].append(np.around(np.mean(dr_val['F1']),8))
            d_epocas['TPR_test'].append(np.around(np.mean(dr_val['TPR']),8))
            d_epocas['PPV_test'].append(np.around(np.mean(dr_val['PPV']),8))
            d_epocas['F2_test'].append(np.around(np.mean(dr_val['F2']),8))
        # =============================================================================
        d_epocas['loss_test'].append(np.around(np.nanmean(lista_loss_val),8))
        d_epocas['dice_test'].append(np.around(np.nanmean(lista_dice_val),8))
        print(f'loss_epoca_test: {d_epocas["loss_test"][-1]} - dice_epoca_test: {d_epocas["dice_test"][-1]}')
        print(f'tiempo_testing:{round(time()-tiempo_testing,2)}[sg]')
        
# =============================================================================
#       Early-stopping
# =============================================================================
        epocas_completadas = epoca
        
        if d['ES']=='T' and epoca >= start_es and flag_es:
            # print('Validation-Earling Stopping...')
            val_F1_promedio = d_epocas['dice_val'][-1]
            contador_epocas_patience+=1
            # print('contador_epocas_patience=',contador_epocas_patience, epoca)
            if val_F1_promedio > promedio_val_F1_maximo:
                promedio_val_F1_maximo = val_F1_promedio
                contador_epocas_patience = 0
            if contador_epocas_patience==patience:
                print('contador_epocas_patience=',contador_epocas_patience, epoca)
                print('Terminar entrenamiento--'*10)
                # nombre_modelo_es = 'k'+str(k)+'_'+str(epoca)+'.pth'
                nombre_modelo_es = 'k'+str(k)+'_es_'+str(epoca)+'.pth'
                # nombre_modelo = nombre_modelo_es
                torch.save(model, os.path.join(path_carpeta_ce,nombre_modelo_es))
                flag_es = False
                break# Termina el entrenamiento
        
        print(f'tiempo_epoca:{round(time()-tiempo_epoca,2)}[sg]')
# =============================================================================
        
# =============================================================================
#   Guardar modelo e historial del entrenamiento
# =============================================================================
    if flag_es:# Si no hubo early-stopping sigue como True y se guarda modelo con 200 epocas de entrenamiento
        nombre_modelo_es = 'k'+str(k)+'_'+str(epoca)+'.pth'
        torch.save(model, os.path.join(path_carpeta_ce,nombre_modelo))
    # =========================================================================
    t_fin = time()
    # print('d_epocas:',d_epocas)
    nombre_ae = 'entrenamiento.txt'
    if nombre_ae not in os.listdir(path_carpeta_ce):
        archivo_e = open(path_carpeta_ce+'/'+nombre_ae, 'w')
    else:
        archivo_e = open(path_carpeta_ce+'/'+nombre_ae, 'a')
    
    archivo_e.write(path_carpeta_ce+'\n')
    archivo_e.write('k'+str(k)+'\n')
    archivo_e.write('epocas_completadas='+str(epocas_completadas)+'\n')
    archivo_e.write('freq_metricas_umbral='+str(freq_metricas)+'\n')
    for llave in d_epocas:
        if len(d_epocas[llave])==0:
            continue
        archivo_e.write(llave+':'+','.join(map(str, np.around(d_epocas[llave], 8)))+'\n')
    # =============================================================================
    # Tiempo
    # =============================================================================
    tiempo_entrenamiento = round(t_fin-t_ini)
    segundos = tiempo_entrenamiento
    horas = segundos // 3600
    segundos %= 3600
    minutos = segundos // 60
    segundos %= 60
    hhmmss = str(horas)+':'+str(minutos)+':'+str(segundos)
    archivo_e.write('tiempo '+str(tiempo_entrenamiento)+'[sg] '+hhmmss+'\n')
    archivo_e.close()
  
    return None

def Testing():
    print('Testing')
    # =============================================================================
    # Abrir modelo
    # =============================================================================
    print(os.listdir(path_carpeta_ce))
    for file in os.listdir(path_carpeta_ce):
        if '.pth' in file:
            nombre_modelo=file
    print('-'*100)
    print(os.path.join(path_carpeta_ce,nombre_modelo))
    model = torch.load(os.path.join(path_carpeta_ce,nombre_modelo))
    # =============================================================================
    # Archivo de texto
    # =============================================================================
    nombre_at = 'testing.txt'
    if nombre_at not in os.listdir(path_carpeta_ce):
        archivo_testing = open(path_carpeta_ce+'/'+nombre_at, 'w')
    else:
        archivo_testing = open(path_carpeta_ce+'/'+nombre_at, 'a')
    
    d_eval_dataloader = {}
    for paciente in pacientes_test:
        if paciente not in d_eval_dataloader:
            d_eval_dataloader[paciente] = DataLoader(ImageDataset(data_path, [paciente], tipo='eval'), batch_size=batch_size, shuffle=False)
    print(d_eval_dataloader)
    
    print('testing...')
    model.eval()
    for paciente in pacientes_test[:]:
        t_1vol = time()
        print(paciente)
        
        Y_paciente_pred = np.zeros((160,2,160,192),dtype=np.float32)
        Y_paciente = np.zeros((160,2,160,192),dtype=np.float32)
        ini = 0
        fin = batch_size
        # for X, Y in d_valtest_dataloader[paciente]:
        for X, Y in d_eval_dataloader[paciente]:
            X, Y = X.to(device), Y.to(device)
            
            if d['loss'] == 'CBL':
                pred, bottle_neck, last_conv = model(X)
            else:
                pred = model(X)
            # print(f'{len(pred)}')
            pred = pred.detach().cpu().numpy()
            Y = Y.detach().cpu().numpy()
            Y_paciente_pred[ini:fin] = pred
            Y_paciente[ini:fin] = Y
            ini = fin
            fin += batch_size
        
        Y_paciente_pred = Y_paciente_pred[:,1,:,:]# NCHW->NHW
        Y_paciente = Y_paciente[:,1,:,:]# NCHW->NHW
        Y_paciente_pred = np.transpose(Y_paciente_pred, axes=(1,2,0)).squeeze()# HWN. Reordenar slides en img 3D
        Y_paciente = np.transpose(Y_paciente, axes=(1,2,0)).squeeze()# HWN
        G = Y_paciente
        S_theta = Y_paciente_pred
        
        tn, fp, fn, tp = bbg.Matriz_confusion(G, S_theta)
        ACC = bbg.Metrica_ACC(tn, fp, fn, tp)
        TPR = bbg.Metrica_TPR(tn, fp, fn, tp)
        PPV = bbg.Metrica_PPV(tn, fp, fn, tp)
        F1 = bbg.Metrica_F1(tn, fp, fn, tp)
        AUC_ROC = bbg.Metrica_AUC_ROC(G.flatten(), S_theta.flatten())
        AUC_PR = bbg.Metrica_AUC_PR(G.flatten(), S_theta.flatten())
        F2 = bbg.Metrica_F2(tn, fp, fn, tp)
        s_real = ('paciente:'+paciente+' TN:'+str(round(tn, 2))+' FP:'+str(round(fp, 2)).rjust(5, '0')+' FN:'+str(round(fn, 2)).rjust(5, '0')+' TP:' +
                  str(round(tp, 2)).rjust(5, '0')+' ACC:'+str(round(ACC, 8)).ljust(6, '0')+' TPR:'+str(round(TPR, 4)).ljust(6, '0')+' PPV:' +
                  str(round(PPV, 4)).ljust(6, '0')+' Dice:'+str(round(F1, 8)).ljust(6, '0')+' AUC_ROC:'+str(round(AUC_ROC, 8)).ljust(6, '0')+
                  ' AUC_PR:'+str(round(AUC_PR, 8)).ljust(6, '0')+' F2:'+str(round(F2, 8)).ljust(6, '0'))
        print('s_real:', s_real)
        
        archivo_testing.write('k'+str(k)+'\n')
        archivo_testing.write(s_real+'\n')
        # =============================================================================
        # Resultados binarizando S_theta (S_theta>umbral=0.5)
        # =============================================================================
        r = Get_s_umbral(S_theta, 0.50, G, paciente)
        print(f'r:{r}')
        archivo_testing.write(r+'\n')
        # =============================================================================
        print('Tiempo t_1vol: '+str(round(time()-t_1vol, 2))+'[sg]')
    archivo_testing.close()
    return None

def mdf_paciente(paciente):
    MDF_array = np.zeros((160,160,192),np.float32)
    for slide in range(160):
        if slide==0 or slide==159:
            mdf = np.ones((160,192),dtype=np.float32)*np.nan
        else:
            Y_inputMDF,AC_inputMDF = ModuloA.Get_inputsMDF(dPAC,data_path,paciente,slide,ady,d['DS'])
            if AC_inputMDF.dtype=='float16':
                AC_inputMDF = AC_inputMDF.astype('float32')
            Y_inputMDF = ModuloA.Filtrar_CC_slides(Y_inputMDF,umbral_cc)
            mdf = ModuloA.MDF(Y_inputMDF, AC_inputMDF, radio,ady,ce,par_ce,prototipo,gamma_MDF,percentil,dist)
        MDF_array[slide]=mdf
    return paciente, MDF_array

# =============================================================================
# Inicio de programa
# =============================================================================

d = Get_parametros_entrada(sys.argv)

# =============================================================================
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu")
print(f"Using {device} device")
print('*'*80+'\nUtilizando GPU:', d['gpu'], '\n'+'*'*80)
# ============================================================================
print(d)
epocas = int(d['epocas'])
batch_size = int(d['batch_size'])
batch_size_val = int(d['batch_size_val'])

k = int(d['k'])
# if d['batch_loss']=='T':
#     batch_loss = True
# else:
#     batch_loss = False

batch_loss = True

loss_function = getattr(LossesAndMetrics,d['loss'])()
print(loss_function)

metric = getattr(LossesAndMetrics,d['metric'])
metrics = []
metrics.append(metric)
print('loss_function:',loss_function,'metrics:',metrics)
# print(loss_function, metrics)

input_dim_2D = getattr(bbg,d['DS']+'_input_dim_2D')
nombre_modelo = 'k'+str(k)+'.pth'
print('input_dim_2D:',input_dim_2D)

print('os.uname()[1]:',os.uname()[1])
if os.uname()[1] == 'f15':# local
    data_path = os.path.join('/media/gustavo/Disco_2/DS_Imagenes/',d['DS'],d['DS']+'_2D_160x192_TN')
    path_caracteristicas =   '/media/gustavo/Disco_2/DS_Imagenes/{DS}/{DS}_160x160x192_caracteristicas/'.format(DS=d['DS'])
    # data_path = os.path.join('<here/your/path>',d['DS'],d['DS']+'_2D_160x192_TN')
    # path_caracteristicas =   '<here/your/path>'.format(DS=d['DS'])
elif '/home' in os.getcwd():# server
    # data_path = os.path.join('/home/'+os.getlogin()+'/DS_Imagenes/',d['DS'],d['DS']+'_2D_160x192_TN')
    # path_caracteristicas =   '/home/'+os.getlogin()+'/DS_Imagenes/{DS}/{DS}_160x160x192_caracteristicas/'.format(DS=d['DS'])
    data_path = os.path.join('<here/your/path>',d['DS'],d['DS']+'_2D_160x192_TN')
    path_caracteristicas =   '<here/your/path>'.format(DS=d['DS'])
# else: # Vast_ai
#     path_DSI = os.path.join('/workspace','DS_Imagenes')
print('data_path:',data_path)

if d['ES']=='T':
    patience = int(d['patience'])
    start_es = int(d['start_es'])
    string_es = '_es'+str(start_es)+'-'+d['patience']
    if d['DS']=='ISBI2015' or d['DS']=='MSSEG2016':
        folds = getattr(bbg,d['folds']+'_trainvaltest')
        pacientes_train, pacientes_val, pacientes_test = folds[k]
    elif d['DS']=='WMH2017':
        pacientes_train, pacientes_val, pacientes_test = getattr(bbg,d['DS']+'_trainvaltest')
    print(f'pacientes_train: #{len(pacientes_train)}, {pacientes_train}')
    print(f'pacientes_val: #{len(pacientes_val)}, {pacientes_val}')
    print(f'pacientes_test: #{len(pacientes_test)}, {pacientes_test}')
else:
    patience = epocas+1
    string_es = ''
    folds = getattr(bbg,d['folds']+'_traintest')
    pacientes_train, pacientes_test = folds[k]
    print('pacientes_train=',pacientes_train)
    print('pacientes_test=',pacientes_test)
print(d['ES'],'patience:',patience)
pacientes = getattr(bbg,'pacientes_'+d['DS'])
print(f'pacientes:{pacientes}')
umbral_vol_training = int(d['umbral_vol_training'])
print('umbral_vol_training:',umbral_vol_training)

alpha_TL = float(d['alpha_TL'])
beta_TL = 1-alpha_TL
gamma = float(d['gamma'])
alpha_HD = float(d['alpha_HD'])
beta_ASL = float(d['beta_ASL'])
w_SEL = float(d['w_SEL'])
alpha_BS = float(d['alpha_BS'])
beta_BS_LC = float(d['beta_BS_LC'])
wa_ABL = float(d['wa_ABL'])
gamma_CBL = float(d['gamma_CBL'])
sc = d['sc']; canal = d['canal']
radio = int(d['r']); ady = int(d['ady'])
ce = d['ce']; par_ce = float(d['par_ce'])
prototipo = d['prototipo']; gamma_MDF = float(d['gamma_MDF'])
percentil = float(d['percentil']); dist = int(d['dist'])
umbral_cc = int(d['umbral_cc'])
parMD_weight = float(d['parMD_weight'])
parMD_pot = float(d['parMD_pot'])
parMD_sq = int(d['parMD_sq'])
parMD_quantil = float(d['parMD_quantil'])
umbral_P_MDF = int(d['umbral_P_MDF'])
batch_size_dtrain=2

if d['loss']=='FT_loss':
    string_loss = d['loss']+'_'+d['alpha_TL']+'_'+d['gamma']
elif d['loss']=='HD_loss':
    string_loss = d['loss']+'_'+d['alpha_HD']
elif d['loss']=='AS_loss':
    string_loss = d['loss']+'_'+d['beta_ASL']
elif d['loss']=='SE_loss':
    string_loss = d['loss']+'_'+d['w_SEL']
elif d['loss']=='BSLoss' or d['loss']=='BSL_LC':
    string_loss = d['loss']+'_'+d['alpha_BS']+'_'+d['beta_BS_LC']
elif d['loss']=='ABL':
    string_loss = d['loss']+'_wa'+d['wa_ABL']
elif d['loss']=='CBL':
    string_loss = d['loss']+'_g'+d['gamma_CBL']
elif 'MD_loss' in d['loss']:
    print('\nEntrenar: MD_loss:')
    print(f'''sc:{sc},canal:{canal},r:{radio},ady:{ady},ce:{ce},par_ce:{par_ce},prototipo:{prototipo},
          gamma_MDF:{gamma_MDF},percentil:{percentil},dist:{dist},umbral_cc:{umbral_cc}''')
    plantilla_MDF = 'MDF_{}_r{}a{}p{}{}g{}p{}_{}_d{}u{}'
    cep = ce+str(par_ce)# Estimador y parámetro de las matrices de covarianzas y precisión
    prot='p' if prototipo=='prototipo_media' else 'm'# o vector de medianas
    carpeta_MDF = plantilla_MDF.format(sc,radio,ady,prot,cep,gamma_MDF,percentil,canal,dist,umbral_cc)
    print(f'carpeta_MDF:{carpeta_MDF}')
    
    listaCaracteristicas_utilizada = ModuloA.Get_lista_características(sc)
    print(f'\nlistaCaracteristicas_utilizada:\n{listaCaracteristicas_utilizada}')
    for feature in listaCaracteristicas_utilizada:
        print(f'feature: {feature}')
    
    if d['loss']=='MD_loss':
        string_loss = d['loss']+'_w'+str(parMD_weight)+'p'+str(parMD_pot)+'-'+carpeta_MDF[4:]+'-'
    else:
        string_loss = d['loss']+'-'+carpeta_MDF[4:]+'-'
else: # BCE, Dice, Boundary loss, ABl, CBL
    string_loss = d['loss']
# string_loss +='_bm'+d['batch_loss']

nombre_carpeta_principal = d['DS']+'_'+d['red']+'_'+string_loss+'_b'+str(
    batch_size)+'_mp'+d['mixed_precision']+'_e'+str(epocas)+string_es#+'_ut'+d['umbral_vol_training']
nombre_carpeta_principal=nombre_carpeta_principal.replace('-_','-')# Para MD_loss
path_carpeta_principal = os.path.join(os.getcwd(),'Experimentos',nombre_carpeta_principal)
print('path_carpeta_principal:',path_carpeta_principal)
if nombre_carpeta_principal not in os.listdir(os.getcwd()+'/Experimentos'):
    os.mkdir(path_carpeta_principal)
nombre_carpeta_ce = 'ce_'+d['corrida']
path_carpeta_ce = os.path.join(path_carpeta_principal,nombre_carpeta_ce)
print('\nnombre_carpeta_principal:',nombre_carpeta_principal)
print('nombre_carpeta_ce:',nombre_carpeta_ce)

# =============================================================================    
if d['tarea'] == 'crear_carpetas':# Si es que un nodo envía resultados antes que se cree la carpeta en el nodo central
    print('#'*10, 'CREAR CARPETAS', '#'*10)
    print('Se creó carpeta:', path_carpeta_principal)
elif d['tarea'] == 'training':
    print('#'*10, 'TRAINING', '#'*10)
    
    if 'MD_loss' in d['loss']:
        dPAC={}# {paciente: Array Características}
        # =============================================================================
        # Obtener diccionario con pacientes como llaves y matriz AC como valor
        # =============================================================================
        for paciente in pacientes[:]:
            AC = ModuloA.Get_ArrayCaracteristicas(path_caracteristicas,paciente,listaCaracteristicas_utilizada)
            if AC.dtype=='float16':
                AC = AC.astype('float32')
            dPAC[paciente]=AC
        print(f'\ndPAC:{dPAC.keys()} {dPAC[paciente].shape}')
        ### np.savez('dPAC',**dPAC)
        # =============================================================================
        # Paralelizar MDFs
        # =============================================================================
        tiempo_MDFs = time()
        nombre_dMDF='dMDF_'+carpeta_MDF[4:]+'_'+d['DS']
        print(f'nombre_dMDF:{nombre_dMDF}')
        if os.uname()[1]== 'f15' or os.uname()[1]=='mineria' or os.uname()[1]=='zealot':
            max_workers=None
        elif os.uname()[1]=='fondecyt1' or os.uname()[1]=='fondecyt2':
            max_workers=4
        print(f'max_workers:{max_workers}')
        
        # =============================================================================
        flag_cargar_dMDF=True
        # flag_cargar_dMDF=False
        # =============================================================================
        if flag_cargar_dMDF:
            # p = '/'.join(os.getcwd().split('/')[:-1])
            # dMDF=dict(np.load(os.path.join(p,nombre_dMDF+'.npz')))
            
            D2='/media/gustavo/Disco_2/pytorch/2025/MDF_Dinamico/'
            # D2='<here/your/path>'
            dMDF=dict(np.load(os.path.join(D2,nombre_dMDF+'.npz')))
            
        else:
            resultados = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                functions = []
                for paciente in pacientes[:]:
                    function = executor.submit(mdf_paciente, paciente)
                    functions.append(function)
                for function in as_completed(functions):
                    resultados.append(function.result())
            print(f'resultados:{len(resultados)} {len(resultados[0])}')
            dMDF={}# {paciente: mdf}
            for paciente, mdf in resultados:
                dMDF[paciente]=mdf
            ### np.savez(nombre_dMDF,**dMDF)
        
        print(f'tiempo_MDFs:{round(time()-tiempo_MDFs,2)}[sg]')
        # =============================================================================
    Training()
elif d['tarea'] == 'testing':
    print('#'*10, 'TESTING', '#'*10)
    Testing()
elif d['tarea'] == 'scp':
    print('#'*10, 'SCP', '#'*10)
    SCP()
