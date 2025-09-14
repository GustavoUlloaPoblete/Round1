#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 20:23:25 2025

@author: gustavo
"""

'''
Simular y_pred utilizando modelo entrenado
'''
import torch
import os
import ModuloA
import Biblioteca_General as bbg
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology

nombre_carpeta_principal='MSSEG2016_Unet_MD_loss_w1-A5GL11_r5a1pmGL0.3g1.0p0.9_FLAIR_d10u0-bmT_e200_b16_Adam_da0.0_mpT_es200-200_ut1'
DS=nombre_carpeta_principal.split('_')[0]
data_path = os.path.join('/media/gustavo/Disco_2/DS_Imagenes/',DS,DS+'_2D_160x192_TN')
path_caracteristicas = os.path.join('/media/gustavo/Disco_2/DS_Imagenes/',DS,DS+'_160x160x192_caracteristicas')
print(f'path_caracteristicas:{path_caracteristicas}')

pacientes = getattr(bbg,'pacientes_'+DS)
print(f'pacientes:{pacientes}')
paciente=pacientes[2]
paciente_respaldo=paciente
epoca='25';slide=110# AV
epoca='25';slide=90

# =============================================================================
# Cargar el modelo
# =============================================================================
path_carpeta_principal = os.path.join(os.getcwd(),'Experimentos',nombre_carpeta_principal)
print('path_carpeta_principal:',path_carpeta_principal)
nombre_carpeta_ce = 'ce_t15_gpu0-0'
path_carpeta_ce = os.path.join(path_carpeta_principal,nombre_carpeta_ce)
print('nombre_carpeta_ce:',nombre_carpeta_ce)
nombre_modelo='k0_e{}.pth'.format(epoca)
model = torch.load(os.path.join(path_carpeta_ce,nombre_modelo))#.to('cpu')

# =============================================================================
# Cargar entrada y target
# =============================================================================
sc='A5GL11';canal='FLAIR';r='5';ady='1';ce='GL';par_ce='0.3'
prototipo='prototipo_mediana';gamma_MDF='1.0';percentil='0.9';dist='10';umbral_cc='0'
listaCaracteristicas_utilizada = ModuloA.Get_lista_características(sc)
print(f'\nlistaCaracteristicas_utilizada:\n{listaCaracteristicas_utilizada}')
for feature in listaCaracteristicas_utilizada:
    print(f'feature: {feature}')

dPAC = ModuloA.Obtener_dPAC(path_caracteristicas,pacientes,listaCaracteristicas_utilizada)
print(f'\ndPAC:{dPAC.keys()} {dPAC[paciente].shape}')
# sc = d['sc'];# canal = d['canal']
r = int(r); ady = int(ady);# ce = d['ce']; 
par_ce = float(par_ce);# prototipo = d['prototipo']; 
gamma_MDF = float(gamma_MDF);percentil = float(percentil)
dist = int(dist);umbral_cc = int(umbral_cc)

ady=1

paciente=paciente_respaldo###

Y_inputMDF,AC_inputMDF = ModuloA.Get_inputsMDF(dPAC,data_path,paciente,slide,ady,DS)
if AC_inputMDF.dtype=='float16':
    AC_inputMDF = AC_inputMDF.astype('float32')
Y_inputMDF = ModuloA.Filtrar_CC_slides(Y_inputMDF,umbral_cc)
print(f'\nY_inputMDF:{Y_inputMDF.shape} {Y_inputMDF.dtype}')
print(f'AC_inputMDF:{AC_inputMDF.shape} {AC_inputMDF.dtype}')

# =============================================================================
# Graficar entrada y target a la CNN
print('\nGraficar entrada y target a la CNN')
# =============================================================================
Y,X = ModuloA.Get_YX_input(data_path,paciente,slide)
print(f'Y: {Y.shape} {Y.dtype}')
print(f'X:{X.shape} {X.dtype}')
# print(f'np.array_equal(Y_inputMDF[1],Y[1]):{np.array_equal(Y_inputMDF[1],Y[1])}')
plt.figure(figsize=(33,25))
plt.subplot(3,3,1);plt.imshow(X[0],cmap='gray');plt.title('T1')
plt.subplot(3,3,2);plt.imshow(X[1],cmap='gray');plt.title('T2')
plt.subplot(3,3,3);plt.imshow(X[2],cmap='gray');plt.title('FLAIR')
plt.subplot(3,3,4);plt.imshow(Y[1],cmap='gray');plt.title('Y')

# =============================================================================
# Obtener MDF de Y y X
# =============================================================================
Y_mdf = ModuloA.MDF(Y_inputMDF, AC_inputMDF, r,ady,ce,par_ce,prototipo,gamma_MDF,percentil,dist)
print(f'Y_mdf:{Y_mdf.shape} {Y_mdf.dtype} mean:{np.mean(Y_mdf)} std:{np.std(Y_mdf)}')
plt.subplot(3,3,5);plt.imshow(Y_mdf,cmap='gray');plt.title('Y_mdf')

# =============================================================================
# Obtener salida P de la CNN
# =============================================================================
X = torch.from_numpy(X)
# Y = torch.from_numpy(Y).to(torch.float32)
print(f'X: {X.shape} {X.dtype}')
# print(f'Y: {Y.shape} {Y.dtype}')
X=torch.unsqueeze(X,0)
# Y=torch.unsqueeze(Y,0)
print(f'\nEntrada a la red:\nX: {X.shape} {X.dtype}')
# print(f'Y: {Y.shape} {Y.dtype}')
X = X.to('cuda')
# Y = Y.to('cuda')


# ady=1
model.eval()
print(f'ady:{ady}')
if ady==0:
    pred = model(X)
    print(f'ady0-pred: {pred.shape} {pred.dtype}')
    # P = torch.zeros((1,160,192),dtype=torch.float32)
    P=pred[:,1]
elif ady==1:
    print(f'ady1')
    P = torch.zeros((3,160,192),dtype=torch.float32)
    for sl in range(3):
        slide_new=slide-ady+sl
        print(f'slide_new:{slide_new}')
        _,X = ModuloA.Get_YX_input(data_path,paciente,slide_new)
        X = torch.from_numpy(X)
        X = torch.unsqueeze(X,0)
        X = X.to('cuda')
        print(f'\nEntrada a la red new:\nX: {X.shape} {X.dtype}')
        pred = model(X)
        print(f'pred: {pred.shape} {pred.dtype}')
        P[sl] = pred[0,1]
print(f'P: {P.shape} {P.dtype}')
P=P.detach().cpu().numpy()
print(f'P: {P.shape} {P.dtype} {np.unique(P)}')
P[P<=0.5]=0;P[P>0.5]=1#Binarizar manteniendo dtype
print(f'P: {P.shape} {P.dtype} {np.unique(P)}')
P = P.astype(np.bool_)
plt.subplot(3,3,6);plt.imshow(P[ady//2],cmap='gray');plt.title('P[0]')
if ady==1:
    plt.subplot(3,3,7);plt.imshow(P[1],cmap='gray');plt.title('P[1]')
    plt.subplot(3,3,8);plt.imshow(P[2],cmap='gray');plt.title('P[2]')

# =============================================================================
# Obtener MDF de P
# =============================================================================
P_inputMDF=np.copy(P)
if AC_inputMDF.dtype=='float16':
    AC_inputMDF = AC_inputMDF.astype('float32')
print(f'\nP_inputMDF:{P_inputMDF.shape} {P_inputMDF.dtype}')
print(f'AC_inputMDF:{AC_inputMDF.shape} {AC_inputMDF.dtype}')
# r=5
P_mdf = ModuloA.MDF(P_inputMDF, AC_inputMDF, r,ady,ce,par_ce,prototipo,gamma_MDF,percentil,dist)
plt.subplot(3,3,9);plt.imshow(P_mdf,cmap='gray');plt.title('P_mdf')


# =============================================================================
# Graficar coloreando TP, FP y FN
print('\nGraficar coloreando TP, FP y FN:')
# =============================================================================
Y=np.copy(Y_inputMDF)
print(f'Y: {Y.shape} {Y.dtype} {type(Y)}')
print(f'P: {P.shape} {P.dtype} {type(P)}')
Y=Y[len(Y)//2]
P=P[len(P)//2]
print(f'\n\nY: {Y.shape} {Y.dtype} {type(Y)}')
print(f'P: {P.shape} {P.dtype} {type(P)}')
print(f'X: {X.shape} {X.dtype} {type(X)}')
X = X.detach().cpu().numpy()
print(f'X: {X.shape} {X.dtype} {type(X)}')
X=np.squeeze(X)
X = X[2]
print(f'X: {X.shape} {X.dtype} {type(X)}')
X_rgb = ModuloA.Gray2RGB(X)
X_rgb = bbg.Normalizar(X_rgb,1)
print(f'X_rgb: {X_rgb.shape} {X_rgb.dtype} {type(X_rgb)}')
mask_TP,mask_FP,mask_FN = ModuloA.comparar_CC(Y,P)
X_rgb = ModuloA.Solapar(X_rgb,[mask_TP,mask_FP,mask_FN],['rojo','amarillo','verde'])
plt.figure(figsize=(25,20))
plt.subplot(1,3,1);plt.imshow(X_rgb,cmap='gray');plt.title('HxW')
plt.subplot(1,3,2);plt.imshow(Y,cmap='gray');plt.title('Y')
plt.subplot(1,3,3);plt.imshow(P,cmap='gray');plt.title('P')

plt.figure(figsize=(33,25))
TP_YLesiones,FN_YLesiones, TP_PLesiones,FP_PLesiones = ModuloA.Obtener_MC_Lesiones(Y,P)
plt.subplot(1,4,1);plt.imshow(TP_YLesiones,cmap='gray');plt.title('TP_YLesiones')
plt.subplot(1,4,2);plt.imshow(FN_YLesiones,cmap='gray');plt.title('FN_YLesiones')
plt.subplot(1,4,3);plt.imshow(TP_PLesiones,cmap='gray');plt.title('TP_PLesiones')
plt.subplot(1,4,4);plt.imshow(FP_PLesiones,cmap='gray');plt.title('FP_PLesiones')


mask_TP,mask_FP,mask_FN=ModuloA.comparar_CC(TP_YLesiones,TP_PLesiones)
plt.figure(figsize=(30,10))
plt.subplot(2,4,1);plt.imshow(mask_TP,cmap='gray');plt.title('mask_TP')
plt.subplot(2,4,2);plt.imshow(mask_FP,cmap='gray');plt.title('mask_FP')
plt.subplot(2,4,3);plt.imshow(mask_FN,cmap='gray');plt.title('mask_FN')

XOR=np.logical_xor(TP_YLesiones, TP_PLesiones)
plt.subplot(2,4,4);plt.imshow(XOR,cmap='gray');plt.title('XOR')

OR_FPFN=np.logical_or(mask_FP, mask_FN)
plt.subplot(2,4,5);plt.imshow(OR_FPFN,cmap='gray');plt.title('OR_FPFN')
# OR=np.logical_or(TP_YLesiones, TP_PLesiones)

# =============================================================================
# Aplicar operaciones morfológicas
# =============================================================================
sq=morphology.square(width=5)
d=morphology.dilation(OR_FPFN,sq)
e=morphology.erosion(OR_FPFN,sq)
plt.subplot(2,4,6);plt.imshow(d,cmap='gray');plt.title('dilation')
plt.subplot(2,4,7);plt.imshow(e,cmap='gray');plt.title('erosion')

# =============================================================================
# Aplicar máscaras de FN y FP al MDF
# =============================================================================






# # testear otra cosas
# Y_inputMDF=np.load('Y_inputMDF.npy')
# P_inputMDF=np.load('P_inputMDF.npy')
# print(f'Y_inputMDF: {Y_inputMDF.shape}')
# print(f'P_inputMDF: {P_inputMDF.shape}')
# # print(Y_inputMDF)
# plt.figure(figsize=(33,15))
# plt.subplot(2,3,1);plt.imshow(Y_inputMDF[0],cmap='gray');plt.title('Y_inputMDF[0]')
# plt.subplot(2,3,2);plt.imshow(Y_inputMDF[1],cmap='gray');plt.title('Y_inputMDF[1]')
# plt.subplot(2,3,3);plt.imshow(Y_inputMDF[2],cmap='gray');plt.title('Y_inputMDF[2]')
# plt.subplot(2,3,4);plt.imshow(P_inputMDF[0],cmap='gray');plt.title('P_inputMDF[0]')
# plt.subplot(2,3,5);plt.imshow(P_inputMDF[1],cmap='gray');plt.title('P_inputMDF[1]')
# plt.subplot(2,3,6);plt.imshow(P_inputMDF[2],cmap='gray');plt.title('P_inputMDF[2]')

