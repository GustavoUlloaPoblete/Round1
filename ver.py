#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 15:23:22 2025

@author: gustavo
"""

import matplotlib.pyplot as plt
import numpy as np, os
import Biblioteca_General as bbg, ModuloA

d={}
d['DS']='MSSEG2016'
sc='A5GL11';canal='FLAIR';r=5;ady=1;ce='GL';par_ce=0.3
prototipo='prototipo_mediana';gamma_MDF=1.0;percentil=0.9;dist=10;umbral_cc=0
data_path = os.path.join('/media/gustavo/Disco_2/DS_Imagenes/',d['DS'],d['DS']+'_2D_160x192_TN')
path_caracteristicas =   '/media/gustavo/Disco_2/DS_Imagenes/{DS}/{DS}_160x160x192_caracteristicas/'.format(DS=d['DS'])
pacientes = getattr(bbg,'pacientes_'+d['DS'])
print(f'pacientes:{pacientes}')
listaCaracteristicas_utilizada = ModuloA.Get_lista_características(sc)
print(f'\nlistaCaracteristicas_utilizada:\n{listaCaracteristicas_utilizada}')

# =============================================================================
# Cargar dAC
# =============================================================================
# Obtener diccionario con pacientes como llaves y matriz AC como valor
# =============================================================================
# dPAC={}# {paciente: Array Características}
# for paciente in pacientes[:]:
#     AC = ModuloA.Get_ArrayCaracteristicas(path_caracteristicas,paciente,listaCaracteristicas_utilizada)
#     if AC.dtype=='float16':
#         AC = AC.astype('float32')
#     dPAC[paciente]=AC
# print(f'\ndPAC:{dPAC.keys()} {dPAC[paciente].shape}')
# np.savez('dPAC',**dPAC)
dPAC=dict(np.load('dPAC'+'.npz'))

# =============================================================================
# Obtener MDF
# =============================================================================
slide=80
paciente = pacientes[1]
Y_inputMDF,AC_inputMDF = ModuloA.Get_inputsMDF(dPAC,data_path,paciente,slide,ady,d['DS'])
if AC_inputMDF.dtype=='float16':
    AC_inputMDF = AC_inputMDF.astype('float32')
# Y_inputMDF = ModuloA.Filtrar_CC_slides(Y_inputMDF,umbral_cc)
# gamma_MDF=0.5
dist=10
mdf = ModuloA.MDF(Y_inputMDF, AC_inputMDF, r,ady,ce,par_ce,prototipo,gamma_MDF,percentil,dist)
Y = np.load(os.path.join(data_path,'{}_{}_target.npy'.format(paciente,slide)))
X = np.load(os.path.join(data_path,'{}_{}.npy'.format(paciente,slide)))
print(f'mdf:{mdf.shape} {mdf.sum()}')

plt.figure(figsize=(33,15))
plt.subplot(2,3,1);plt.imshow(X[2],cmap='gray');plt.title('FLAIR')
plt.subplot(2,3,2);plt.imshow(Y,cmap='gray');plt.title('Y')
plt.subplot(2,3,3);plt.imshow(mdf,cmap='gray');plt.title('mdf')
mdf_05 = mdf**0.5
mdf_15 = mdf**1.5
mdf_20 = mdf**2.0
plt.subplot(2,3,4);plt.imshow(mdf_05,cmap='gray');plt.title('mdf_05')
plt.subplot(2,3,5);plt.imshow(mdf_15,cmap='gray');plt.title('mdf_15')
plt.subplot(2,3,6);plt.imshow(mdf_20,cmap='gray');plt.title('mdf_20')

# # X_antes=np.load('X_antes.npy')
# # X_despues=np.load('X_despues.npy')
# # pred_antes=np.load('pred_antes.npy')
# # pred_despues=np.load('pred_despues.npy')

# # plt.figure(figsize=(33,15))
# # plt.subplot(2,2,1);plt.imshow(X_antes[0,2],cmap='gray');plt.title('X_antes[0,2]')
# # plt.subplot(2,2,2);plt.imshow(X_despues[0,2],cmap='gray');plt.title('X_despues[0,2]')

# # plt.subplot(2,2,3);plt.imshow(pred_antes[0,1],cmap='gray');plt.title('pred_antes[0,1]')
# # plt.subplot(2,2,4);plt.imshow(pred_despues[0,1],cmap='gray');plt.title('pred_despues[0,1]')