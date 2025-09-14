#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 09:18:12 2024

@author: gustavo
"""
# from IPython.display import display, HTML
# display(HTML("<style>.container { width:100% !important; }</style>"))

# import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import find_boundaries
from scipy import ndimage
from sklearn.covariance import EmpiricalCovariance, OAS, LedoitWolf, GraphicalLasso, GraphicalLassoCV
# import mahotas
# import pyfeats
from scipy.stats import describe
import os
from sklearn.decomposition import PCA
import Biblioteca_General as bbg
# from skimage import measure
from skimage.measure import label

import warnings
from sklearn.exceptions import ConvergenceWarning

def Obtener_dPAC(path_caracteristicas,pacientes,listaCaracteristicas_utilizada):
    '''
    Obtener diccionario con pacientes como llaves y matriz AC como valor
    '''
    dPAC={}
    for paciente in pacientes[:]:
        AC = Get_ArrayCaracteristicas(path_caracteristicas,paciente,listaCaracteristicas_utilizada)
        if AC.dtype=='float16':
            AC = AC.astype('float32')
        dPAC[paciente]=AC
    return dPAC

# =============================================================================
# Para obtener máscaras con FP y FN
# =============================================================================
# def comparar_CC(Y,P):
#     '''
#     Y:2xHxW , P:2xHxW
#     '''
#     mask_TP=Y[1]*P[1]
#     mask_FP=Y[0]*P[1]
#     mask_FN=Y[1]*P[0]
#     return mask_TP,mask_FP,mask_FN
def comparar_CC(Y,P):
    '''
    Y:HxW , P:HxW 
    '''
    mask_TP=Y*P
    mask_FP=np.logical_not(Y)*P
    mask_FN=Y*np.logical_not(P)
    return mask_TP,mask_FP,mask_FN
def Obtener_TPFN(Y,P,umbral_cc=0):
    Y_cc,Y_Nl = label(Y, return_num=True)
    P_cc,P_Nl = label(P, return_num=True)
    TP_YLesiones=np.zeros_like(Y)
    FN_YLesiones=np.zeros_like(Y)
    for Y_nl in range(1,Y_Nl+1):
        Y_lesion = Y_cc==Y_nl
        Y_FN=True
        for P_nl in range(1,P_Nl+1):
            P_lesion = P_cc==P_nl
            mask_TP,mask_FP,mask_FN=comparar_CC(Y_lesion,P_lesion)
            if mask_TP.sum()>umbral_cc:#Hay solapamiento, determinar umbral de solapamiento
                TP_YLesiones+=Y_lesion
                Y_FN=False
        if Y_FN:
            FN_YLesiones+=Y_lesion
    return TP_YLesiones,FN_YLesiones
def Obtener_MC_Lesiones(Y,P):
    '''
    Retorna máscaras con lesiones TP de Y y P, como también FN, FP 
    Y: bool_-HxW_, P: bool_-HxW
    '''
    TP_YLesiones,FN_YLesiones=Obtener_TPFN(Y,P)
    TP_PLesiones,FP_PLesiones=Obtener_TPFN(P,Y)
    return TP_YLesiones,FN_YLesiones, TP_PLesiones,FP_PLesiones

def Gray2RGB(I):
    if len(I.shape)==2:
        IRGB=np.zeros((I.shape[0],I.shape[1],3),np.float32)
        IRGB[:,:,0]=I;IRGB[:,:,1]=I;IRGB[:,:,2]=I
    elif len(I.shape)==3:
        IRGB=np.zeros((I.shape[0],I.shape[1],I.shape[2],3),np.float32)
        IRGB[:,:,:,0]=I;IRGB[:,:,:,1]=I;IRGB[:,:,:,2]=I
    return IRGB
def Solapar(I,mascaras,colores):
    for i in range(len(mascaras)):
        mask=mascaras[i]
        color=colores[i]
        if color=='negro':
            rgb=(0.0,0.0,0.0)
        elif color=='blanco':
            rgb=(1.0,1.0,1.0)
        elif color=='rojo':
            rgb=(1.0,0.0,0.0)
        elif color=='verde':
            rgb=(0.0,1.0,0.0)
        elif color=='azul':
            rgb=(0.0,0.0,1.0)
        elif color=='cian':
            rgb=(0.0,1.0,1.0)
        elif color=='magenta':
            rgb=(1.0,0.0,1.0)
        elif color=='amarillo':
            rgb=(1.0,1.0,0.0)
        mask=np.argwhere(mask)
        if len(I.shape[:-1])==2:
            for i in range(len(mask)):
                I[mask[i,0],mask[i,1],:]=rgb
        elif len(I.shape[:-1])==3:
            for i in range(len(mask)):
                I[mask[i,0],mask[i,1],mask[i,2],:]=rgb
    return I

# P_inputMDF,AC_inputMDF = ModuloA.Get_inputsMDF_Pred(dp,dPAC,data_path,paciente,ady,d['DS'])
# P_inputMDF,AC_inputMDF = ModuloA.Get_inputsMDF_Pred(dp,dPAC,data_path,paciente,slide,ady,d['DS'])
# def Get_inputsMDF_Pred(dp,dPAC,data_path,paciente,slide,ady,DS):
def Get_inputsMDF_Pred(dp,dpac,data_path,paciente,slide,ady,DS):
    if ady==0:
        # AC = dPAC[paciente][:,slide:slide+1]
        AC = dpac
        Y = np.zeros((1,160,192),dtype=np.bool_)
        Y[0] = dp
    elif ady==1:
        # AC = dPAC[paciente][:,slide-1:slide+2]
        AC = dpac
        Y = dp
        # for i in range(3):
        #     Y[i] = dPpred[paciente][slide-1+i]
        #     # Y[i] = np.load(os.path.join(path_base,'{}_{}_target.npy'.format(paciente,slide-1+i)))
    return Y, AC

def Get_inputsMDF(d,path_base,paciente,slide,ady,DS):
    if ady==0:
        AC = d[paciente][:,slide:slide+1]
        Y = np.zeros((1,160,192),dtype=np.bool_)
        Y[0] = np.load(os.path.join(path_base,'{}_{}_target.npy'.format(paciente,slide)))
    elif ady==1:
        AC = d[paciente][:,slide-1:slide+2]
        Y = np.zeros((3,160,192),dtype=np.bool_)
        for i in range(3):
            Y[i] = np.load(os.path.join(path_base,'{}_{}_target.npy'.format(paciente,slide-1+i)))
    return Y, AC

# def Get_YX_inputAdy(path_base,paciente,slide,ady,DS,canal):
#     if ady==0:
#         Y = np.zeros((1,160,192),dtype=np.bool_)
#         X = np.zeros((1,160,192),dtype=np.float32)
#         Y[0] = np.load(os.path.join(path_base,'{}_{}_target.npy'.format(paciente,slide)))
#         X[0] = np.load(os.path.join(path_base,'{}_{}.npy'.format(paciente,slide)))
#     elif ady==1:
#         Y = np.zeros((3,160,192),dtype=np.bool_)
#         X = np.zeros((3,160,192),dtype=np.float32)
#         for i in range(3):
#             Y[i] = np.load(os.path.join(path_base,'{}_{}_target.npy'.format(paciente,slide-1+i)))
#             X[i] = Get_canal(np.load(os.path.join(path_base,'{}_{}.npy'.format(paciente,slide-1+i))),DS,canal)
#     return Y, X

def Get_YX_input(path_base,paciente,slide):
    # Y = np.zeros((1,160,192),dtype=np.bool_)
    # X = np.zeros((3,160,192),dtype=np.float32)
    target = np.load(os.path.join(path_base,'{}_{}_target.npy'.format(paciente,slide)))
    H,W=target.shape
    Y = np.zeros((2,H,W),dtype=np.float32)
    Y[0,:,:] = 1-target
    Y[1,:,:] = target
    
    X = np.load(os.path.join(path_base,'{}_{}.npy'.format(paciente,slide)))
    return Y, X

def Get_canal(X,DS,canal):
    # print(f'X.sum():{X.sum()}, {X.shape,DS,canal}')
    if canal=="T1":
        X = X[0]
    elif canal=="T2":# 'WMH2017' no tiene
        if DS == 'MSSEG2016' or DS == 'ISBI2015':
            X = X[1]
        elif DS == 'WMH2017':
            print('ERROR!, WMH2017 no tiene T2')
    elif canal=="FLAIR":
        if DS == 'MSSEG2016' or DS == 'ISBI2015':
            X = X[2]
        elif DS == 'WMH2017':
            X = X[1]
    return X
def Get_YAC_inputAdy(d,path_base,paciente,slide,ady,DS,canal):
    if ady==0:
        AC = d[paciente][:,slide:slide+1]
        Y = np.zeros((1,160,192),dtype=np.bool_)
        # X = np.zeros((1,160,192),dtype=np.float32)
        Y[0] = np.load(os.path.join(path_base,'{}_{}_target.npy'.format(paciente,slide)))
        # X[0] = np.load(os.path.join(path_base,'{}_{}.npy'.format(paciente,slide)))
    elif ady==1:
        AC = d[paciente][:,slide-1:slide+2]
        Y = np.zeros((3,160,192),dtype=np.bool_)
        # X = np.zeros((3,160,192),dtype=np.float32)
        for i in range(3):
            Y[i] = np.load(os.path.join(path_base,'{}_{}_target.npy'.format(paciente,slide-1+i)))
            # X[i] = Get_canal(np.load(os.path.join(path_base,'{}_{}.npy'.format(paciente,slide-1+i))),DS,canal)
    return Y, AC#, X
def Get_YACX_inputAdy(d,path_base,paciente,slide,ady,DS,canal):
    if ady==0:
        AC = d[paciente][:,slide:slide+1]
        Y = np.zeros((1,160,192),dtype=np.bool_)
        X = np.zeros((1,160,192),dtype=np.float32)
        Y[0] = np.load(os.path.join(path_base,'{}_{}_target.npy'.format(paciente,slide)))
        X[0] = np.load(os.path.join(path_base,'{}_{}.npy'.format(paciente,slide)))
    elif ady==1:
        AC = d[paciente][:,slide-1:slide+2]
        Y = np.zeros((3,160,192),dtype=np.bool_)
        X = np.zeros((3,160,192),dtype=np.float32)
        for i in range(3):
            Y[i] = np.load(os.path.join(path_base,'{}_{}_target.npy'.format(paciente,slide-1+i)))
            X[i] = Get_canal(np.load(os.path.join(path_base,'{}_{}.npy'.format(paciente,slide-1+i))),DS,canal)
    return Y, AC, X

def Filtrar_CC_slides(Y,umbral_cc):
    '''
    input: NxHxW
    output: NxHxW
    Filtrar solo la slide central
    '''
    # print(f'Filtrar_CC_slides:')
    Y_copy = np.copy(Y)
    # print(f'Filtrar_CC_slides\nY_copy:{Y_copy.shape[0]}')
    slide=Y_copy.shape[0]//2
    # for slide in range(Y_copy.shape[0]):
    cc,Nl = label(Y_copy[slide], return_num=True)
    # print(f'Nl:{Nl}, 0:{np.sum(cc==0)}, 1:{np.sum(cc==1)}, 2:{np.sum(cc==2)}, 3:{np.sum(cc==3)}, 4:{np.sum(cc==4)}')
    for nl in range(1,Nl+1):
        # print(f'nl:{nl} {np.sum(cc==nl)}, umbral_cc:{umbral_cc}')
        if np.sum(cc==nl)<=umbral_cc:
            # print(f'IF: nl:{nl}')
            Y_copy[slide][cc==nl]=0
    return Y_copy

def Mascara_dist(array,MP,d):
    array_copy = array.copy()
    array_copy_max = array_copy.max()
    mascara_dist = MP > d
    array_copy[mascara_dist]=array_copy_max
    return array_copy

def Get_XY_slides(paciente,DS,path_base,canal=None):
    size_inputs = getattr(bbg,DS+'_input_dim_3D')
    # print(f'size_inputs:{size_inputs}')
    size_X = (size_inputs[3],size_inputs[0],size_inputs[1],size_inputs[2])
    size_Y = (size_inputs[3],size_inputs[1],size_inputs[2])
    # print(f'size_X:{size_X}, size_Y:{size_Y}')
    X_canales_slides = np.zeros(size_X,dtype=np.float32)
    Y = np.zeros(size_Y,dtype=np.bool_)
    for slide in range(Y.shape[0]):
        X_name = '{PACIENTE}_{SLIDE}.npy'.format(PACIENTE=paciente,SLIDE=slide)
        Y_name = '{PACIENTE}_{SLIDE}_target.npy'.format(PACIENTE=paciente,SLIDE=slide)
        X_canales_slides[slide] = np.load(os.path.join(path_base,X_name))
        Y[slide] = np.load(os.path.join(path_base,Y_name))
    # print(f'X_canales_slides:{X_canales_slides.shape}{X_canales_slides.dtype}, Y:{Y.shape}{Y.dtype}')
    X_canales_slides = np.transpose(X_canales_slides,(1,0,2,3))
    # print(f'X_canales_slides:{X_canales_slides.shape}{X_canales_slides.dtype}, Y:{Y.shape}{Y.dtype}')
    if canal=="T1":
        X = X_canales_slides[0]
    elif canal=="T2":# 'WMH2017' no tiene
        if DS == 'MSSEG2016' or DS == 'ISBI2015':
            X = X_canales_slides[1]
        elif DS == 'WMH2017':
            print('ERROR!, WMH2017 no tiene T2')
    elif canal=="FLAIR":
        if DS == 'MSSEG2016' or DS == 'ISBI2015':
            X = X_canales_slides[2]
        elif DS == 'WMH2017':
            X = X_canales_slides[1]
    else:
        return X_canales_slides, Y
    # print(f'X:{X.shape} {X.dtype}, Y:{Y.shape} {Y.dtype}')
    return X, Y

def Get_lista_características(tipo_caracteristicas):
    if tipo_caracteristicas=='A5':
        listaCaracteristicas_utilizada = ['coord_i','coord_j','intensidad','varianza','gradMag']
    elif tipo_caracteristicas=='A6':
        listaCaracteristicas_utilizada = ['coord_i','coord_j','intensidad','varianza_5x5','gradMag']
    
    elif tipo_caracteristicas=='A5GL0':
        listaCaracteristicas_utilizada = ['coord_i','coord_j','intensidad','varianza','gradMag']
    
    
    elif tipo_caracteristicas=='A5GL1':
        listaCaracteristicas_utilizada = ['coord_i','coord_j','intensidad','varianza','gradMag',
                                           'GLCM_32_9x9_FLAIR_dAPCA:0']
    elif tipo_caracteristicas=='A5GL2':
        listaCaracteristicas_utilizada = ['coord_i','coord_j','intensidad','varianza','gradMag',
                                           'GLCM_32_9x9_FLAIR_dAPCA:0','GLCM_32_9x9_FLAIR_dAPCA:1']
    elif tipo_caracteristicas=='A5GL3':
        listaCaracteristicas_utilizada = ['coord_i','coord_j','intensidad','varianza','gradMag',
                                           'GLCM_32_9x9_FLAIR_dAPCA:0','GLCM_32_9x9_FLAIR_dAPCA:1','GLCM_32_9x9_FLAIR_dAPCA:2']
    
    
    elif tipo_caracteristicas=='A5GL4':
        listaCaracteristicas_utilizada = ['coord_i','coord_j','intensidad','varianza','gradMag',
                                          'GLRLM_32_9x9_FLAIR_dAPCA:0']
    elif tipo_caracteristicas=='A5GL5':
        listaCaracteristicas_utilizada = ['coord_i','coord_j','intensidad','varianza','gradMag',
                                          'GLRLM_32_9x9_FLAIR_dAPCA:0','GLRLM_32_9x9_FLAIR_dAPCA:1']
    elif tipo_caracteristicas=='A5GL6':
        listaCaracteristicas_utilizada = ['coord_i','coord_j','intensidad','varianza','gradMag',
                                          'GLRLM_32_9x9_FLAIR_dAPCA:0','GLRLM_32_9x9_FLAIR_dAPCA:1','GLRLM_32_9x9_FLAIR_dAPCA:2']
    
    elif tipo_caracteristicas=='A5GL7':
        listaCaracteristicas_utilizada = ['coord_i','coord_j','intensidad','varianza','gradMag',
                                          'GLSZM_32_9x9_FLAIR_dAPCA:0']
    elif tipo_caracteristicas=='A5GL8':
        listaCaracteristicas_utilizada = ['coord_i','coord_j','intensidad','varianza','gradMag',
                                          'GLSZM_32_9x9_FLAIR_dAPCA:0','GLSZM_32_9x9_FLAIR_dAPCA:1']
    elif tipo_caracteristicas=='A5GL9':
        listaCaracteristicas_utilizada = ['coord_i','coord_j','intensidad','varianza','gradMag',
                                          'GLSZM_32_9x9_FLAIR_dAPCA:0','GLSZM_32_9x9_FLAIR_dAPCA:1','GLSZM_32_9x9_FLAIR_dAPCA:2']
    
    elif tipo_caracteristicas=='A5GL10':
        listaCaracteristicas_utilizada = ['coord_i','coord_j','intensidad','varianza','gradMag',
                                          'GLCM_32_9x9_FLAIR_dAPCA:0',
                                          'GLSZM_32_9x9_FLAIR_dAPCA:0']
    # elif tipo_caracteristicas=='A5GL11':
    #     listaCaracteristicas_utilizada = ['coord_i','coord_j','intensidad','varianza','gradMag',
    #                                       'GLCM_32_9x9_FLAIR_dAPCA:0', 'GLCM_32_9x9_FLAIR_dAPCA:1',
    #                                       'GLSZM_32_9x9_FLAIR_dAPCA:0','GLSZM_32_9x9_FLAIR_dAPCA:1']
    # elif tipo_caracteristicas=='A5GL11':
    #     listaCaracteristicas_utilizada = ['coord_i','coord_j','intensidad','varianza_5x5','gradMag',
    #                                       'GLCM_32_9x9_FLAIR_dAPCA:0', 'GLCM_32_9x9_FLAIR_dAPCA:1',
    #                                       'GLSZM_32_9x9_FLAIR_dAPCA:0','GLSZM_32_9x9_FLAIR_dAPCA:1']
    elif tipo_caracteristicas=='A5GL11':
        listaCaracteristicas_utilizada = ['coord_i','coord_j','intensidad','gradMag',
                                          'GLCM_32_9x9_FLAIR_dAPCA:0', 'GLCM_32_9x9_FLAIR_dAPCA:1',
                                          'GLSZM_32_9x9_FLAIR_dAPCA:0','GLSZM_32_9x9_FLAIR_dAPCA:1']
    
    
    elif tipo_caracteristicas=='A5GL12':
        listaCaracteristicas_utilizada = ['coord_i','coord_j','intensidad','varianza','gradMag',
                                          'GLCM_32_9x9_FLAIR_dAPCA:0','GLCM_32_9x9_FLAIR_dAPCA:1', 'GLCM_32_9x9_FLAIR_dAPCA:2',
                                          'GLSZM_32_9x9_FLAIR_dAPCA:0','GLSZM_32_9x9_FLAIR_dAPCA:1','GLSZM_32_9x9_FLAIR_dAPCA:2']
    
    
    # elif tipo_caracteristicas=='A5GL3':
    #     listaCaracteristicas_utilizada = ['coord_i','coord_j','intensidad','varianza','gradMag',
    #                                       'GLSZM_32_9x9_FLAIR_dAPCA:0']
    # elif tipo_caracteristicas=='A5GL4':
    #     listaCaracteristicas_utilizada = ['coord_i','coord_j','intensidad','varianza','gradMag',
    #                                       'GLCM_32_9x9_FLAIR_dAPCA:0',
    #                                       'GLRLM_32_9x9_FLAIR_dAPCA:0']
    
    elif tipo_caracteristicas=='A5GL13':
        listaCaracteristicas_utilizada = ['coord_i','coord_j','intensidad','varianza','gradMag',
                                          'GLCM_32_9x9_FLAIR_dAPCA:0',
                                          'GLRLM_32_9x9_FLAIR_dAPCA:0',
                                          'GLSZM_32_9x9_FLAIR_dAPCA:0']
    elif tipo_caracteristicas=='A5GL14':
        listaCaracteristicas_utilizada = ['coord_i','coord_j','intensidad','varianza','gradMag',
                                          'GLCM_32_9x9_FLAIR_dAPCA:0','GLCM_32_9x9_FLAIR_dAPCA:1',
                                          'GLRLM_32_9x9_FLAIR_dAPCA:0','GLRLM_32_9x9_FLAIR_dAPCA:1',
                                          'GLSZM_32_9x9_FLAIR_dAPCA:0','GLSZM_32_9x9_FLAIR_dAPCA:1']
    elif tipo_caracteristicas=='A5GL15':
        listaCaracteristicas_utilizada = ['coord_i','coord_j','intensidad','varianza','gradMag',
                                          'GLCM_32_9x9_FLAIR_dAPCA:0','GLCM_32_9x9_FLAIR_dAPCA:1','GLCM_32_9x9_FLAIR_dAPCA:2',
                                          'GLRLM_32_9x9_FLAIR_dAPCA:0','GLRLM_32_9x9_FLAIR_dAPCA:1','GLRLM_32_9x9_FLAIR_dAPCA:2',
                                          'GLSZM_32_9x9_FLAIR_dAPCA:0','GLSZM_32_9x9_FLAIR_dAPCA:1','GLSZM_32_9x9_FLAIR_dAPCA:2']
    
    # elif tipo_caracteristicas=='A5GL13':
    #     listaCaracteristicas_utilizada = ['coord_i','coord_j','intensidad','varianza','gradMag',
    #                                       'GLCM_32_9x9_FLAIR_dAPCA:0','GLCM_32_9x9_FLAIR_dAPCA:1',
    #                                       'GLSZM_32_9x9_FLAIR_dAPCA:0','GLSZM_32_9x9_FLAIR_dAPCA:1']
    return listaCaracteristicas_utilizada

def Get_Xdc(dc,lista_caracteristicas=None):###cambiar
    if lista_caracteristicas!=None:
        C=len(lista_caracteristicas)
    else:
        C = len(dc)
        lista_caracteristicas=list(dc.keys())
    S, H, W = dc[list(dc.keys())[0]].shape
    # print(f'C:{C}, S:{S}, H:{H}, W:{W}')
    X = np.zeros([C, S, H, W], dtype=np.float32)
    # print(f'X: {X.shape}')
    
    for i in range(C):
        caracteristica=lista_caracteristicas[i]
        # caracteristica = list(dc.keys())[i]
        # print(f'caracteristica: {caracteristica}, {dc[caracteristica].shape, dc[caracteristica].dtype}')
        X[i]=dc[caracteristica]
    return X
def Get_Xdc_sample(N,Xdc):
    np.random.seed(0)
    C,S,H,W=Xdc.shape
    X_sample = np.zeros((N,C),dtype=Xdc.dtype)
    n = 0
    while n < N:
        s = np.random.randint(0,S)
        h = np.random.randint(0,H)
        w = np.random.randint(0,W)
        # print('aer:',s,h,w,Xdc[:,s,h,w])
        if np.isnan(Xdc[:,s,h,w]).sum()==0:
            X_sample[n] = Xdc[:,s,h,w]
            n+=1
    return X_sample

def Print_Resume(array):
    print('min:',round(np.nanmin(array),2),'max:',round(np.nanmax(array),2),'mean:',round(np.nanmean(array),2),
          'q1:',round(np.quantile(array,0.25),2),'median:',round(np.nanmedian(array),2),'q3:',round(np.quantile(array,0.75),2),'varianza:',round(np.nanvar(array),2))
    return None

def Get_Px(paciente, Xdc):
    for i in range(len(Xdc)):
        print(f'\ni:{i}')
        print('min:',np.nanmin(Xdc[i]),'max:',np.nanmax(Xdc[i]),'mean:',np.nanmean(Xdc[i]),
              'median:',np.nanmedian(Xdc[i]),'varianza:',np.nanvar(Xdc[i]))
        m = np.nanmean(Xdc[i])
        s = np.nanstd(Xdc[i])
        Xdc[i] -= m
        Xdc[i] /= s
    # print(f'Xdc:{Xdc.shape}')
    '''Se eliminan características debido a la altísima varianza e14-e30.
    causado por utilizar RoIs pequeñas de 9x9, en lugar de toda la imagen.'''
    # if fold_textura == 'GLRLM_32_9x9_FLAIR':
    #     Xdc = np.delete(Xdc,(5,6,7,8,9,10),0)
    # elif fold_textura == 'GLSZM_32_9x9_FLAIR':
    #     Xdc = np.delete(Xdc,(2,5,7,10,11,12,13),0)
    # elif fold_textura == 'GLCM_32_9x9_FLAIR':
    #     Xdc = np.delete(Xdc,(3,6,9),0)
    C,S,H,W = Xdc.shape
    # print(f'GL-Xdc:{Xdc.shape}')
    # =============================================================================
    # Ajustar PCA: Obtener muestra NxC para ajustar PCA y ajustar
    # =============================================================================
    N=int(2.5e6)
    N=int(1e5)
    X_sample = Get_Xdc_sample(N,Xdc)
    # print(f'X_sample:{X_sample.shape}')
    pca = PCA()
    pca.fit(X_sample)
    # print(f'explained_variance_:\n{pca.explained_variance_}')
    # print(f'explained_variance_ratio_:\n{pca.explained_variance_ratio_}')
    print(f'CUMSUM: explained_variance_ratio_:\n{np.cumsum(pca.explained_variance_ratio_[:6])}')
    # print(f'pca.components_: {pca.components_.shape},\n{pca.components_}')
    Xdc = Xdc.reshape(C,-1); Xdc = Xdc.T
    # print(f'Xdc:{Xdc.shape}')

    indices_nans = np.isnan(Xdc[:,0])
    Xdc=Xdc.copy()
    Xdc[indices_nans]=np.nanmedian(Xdc)
    PXdc = pca.transform(Xdc)
    # PXdc[indices_nans]=np.median(PXdc)
    # PXdc[indices_nans]=np.max(PXdc)*1.0
    PXdc[indices_nans]=np.nan

    PXdc = PXdc.T; PXdc = PXdc.reshape(C,S,H,W)
    # print(f'PXdc:{PXdc.shape}')
    return PXdc

def Get_ArrayCaracteristicas(path_caracteristicas, paciente, lista_caracteristicas):
    canal = 'FLAIR'# Parametro constante. Después se puede generalizar a >1 canal
    fold_A = 'A-3x3_{}'.format(canal)
    dc = dict(np.load(os.path.join(path_caracteristicas,fold_A,paciente+'.npz')))
    # print(f'dc fold_A: {list(dc.keys())}\n')
    C=len(lista_caracteristicas)
    S,H,W = dc[list(dc.keys())[0]].shape
    # print(f'C:{C}, S:{S}, H:{H}, W:{W}')
    # AC = np.zeros((C,S,H,W),dtype=dc[list(dc.keys())[0]].dtype)
    AC = np.zeros((C,S,H,W),dtype=np.float16)
    # print(f'AC:{AC.shape, AC.dtype}\n')
    
    # for feature in lista_caracteristicas:
    dGLCM={'0':'GLCM_ASM_Mean','1':'GLCM_Contrast_Mean','2':'GLCM_Correlation_Mean','3':'GLCM_SumOfSquaresVariance_Mean','4':'GLCM_InverseDifferenceMoment_Mean','5':'GLCM_SumAverage_Mean',
       '6':'GLCM_SumVariance_Mean','7':'GLCM_SumEntropy_Mean','8':'GLCM_Entropy_Mean','9':'GLCM_DifferenceVariance_Mean','10':'GLCM_DifferenceEntropy_Mean','11':'GLCM_Information1_Mean','12':'GLCM_Information2_Mean'}
    for indice in range(len(lista_caracteristicas)):
        feature=lista_caracteristicas[indice]
        # print(f'\nfeature: {feature}')
        if 'GL' in feature:
            lista=feature.split('_')
            fold_textura='_'.join(lista[:-1])
            # print(f'fold_textura: {fold_textura}')
            tipoCaract, i = lista[-1].split(':')
            # print(f'tipoCaract: {tipoCaract}, i:{i}')
            if 'PCA' in tipoCaract:
                fold_textura_PCA = '{}_{}'.format(fold_textura,tipoCaract)
                # print(f'fold_textura_PCA:{fold_textura_PCA}, i:{i}')
                file = '{}_Px{}.npy'.format(paciente,i)
                # print(f'file:{file}')
                Pxi = np.load(os.path.join(path_caracteristicas,fold_textura_PCA,file))
                # print(f'Px{i}:,{Pxi.shape,Pxi.dtype}')
                AC[indice]=Pxi
            elif tipoCaract=='C':# Si decido utilizar características específicas
                file = '{}.npz'.format(paciente)
                GLi = dict(np.load(os.path.join(path_caracteristicas,fold_textura,file)))
                # print(f'GLi:{GLi.keys()}')
                # print(f'i:{i}, {dGLCM[i], GLi[dGLCM[i]].shape, GLi[dGLCM[i]].dtype}')
                AC[indice]=GLi[dGLCM[i]]
        else:#Proveniente de A-3x3...
            AC[indice]=dc[feature]
    return AC
def MDF(G_original, AC, r, ady, ce, par_ce, prototipo, gamma, percentil,dist):
    '''
    input: NxHxW (N=1 o 3, depende de ady)
    output: HxW
    '''
    # print('\nDentro de MDF:')
    G = np.copy(G_original)
    notG = np.logical_not(G)
    # mdf3D = np.zeros([2]+list(G.shape[1:]),dtype=AC.dtype)
    mdf3D = np.zeros([2]+list(G.shape[1:]),dtype=np.float32)
    if ady=='circulo' or ady==0:
        # print("Vecindario circular")
        ady = 0
    # elif ady=='esfera' or ady==None or ady>=r-1:
    #     print("Vecindario esférico")
    #     ady = r-1
    # else:
        # print("Círcular con {} slide de adyacencia!".format(ady))
    coordenadas_ventana, offset = Get_coordenadas_esfera_ady(r, ady)# función general
    slide_nans = np.ones(AC.shape[-2:],dtype=np.float32)*np.nan
    # slide_nans = np.ones((160,192),dtype=np.float32)*np.nan
    if ady==0 and G.shape[0]==1:
        k=0
    elif ady==1 and G.shape[0]==3:
        k=1
    else:
        print('ERROR en parámetro ady')
        return None
    G_slide = np.copy(G[k])
    notG_slide = np.copy(notG[k])
    if G_slide.sum()>0: # slides con lesiones
        # ========================================================================
        # MDF para píxeles fuera de la lesión, calcula distancias al vecindario local más cercano de clase lesión
        # ========================================================================
        boundaries = find_boundaries(G_slide, mode='inner', connectivity=1)
        coordenadas_boundaries = np.argwhere(boundaries)
        # print(f'coordenadas_boundaries:{coordenadas_boundaries.shape}')
        sdf_Gin, ind = SDF(G_slide, return_indices=True)
        dBordes = Get_dBordes(G,k,coordenadas_boundaries,AC,coordenadas_ventana,offset, ce,par_ce)
        
        mascara_dist = sdf_Gin<=dist
        # coordenadas = np.argwhere(mascara_dist)
        # for i,j in coordenadas:#Esto se puede hacer más eficiente
        
        for i,j in np.argwhere((0<sdf_Gin) & (sdf_Gin<=dist)):
            i_ind,j_ind = ind[:,i,j]
            if dBordes[i_ind,j_ind]['esNan']:
                mdf3D[1,i,j]=abs(sdf_Gin[i,j])
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", category=RuntimeWarning)
                    try:
                        mdf3D[1,i,j]=np.sqrt( dMahalanobis(AC[:,k,i,j], dBordes[i_ind,j_ind][prototipo], dBordes[i_ind,j_ind]['precision']))
                    except RuntimeWarning:
                        print('RuntimeWarning, except sqrt 1!!!!!')
                        print(f'mahalanobis:{dMahalanobis(AC[:,k,i,j], dBordes[i_ind,j_ind][prototipo], dBordes[i_ind,j_ind]['precision'])}')
                        mdf3D[1,i,j]=abs(sdf_Gin[i,j])
                    except:
                        print('Otra except!')
        # ========================================================================
        # MDF para píxeles dentro de la lesión, calcula distancia al vecindario local de clase no-lesión más cercano
        # ========================================================================
        boundaries = find_boundaries(notG_slide, mode='inner', connectivity=1)# Son las llaves, si faltan, aumento conectividad a 2
        coordenadas_boundaries = np.argwhere(boundaries)
        # print(f'coordenadas_boundaries:{coordenadas_boundaries.shape}')
        sdf, ind = SDF(notG_slide, return_indices=True)
        dBordes = Get_dBordes(notG,k,coordenadas_boundaries,AC,coordenadas_ventana,offset, ce,par_ce)
        # for i,j in coordenadas:
        
        for i,j in np.argwhere(sdf_Gin<=0):
            i_ind,j_ind = ind[:,i,j]
            if dBordes[i_ind,j_ind]['esNan']:# Se espera que no se ingrese aquí
                mdf3D[0,i,j]=abs(sdf[i,j])
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", category=RuntimeWarning)
                    try:
                        mdf3D[0,i,j]=np.sqrt( dMahalanobis(AC[:,k,i,j], dBordes[i_ind,j_ind][prototipo], dBordes[i_ind,j_ind]['precision']))
                    except RuntimeWarning:
                        print('RuntimeWarning, except sqrt 0!!!!!')
                        print(f'mahalanobis:{dMahalanobis(AC[:,k,i,j], dBordes[i_ind,j_ind][prototipo], dBordes[i_ind,j_ind]['precision'])}')
                        mdf3D[0,i,j]=abs(sdf[i,j])
                    except:
                        print('Otra except!')
    else: # En slides sin lesiones no es posible calcular MDF
        # print('* *'*10,'slide sin lesiones','* *'*10)
        mdf3D = slide_nans
        return mdf3D
    mdf_final = MDF_step2(mdf3D,G_slide,gamma,percentil,mascara_dist)
    return mdf_final
def MDF_step2(mdf,Y,gamma,percentil,mascara_dist):
    mdf0 = mdf[0]
    mdf1 = mdf[1]
    notY=np.logical_not(Y)
    mdf_fore = mdf0*Y
    mdf_back = mdf1*notY
    mdf_union = mdf_back+mdf_fore
    
    # Utilizar transformación gamma y luego lugar de truncar
    maximo = max(mdf_union[mascara_dist])
    minimo = min(mdf_union[mascara_dist])
    if gamma!=1.0:
        for i,j in np.argwhere(mascara_dist):
            mdf_union[i,j] = minimo+(maximo-minimo)*((mdf_union[i,j]-minimo)/(maximo-minimo))**gamma
        # print('Gamma: ModuloA.Print_Resume(mdf_union[mascara_dist]):')
        # Print_Resume(mdf_union[mascara_dist])
        mdf_union[np.logical_not(mascara_dist)]=np.max(mdf_union[mascara_dist])
    cuantil = np.quantile(mdf_union[mascara_dist], percentil)
    mdf_union[np.logical_not(mascara_dist)]=cuantil
    mdf_union[mdf_union>cuantil]=cuantil
    return mdf_union

def Truncar_stack_superior(array_original, p):
    array = np.copy(array_original)
    for k in range(array.shape[0]):
        slide = array[k]
        if np.isnan(slide.sum()):
            continue
        array[k] = Truncar_superior(slide, p)
    return array

def Truncar_superior(array_original, p):
    array = np.copy(array_original)
    cuantil = np.quantile(array, p)
    # print(f'Truncar_superior:{p,cuantil}')
    array[array>cuantil]=cuantil
    return array

def Truncar_superior_nans(array_original, p):
    array = np.copy(array_original)
    array_sNans = array[np.logical_not(np.isnan(array))]
    cuantil = np.quantile(array_sNans, p)
    array[array>cuantil]=cuantil
    return array

def Get_dBordes(G,k,coordenadas_boundaries,AC,coordenadas_ventana,offset, ce,par_ce):
    d = dict()
    for i,j in coordenadas_boundaries[:]:
        coordenadas_RoI = Get_coordenadas_RoI_3D(G,(k,i,j),coordenadas_ventana,offset)
        mc = Matriz_caracteristicas_3D(AC,coordenadas_RoI)
        # print(f'i,j:{i,j}, mcc:{mc.shape}\n{np.around(mc,5)}')
        # print(np.around(mc[:,0],2))
            
        if mc.shape[0]<6: # Ajustar umbral. Si es 1 se cae GL.
            d[(i,j)] = {'prototipo_media':np.nan,'prototipo_mediana':np.nan,'precision':np.nan,'esNan':True}
            continue
        else:
            flag_OAS = False
            for col in range(mc.shape[1]):
                if len(set(mc[:,col])) <= 2:
                    flag_OAS = True
            if flag_OAS:
                cov = OAS().fit(mc)
            elif ce=='GL':
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", category=ConvergenceWarning)
                    try:
                        cov = GraphicalLasso(alpha=par_ce,tol=0.001, max_iter=300).fit(mc)
                        # cov = OAS().fit(mc)###
                    except FloatingPointError:
                        print('FloatingPointError!!')
                        print(f'i,j:{i,j}, mcc:{mc.shape}')
                        d[(i,j)] = {'prototipo_media':np.nan,'prototipo_mediana':np.nan,'precision':np.nan,'esNan':True}
                        continue
                    except ConvergenceWarning:
                        print('ConvergenceWarning!')
                        print(f'i,j:{i,j}, mcc:{mc.shape}')
                        cov = OAS().fit(mc)
                    except ValueError:#Los NaNs
                        print('ValueError')
                        print(f'i:{i}, j:{j}')
                        print(mc)
                        print(np.mean(mc))
                        cov = OAS().fit(mc)
                        
            prototipo_media = np.mean(mc, axis=0)
            prototipo_mediana = np.median(mc, axis=0)
            d[(i,j)] = {'prototipo_media':prototipo_media,'prototipo_mediana':prototipo_mediana,'precision':cov.precision_.astype(np.float32),'esNan':False}
    return d

def Replace_nans(a,b):
    '''array a contiene los nans'''
    c = np.zeros_like(a)
    a_nan = np.isnan(a)
    if a.ndim == 2:
        for i, j in np.argwhere(a_nan):
            c[i,j] = b[i,j]
        for i, j in np.argwhere(np.logical_not(a_nan)):
            c[i,j] = a[i,j]
    elif a.ndim == 3:
        for k, i, j in np.argwhere(a_nan):
            c[k, i,j] = b[k, i,j]
        for k, i, j in np.argwhere(np.logical_not(a_nan)):
            c[k, i,j] = a[k, i,j]
    return c
def Suavizar(array,k_size=(3,3,3)):
    dim = len(k_size)
    mascara, offset, coordenadas = Get_ventana(k_size)
    print(offset,array.shape)
    array_new = np.zeros_like(array)
    if dim ==2:
        offset_axis0,offset_axis1=offset
        array_pad = np.pad(array,pad_width=((offset_axis0,offset_axis0),(offset_axis1,offset_axis1)),mode='edge')
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                i_pad = i+offset_axis0; j_pad = j+offset_axis1
                array_new[i,j] = np.mean(array_pad[i_pad+coordenadas[:,0],j_pad+coordenadas[:,1]])
    elif dim==3:
        offset_axis0,offset_axis1,offset_axis2=offset
        array_pad = np.pad(array,pad_width=((offset_axis0,offset_axis0),(offset_axis1,offset_axis1),(offset_axis2,offset_axis2)),mode='edge')
        for k in range(array.shape[0]):
            for i in range(array.shape[1]):
                for j in range(array.shape[2]):
                    k_pad = k+offset_axis0; i_pad = i+offset_axis1; j_pad = j+offset_axis2
                    array_new[k,i,j] = np.mean(array_pad[k_pad+coordenadas[:,0],i_pad+coordenadas[:,1],j_pad+coordenadas[:,2]])
    return array_new

def Cuantizar(array, niveles=32, lim_inf_zero=True):
    '''Manera simple de cuantizar'''
    if lim_inf_zero:
        return np.around( (array-array.min())/(array.max()-array.min()) * niveles).astype(np.int16)
    return np.around(array/array.max()*niveles).astype(np.int16)

def Get_coordenadas_circulo(r):
    d = 2*r-1
    mascara, offset, coordenadas = Get_ventana((d,d))
    lista = list()
    for i, j in coordenadas:
        if np.linalg.norm(np.array([i,j])) <= r-1:
            lista.append([i,j])
    return np.array(lista), offset
def Get_coordenadas_esfera(r):
    d = 2*r-1
    mascara, offset, coordenadas = Get_ventana((d,d,d))
    lista = list()
    for k, i, j in coordenadas:
#         if np.linalg.norm(np.array([k,i,j])) <= r-1 and -1<=k<=1:
        if np.linalg.norm(np.array([k,i,j])) <= r-1:
            lista.append([k,i,j])
    return np.array(lista), offset

def Get_coordenadas_esfera_ady(r,ady=None):
    if ady==None:
        ady=r-1
    d = 2*r-1
    mascara, offset, coordenadas = Get_ventana((d,d,d))
    lista = list()
    for k, i, j in coordenadas:
        if np.linalg.norm(np.array([k,i,j])) <= r-1 and -ady<=k<=ady:
#         if np.linalg.norm(np.array([k,i,j])) <= r-1:
            lista.append([k,i,j])
    return np.array(lista), offset

def dMahalanobis(vector, prototipo, precision):
    v = vector-prototipo
    return np.matmul(v, np.matmul(precision,v))

def SDF(array_orig, return_indices=False):
    '''
    input: HxW o NxHxW
    output: HxW o NxHxW
    '''
    if array_orig.sum()==0:
        return np.ones_like(array_orig)*np.nan
    array = array_orig.copy()
    boundaries = find_boundaries(array, mode='inner', connectivity=1)
    array[boundaries]=False
    Gin_edt, Gin_ind = ndimage.distance_transform_edt(array, return_indices=True)
    
    not_array = np.logical_not(array_orig)
    Gout_edt, Gout_ind = ndimage.distance_transform_edt(not_array, return_indices=True)
    
    sdf = Gout_edt-Gin_edt
    
    if return_indices:
        ind = np.zeros_like(Gin_ind)
        ind[0] = Gout_ind[0]*not_array + Gin_ind[0]*array_orig
        ind[1] = Gout_ind[1]*not_array + Gin_ind[1]*array_orig
        return sdf.astype(np.float32), ind
    return sdf.astype(np.float32)
def PDF(array_orig, return_indices=False):# Lo mismo que DTM
    '''
    input: HxW o NxHxW
    output: HxW o NxHxW
    '''
    if array_orig.sum()==0:
        return np.ones_like(array_orig)*np.nan
    array = array_orig.copy()
    boundaries = find_boundaries(array, mode='inner', connectivity=1)
    array[boundaries]=False
    Gin_edt, Gin_ind = ndimage.distance_transform_edt(array, return_indices=True)
    
    not_array = np.logical_not(array_orig)
    Gout_edt, Gout_ind = ndimage.distance_transform_edt(not_array, return_indices=True)
    
    pdf = Gout_edt+Gin_edt
    
    if return_indices:
        ind = np.zeros_like(Gin_ind)
        ind[0] = Gout_ind[0]*not_array + Gin_ind[0]*array_orig
        ind[1] = Gout_ind[1]*not_array + Gin_ind[1]*array_orig
        return pdf.astype(np.float32), ind
    return pdf.astype(np.float32)

def DTM(array_orig, return_indices=False):# Lo mismo que PDF
    '''
    input: HxW o NxHxW
    output: HxW o NxHxW
    '''
    if array_orig.sum()==0:
        return np.ones_like(array_orig)*np.nan
    array = array_orig.copy()
    boundaries = find_boundaries(array, mode='inner', connectivity=1)
    array[boundaries]=False
    Gin_edt, Gin_ind = ndimage.distance_transform_edt(array, return_indices=True)
    
    not_array = np.logical_not(array_orig)
    Gout_edt, Gout_ind = ndimage.distance_transform_edt(not_array, return_indices=True)
    
    dtm = Gout_edt+Gin_edt
    
    if return_indices:
        ind = np.zeros_like(Gin_ind)
        ind[0] = Gout_ind[0]*not_array + Gin_ind[0]*array_orig
        ind[1] = Gout_ind[1]*not_array + Gin_ind[1]*array_orig
        return dtm.astype(np.float32), ind
    return dtm.astype(np.float32)

def Matriz_caracteristicas_3D(AC,coordenadas_RoI):
    # 'Obtener matriz de N(píxeles) X NC(caraterísticas), para la máscara(RoI)'
    mc = list()
    for k,i,j in coordenadas_RoI:
        mc.append(AC[:,k,i,j])
    mc = np.array(mc)
    return mc

def mascara_kij(size,coord,coordenadas_ventana,offset):
    '''
    Obtener máscara en array 3D, centrada en coordenadas k,i,j.
    '''
    
    offset_axis_k,offset_axis_i,offset_axis_j = offset
    k,i,j=coord
    ventana_kij = np.zeros(size,dtype=np.bool_)# ventana centrada en coordenadas borde p=(i,j)
#     ventana_kij = np.pad(ventana_kij, pad_width=offset)
    ventana_kij = np.pad(ventana_kij, pad_width=((offset_axis_k,offset_axis_k),(offset_axis_i,offset_axis_i),(offset_axis_j,offset_axis_j)),mode='edge')
    
#     k_pad = k+offset; i_pad = i+offset; j_pad = j+offset
    k_pad = k+offset_axis_k; i_pad = i+offset_axis_i; j_pad = j+offset_axis_j
    
    ventana_kij[k_pad+coordenadas_ventana[:,0],i_pad+coordenadas_ventana[:,1],j_pad+coordenadas_ventana[:,2]]=True
    if offset != 0:
        ventana_kij = ventana_kij[offset_axis_k:-offset_axis_k,offset_axis_i:-offset_axis_i,offset_axis_j:-offset_axis_j]
    return ventana_kij

def Get_coordenadas_RoI_3D(G3D,coord,coordenadas_ventana, offset, return_RoI=False):
#     print(f'Get_coordenadas_RoI_3D:{offset}')
    k,i,j = coord
    ventana_kij = mascara_kij(G3D.shape,(k,i,j),coordenadas_ventana,offset)
    RoI = G3D*ventana_kij
    coordenadas_RoI = np.argwhere(RoI)
    if return_RoI:
        return coordenadas_RoI, RoI
    return coordenadas_RoI

def Get_ventana(size_mascara):
    offset_axis0 = size_mascara[0]//2
    offset_axis1 = size_mascara[1]//2
    dim = len(size_mascara)
    largo = np.prod(size_mascara)
    coordenadas = np.zeros((largo, dim),dtype=np.int16)
    indice = 0
    if dim==2:
        offset = (offset_axis0,offset_axis1)
        mascara = np.mgrid[0:size_mascara[0], 0:size_mascara[1]]#-offset
        mascara[0] -= offset_axis0
        mascara[1] -= offset_axis1
        for i in range(mascara.shape[1]):# fila
            for j in range(mascara.shape[2]):# columna
                coordenadas[indice] = mascara[:,i,j]
                indice += 1
    elif dim==3:
        offset_axis2 = size_mascara[2]//2
        offset = (offset_axis0,offset_axis1,offset_axis2)
        mascara = np.mgrid[0:size_mascara[0], 0:size_mascara[1], 0:size_mascara[2]]#-offset
        mascara[0] -= offset_axis0
        mascara[1] -= offset_axis1
        mascara[2] -= offset_axis2
        for k in range(mascara.shape[1]):# 1er dimensión corresponde a número de slide
            for i in range(mascara.shape[2]):# fila
                for j in range(mascara.shape[3]):# columna
                    coordenadas[indice] = mascara[:,k,i,j]
                    indice += 1
    return mascara, offset, coordenadas

def dCaracteristicas_tipo(X3D, Y3D, tipo_caracteristicas=None, size_ventana_A=None, size_ventana_hf=None, niveles_hf=None):
    '''
    Asumiendo imagen 3D de un canal y target.
    tipo_caracteristicas=='A': características de bajo-mediano costo computacional.
    tipo_caracteristicas=='B': características de Haralick de alto costo computacional. Require size_ventana_hf y Xhf.
    Retorna diccionario con características para cada vóxel
    '''
    print(f'tipo_caracteristicas:{tipo_caracteristicas}, size_ventana_A:{size_ventana_A}, size_ventana_hf:{size_ventana_hf}, niveles_hf:{niveles_hf}')
    if tipo_caracteristicas==None:
        print('Error')
        return None
    elif (tipo_caracteristicas=='A' and size_ventana_A==None) or (tipo_caracteristicas=='A' and (size_ventana_hf!=None or niveles_hf!=None)):
        print('Error')
        return None
    elif 'HF' in tipo_caracteristicas or 'GLRLM' in tipo_caracteristicas or 'GLSZM' in tipo_caracteristicas:
        if (size_ventana_hf==None or niveles_hf==None) or size_ventana_A!=None:
            print('Error')
            return None
    d = {}
    float_tipo = 'float16'
    # float_tipo = 'float32'
    if tipo_caracteristicas=='GLCM' or tipo_caracteristicas=='GLRLM' or tipo_caracteristicas=='GLSZM':
        Xhf = Cuantizar(np.copy(X3D), niveles_hf)
        dim = len(size_ventana_hf)
        if dim==2:
            mascara_hf, offset_hf, coordenadas_hf = Get_ventana(size_ventana_hf)
            offset_hf_axis_i, offset_hf_axis_j = offset_hf
            offset_hf_axis_k = 0
            Xhf_pad = np.pad(Xhf, pad_width=((offset_hf_axis_k,offset_hf_axis_k),(offset_hf_axis_i,offset_hf_axis_i),(offset_hf_axis_j,offset_hf_axis_j)),mode='edge')
        elif dim==3:
            mascara_hf, offset_hf, coordenadas_hf = Get_ventana(size_ventana_hf)
            offset_hf_axis_k, offset_hf_axis_i, offset_hf_axis_j = offset_hf
            Xhf_pad = np.pad(Xhf, pad_width=((offset_hf_axis_k,offset_hf_axis_k),(offset_hf_axis_i,offset_hf_axis_i),(offset_hf_axis_j,offset_hf_axis_j)),mode='edge')
    if tipo_caracteristicas=='A':
        coord_k = np.zeros(X3D.shape,dtype=np.uint8)
        coord_i = np.zeros(X3D.shape,dtype=np.uint8)
        coord_j = np.zeros(X3D.shape,dtype=np.uint8)
        d['intensidad'] = np.copy(X3D).astype(float_tipo)
        sdf3D = SDF(Y3D)
        sdf = np.zeros(Y3D.shape,dtype=float_tipo)
        intensidad_media = np.zeros(X3D.shape,dtype=float_tipo)
        var = np.zeros(X3D.shape,dtype=float_tipo)
        var_5x5 = np.zeros(X3D.shape,dtype=float_tipo)
        gradh = np.zeros(X3D.shape,dtype=float_tipo)
        gradv = np.zeros(X3D.shape,dtype=float_tipo)
        laplacian = np.zeros(X3D.shape,dtype=float_tipo)
        k_h = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
        k_v = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        k_lpc = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    # elif tipo_caracteristicas=='B':
    #     coord_k = np.zeros(X3D.shape,dtype=np.uint8)
    #     coord_i = np.zeros(X3D.shape,dtype=np.uint8)
    #     coord_j = np.zeros(X3D.shape,dtype=np.uint8)
    #     d['intensidad'] = np.copy(X3D)
    #     sdf3D = SDF(Y3D)
    #     sdf = np.zeros(Y3D.shape,dtype=np.float32)
    #     intensidad_media = np.zeros(X3D.shape,dtype=np.float32)
    #     var = np.zeros(X3D.shape,dtype=np.float32)
    #     var_5x5 = np.zeros(X3D.shape,dtype=np.float32)
    #     gradh = np.zeros(X3D.shape,dtype=np.float32)
    #     gradv = np.zeros(X3D.shape,dtype=np.float32)
    #     laplacian = np.zeros(X3D.shape,dtype=np.float32)
    #     k_h = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    #     k_v = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    #     k_lpc = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    elif tipo_caracteristicas=='GLCM':
        GLCM_ASM_Mean = np.zeros(X3D.shape, dtype=float_tipo)# (0) or Energy
        GLCM_Contrast_Mean = np.zeros(X3D.shape, dtype=float_tipo)# (1)
        GLCM_Correlation_Mean = np.zeros(X3D.shape, dtype=float_tipo)# (2)
        GLCM_SumOfSquaresVariance_Mean = np.zeros(X3D.shape, dtype=float_tipo)# (3)
        GLCM_InverseDifferenceMoment_Mean = np.zeros(X3D.shape, dtype=float_tipo)# (4)
        GLCM_SumAverage_Mean = np.zeros(X3D.shape, dtype=float_tipo)# (5)
        GLCM_SumVariance_Mean = np.zeros(X3D.shape, dtype=float_tipo)# (6)
        GLCM_SumEntropy_Mean = np.zeros(X3D.shape, dtype=float_tipo)# (7)
        GLCM_Entropy_Mean = np.zeros(X3D.shape, dtype=float_tipo)# (8)
        GLCM_DifferenceVariance_Mean = np.zeros(X3D.shape, dtype=float_tipo)# (9)
        GLCM_DifferenceEntropy_Mean = np.zeros(X3D.shape, dtype=float_tipo)# (10)
        GLCM_Information1_Mean = np.zeros(X3D.shape, dtype=float_tipo)# (11)
        GLCM_Information2_Mean = np.zeros(X3D.shape, dtype=float_tipo)# (12)
        # GLCM_MaximalCorrelationCoefficient_Mean = np.zeros(X3D.shape, dtype=np.float32)# (13)# arroja error
    elif tipo_caracteristicas=='GLRLM':
        GLRLM_ShortRunEmphasis = np.zeros(X3D.shape, dtype=float_tipo)# (0)
        GLRLM_LongRunEmphasis = np.zeros(X3D.shape, dtype=float_tipo)# (1)
        GLRLM_GrayLevelNo_Uniformity = np.zeros(X3D.shape, dtype=float_tipo)# (2)
        GLRLM_RunLengthNonUniformity = np.zeros(X3D.shape, dtype=float_tipo)# (3)
        GLRLM_RunPercentage = np.zeros(X3D.shape, dtype=float_tipo)# (4)
        GLRLM_LowGrayLevelRunEmphasis = np.zeros(X3D.shape, dtype=float_tipo)# (5)
        GLRLM_HighGrayLevelRunEmphasis = np.zeros(X3D.shape, dtype=float_tipo)# (6)
        GLRLM_Short_owGrayLevelEmphasis = np.zeros(X3D.shape, dtype=float_tipo)# (7)
        GLRLM_ShortRunHighGrayLevelEmphasis = np.zeros(X3D.shape, dtype=float_tipo)# (8)
        GLRLM_LongRunLowGrayLevelEmphasis = np.zeros(X3D.shape, dtype=float_tipo)# (9)
        GLRLM_LongRunHighGrayLevelEmphasis = np.zeros(X3D.shape, dtype=float_tipo)# (10)
    elif tipo_caracteristicas=='GLSZM':
        GLSZM_SmallZoneEmphasis = np.zeros(X3D.shape, dtype=float_tipo)# (0)
        GLSZM_LargeZoneEmphasis = np.zeros(X3D.shape, dtype=float_tipo)# (1)
        GLSZM_GrayLevelNonuniformity = np.zeros(X3D.shape, dtype=float_tipo)# (2)
        GLSZM_ZoneSizeNonuniformity = np.zeros(X3D.shape, dtype=float_tipo)# (3)
        GLSZM_ZonePercentage = np.zeros(X3D.shape, dtype=float_tipo)# (4)
        GLSZM_LowGrayLeveLZoneEmphasis = np.zeros(X3D.shape, dtype=float_tipo)# (5)
        GLSZM_HighGrayLevelZoneEmphasis = np.zeros(X3D.shape, dtype=float_tipo)# (6)
        GLSZM_SmallZoneLowGrayLevelEmphasis = np.zeros(X3D.shape, dtype=float_tipo)# (7)
        GLSZM_SmallZoneHighGrayLevelEmphasis = np.zeros(X3D.shape, dtype=float_tipo)# (8)
        GLSZM_LargeZoneLowGrayLevelEmphassis = np.zeros(X3D.shape, dtype=float_tipo)# (9)
        GLSZM_LargeZoneHighGrayLevelEmphasis = np.zeros(X3D.shape, dtype=float_tipo)# (10)
        GLSZM_GrayLevelVariance = np.zeros(X3D.shape, dtype=float_tipo)# (11)
        GLSZM_ZoneSizeVariance = np.zeros(X3D.shape, dtype=float_tipo)# (12)
        GLSZM_ZoneSizeEntropy = np.zeros(X3D.shape, dtype=float_tipo)# (13)
    for k in range(X3D.shape[0]):
        print(f'Procesando slide: {k}...')
        if tipo_caracteristicas=='A':
            mascara2D, offset2D, coordenadas2D = Get_ventana(size_ventana_A)
            offset2D_axis_i, offset2D_axis_j = offset2D
            mascara2D_5x5, offset2D_5x5, coordenadas2D_5x5 = Get_ventana((5,5))
            offset2D_axis_i_5x5, offset2D_axis_j_5x5 = offset2D_5x5
            X2D = X3D[k]
            Y2D = Y3D[k]
            X2D_pad = np.pad(X2D,pad_width=((offset2D_axis_i,offset2D_axis_i),(offset2D_axis_j,offset2D_axis_j)),mode='edge')
            X2D_pad_5x5 = np.pad(X2D,pad_width=((offset2D_axis_i_5x5,offset2D_axis_i_5x5),(offset2D_axis_j_5x5,offset2D_axis_j_5x5)),mode='edge')
            sdf[k] = SDF(Y2D)
        for i in range(X3D.shape[1]):
            for j in range(X3D.shape[2]):
                if tipo_caracteristicas=='GLCM' or tipo_caracteristicas=='GLRLM' or tipo_caracteristicas=='GLSZM':
                    k_pad_hf = k+offset_hf_axis_k; i_pad_hf = i+offset_hf_axis_i; j_pad_hf = j+offset_hf_axis_j
                    if dim==2:
                        intensidades_hf = Xhf_pad[k_pad_hf,i_pad_hf+mascara_hf[0],j_pad_hf+mascara_hf[1]]
                    elif dim==3:# si solo se usa slide adyacentes, limitar arreglo 3D intensidades_hf
                        intensidades_hf = Xhf_pad[k_pad_hf+mascara_hf[0],i_pad_hf+mascara_hf[1],j_pad_hf+mascara_hf[2]]
                if tipo_caracteristicas=='A':
                    coord_k[k,i,j]=k
                    coord_i[k,i,j]=i
                    coord_j[k,i,j]=j
                    i_pad = i+offset2D_axis_i; j_pad = j+offset2D_axis_j
                    intensidades2D = X2D_pad[i_pad+mascara2D[0],j_pad+mascara2D[1]]
                    i_pad_5x5 = i+offset2D_axis_i_5x5; j_pad_5x5 = j+offset2D_axis_j_5x5
                    intensidades2D_5x5 = X2D_pad_5x5[i_pad_5x5+mascara2D_5x5[0],j_pad_5x5+mascara2D_5x5[1]]
                    intensidad_media[k,i,j] = np.mean(intensidades2D)
                    var[k,i,j] = np.var(intensidades2D)
                    var_5x5[k,i,j] = np.var(intensidades2D_5x5)
                    gradh[k,i,j] = np.sum(intensidades2D*k_h)/4
                    gradv[k,i,j] = np.sum(intensidades2D*k_v)/4
                    laplacian[k,i,j] = np.sum(intensidades2D*k_lpc)
                elif tipo_caracteristicas=='GLCM':
                    features = mahotas.features.haralick(intensidades_hf,return_mean=True)
                    # print(f'GLCM: features: {features}')
                    GLCM_ASM_Mean[k,i,j] = features[0]
                    GLCM_Contrast_Mean[k,i,j] = features[1]
                    GLCM_Correlation_Mean[k,i,j] = features[2]
                    GLCM_SumOfSquaresVariance_Mean[k,i,j] = features[3]
                    GLCM_InverseDifferenceMoment_Mean[k,i,j] = features[4]
                    GLCM_SumAverage_Mean[k,i,j] = features[5]
                    GLCM_SumVariance_Mean[k,i,j] = features[6]
                    GLCM_SumEntropy_Mean[k,i,j] = features[7]
                    GLCM_Entropy_Mean[k,i,j] = features[8]
                    GLCM_DifferenceVariance_Mean[k,i,j] = features[9]
                    GLCM_DifferenceEntropy_Mean[k,i,j] = features[10]
                    GLCM_Information1_Mean[k,i,j] = features[11]
                    GLCM_Information2_Mean[k,i,j] = features[12]
                    # GLCM_MaximalCorrelationCoefficient_Mean[k,i,j] = features[13]
                elif tipo_caracteristicas=='GLRLM':
                    features, labels = pyfeats.glrlm_features(intensidades_hf, mask=None, Ng=64)
                    # if i==60 and j==60:
                    #     print(i,j)
                    #     print(f'GLRLM: features: {features}')
                    #     return None
                    # print(f'GLRLM: features: {features}')
                    GLRLM_ShortRunEmphasis[k,i,j] = features[0]
                    GLRLM_LongRunEmphasis[k,i,j] = features[1]
                    GLRLM_GrayLevelNo_Uniformity[k,i,j] = features[2]
                    GLRLM_RunLengthNonUniformity[k,i,j] = features[3]
                    GLRLM_RunPercentage[k,i,j] = features[4]
                    GLRLM_LowGrayLevelRunEmphasis[k,i,j] = features[5]
                    GLRLM_HighGrayLevelRunEmphasis[k,i,j] = features[6]
                    GLRLM_Short_owGrayLevelEmphasis[k,i,j] = features[7]
                    GLRLM_ShortRunHighGrayLevelEmphasis[k,i,j] = features[8]
                    GLRLM_LongRunLowGrayLevelEmphasis[k,i,j] = features[9]
                    GLRLM_LongRunHighGrayLevelEmphasis[k,i,j] = features[10]
                elif tipo_caracteristicas=='GLSZM':
                    features, labels = pyfeats.glszm_features(intensidades_hf, mask=None)
                    # print(f'GLSZM: features:{features}')
                    GLSZM_SmallZoneEmphasis[k,i,j] = features[0]
                    GLSZM_LargeZoneEmphasis[k,i,j] = features[1]
                    GLSZM_GrayLevelNonuniformity[k,i,j] = features[2]
                    GLSZM_ZoneSizeNonuniformity[k,i,j] = features[3]
                    GLSZM_ZonePercentage[k,i,j] = features[4]
                    GLSZM_LowGrayLeveLZoneEmphasis[k,i,j] = features[5]
                    GLSZM_HighGrayLevelZoneEmphasis[k,i,j] = features[6]
                    GLSZM_SmallZoneLowGrayLevelEmphasis[k,i,j] = features[7]
                    GLSZM_SmallZoneHighGrayLevelEmphasis[k,i,j] = features[8]
                    GLSZM_LargeZoneLowGrayLevelEmphassis[k,i,j] = features[9]
                    GLSZM_LargeZoneHighGrayLevelEmphasis[k,i,j] = features[10]
                    GLSZM_GrayLevelVariance[k,i,j] = features[11]
                    GLSZM_ZoneSizeVariance[k,i,j] = features[12]
                    GLSZM_ZoneSizeEntropy[k,i,j] = features[13]
    if tipo_caracteristicas=='A':
        gradMag = np.sqrt((gradh)**2/2+(gradv)**2/2)
        gradAng = np.arctan2(gradv, gradh)
        # d['intensidad_media'] = intensidad_media
        d['varianza'] = var
        # d['varianza_5x5'] = var_5x5
        # d['gradh'] = gradh
        # d['gradv'] = gradv
        d['gradMag'] = gradMag
        # d['gradAng'] = gradAng
        # d['laplacian'] = laplacian
        # d['sdf'] = sdf
        # d['sdf3D'] = sdf3D
        # d['coord_k'] = coord_k
        d['coord_i'] = coord_i
        d['coord_j'] = coord_j
    # elif tipo_caracteristicas=='B':
    #     gradMag = np.sqrt((gradh)**2/2+(gradv)**2/2)
    #     gradAng = np.arctan2(gradv, gradh)
    #     # d['intensidad_media'] = intensidad_media
    #     d['varianza'] = var
    #     # d['varianza_5x5'] = var_5x5
    #     # d['gradh'] = gradh
    #     # d['gradv'] = gradv
    #     d['gradMag'] = gradMag
    #     # d['gradAng'] = gradAng
    #     # d['laplacian'] = laplacian
    #     # d['sdf'] = sdf
    #     # d['sdf3D'] = sdf3D
    #     # d['coord_k'] = coord_k
    #     d['coord_i'] = coord_i
    #     d['coord_j'] = coord_j
    elif tipo_caracteristicas=='GLCM':
        d['GLCM_ASM_Mean'] = GLCM_ASM_Mean
        d['GLCM_Contrast_Mean'] = GLCM_Contrast_Mean
        d['GLCM_Correlation_Mean'] = GLCM_Correlation_Mean
        d['GLCM_SumOfSquaresVariance_Mean'] = GLCM_SumOfSquaresVariance_Mean
        d['GLCM_InverseDifferenceMoment_Mean'] = GLCM_InverseDifferenceMoment_Mean
        d['GLCM_SumAverage_Mean'] = GLCM_SumAverage_Mean
        d['GLCM_SumVariance_Mean'] = GLCM_SumVariance_Mean
        d['GLCM_SumEntropy_Mean'] = GLCM_SumEntropy_Mean
        d['GLCM_Entropy_Mean'] = GLCM_Entropy_Mean
        d['GLCM_DifferenceVariance_Mean'] = GLCM_DifferenceVariance_Mean
        d['GLCM_DifferenceEntropy_Mean'] = GLCM_DifferenceEntropy_Mean
        d['GLCM_Information1_Mean'] = GLCM_Information1_Mean
        d['GLCM_Information2_Mean'] = GLCM_Information2_Mean
        # d['GLCM_MaximalCorrelationCoefficient_Mean'] = GLCM_MaximalCorrelationCoefficient_Mean
    elif tipo_caracteristicas=='GLRLM':
        d['GLRLM_ShortRunEmphasis'] = GLRLM_ShortRunEmphasis
        d['GLRLM_LongRunEmphasis'] = GLRLM_LongRunEmphasis
        d['GLRLM_GrayLevelNo_Uniformity'] = GLRLM_GrayLevelNo_Uniformity
        d['GLRLM_RunLengthNonUniformity'] = GLRLM_RunLengthNonUniformity
        d['GLRLM_RunPercentage'] = GLRLM_RunPercentage
        d['GLRLM_LowGrayLevelRunEmphasis'] = GLRLM_LowGrayLevelRunEmphasis
        d['GLRLM_HighGrayLevelRunEmphasis'] = GLRLM_HighGrayLevelRunEmphasis
        d['GLRLM_Short_owGrayLevelEmphasis'] = GLRLM_Short_owGrayLevelEmphasis
        d['GLRLM_ShortRunHighGrayLevelEmphasis'] = GLRLM_ShortRunHighGrayLevelEmphasis
        d['GLRLM_LongRunLowGrayLevelEmphasis'] = GLRLM_LongRunLowGrayLevelEmphasis
        d['GLRLM_LongRunHighGrayLevelEmphasis'] = GLRLM_LongRunHighGrayLevelEmphasis
    elif tipo_caracteristicas=='GLSZM':
        d['GLSZM_SmallZoneEmphasis'] = GLSZM_SmallZoneEmphasis
        d['GLSZM_LargeZoneEmphasis'] = GLSZM_LargeZoneEmphasis
        d['GLSZM_GrayLevelNonuniformity'] = GLSZM_GrayLevelNonuniformity
        d['GLSZM_ZoneSizeNonuniformity'] = GLSZM_ZoneSizeNonuniformity
        d['GLSZM_ZonePercentage'] = GLSZM_ZonePercentage
        d['GLSZM_LowGrayLeveLZoneEmphasis'] = GLSZM_LowGrayLeveLZoneEmphasis
        d['GLSZM_HighGrayLevelZoneEmphasis'] = GLSZM_HighGrayLevelZoneEmphasis
        d['GLSZM_SmallZoneLowGrayLevelEmphasis'] = GLSZM_SmallZoneLowGrayLevelEmphasis
        d['GLSZM_SmallZoneHighGrayLevelEmphasis'] = GLSZM_SmallZoneHighGrayLevelEmphasis
        d['GLSZM_LargeZoneLowGrayLevelEmphassis'] = GLSZM_LargeZoneLowGrayLevelEmphassis
        d['GLSZM_LargeZoneHighGrayLevelEmphasis'] = GLSZM_LargeZoneHighGrayLevelEmphasis
        d['GLSZM_GrayLevelVariance'] = GLSZM_GrayLevelVariance
        d['GLSZM_ZoneSizeVariance'] = GLSZM_ZoneSizeVariance
        d['GLSZM_ZoneSizeEntropy'] = GLSZM_ZoneSizeEntropy
    return d