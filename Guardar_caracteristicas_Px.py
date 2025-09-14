#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 08:59:55 2024

@author: gustavo
"""

import matplotlib.pyplot as plt
import numpy as np, os, Biblioteca_General as bbg
import ModuloA
from time import time

def Get_delete(tipoGL,GL_del):
    if tipoGL=='GLCM':
        if GL_del=='A':
            delete = (3,6,7,9,10,11,12)
    elif tipoGL=='GLRLM':
        if GL_del=='A':
            delete = (5,6,7,8,9,10)
    elif tipoGL=='GLSZM':
        if GL_del=='A':
            delete = (1,2,5,6,7,10,11,12,13)
    return delete

# =============================================================================
# Guardar DS
# =============================================================================
print('Guardar Caracteristicas:')

float_tipo = 'float16'
# float_tipo = 'float32'
# DS = 'MSSEG2016'
# DS = 'ISBI2015'
# DS = 'WMH2017'
# for DS in ['MSSEG2016']:
# for DS in ['ISBI2015']:
for DS in ['WMH2017']:
# for DS in ['MSSEG2016','ISBI2015']:
# for DS in ['MSSEG2016','ISBI2015','WMH2017']:

    if os.uname()[1]=='f15':
        path_base = '/media/gustavo/Disco_2/DS_Imagenes/{DS}/{DS}_2D_160x192_TN/'.format(DS=DS)
        path_caracteristicas = '/media/gustavo/Disco_2/DS_Imagenes/{DS}/{DS}_160x160x192_caracteristicas/'.format(DS=DS)
        # path_MDFs = '/media/gustavo/Disco_2/DS_Imagenes/{DS}/{DS}_160x160x192_MDFs/'.format(DS=DS)
    elif os.uname()[1]=='postgrado01':
        path_base = '/home/gustavo/DS_Imagenes/{DS}/{DS}_2D_160x192_TN/'.format(DS=DS)
        path_caracteristicas = '/home/gustavo/DS_Imagenes/{DS}/{DS}_160x160x192_caracteristicas/'.format(DS=DS)
        # path_MDFs = '/home/gustavo/DS_Imagenes/{DS}/{DS}_160x160x192_MDFs/'.format(DS=DS)
    
    # path_base = '/home/gustavo/DS_Imagenes/{DS}/{DS}_2D_160x192_TN/'.format(DS=DS)
    # path_caracteristicas = '/home/gustavo/DS_Imagenes/{DS}/{DS}_160x160x192_caracteristicas/'.format(DS=DS)
    prefijo ='{DS}_160x160x192_'.format(DS=DS)
    print(path_base)
    pacientes = getattr(bbg,'pacientes_'+DS)
    print(f'pacientes:{len(pacientes)} {pacientes}')
    size_inputs = getattr(bbg,DS+'_input_dim_3D')
    print(f'size_inputs:{size_inputs}')
    size_X = (size_inputs[3],size_inputs[0],size_inputs[1],size_inputs[2])
    size_Y = (size_inputs[3],size_inputs[1],size_inputs[2])
    print(f'size_X:{size_X}, size_Y:{size_Y}')
    # plantilla_A_nombre = '{}_{}_{}_{}'
    # ========================================================================
    # canal = 'FLAIR'
    # size = '9x9'
    # tipoGL = 'GLCM'
    # tipoGL = 'GLRLM'
    # tipoGL = 'GLSZM'
    # GLCM_del=GLRLM_del=GLSZM_del=None
    # GL_del = 'A'
    
    # Pout1=('GLCM', '11x11','FLAIR','A')
    # Pout2=('GLRLM','11x11','FLAIR','A')
    # Pout3=('GLSZM','11x11','FLAIR','A')
    Pout1=('GLCM','9x9','FLAIR','A')
    Pout2=('GLRLM','9x9','FLAIR','A')
    Pout3=('GLSZM','9x9','FLAIR','A')
    # Pout1=('GLCM','7x7','FLAIR','A')
    # Pout2=('GLRLM','7x7','FLAIR','A')
    # Pout3=('GLSZM','7x7','FLAIR','A')
    for Pout in [Pout1,Pout2,Pout3][:]:
        tipoGL,size,canal,GL_del=Pout
        print(f'tipoGL:{tipoGL}, size:{size}, canal:{canal}, GL_del:{GL_del}')
        delete=Get_delete(tipoGL,GL_del)
        print(f'delete:{delete}')
        fold_textura = '{}_32_{}_{}'.format(tipoGL,size,canal)
        print(f'fold_textura:{fold_textura}')
        carpeta_guardar = '{}_d{}PCA'.format(fold_textura,GL_del)
        print(f'carpeta_guardar:{carpeta_guardar}')
        if carpeta_guardar not in os.listdir(path_caracteristicas):
            os.mkdir(os.path.join(path_caracteristicas,carpeta_guardar))
        # =============================================================================
        for paciente in pacientes[:1]:#CPU0
        # for paciente in pacientes[4:8]:#CPU1
        # for paciente in pacientes[8:12]:#CPU2
        # for paciente in pacientes[12:16]:#CPU3
        # for paciente in pacientes[16:]:#CPU4
            print(f'paciente: {paciente}')
            paciente='11_id11'
            print(f'paciente: {paciente}')
            dcGL = dict(np.load(os.path.join(path_caracteristicas,fold_textura,paciente+'.npz')))
            Xdc = ModuloA.Get_Xdc(dcGL)
            print(f'Xdc:{Xdc.shape}')
            
            MRIs,Y = ModuloA.Get_XY_slides(paciente,DS,path_base)# retorna X.shape=(C,A,H,W). C:Channels, A:Axial, H:High, W:Wide
            print(f'MRIs:{MRIs.shape}{MRIs.dtype}, Y:{Y.shape}{Y.dtype}')
            # slide=95
            # plt.figure(figsize=(33,25))
            # plt.subplot(4,4,1);plt.imshow(MRIs[2,slide],cmap='gray');plt.axis('off')
            # plt.subplot(4,4,2);plt.imshow(Y[slide],cmap='gray');plt.axis('off')
            # for i in range(len(Xdc)):
            #     plt.subplot(4,4,2+i+1);plt.imshow(Xdc[i,slide],cmap='gray');plt.axis('off')
            #     plt.title(f'i:{i}, {paciente}, slide:{slide}',fontsize=20)
            
            Xdc = np.delete(Xdc,delete,0)
            Px = ModuloA.Get_Px(paciente, Xdc)
            print(f'Px:{Px.shape, Px.dtype}')
            # plt.figure(figsize=(33,25))
            # plt.subplot(4,4,1);plt.imshow(MRIs[2,slide],cmap='gray');plt.axis('off')
            # plt.subplot(4,4,2);plt.imshow(Y[slide],cmap='gray');plt.axis('off')
            # for i in range(len(Px)):
            #     plt.subplot(4,4,2+i+1);plt.imshow(Px[i,slide],cmap='gray');plt.axis('off')
            #     plt.title(f'i:{i}, {paciente}, slide:{slide}',fontsize=20)
            
            for i in range(len(Px)):
                if i<2:
                    nombre_Pxi = '{}_Px{}'.format(paciente,i)
                    print(f'nombre_Pxi: {nombre_Pxi}')
                    np.save(os.path.join(path_caracteristicas,carpeta_guardar,nombre_Pxi),Px[i].astype(float_tipo))
