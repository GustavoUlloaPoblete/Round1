#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 08:59:55 2024

@author: gustavo
"""

# import matplotlib.pyplot as plt
import numpy as np, os, Biblioteca_General as bbg
import ModuloA
from time import time

# =============================================================================
# Guardar DS
# =============================================================================
print('Guardar Caracteristicas:')

# DS = 'MSSEG2016'
# DS = 'ISBI2015'
# DS = 'WMH2017'

# for DS in ['MSSEG2016']:
# for DS in ['ISBI2015']:
for DS in ['WMH2017']:
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
    plantilla_A_nombre = '{}_{}_{}_{}'
    plantilla_HF_nombre = '{}_{}_{}_{}_{}'
    canal = 'FLAIR'
    # canal = 'T1'
    # canal = 'T2'
    # =============================================================================
    
    size_ventana_A=None;niveles_hf=None;size_ventana_hf=None
    # tipo_caracteristicas='A'; size_ventana_A = (3,3)
    # tipo_caracteristicas='B'
    for tipo_caracteristicas,size_ventana_A,niveles_hf,size_ventana_hf in [('A',(3,3),None,None)]:
    
    # tipo_caracteristicas='GLCM'; niveles_hf=32;size_ventana_hf=(7,7)
    # tipo_caracteristicas='GLRLM'; niveles_hf=32;size_ventana_hf=(7,7)
    # tipo_caracteristicas='GLSZM'; niveles_hf=32;size_ventana_hf=(7,7)
    
    # tipo_caracteristicas,niveles_hf,size_ventana_hf=['GLCM',32,(7,7)]
    # tipo_caracteristicas,niveles_hf,size_ventana_hf=['GLRLM',32,(7,7)]
    # tipo_caracteristicas,niveles_hf,size_ventana_hf=['GLSZM',32,(7,7)]
    
    
    # for tipo_caracteristicas,niveles_hf,size_ventana_hf in [('GLRLM',32,(7,7))]:
    # for tipo_caracteristicas,niveles_hf,size_ventana_hf in [('GLSZM',32,(9,9))]:
    # for tipo_caracteristicas,niveles_hf,size_ventana_hf in [('GLCM',32,(7,7)),('GLRLM',32,(7,7)),('GLSZM',32,(7,7)),
                                                            # ('GLCM',32,(9,9)),('GLRLM',32,(9,9)),('GLSZM',32,(9,9)),
                                                            # ('GLCM',32,(11,11)),('GLRLM',32,(11,11)),('GLSZM',32,(11,11))]:
        print(tipo_caracteristicas,niveles_hf,size_ventana_hf)
        # break
    
        # Para MSSEG2016
        # for paciente in pacientes[:1]:
        # for paciente in pacientes[:3]:
        # for paciente in pacientes[3:6]:
        # for paciente in pacientes[6:9]:
        # for paciente in pacientes[9:12]:
        # for paciente in pacientes[12:]:
            
        # Para ISBI2015
        for paciente in pacientes[:1]:
        # for paciente in pacientes[5:10]:
        # for paciente in pacientes[10:15]:
        # for paciente in pacientes[15:20]:
        # for paciente in pacientes[20:25]:
        # for paciente in pacientes[25:30]:
        # for paciente in pacientes[30:35]:
        # for paciente in pacientes[35:40]:
        # for paciente in pacientes[40:45]:
        # for paciente in pacientes[45:50]:
        # for paciente in pacientes[50:55]:
        # for paciente in pacientes[55:60]:
            
        # for paciente in pacientes[0:10]:
        # for paciente in pacientes[10:20]:
        # for paciente in pacientes[20:30]:
        # for paciente in pacientes[30:40]:
        # for paciente in pacientes[40:50]:
        # for paciente in pacientes[50:60]:
        # break
    
        # for paciente in pacientes[23:24]:#CPU0
        # for paciente in pacientes[41:42]:#CPU0
        # for paciente in pacientes[0:2]:#CPU0
        # for paciente in pacientes[2:4]:#CPU1
        # for paciente in pacientes[4:6]:#CPU2
        # for paciente in pacientes[6:8]:#CPU3
        # for paciente in pacientes[8:10]:#CPU4
        # for paciente in pacientes[10:12]:#CPU5
        # for paciente in pacientes[12:14]:#CPU6
        # for paciente in pacientes[14:16]:#CPU7
        # for paciente in pacientes[16:18]:#CPU8
        # for paciente in pacientes[18:20]:#CPU9
        # for paciente in pacientes[20:]:#CPU10
            print('\n',paciente)
            paciente='11_id11'
            print('\n',paciente)
            # break
            X_canales_slides = np.zeros(size_X,dtype=np.float32)
            Y = np.zeros(size_Y,dtype=np.bool_)
            for slide in range(Y.shape[0]):
                X_name = '{PACIENTE}_{SLIDE}.npy'.format(PACIENTE=paciente,SLIDE=slide)
                Y_name = '{PACIENTE}_{SLIDE}_target.npy'.format(PACIENTE=paciente,SLIDE=slide)
                X_canales_slides[slide] = np.load(os.path.join(path_base,X_name))
                Y[slide] = np.load(os.path.join(path_base,Y_name))
            print(f'X_canales_slides:{X_canales_slides.shape}{X_canales_slides.dtype}, Y:{Y.shape}{Y.dtype}')
            X_canales_slides = np.transpose(X_canales_slides,(1,0,2,3))
            print(f'X_canales_slides:{X_canales_slides.shape}{X_canales_slides.dtype}, Y:{Y.shape}{Y.dtype}')

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
            print(f'X:{X.shape} {X.dtype}, Y:{Y.shape} {Y.dtype}')
            
            t_start = time()
            # X=X[81:82]; Y=Y[81:82]#107,112
            if tipo_caracteristicas=='A' or tipo_caracteristicas=='B':
                dc = ModuloA.dCaracteristicas_tipo(X, Y, tipo_caracteristicas=tipo_caracteristicas, size_ventana_A=size_ventana_A)
                sufijo = '{}-{}_{}'.format(tipo_caracteristicas,'x'.join(list(map(str,size_ventana_A))),canal)
            elif 'GLCM'==tipo_caracteristicas or 'GLRLM'==tipo_caracteristicas or 'GLSZM'==tipo_caracteristicas:
                dc = ModuloA.dCaracteristicas_tipo(X, Y, tipo_caracteristicas=tipo_caracteristicas, size_ventana_hf=size_ventana_hf, niveles_hf=niveles_hf)
                sufijo = '{}_{}_{}_{}'.format(tipo_caracteristicas,niveles_hf,'x'.join(list(map(str,size_ventana_hf))),canal)
            print(round(time()-t_start,2),'[sg]')
            
            carpeta_guardar = sufijo
            print(f'carpeta_guardar:{carpeta_guardar}')
            if carpeta_guardar not in os.listdir(path_caracteristicas):
                os.mkdir(os.path.join(path_caracteristicas,carpeta_guardar))
            
            nombre_file = paciente
            print(os.path.join(path_caracteristicas,carpeta_guardar,nombre_file))
            np.savez(os.path.join(path_caracteristicas,carpeta_guardar,nombre_file),**dc)
        # =============================================================================
    
    # print(dc)
