#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 20:58:33 2024

@author: gustavo
"""

import matplotlib.pyplot as plt
import numpy as np, os, Biblioteca_General as bbg
import ModuloA
from time import time
from skimage import measure

# =============================================================================
# Guardar MDFs
# =============================================================================
# for DS in ['MSSEG2016','ISBI2015']:
for DS in ['MSSEG2016']:
# for DS in ['ISBI2015']:
# for DS in ['WMH2017']:
    if os.uname()[1]=='f15':
        path_base = '/media/gustavo/Disco_2/DS_Imagenes/{DS}/{DS}_2D_160x192_TN/'.format(DS=DS)
        path_caracteristicas = '/media/gustavo/Disco_2/DS_Imagenes/{DS}/{DS}_160x160x192_caracteristicas/'.format(DS=DS)
        path_MDFs = '/media/gustavo/Disco_2/DS_Imagenes/{DS}/{DS}_160x160x192_MDFs/'.format(DS=DS)
        path_MDFs_2D = '/media/gustavo/Disco_2/DS_Imagenes/{DS}/{DS}_2D_160x192_MDFs/'.format(DS=DS)
    elif os.uname()[1]=='postgrado01':
        path_base = '/home/gustavo/DS_Imagenes/{DS}/{DS}_2D_160x192_TN/'.format(DS=DS)
        path_caracteristicas = '/home/gustavo/DS_Imagenes/{DS}/{DS}_160x160x192_caracteristicas/'.format(DS=DS)
        path_MDFs = '/home/gustavo/DS_Imagenes/{DS}/{DS}_160x160x192_MDFs/'.format(DS=DS)
        path_MDFs_2D = '/home/gustavo/DS_Imagenes/{DS}/{DS}_2D_160x192_MDFs/'.format(DS=DS)
    elif os.uname()[1]=='fondecyt1':
        path_base = '/home/gulloa/DS_Imagenes/{DS}/{DS}_2D_160x192_TN/'.format(DS=DS)
        path_caracteristicas = '/home/gulloa/DS_Imagenes/{DS}/{DS}_160x160x192_caracteristicas/'.format(DS=DS)
        path_MDFs = '/home/gulloa/DS_Imagenes/{DS}/{DS}_160x160x192_MDFs/'.format(DS=DS)
    
    prefijo ='{DS}_160x160x192_'.format(DS=DS)
    print(f'path_base: {path_base}')
    pacientes = getattr(bbg,'pacientes_'+DS)
    print(f'pacientes:{len(pacientes)} {pacientes}')
    
    # =============================================================================
    # Parámetros
    # =============================================================================
    # r, ady, ce, par_ce, prototipo, gamma, percentil, dist, umbral_cc
    d_POUT0={'sc':'A5GL0','canal':'FLAIR','r':5,'ady':1,'ce':'GL','par_ce':0.1,
        'prototipo':'prototipo_media','gamma_MDF':1.0,'percentil':0.9,'dist':10,'umbral_cc':4}
    d_POUT1={'sc':'A5GL1','canal':'FLAIR','r':5,'ady':1,'ce':'GL','par_ce':0.1,
        'prototipo':'prototipo_mediana','gamma_MDF':1.0,'percentil':0.9,'dist':10,'umbral_cc':4}
    d_POUT2={'sc':'A5GL2','canal':'FLAIR','r':5,'ady':1,'ce':'GL','par_ce':0.1,
        'prototipo':'prototipo_mediana','gamma_MDF':1.0,'percentil':0.9,'dist':10,'umbral_cc':4}
    d_POUT3={'sc':'A5GL3','canal':'FLAIR','r':5,'ady':1,'ce':'GL','par_ce':0.1,
        'prototipo':'prototipo_mediana','gamma_MDF':1.0,'percentil':0.9,'dist':10,'umbral_cc':4}
    
    
    d_POUT4={'sc':'A5GL4','canal':'FLAIR','r':5,'ady':1,'ce':'GL','par_ce':0.1,
        'prototipo':'prototipo_mediana','gamma_MDF':1.0,'percentil':0.9,'dist':10,'umbral_cc':4}
    d_POUT5={'sc':'A5GL5','canal':'FLAIR','r':5,'ady':1,'ce':'GL','par_ce':0.1,
        'prototipo':'prototipo_mediana','gamma_MDF':1.0,'percentil':0.9,'dist':10,'umbral_cc':4}
    d_POUT6={'sc':'A5GL6','canal':'FLAIR','r':5,'ady':1,'ce':'GL','par_ce':0.1,
        'prototipo':'prototipo_mediana','gamma_MDF':1.0,'percentil':0.9,'dist':10,'umbral_cc':4}
    
    d_POUT7={'sc':'A5GL7','canal':'FLAIR','r':5,'ady':1,'ce':'GL','par_ce':0.1,
        'prototipo':'prototipo_mediana','gamma_MDF':1.0,'percentil':0.9,'dist':10,'umbral_cc':4}
    d_POUT8={'sc':'A5GL8','canal':'FLAIR','r':5,'ady':1,'ce':'GL','par_ce':0.1,
        'prototipo':'prototipo_mediana','gamma_MDF':1.0,'percentil':0.9,'dist':10,'umbral_cc':4}
    d_POUT9={'sc':'A5GL9','canal':'FLAIR','r':5,'ady':1,'ce':'GL','par_ce':0.1,
        'prototipo':'prototipo_mediana','gamma_MDF':1.0,'percentil':0.9,'dist':10,'umbral_cc':4}
    
    d_POUT10={'sc':'A5GL10','canal':'FLAIR','r':5,'ady':1,'ce':'GL','par_ce':0.1,
        'prototipo':'prototipo_mediana','gamma_MDF':1.0,'percentil':0.9,'dist':10,'umbral_cc':4}
    d_POUT11={'sc':'A5GL11','canal':'FLAIR','r':5,'ady':1,'ce':'GL','par_ce':0.1,
        'prototipo':'prototipo_mediana','gamma_MDF':1.0,'percentil':0.9,'dist':10,'umbral_cc':4}
    d_POUT12={'sc':'A5GL12','canal':'FLAIR','r':5,'ady':1,'ce':'GL','par_ce':0.1,
        'prototipo':'prototipo_mediana','gamma_MDF':1.0,'percentil':0.9,'dist':10,'umbral_cc':4}
    
    d_POUT13={'sc':'A5GL13','canal':'FLAIR','r':5,'ady':1,'ce':'GL','par_ce':0.1,
        'prototipo':'prototipo_mediana','gamma_MDF':1.0,'percentil':0.9,'dist':10,'umbral_cc':4}
    d_POUT14={'sc':'A5GL14','canal':'FLAIR','r':5,'ady':1,'ce':'GL','par_ce':0.1,
        'prototipo':'prototipo_mediana','gamma_MDF':1.0,'percentil':0.9,'dist':10,'umbral_cc':4}
    d_POUT15={'sc':'A5GL15','canal':'FLAIR','r':5,'ady':1,'ce':'GL','par_ce':0.1,
        'prototipo':'prototipo_mediana','gamma_MDF':1.0,'percentil':0.9,'dist':10,'umbral_cc':4}
    
    
    # d_POUT1={'sc':'A5GL0','canal':'FLAIR','r':5,'ady':0,'ce':'GL','par_ce':0.1,'prototipo':'prototipo_media','gamma_MDF':1.0,'percentil':0.9,'dist':15,'umbral_cc':4}
    # d_POUT2={'sc':'A5GL0','canal':'FLAIR','r':5,'ady':1,'ce':'GL','par_ce':0.1,'prototipo':'prototipo_media','gamma_MDF':1.0,'percentil':0.9,'dist':15,'umbral_cc':4}
    # d_POUT2={'sc':'A5GL0','canal':'FLAIR','r':5,'ady':0,'ce':'GL','par_ce':0.2,'prototipo':'prototipo_mediana','gamma_MDF':1.0,'percentil':0.9,'dist':10,'umbral_cc':6}
    # POUT1=(5,0,'GL',0.1,'prototipo_media',1.0,0.9,15,4)
    # POUT2=(5,1,'GL',0.1,'prototipo_media',1.0,0.9,15,4)
    plantilla_MDF = 'MDF_{}_r{}a{}p{}{}g{}p{}_{}_d{}u{}'
    # path_imgSave='/media/gustavo/Disco_2/Tesis/Codigos/Images/'
    cont=0
    # for parametros_out in [d_POUT13,d_POUT14,d_POUT15]:
    # for parametros_out in [d_POUT10,d_POUT11,d_POUT12]:
    # for parametros_out in [d_POUT7,d_POUT8,d_POUT9]:
    # for parametros_out in [d_POUT5,d_POUT6,d_POUT7]:
    # for parametros_out in [d_POUT1,d_POUT2,d_POUT3]:
    # for parametros_out in [d_POUT1,d_POUT2,d_POUT3,d_POUT10,d_POUT11,d_POUT12,d_POUT13,d_POUT14,d_POUT15]:
    for parametros_out in [d_POUT11]:
        # Extraer características de setting
        sc = parametros_out['sc']; canal = parametros_out['canal']
        r = parametros_out['r']; ady = parametros_out['ady']
        ce = parametros_out['ce']; par_ce = parametros_out['par_ce'];par_ce = 0.3
        prototipo = parametros_out['prototipo']; gamma_MDF = parametros_out['gamma_MDF']
        percentil = parametros_out['percentil']; dist = parametros_out['dist']
        umbral_cc = parametros_out['umbral_cc']
        
        listaCaracteristicas_utilizada = ModuloA.Get_lista_características(sc)
        dPAC={}# {paciente: Array Características}
        # =============================================================================
        # Obtener diccionario con pacientes como llaves y matriz AC como valor
        # =============================================================================
        # for paciente in pacientes:
        # for paciente in pacientes[6:7]:
        p=1
        for paciente in pacientes[p:p+1]:# CPU0
            AC = ModuloA.Get_ArrayCaracteristicas(path_caracteristicas,paciente,listaCaracteristicas_utilizada)
            print(f'AC:{AC.shape}\n')
            dPAC[paciente]=AC
        print(f'dPAC:{dPAC.keys()}')
        # =============================================================================
        # Guardar
        # =============================================================================
        print(f'\nPARÁMETROS:\nr:{r},a:{ady},ce:{ce},par_ce:{par_ce},prototipo:{prototipo}, gamma_MDF:{gamma_MDF}, percentil:{percentil}, dist:{dist}, umbral_cc:{umbral_cc}')
        print(f'listaCaracteristicas_utilizada:{listaCaracteristicas_utilizada}')
        cep = ce+str(par_ce)# Estimador y parámetro de las matrices de covarianzas y precisión
        prot='p' if prototipo=='prototipo_media' else 'm'# o vector de medianas
        # carpeta_guardar = plantilla_MDF.format(sc,r,ady,prot,cep,gamma_MDF,percentil,canal,dist,umbral_cc)
        # print(f'carpeta_guardar:{carpeta_guardar}')
        # if carpeta_guardar not in os.listdir(path_MDFs_2D):
        #     os.mkdir(os.path.join(path_MDFs_2D,carpeta_guardar))
        
        # for paciente in pacientes:# CPU0
        # for paciente in pacientes[6:7]:# CPU0
        for paciente in pacientes[p:p+1]:# CPU0
            t1=time()
            print(f'\npaciente: {paciente}')
            # slide=85
            # for slide in range(160):
            
            s=90
            # for slide in range(40,41):
            for slide in range(s,s+1):
                cont+=1
                print(f'slide: {slide}...')
                nombre_slide = paciente+'_'+str(slide)+'_MDF'
                if slide==0 or slide==159:
                    mdf = np.ones(AC.shape[-2:],dtype=np.float32)*np.nan
                else:
                    Y_input,AC_input,X_input = ModuloA.Get_inputAdy(dPAC,path_base,paciente,slide,ady,DS,canal,input_Y=True,input_AC=True,input_X=True)
                    print(f'Y_input:{Y_input.shape}, AC_input:{AC_input.shape}, X_input:{X_input.shape}')
                    print(f'Y_input:{Y_input.dtype}, AC_input:{AC_input.dtype}, X_input:{X_input.dtype}')
                    if AC_input.dtype=='float16':
                        AC_input = AC_input.astype('float32')
                    print(f'Y_input:{Y_input.dtype}, AC_input:{AC_input.dtype}, X_input:{X_input.dtype}')
                    if cont==1:
                        # Y_input,AC_input,X_input = ModuloA.Get_inputAdy(dPAC,path_base,paciente,slide,ady,DS,canal,input_Y=True,input_AC=True,input_X=True)
                        # print(f'Y_input:{Y_input.shape}, AC_input:{AC_input.shape}, X_input:{X_input.shape}')
                        plt.figure(figsize=(30,40))
                        plt.subplot(4,3,1);plt.imshow(X_input[ady],cmap='gray')
                        plt.subplot(4,3,2);plt.imshow(Y_input[ady],cmap='gray')
                        Y_input = ModuloA.Filtrar_CC_slides(Y_input,umbral_cc)
                        plt.subplot(4,3,3);plt.imshow(Y_input[ady],cmap='gray')
                    # sdf_Gin, ind = ModuloA.SDF(Y_input[ady], return_indices=True)
                    # print(f'sdf_Gin:{sdf_Gin.shape}')
                    # plt.subplot(2,3,4);plt.imshow(sdf_Gin,cmap='gray')
                    # pdf_Gin, ind = ModuloA.PDF(Y_input[ady], return_indices=True)
                    # plt.subplot(2,3,5);plt.imshow(pdf_Gin,cmap='gray')
                    
                    mdf = ModuloA.MDF(Y_input, AC_input, r,ady,ce,par_ce,prototipo,gamma_MDF,percentil,dist)
                    print(f'mdf:{mdf.shape}')
                    print(f'cont:{cont}')
                    plt.subplot(4,3,3+cont);plt.imshow(mdf,cmap='gray');plt.title(f'sc:{sc}')
                    mdf = ModuloA.Suavizar(mdf,(3,3))
                    plt.subplot(4,3,3+cont+1);plt.imshow(mdf,cmap='gray');plt.title(f'sc:{sc}')
                # np.save(os.path.join(path_MDFs_2D,carpeta_guardar,nombre_slide),mdf)
                # print(f'\nmdf:{mdf.shape} {mdf.dtype}')
                print(f'nombre_slide: {nombre_slide}')
            print(f'time: {round(time()-t1,2)}')
            # mdf_features__f32__sinvar=np.copy(mdf)