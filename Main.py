#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 17:46:35 2024

@author: gustavo
"""

import os
from time import sleep

# =============================================================================
# Enable training, then testing and sending folders to the central node
# =============================================================================
crear_carpetas=False; training=False; testing=False; scp=False; enfriar=False
# crear_carpetas=True
if not crear_carpetas:
    training=True
    # testing=True
    # scp=True

# enfriar=True # cool down or pause computations

# =============================================================================
# Indicate server, GPU and runs
# =============================================================================

# servidor = 'mineria';  gpu='0'; Corridas=['1','2','3','4']
# servidor = 'zealot'; gpu='0'; Corridas=['1','2','3','4']
# servidor = 'fondecyt1' ; gpu='0'; Corridas=['1','2','3'] ;gpu='1'; Corridas=['1','2','3']
# servidor = 'fondecyt2' ; gpu='0'; Corridas=['1','2','3'] ;gpu='1'; Corridas=['1','2','3']

servidor = 't15'    ; gpu='0'; Corridas=['1'] # local
# servidor = 't15'    ; gpu='cpu'; Corridas=['0']
for indice in range(len(Corridas)):
    Corridas[indice] = servidor+'_gpu'+gpu+'-'+Corridas[indice]
print(Corridas)

os.environ["CUDA_VISIBLE_DEVICES"]=gpu

# =============================================================================
# Training parameters
# =============================================================================
tasa_da = '0.0'
umbral_vol_training = '1'
batch_size='16'
metric = 'Dice_metric'
opt = 'Adam'
red= 'Unet'
mixed_precision = 'T' #;mixed_precision = 'F'
ES = 'T' #;ES = 'F'#Early-stopping

# =============================================================================
# Loss function parameters
# =============================================================================
gamma = '1.0' #;gamma = 'nan'
alpha_HD = '2.0' #;alpha_HD = 'nan' # Hausdorff-distance loss
potencia_assd ='1.0' ;potencia_assd ='nan'
alpha_TL = '0.3' #;alpha_TL = 'nan'# alpha de Tversky loss
beta_ASL = '2.0' #;beta_ASL = 'nan'# Asymmetric similarity loss
w_SEL = '0.1' #;w_SEL = 'nan'# sensitivity-especifity loss
p_assd = '1.0' ;p_assd = 'nan'
alpha_BS = '0.8' # for Boundary-sensitive loss
beta_BS_LC = '0.9'
wa_ABL = '1.0' # for Active boundary loss
batch_loss = 'T' #;batch_loss = 'F'
gamma_CBL = '2.0' # for Conditional boundary loss

umbral_cc = '0' # only slides with number of lesion > 0

parMD_weight = '1.0'
parMD_pot = '2.0'# lambda parameter in the paper
parMD_sq = '1'
umbral_P_MDF = '150'
parMD_quantil = '1.0'

start_es = '40'
patience = '30'

# =============================================================================
# Training cycle
# =============================================================================
# for DS in ['ISBI2015','MSSEG2016']:
# for DS in ['MSSEG2016','ISBI2015']:
for DS in ['MSSEG2016']:
# for DS in ['ISBI2015']:
    for epocas in ['200']:
        folds = DS+'_5folds'
        for loss in ['MD_loss']:
        
        # for loss in ['GDL']:
        # for loss in ['Boundary_loss']:
        # for loss in ['HD_loss']:
        # for loss in ['ABL']:
            
            sc='A5GL11';canal='FLAIR';r='5';ady='1';ce='GL';par_ce='0.3'
            prototipo='prototipo_mediana';gamma_MDF='1.0';percentil='0.9'
            for parMD_weight in ['1.0']:
                
                for parMD_pot in ['3.0']:# lambda parameter in the paper
                    # for dist in ['5','10','15']:
                    for dist in ['10']:
                        if DS=='MSSEG2016' and dist=='5' or DS=='ISBI2015' and (dist=='10' or dist=='15'):
                            continue
                # for parMD_pot in ['2.5','3.0']:
                #     for dist in ['5','10','15']:
                        print(f'DS:{DS}, dist:{dist} pot:{parMD_pot}')
                        lista_MDF = ['sc:'+sc,'canal:'+canal,'r:'+r,'ady:'+ady,'ce:'+ce,'par_ce:'+par_ce,'dist:'+dist,
                            'prototipo:'+prototipo,'gamma_MDF:'+gamma_MDF,'percentil:'+percentil,'umbral_cc:'+umbral_cc,
                            'parMD_weight:'+parMD_weight,'parMD_sq:'+parMD_sq,'umbral_P_MDF:'+umbral_P_MDF,'parMD_pot:'+parMD_pot,
                            'parMD_quantil:'+parMD_quantil]
                        print(f'lista_MDF: {lista_MDF}')
                        
                        lista_parametros = [
                            'loss:'+loss,'epocas:'+epocas,'batch_size:'+batch_size,'opt:'+opt,
                            'DS:'+DS,'red:'+red,'gpu:'+gpu,'tasa_da:'+tasa_da,'mixed_precision:'+mixed_precision,
                            'alpha_TL:'+alpha_TL,'folds:'+folds,'metric:'+metric,'gamma:'+gamma,'alpha_HD:'+alpha_HD,
                            'ES:'+ES,'patience:'+patience,'start_es:'+start_es,'potencia_assd:'+potencia_assd,'beta_ASL:'+beta_ASL,
                            'w_SEL:'+w_SEL,'p_assd:'+p_assd,'umbral_vol_training:'+umbral_vol_training,
                            'alpha_BS:'+alpha_BS,'beta_BS_LC:'+beta_BS_LC,'wa_ABL:'+wa_ABL,'batch_loss:'+batch_loss,
                            'gamma_CBL:'+gamma_CBL]+lista_MDF
                        parametros_base = ' '.join(lista_parametros)
                        print(f'parametros_base: {parametros_base}')
                        if crear_carpetas and os.uname()[1] == 'mineria':
                            parametros_base+=' tarea:crear_carpetas k:'+'1'+' corrida:'+'0'
                            os.system('python Entrenar.py '+parametros_base)
                            continue # Algún ciclo for
                        for corrida in Corridas:
                            if DS=='MSSEG2016' or DS=='ISBI2015':
                                lista_k = ['0']
                            elif DS=='WMH2017':
                                lista_k = ['1']
                            for k in lista_k:
                                if training:
                                    parametros_base+=' tarea:training k:'+k+' corrida:'+corrida
                                    print(f'parametros_base: {parametros_base}')
                                    os.system('python Entrenar.py '+parametros_base)
                                if testing:
                                    parametros_base+=' tarea:testing k:'+k+' corrida:'+corrida
                                    os.system('python Entrenar.py '+parametros_base)
                                if enfriar and not crear_carpetas:
                                    sleep(60)
                            if scp:
                                parametros_base+=' tarea:scp k:'+'1'+' corrida:'+corrida
                                os.system('python Entrenar.py '+parametros_base)
