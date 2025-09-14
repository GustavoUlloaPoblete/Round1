#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 22:33:42 2022

@author: gustavo
"""

import numpy
# import os
# from skimage import measure
from skimage.segmentation import find_boundaries
from scipy import ndimage
from skimage.transform import rotate
# from skimage.exposure import histogram
# from random import choice#, randint
# import elasticdeform
from sklearn.metrics import roc_curve, auc, precision_recall_curve
# import SimpleITK as sitk

eps = 1e-7
# smooth = 1

def scheduler(epoch, lr):
    if epoch!=0 and epoch%80==0:
        lr = lr*0.9
    return lr

def Cropping(imagen,shape_target):
    shape_input = imagen.shape
    # print(imagen.shape,shape_target)
    d_cortar={0:[],1:[],2:[]}
    diff_0 = shape_input[0] - shape_target[0]
    crop_0_left = diff_0//2
    crop_0_right = diff_0 - crop_0_left
    # print(crop_0_left,crop_0_right)
    d_cortar[0]=[crop_0_left,crop_0_right]
    diff_1 = shape_input[1] - shape_target[1]
    crop_1_left = diff_1//2
    crop_1_right = diff_1 - crop_1_left
    # print(crop_1_left,crop_1_right)
    d_cortar[1]=[crop_1_left,crop_1_right]
    diff_2 = shape_input[2] - shape_target[2]
    crop_2_left = diff_2//2
    crop_2_right = diff_2 - crop_2_left
    # print(crop_2_left,crop_2_right)
    d_cortar[2]=[crop_2_left,crop_2_right]
    return d_cortar

def rotate_3D(I,angulo,order):#channels last
    I_aug = numpy.zeros(I.shape,dtype=numpy.float32)
    for ch in range(I.shape[-1]):
        I_aug[:,:,ch] = rotate(I[:,:,ch],angulo,preserve_range=True,order=order)
    return I_aug
def flip_3D(I,eje):#channels last
    I_aug = numpy.zeros(I.shape,dtype=numpy.float32)
    for ch in range(I.shape[-1]):
        I_aug[:,:,ch] = numpy.flip(I[:,:,ch],eje)
    return I_aug
def Get_d_channels(path_DS):
    if path_DS.split('/')[-2] == 'MSSEG2016':
        # print('MSSEG2016')
        d = {'T1':'T1_preprocessed.npy','T2':'T2_preprocessed.npy','FLAIR':'FLAIR_preprocessed.npy','GADO':'GADO_preprocessed.npy','DP':'DP_preprocessed.npy','MASCARA':'Consensus.npy'}
        channels = ['T1','T2','FLAIR']
    elif path_DS.split('/')[-2] == 'ISBI2015':
        # print('ISBI2015')
        d = {'T1':'t1.npy','T2':'t2.npy','FLAIR':'flair.npy','DP':'pd.npy','MASCARA':'segmentation_1.npy'}
        channels = ['T1','T2','FLAIR']
    return d,channels

def Get_lambda_HD_loss(Y_train, Y_hat, alpha_HD):
    # print('\ndentro de: Get_lambda_HD_loss...')
    tn, fp, fn, tp = Matriz_confusion(Y_train[:,:,:,0], Y_hat[:,:,:,0])
    # print(tn, fp, fn, tp)
    Train_Dice_loss = (1-Metrica_F1(tn, fp, fn, tp))
    # print('Train_Dice_loss:',Train_Dice_loss)
    
    G_dtm = Y_train[:,:,:,1]
    S_dtm = Y_train[:,:,:,2]
    loss = numpy.power((Y_hat[:,:,:,0]-Y_train[:,:,:,0]),2)*(numpy.power(G_dtm,alpha_HD)+numpy.power(S_dtm,alpha_HD))
    # print('loss:',loss.shape,loss.sum())
    Train_hd_loss = loss.sum()/loss.size
    # hd_loss = tf.reduce_sum(loss)/tf.cast(tf.size(loss),dtype=tf.float32)
    # print('Train_hd_loss:',Train_hd_loss)
    labmda_HD_loss = Train_hd_loss/Train_Dice_loss
    # print('labmda_HD_loss:',labmda_HD_loss)
    return labmda_HD_loss

# =============================================================================
# Preprocesar
# =============================================================================
'''
Para no importar sitk
def N4(path_input,path_output):
    print('Procesando N4 Bias Field Correction.....')
    inputImage = sitk.ReadImage(path_input, sitk.sitkFloat32)
    maskImage = sitk.OtsuThreshold(inputImage,0,1,200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    output = corrector.Execute(inputImage,maskImage)
    sitk.WriteImage(output,path_output)
    return None
'''

def Truncar_simetrico(volumen,umbral_inferior,umbral_superior):
    volumen[volumen<umbral_inferior]=umbral_inferior
    volumen[volumen>umbral_superior]=umbral_superior
    return volumen

def Truncar(volumen,mask,p=0.9995):
    vector = volumen[mask==1]
    cuantil = numpy.quantile(vector,p)
    volumen[volumen>cuantil]=cuantil
    return volumen

def Truncar_superior(volumen,p=0.95):
    volumen = volumen.copy()
    # vector = volumen[mask==1]
    cuantil = numpy.quantile(volumen,p)
    volumen[volumen>cuantil]=cuantil
    return volumen

def Normalizar(img, bound = 1):
    return (img-numpy.min(img))/(numpy.max(img)-numpy.min(img))*bound

def Estandarizar(img):
    return (img-numpy.mean(img))/numpy.std(img)

# =============================================================================
# Métricas de Surface-Distances
# =============================================================================

def Metricas_borde(Q,P,metricas):
    
    BQ = find_boundaries(Q,mode='inner')
    BP = find_boundaries(P,mode='inner')
    not_BQ = numpy.logical_not(BQ)
    not_BP = numpy.logical_not(BP)
    
    DBQ = ndimage.distance_transform_edt(not_BQ).astype(numpy.float32)
    DBP = ndimage.distance_transform_edt(not_BP).astype(numpy.float32)
    
    d_BP2BQ = DBP[numpy.nonzero(BQ)]
    d_BQ2BP = DBQ[numpy.nonzero(BP)]
    
    d = {}
    for metrica in metricas:
        if metrica == 'hd':
            hd = max(max(d_BP2BQ),max(d_BQ2BP))
            d['hd']= hd
        elif metrica == 'hd95':
            cuantil = 0.95
            hd95 = max(numpy.quantile(d_BP2BQ,cuantil),numpy.quantile(d_BQ2BP,cuantil))
            d['hd95']= hd95
        elif metrica == 'hd90':
            cuantil = 0.90
            hd90 = max(numpy.quantile(d_BP2BQ,cuantil),numpy.quantile(d_BQ2BP,cuantil))
            d['hd90']= hd90
        elif metrica == 'assd':
            assd = (d_BP2BQ.sum()+d_BQ2BP.sum())/(BQ.sum()+BP.sum())
            d['assd']= assd
        elif metrica == 'assd95':
            cuantil = 0.95
            assd95 = (d_BP2BQ[d_BP2BQ<=numpy.quantile(d_BP2BQ,cuantil)].sum()+d_BQ2BP[d_BQ2BP<=numpy.quantile(d_BQ2BP,cuantil)].sum())/(BQ.sum()+BP.sum())
            d['assd95']= assd95
    return d

# G=numpy.load('segmentation_1.npy')
# S=numpy.load('segmentation_2.npy')
# print(f'G:{G.shape}')
# print(f'S:{S.shape}')
# d=Metricas_borde(G,S,['hd', 'hd95', 'assd', 'assd95'])
# print(d)

# =============================================================================
# Metrica Relative volume difference
# =============================================================================
def RVD(G,S):
    return abs(G.sum()-S.sum())/G.sum()

# =============================================================================
# Metricas Spatial-Overlap
# =============================================================================
def Metrica_AUC_ROC(y_true,y_pred):
    fpr, tpr, thresholds = roc_curve(y_true,y_pred)
    auc_roc = auc(fpr, tpr)
    return auc_roc

def Metrica_AUC_PR(y_true,y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true,y_pred)
    auc_pr = auc(recall, precision)
    return auc_pr

def Metrica_F2(tn,fp,fn,tp):
    f2 = (5.0*tp + eps)/(5.0*tp + 4.0*fn+fp + eps)
    return f2

def Metrica_F1(tn,fp,fn,tp):
    f1 = (2.0*tp + eps)/(2.0*tp+fn+fp + eps)
    return f1

def Metrica_ACC(tn,fp,fn,tp):
    acc = (tp+tn + eps)/(tp+tn+fp+fn + eps)
    return acc

def Metrica_ACC_r(y_true,y_pred): # Trabaja igual que el 'accuracy' de tensorflow
    return numpy.mean(y_true==numpy.round(y_pred))

def Metrica_TPR(tn,fp,fn,tp):
    tpr = (tp + eps)/(float(tp+fn) + eps)
    return tpr
    
def Metrica_PPV(tn,fp,fn,tp):
    ppv = (tp + eps)/(float(tp+fp) + eps)
    return ppv

def Metrica_FPR(tn,fp,fn,tp):
    fpr = (fp + eps)/(float(fp+tn) + eps)
    return fpr

def Metrica_FNR(tn,fp,fn,tp):
    fnr = (fn + eps)/(float(fn+tp) + eps)
    return fnr

def Metrica_TNR(tn,fp,fn,tp):
    tnr = (tn + eps)/(float(tn+fp) + eps)
    return tnr

def Redondear_Lista(lista):# Borrar
    lista_2 = []
    for i in range(len(lista)):
        lista_2.append(round(lista[i],4))
    return lista_2

def Matriz_confusion(G, P, axis=None):# un ejemplo a la vez, sin batch!, borrar axis
    TP = numpy.sum(numpy.multiply(G, P), axis)
    FP = numpy.sum(numpy.multiply((1-G), P), axis)
    FN = numpy.sum(numpy.multiply(G, (1-P)), axis)
    TN = numpy.sum(numpy.multiply((1-G), (1-P)), axis)
    return TN, FP, FN, TP

def Dice_metric(G, P):# un ejemplo a la vez, sin batch!
    numerador = 2*numpy.sum(numpy.multiply(G, P)) + eps
    denominador = numpy.sum(G) + numpy.sum(P) + eps
    dice = numerador/denominador
    # print(f'dice:{dice.shape}')
    return dice

# G1 = numpy.load('segmentation_1.npy')
# G2 = numpy.load('segmentation_2.npy')


# =============================================================================
# Mapas de transformación de distancias
# =============================================================================
def DTM_inner(array_orig,mode='inner'):
    array=array_orig.copy()
    if mode=='inner':
        IA_boundaries = find_boundaries(array, mode=mode, connectivity=1)
        array[IA_boundaries]=0
    G_in = ndimage.distance_transform_edt(array).astype(numpy.float32)
    return G_in

def DTM(array_orig,mode='inner'):
    G_in = DTM_inner(array_orig,mode)
    array = array_orig.copy()
    if mode=='outer':
        IA_boundaries = find_boundaries(array, mode=mode, connectivity=1)
        array[IA_boundaries]=1
    not_array = numpy.logical_not(array)
    G_out = ndimage.distance_transform_edt(not_array).astype(numpy.float32)
    return G_out+G_in

def DTM_2D(array_orig,mode='inner'):#Retorna DTM de cada elemento (array 2D) del eje 0
    array=array_orig.copy()
    array_DTM_2D = numpy.zeros(array.shape,dtype=numpy.float32)
    for n in range(array_DTM_2D.shape[0]):
        array_DTM_2D[n] = DTM(array[n].astype(numpy.int8),mode)
    return array_DTM_2D

def SDF(array_orig,mode='inner'):
    G_in = DTM_inner(array_orig,mode)
    array = array_orig.copy()
    if mode=='outer':
        IA_boundaries = find_boundaries(array, mode=mode, connectivity=1)
        array[IA_boundaries]=1
    not_array = numpy.logical_not(array)
    G_out = ndimage.distance_transform_edt(not_array).astype(numpy.float32)
    return G_out-G_in

def SDF_2D(array_orig,mode='inner'):
    array=array_orig.copy()
    array_SDF_2D = numpy.zeros(array.shape,dtype=numpy.float32)
    for n in range(array_SDF_2D.shape[0]):
        array_SDF_2D[n] = SDF(array[n].astype(numpy.int8),mode)
    return array_SDF_2D

# def PDF(array_orig,mode='inner'):
#     G_in = DTM(array_orig,mode)
#     array = array_orig.copy()
#     if mode=='outer':
#         IA_boundaries = find_boundaries(array, mode=mode, connectivity=1)
#         array[IA_boundaries]=1
#     not_array = numpy.logical_not(array)
#     G_out = ndimage.distance_transform_edt(not_array).astype(numpy.float32)
#     return G_out+G_in

# def PDF_TS(array_orig,p,mode='inner'):
#     pdf = PDF(array_orig)
#     pdf_TS = Truncar_superior(pdf,p)
#     return pdf_TS
    
# =============================================================================
# Para Cross-Validation
# =============================================================================
# LITS

# if os.uname()[1]=='ASUS-TUF':
#     path_DSI = '/media/gustavo/wd/DS_Imagenes'
# else:
#     path_DSI = '/home/'+os.getlogin()+'/DS_Imagenes'

# path_MICCAI2017LITS_DS = path_DSI+'/MICCAI2017LITS/MICCAI2017LITS_2x2x2_192x192x128_npy'
# pacientes_MICCAI2017LITS = os.listdir(path_MICCAI2017LITS_DS)
# # print(pacientes_MICCAI2017LITS)
# numpy.random.seed(33);numpy.random.shuffle(pacientes_MICCAI2017LITS)
# # print(pacientes_MICCAI2017LITS)
# folds = {1:pacientes_MICCAI2017LITS[0:26],
#           2:pacientes_MICCAI2017LITS[26:52],
#           3:pacientes_MICCAI2017LITS[52:78],
#           4:pacientes_MICCAI2017LITS[78:104],
#           5:pacientes_MICCAI2017LITS[104:131]}
# Group_5fold_MICCAI2017LITS = {
#     1:[folds[2]+folds[3]+folds[4]+folds[5], folds[1]],
#     2:[folds[1]+folds[3]+folds[4]+folds[5], folds[2]],
#     3:[folds[1]+folds[2]+folds[4]+folds[5], folds[3]],
#     4:[folds[1]+folds[2]+folds[3]+folds[5], folds[4]],
#     5:[folds[1]+folds[2]+folds[3]+folds[4], folds[5]]}

# =============================================================================
# Crear 4fold cross validation
# =============================================================================
#k, 5,4,3
# - izquierda-arriba:0-47=48,9,12,16
# - izquierda-abajo:68-82=15,3,3-4,5
# - derecha-arriba:53-67 + 83-130=63,12,15-16,21

# =============================================================================
# Muestreo estratificado
# =============================================================================
# izquierda_arriba=list(range(0,48))
# izquierda_abajo=list(range(68,83))
# derecha_arriba=list(range(53,68))+list(range(83,131))

# =============================================================================
# def Get_muestra(lista,N):
#     muestra=[]
#     for i in list(range(N)):
#         elemento=choice(lista)
#         muestra.append(elemento)
#         lista.remove(elemento)
#     return muestra,lista

# seed(11)#Mejor no usar semilla
# fold1=[]
# muestra,izquierda_arriba = Get_muestra(izquierda_arriba,12)
# fold1+=muestra
# muestra,izquierda_abajo  = Get_muestra(izquierda_abajo,4)
# fold1+=muestra
# muestra,derecha_arriba   = Get_muestra(derecha_arriba,16)
# fold1+=muestra

# fold2=[]
# muestra,izquierda_arriba = Get_muestra(izquierda_arriba,12)
# fold2+=muestra
# muestra,izquierda_abajo  = Get_muestra(izquierda_abajo,4)
# fold2+=muestra
# muestra,derecha_arriba   = Get_muestra(derecha_arriba,16)
# fold2+=muestra

# fold3=[]
# muestra,izquierda_arriba = Get_muestra(izquierda_arriba,12)
# fold3+=muestra
# muestra,izquierda_abajo  = Get_muestra(izquierda_abajo,3)
# fold3+=muestra
# muestra,derecha_arriba   = Get_muestra(derecha_arriba,16)
# fold3+=muestra

# fold4=[]
# muestra,izquierda_arriba = Get_muestra(izquierda_arriba,12)
# fold4+=muestra
# muestra,izquierda_abajo  = Get_muestra(izquierda_abajo,4)
# fold4+=muestra
# muestra,derecha_arriba   = Get_muestra(derecha_arriba,15)
# fold4+=muestra


# folds={1:[],2:[],3:[],4:[]}
# plantilla_id = 'paciente_id{}'
# for n in fold1:
#     folds[1].append(plantilla_id.format(n))
# for n in fold2:
#     folds[2].append(plantilla_id.format(n))
# for n in fold3:
#     folds[3].append(plantilla_id.format(n))
# for n in fold4:
#     folds[4].append(plantilla_id.format(n))
# MICCAI2017LITS_4folds = {
#     1:[folds[2]+folds[3]+folds[4], folds[1]],
#     2:[folds[1]+folds[3]+folds[4], folds[2]],
#     3:[folds[1]+folds[2]+folds[4], folds[3]],
#     4:[folds[1]+folds[2]+folds[3], folds[4]]}
# print(folds)

folds={1: ['paciente_id28', 'paciente_id36', 'paciente_id30', 'paciente_id29', 'paciente_id35', 'paciente_id42', 'paciente_id12', 'paciente_id11', 'paciente_id39', 'paciente_id37', 'paciente_id13', 'paciente_id6', 'paciente_id75', 'paciente_id72', 'paciente_id70', 'paciente_id69', 'paciente_id102', 'paciente_id120', 'paciente_id126', 'paciente_id113', 'paciente_id109', 'paciente_id55', 'paciente_id108', 'paciente_id94', 'paciente_id98', 'paciente_id116', 'paciente_id124', 'paciente_id114', 'paciente_id118', 'paciente_id64', 'paciente_id117', 'paciente_id53'], 
       2: ['paciente_id45', 'paciente_id4', 'paciente_id3', 'paciente_id2', 'paciente_id19', 'paciente_id44', 'paciente_id14', 'paciente_id31', 'paciente_id0', 'paciente_id43', 'paciente_id24', 'paciente_id20', 'paciente_id79', 'paciente_id82', 'paciente_id74', 'paciente_id76', 'paciente_id122', 'paciente_id89', 'paciente_id106', 'paciente_id54', 'paciente_id130', 'paciente_id61', 'paciente_id107', 'paciente_id91', 'paciente_id104', 'paciente_id127', 'paciente_id62', 'paciente_id92', 'paciente_id99', 'paciente_id88', 'paciente_id129', 'paciente_id97'], 
       3: ['paciente_id1', 'paciente_id8', 'paciente_id40', 'paciente_id10', 'paciente_id27', 'paciente_id15', 'paciente_id25', 'paciente_id34', 'paciente_id9', 'paciente_id5', 'paciente_id47', 'paciente_id38', 'paciente_id68', 'paciente_id73', 'paciente_id77', 'paciente_id125', 'paciente_id128', 'paciente_id57', 'paciente_id95', 'paciente_id87', 'paciente_id115', 'paciente_id90', 'paciente_id96', 'paciente_id59', 'paciente_id112', 'paciente_id123', 'paciente_id67', 'paciente_id85', 'paciente_id100', 'paciente_id60', 'paciente_id101'], 
       4: ['paciente_id22', 'paciente_id7', 'paciente_id32', 'paciente_id17', 'paciente_id21', 'paciente_id18', 'paciente_id46', 'paciente_id16', 'paciente_id23', 'paciente_id26', 'paciente_id41', 'paciente_id33', 'paciente_id78', 'paciente_id81', 'paciente_id71', 'paciente_id80', 'paciente_id93', 'paciente_id65', 'paciente_id119', 'paciente_id63', 'paciente_id103', 'paciente_id105', 'paciente_id58', 'paciente_id111', 'paciente_id84', 'paciente_id66', 'paciente_id56', 'paciente_id110', 'paciente_id121', 'paciente_id86', 'paciente_id83']}
MICCAI2017LITS_4folds = {
    1:[folds[2]+folds[3]+folds[4], folds[1]],
    2:[folds[1]+folds[3]+folds[4], folds[2]],
    3:[folds[1]+folds[2]+folds[4], folds[3]],
    4:[folds[1]+folds[2]+folds[3], folds[4]]}
# =============================================================================
# ISBI2015
# =============================================================================
pacientes_ISBI2015 = ['paciente_id11','paciente_id12','paciente_id13','paciente_id14',
              'paciente_id21','paciente_id22','paciente_id23','paciente_id24',
              'paciente_id31','paciente_id32','paciente_id33','paciente_id34','paciente_id35',
              'paciente_id41','paciente_id42','paciente_id43','paciente_id44',
              'paciente_id51','paciente_id52','paciente_id53','paciente_id54']
# channels_ISBI2015 = ['t1','t2','flair','pd','rater_1','rater_2']
folds={1:[],2:[],3:[],4:[],5:[]}
folds[1]=['paciente_id11','paciente_id12','paciente_id13','paciente_id14']
folds[2]=['paciente_id21','paciente_id22','paciente_id23','paciente_id24']
folds[3]=['paciente_id31','paciente_id32','paciente_id33','paciente_id34','paciente_id35']
folds[4]=['paciente_id41','paciente_id42','paciente_id43','paciente_id44']
folds[5]=['paciente_id51','paciente_id52','paciente_id53','paciente_id54']
ISBI2015_5folds_traintest = {
    1:[folds[2]+folds[3]+folds[4]+folds[5], folds[1]],
    2:[folds[1]+folds[3]+folds[4]+folds[5], folds[2]],
    3:[folds[1]+folds[2]+folds[4]+folds[5], folds[3]],
    4:[folds[1]+folds[2]+folds[3]+folds[5], folds[4]],
    5:[folds[1]+folds[2]+folds[3]+folds[4], folds[5]]}
ISBI2015_5folds_trainvaltest = {
    # 0:[folds[1]+folds[3]+folds[4],folds[2], folds[5]],
    0:[folds[1]+folds[3]+folds[5],folds[2], folds[4]],
    
    1:[folds[2]+folds[3]+folds[4],folds[5], folds[1]],
    2:[folds[1]+folds[3]+folds[5],folds[4], folds[2]],
    3:[folds[1]+folds[4]+folds[5],folds[2], folds[3]],
    4:[folds[1]+folds[2]+folds[5],folds[3], folds[4]],
    5:[folds[2]+folds[3]+folds[4],folds[1], folds[5]]}
ISBI2015_input_dim_3D = (3, 160, 192, 160)
ISBI2015_input_dim_2D = (3, 160, 192)
# ISBI2015_input_dim_3D = (160, 192, 160, 3)
# ISBI2015_input_dim_2D = (160, 192, 3)

# (181,217,181)
# =============================================================================
# MSSEG2016
# =============================================================================
pacientes_MSSEG2016 = ['01016SACH_id1016', '01038PAGU_id1038', '01039VITE_id1039', '01040VANE_id1040', '01042GULE_id1042',
              '07001MOEL_id7001', '07003SATH_id7003', '07010NABO_id7010', '07040DORE_id7040', '07043SEME_id7043',
              '08002CHJE_id8002', '08027SYBR_id8027', '08029IVDI_id8029', '08031SEVE_id8031', '08037ROGU_id8037']

folds={1:[],2:[],3:[],4:[],5:[]}
folds[1]=['01016SACH_id1016', '07010NABO_id7010', '08027SYBR_id8027']
folds[2]=['01038PAGU_id1038', '07001MOEL_id7001', '08037ROGU_id8037']
folds[3]=['01039VITE_id1039', '07043SEME_id7043', '08029IVDI_id8029']
folds[4]=['01040VANE_id1040', '07003SATH_id7003', '08031SEVE_id8031']
folds[5]=['01042GULE_id1042', '07040DORE_id7040', '08002CHJE_id8002']
MSSEG2016_5folds_traintest = {
    1:[folds[2]+folds[3]+folds[4]+folds[5], folds[1]],
    2:[folds[1]+folds[3]+folds[4]+folds[5], folds[2]],
    3:[folds[1]+folds[2]+folds[4]+folds[5], folds[3]],
    4:[folds[1]+folds[2]+folds[3]+folds[5], folds[4]],
    5:[folds[1]+folds[2]+folds[3]+folds[4], folds[5]]}
MSSEG2016_5folds_trainvaltest = {
    0:[folds[2]+folds[3]+folds[4],folds[1], folds[5]],
    
    1:[folds[2]+folds[3]+folds[4],folds[5], folds[1]],
    2:[folds[1]+folds[3]+folds[5],folds[4], folds[2]],
    3:[folds[1]+folds[4]+folds[5],folds[2], folds[3]],
    4:[folds[1]+folds[2]+folds[5],folds[3], folds[4]],
    5:[folds[2]+folds[3]+folds[4],folds[1], folds[5]]}
MSSEG2016_input_dim_3D = (3, 160, 192, 160)
MSSEG2016_input_dim_2D = (3, 160, 192)
# MSSEG2016_input_dim_3D = (160, 192, 160, 3)
# MSSEG2016_input_dim_2D = (160, 192, 3)

# =============================================================================
# WMH2017
# =============================================================================
# folds = {1: ['19_id19', '49_id49', '41_id41', '8_id8', '61_id61', '66_id66',
#       '52_id52', '50_id50', '116_id116', '108_id108', '144_id144', '107_id107'],
#   2: ['21_id21', '33_id33', '25_id25', '29_id29', '69_id69', '58_id58', '64_id64',
#       '54_id54', '109_id109', '114_id114', '102_id102', '113_id113'],
#   3: ['23_id23', '0_id0', '4_id4', '11_id11', '65_id65', '51_id51',
#       '59_id59', '53_id53', '105_id105', '115_id115', '137_id137', '112_id112'],
#   4: ['37_id37', '27_id27', '39_id39', '35_id35', '68_id68', '56_id56', '60_id60',
#       '55_id55', '100_id100', '103_id103', '110_id110', '104_id104'],
#   5: ['17_id17', '31_id31', '6_id6', '2_id2', '67_id67', '63_id63',
#       '62_id62', '57_id57', '132_id132', '126_id126', '101_id101', '106_id106']}
# WMH2017_5folds = {
#     1:[folds[2]+folds[3]+folds[4]+folds[5], folds[1]],
#     2:[folds[1]+folds[3]+folds[4]+folds[5], folds[2]],
#     3:[folds[1]+folds[2]+folds[4]+folds[5], folds[3]],
#     4:[folds[1]+folds[2]+folds[3]+folds[5], folds[4]],
#     5:[folds[1]+folds[2]+folds[3]+folds[4], folds[5]]}

# =============================================================================
# Realizado en el laptop, Dropbox/.../Codigos_Tesis_General/2024/Revisar WHM2017/wmh2017.py
# Sin k-fold ya que hay más ejemplos(60) que en ISBI2015(21) y MSSEG2016(15) 
# =============================================================================
# import os
# path = '/media/gustavo/Disco_2/DS_Imagenes/WMH2017/'
# nombre_c1 = 'Amsterdam_GE3T'
# nombre_c2 = 'Singapore'
# nombre_c3 = 'Utrecht'

# c1 = os.listdir(os.path.join(path,nombre_c1))
# c2 = os.listdir(os.path.join(path,nombre_c2))
# c3 = os.listdir(os.path.join(path,nombre_c3))

# for i in range(len(c1)):
#     c1[i]+='_id'+c1[i]
# for i in range(len(c2)):
#     c2[i]+='_id'+c2[i]
# for i in range(len(c3)):
#     c3[i]+='_id'+c3[i]

# # training_set = [12,12,11]
# # validation_set = [3,3,4]
# # testing_set = [5,5,5]

# # training_set = [12,12,12]
# # validation_set = [4,4,4]
# # testing_set = [4,4,4]

# numpy.random.seed(3)
# numpy.random.shuffle(c1)
# numpy.random.shuffle(c2)
# numpy.random.shuffle(c3)

# # WMH2017_trainvaltest = [c1[:12]+c2[:12]+c3[:11],
# #                         c1[12:12+3]+c2[12:12+3]+c3[11:11+4],
# #                         c1[12+3:12+3+5]+c2[12+3:12+3+5]+c3[11+4:11+4+5]]

# WMH2017_trainvaltest = [c1[:12]+c2[:12]+c3[:12],
#                         c1[12:12+4]+c2[12:12+4]+c3[12:12+4],
#                         c1[12+4:12+4+4]+c2[12+4:12+4+4]+c3[11+4:11+4+4]]


# Resultados:
    ### SE REPITE PACIENTE 2 EN VALIDATION Y TESTING. y FALTA EL 11 DE TODO.###
    ### ASI ENTRENÉ, HAY QUE MANTENER NOMÁS, ES SOLO 1 PACIENTE DE 60... Y NO TENGO PODER DE COMPUTO DE SOBRA ###
WMH2017_training = ['115_id115', '102_id102', '101_id101', '132_id132', '104_id104', 
            '126_id126', '106_id106', '107_id107', '116_id116', '113_id113', 
            '109_id109', '112_id112', '54_id54', '59_id59', '57_id57', 
            '56_id56', '69_id69', '50_id50', '66_id66', '61_id61', '63_id63', 
            '58_id58', '55_id55', '60_id60', '35_id35', '31_id31', '29_id29', 
            '19_id19', '33_id33', '17_id17', '37_id37', '27_id27', '0_id0',
            '23_id23', '25_id25', '6_id6']
WMH2017_validation = ['144_id144', '137_id137', '114_id114', '105_id105', '53_id53',
              '51_id51', '62_id62', '65_id65', '49_id49', '8_id8', '39_id39', '2_id2']
WMH2017_testing = ['100_id100', '108_id108', '103_id103', '110_id110', '68_id68', 
           '52_id52', '67_id67', '64_id64', '2_id2', '21_id21', '41_id41', '4_id4']
pacientes_WMH2017 = WMH2017_training+WMH2017_validation+WMH2017_testing
WMH2017_trainvaltest = [WMH2017_training,WMH2017_validation,WMH2017_testing]
WMH2017_input_dim_3D = (2, 160, 192, 160)
WMH2017_input_dim_2D = (2, 160, 192)

# pacientes_WMH2017 = ['115_id115', '102_id102', '101_id101', '132_id132', '104_id104', 
#             '126_id126', '106_id106', '107_id107', '116_id116', '113_id113', '109_id109', '112_id112', '54_id54', '59_id59', '57_id57', 
#             '56_id56', '69_id69', '50_id50', '66_id66', '61_id61', '63_id63', '58_id58', '55_id55', '60_id60', '35_id35', '31_id31', '29_id29', 
#             '19_id19', '33_id33', '17_id17', '37_id37', '27_id27', '0_id0', '23_id23', '25_id25', '6_id6',
#             '144_id144', '137_id137', '114_id114', '105_id105', '53_id53',
#             '51_id51', '62_id62', '65_id65', '49_id49', '8_id8', '39_id39', '2_id2',
#             '100_id100', '108_id108', '103_id103', '110_id110', '68_id68', 
#             '52_id52', '67_id67', '64_id64', '11_id11', '21_id21', '41_id41', '4_id4']
# l=[]
# for paciente in pacientes_WMH2017:
#     l.append([int(paciente.split('_')[0]),paciente])
# l.sort()
# l2=[]
# for _,paciente in l:
#     l2.append(paciente´)
pacientes_WMH2017=[
    '0_id0', '2_id2', '4_id4', '6_id6', '8_id8', '11_id11', '17_id17', '19_id19', '21_id21', '23_id23', 
 '25_id25', '27_id27', '29_id29', '31_id31', '33_id33', '35_id35', '37_id37', '39_id39', '41_id41', 
 '49_id49', '50_id50', '51_id51', '52_id52', '53_id53', '54_id54', '55_id55', '56_id56', '57_id57', 
 '58_id58', '59_id59', '60_id60', '61_id61', '62_id62', '63_id63', '64_id64', '65_id65', '66_id66', 
 '67_id67', '68_id68', '69_id69', '100_id100', '101_id101', '102_id102', '103_id103', '104_id104', 
 '105_id105', '106_id106', '107_id107', '108_id108', '109_id109', '110_id110', '112_id112', '113_id113', 
 '114_id114', '115_id115', '116_id116', '126_id126', '132_id132', '137_id137', '144_id144']

#Resultado, NO ME ACUERDO DE ESTO
# training = ['137_id137','101_id101','144_id144','108_id108','110_id110','132_id132','106_id106','114_id114',
#             '104_id104','102_id102','105_id105','115_id115','61_id61','51_id51','68_id68','67_id67','52_id52',
#             '62_id62','69_id69','66_id66','60_id60','50_id50','53_id53','54_id54','4_id4','37_id37','35_id35',
#             '21_id21','33_id33','17_id17','27_id27','23_id23','19_id19','49_id49','2_id2']
# validation = ['109_id109','107_id107','126_id126','65_id65','58_id58','63_id63','31_id31','41_id41','6_id6','29_id29']
# testing = ['112_id112','103_id103','100_id100','116_id116','113_id113','59_id59','55_id55','64_id64',
#            '57_id57','56_id56','11_id11','0_id0','25_id25','39_id39','8_id8']

# WMH2017_trainvaltest = [training,validation,testing]
# WMH2017_input_dim_3D = (160, 192, 160, 2)
# WMH2017_input_dim_2D = (160, 192, 2)

# =============================================================================
# MICCAI2008
# =============================================================================
# ['CHB_train_Case03_id3', 'CHB_train_Case04_id4', 'CHB_train_Case06_id6', 'CHB_train_Case09_id9', 
#  'CHB_train_Case07_id7', 'CHB_train_Case01_id1', 'CHB_train_Case10_id10', 'CHB_train_Case05_id5', 
#  'CHB_train_Case08_id8', 'CHB_train_Case02_id2']
# ['UNC_train_Case07_id17', 'UNC_train_Case10_id20', 'UNC_train_Case08_id18', 'UNC_train_Case09_id19', 
#  'UNC_train_Case01_id11', 'UNC_train_Case03_id13', 'UNC_train_Case04_id14', 'UNC_train_Case02_id12', 
#  'UNC_train_Case05_id15', 'UNC_train_Case06_id16']

folds={1:[],2:[],3:[],4:[],5:[]}
folds[1]=['CHB_train_Case03_id3', 'CHB_train_Case04_id4', 'UNC_train_Case07_id17', 'UNC_train_Case10_id20']
folds[2]=['CHB_train_Case06_id6', 'CHB_train_Case09_id9', 'UNC_train_Case08_id18', 'UNC_train_Case09_id19']
folds[3]=['CHB_train_Case07_id7', 'CHB_train_Case01_id1', 'UNC_train_Case01_id11', 'UNC_train_Case03_id13']
folds[4]=['CHB_train_Case10_id10', 'CHB_train_Case05_id5', 'UNC_train_Case04_id14', 'UNC_train_Case02_id12']
folds[5]=['CHB_train_Case08_id8', 'CHB_train_Case02_id2', 'UNC_train_Case05_id15', 'UNC_train_Case06_id16']
MICCAI2008_5folds = {
    1:[folds[2]+folds[3]+folds[4]+folds[5], folds[1]],
    2:[folds[1]+folds[3]+folds[4]+folds[5], folds[2]],
    3:[folds[1]+folds[2]+folds[4]+folds[5], folds[3]],
    4:[folds[1]+folds[2]+folds[3]+folds[5], folds[4]],
    5:[folds[1]+folds[2]+folds[3]+folds[4], folds[5]]}