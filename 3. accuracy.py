# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 15:40:00 2023

@author: alexd
"""
import numpy as np
from sklearn.metrics import confusion_matrix

#accuracy = producer's accuracy

def mask_acc(pred_res, X, y, perm):
  
  x_y = X - y 
  idx = np.array(np.where(x_y != 0))
  idx = np.transpose(idx)
  perm = np.transpose(perm)
  perm = perm[:,1:3]
  #intersection between non-permanent water and gapped pixels
  aset = set([tuple(x) for x in idx])
  bset = set([tuple(x) for x in perm])
  x = np.array([x for x in aset & bset])

  intersection = tuple(e for e in np.transpose(x))
  
  acc = []
  
  a = pred_res
  b = y
  
  Confusion_Matrix = confusion_matrix(a[intersection], b[intersection])
  if len(Confusion_Matrix) == 2:
      a1 = (Confusion_Matrix[1][1]/(Confusion_Matrix[1][1] + Confusion_Matrix[1][0]))
  else:
      a1 = (Confusion_Matrix[2][2]/(Confusion_Matrix[2][2] + Confusion_Matrix[2][1])) 
  acc.append(a1)
    
  return acc

#overestimation = false postive rate

def overestimation(pred_res, X, y, perm):
  
  x_y = X - y 
  idx = np.array(np.where(x_y != 0))
  idx = np.transpose(idx)
  perm = np.transpose(perm)
  perm = perm[:,1:3]
  #intersection between non-permanent water and gapped pixels
  aset = set([tuple(x) for x in idx])
  bset = set([tuple(x) for x in perm])
  x = np.array([x for x in aset & bset])

  intersection = tuple(e for e in np.transpose(x))
  
  acc = []
  
  a = pred_res
  b = y
  
  Confusion_Matrix = confusion_matrix(a[intersection], b[intersection])
  if len(Confusion_Matrix) == 2:
      #a1 = (Confusion_Matrix[0][1]/(Confusion_Matrix[0][1] + Confusion_Matrix[0][0]))
      a1 = (Confusion_Matrix[0][1]/(Confusion_Matrix[0][1] + Confusion_Matrix[0][0]
            + Confusion_Matrix[1][0] + Confusion_Matrix[1][1]))
  else:
      #a1 = (Confusion_Matrix[1][2]/(Confusion_Matrix[1][2] + Confusion_Matrix[1][1])) 
      a1 = (Confusion_Matrix[1][2]/(Confusion_Matrix[1][2] + Confusion_Matrix[1][1]
            + Confusion_Matrix[2][1] + Confusion_Matrix[2][2]))
  acc.append(a1)
    
  return acc


#underestimation = false omission rate
 
def underestimation(pred_res, X, y, perm):
  
  x_y = X - y 
  idx = np.array(np.where(x_y != 0))
  idx = np.transpose(idx)
  perm = np.transpose(perm)
  perm = perm[:,1:3]
  #intersection between non-permanent water and gapped pixels
  aset = set([tuple(x) for x in idx])
  bset = set([tuple(x) for x in perm])
  x = np.array([x for x in aset & bset])

  intersection = tuple(e for e in np.transpose(x))
  
  acc = []
  
  a = pred_res
  b = y
  
  Confusion_Matrix = confusion_matrix(a[intersection], b[intersection])
  if len(Confusion_Matrix) == 2:
      #a1 = (Confusion_Matrix[1][0]/(Confusion_Matrix[1][0] + Confusion_Matrix[0][0]))
      a1 = (Confusion_Matrix[1][0]/(Confusion_Matrix[0][1] + Confusion_Matrix[0][0]
            + Confusion_Matrix[1][0] + Confusion_Matrix[1][1]))
  else:
      #a1 = (Confusion_Matrix[2][1]/(Confusion_Matrix[2][1] + Confusion_Matrix[1][1])) 
      a1 = (Confusion_Matrix[2][1]/(Confusion_Matrix[1][2] + Confusion_Matrix[1][1]
            + Confusion_Matrix[2][1] + Confusion_Matrix[2][2]))
  acc.append(a1)
    
  return acc