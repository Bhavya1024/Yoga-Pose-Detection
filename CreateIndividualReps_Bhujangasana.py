#!/usr/bin/env python
# coding: utf-8

import os
import random
import time
import pandas as pd
import statsmodels.api as sm
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from scipy.signal import find_peaks, peak_prominences, chirp, peak_widths

from os import walk
from pathlib import Path

# gautam had maybe not accounted for inverse y relation his peaks were actually lows but his overall calculation was correct

USE_CUDA = False
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  





import matplotlib.pyplot as plt


def find_before(val,index,data):
  '''
  This takes val as target value before which other value should be smaller,
  index as which before which it should look
  and data as list.
  '''
  ans = index
  for i in range(index,-1,-1):
    if data[i] > val:
      ans = i
    else:
      break
  return ans

def find_after(val,index,data):
  '''
  This takes val as target value from which other value should be smaller,
  index as which after which it should look
  and data as list.
  '''
  ans = index
  for i in range(index,len(data)):
    if data[i] > val:
      ans = i
    else:
      break
  return ans





def fpeak(data,plot = True,Chitresh=False):
  #  for chitresh only split using kneews
  if Chitresh:
    y1 =1-np.array( data.iloc[:,1+4*25].tolist() )# 1 offset. 1 for y coord ,x,y,z,visibility
  # y2 =1-np.array( data.iloc[:,1+4*16+1].tolist() )# 1 offset. 1 for y coord ,x,y,z,visibility
    y=y1
  else:
    y1 =1-np.array( data.iloc[:,1+4*15+1].tolist() )# 1 offset. 1 for y coord ,x,y,z,visibility
    y2 =1-np.array( data.iloc[:,1+4*16+1].tolist() )# 1 offset. 1 for y coord ,x,y,z,visibility
    y=y1+y2
  # import pdb;pdb.set_trace()

  #   t=np.arange(20)
#   y=np.sin(t)
#   print(y[:30])

  peaks,_ = find_peaks(y,prominence=0.005)
  print(peaks)
  prominences = peak_prominences(y, peaks)[0]
  std_dev = prominences.std()
    
  ## single end start.
  # good_peaks = [0]
  # good_peaks = [[0]]
  good_peaks = []#abhishekj
  threshold = 1*std_dev
  for i in range(len(prominences)):
    if prominences[i]  >threshold:
      # seprate end and start
      key = y[peaks[i]] - ( (prominences[i]) * 0.93 ) # This .9 threshold can be increases to .95 and so on to make the rep complete
#       print(peaks[i])
      print(prominences[i])
      e0 = find_before(key,peaks[i],y)
      good_peaks.append([e0])#abhishek
      s1 = find_after(key,peaks[i],y)
      good_peaks[-1].append(s1)#abhishek
      #### good_peaks[-1].append(e0)

      ### good_peaks.append([s1])

      ## single end start.
      # good_peaks.append(peaks[i])

  # seprate end and start     
  # good_peaks[-1].append(len(y)-1)   # not REQUIRED IN ABHIHSEK


  # if len(good)
  # good_peaks = good_peaks[1:-1] # remove first and last peaks, prone to errors
  ## if single start stop
  # good_peaks.append(len(y))
  print("good peaks count:" , len(good_peaks))
  if plot:
    print("plotting")
    contour_heights = y[peaks] -  prominences
      
    fig,ax=plt.subplots(figsize=(20, 15))
    plt.plot(np.arange(0,len(y)),y,c='y')
    pt=[peaks[i]  for i in range(len(peaks))  if prominences[i]  > threshold]
#     Good peakso only
    # print(good_peaks)
    # print(y[peaks])
    # plt.scatter(pt, y[pt], c= 'b')
    plt.scatter(peaks, y[peaks], c= 'b')
    plt.scatter(np.ravel(good_peaks), y [np.ravel(good_peaks)],s=108)
    # plt.scatter(start_end_x, start_end_y, c= 'y')
      
    plt.vlines(x=peaks, ymin=contour_heights, ymax= y[peaks], color = 'red')
      
      #plt.hlines(*results_half[1:],color = 'purple')
      #plt.hlines(*results_full[1:],color = 'black')
      
    plt.legend(['real','peaks','prominences','half width', 'full width'])
#     plt.close()
    plt.show()

  return good_peaks


f = []
for (root, dirs, filenames) in walk('csv_vis/'):
    for dir in dirs:
        try:
          shutil.rmtree( os.path.join('csv_vis_reps/', dir))
          print(" removed ", os.path.join('csv_vis_reps/', dir,))
        except OSError as e:
          print("Error: %s - %s." % (e.filename, e.strerror))
        # if 'unseen_test_class' not in dir:
        #   continue
        print("dir",dir,filenames)
        # writer = pd.ExcelWriter('data/'+dir+'.xlsx', engine='xlsxwriter')

        sheet_count=0
        csv_count=0
        # writer = pd.ExcelWriter('csv_vis/'+dir+'/'+dr+'.xlsx', engine='xlsxwriter')
        for (_, d2, filenames) in walk(os.path.join(root, dir)):
                for filename in filenames:
                    # if "Chitresh" not in dir:
                    #   print(filename ,"skipped")
                    #   continue
                    mypath  =  os.path.join('csv_vis_reps/', dir, filename[:-4])
                    

                    if not os.path.isdir(mypath):
                      os.makedirs(mypath)
                    f.append(os.path.join(dir, filename))
                    filepath = os.path.join(root, dir, filename)
                    print(filepath)

                    # import pdb;pdb.set_trace()
                    df=pd.read_csv(filepath)
                    if 'Chitresh' in filepath and 'new_cor' not in filepath:
                      peaks = fpeak( df,plot=False,Chitresh=True)
                    else:
                      peaks = fpeak( df,plot=False,Chitresh=False)
                    pre=[]
                    file_index = 1
                    for i in range(len(peaks)-1):
                        # inpt = df.iloc[peaks[i],:]
                        pre_val=[]
                        # import pdb;pdb.set_trace()
                        temp =  df.iloc[peaks[i][0]:peaks[i][1]]
                        # temp =  df.iloc[peaks[i]:peaks[i+1]]#single start end

                        cols = []
                        for t in range(33):
                          cols.append('x_'+str(t))
                        # if np.sum(temp.loc[:,'x_0']) > np.sum(temp.loc[:,'x_23']):
                        #   print("Have to reverse ",filepath)
                        #   for col in cols:
                        #     temp.loc[:,col] = 1- temp.loc[:,col]
                        ## if single start stop
                        # temp =  df.iloc[peaks[i]:peaks[i+1]]
                        # temp.to_csv(mypath+str(csv_count), index=False)
                        # csv_count+=1
                        print("dest path",os.path.join(mypath,str(i)))
                        temp.to_csv(os.path.join(mypath,str(i)), index=False)
                    if len(peaks) == 0 :
                      print("dest path",os.path.join(mypath,str(0)))
                      df.to_csv(os.path.join(mypath,str(0)), index=False)


