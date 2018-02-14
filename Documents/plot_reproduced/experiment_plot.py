#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 10:44:59 2017

@author: yingxuezhang
"""
import matplotlib.pyplot as plt
import numpy as np
import re
#================================Load point net data==================================
folder = 'point_net_log/'
pointnet_loss = []
pointnet_acc = []
pointnet_mean_acc = []
matchers = ['eval mean loss', 'eval accuracy', 'eval avg class acc'] 

filenames= ['train.sh.o3537','train.sh.o3317','train.sh.o3159','train.sh.o4012','train.sh.o5177']

for filename in filenames:
    with open(folder+filename,'r') as f:
        data = f.readlines()
    record = [s[-9:-1] for s in data if any(xs in s for xs in matchers)]        
    record = [float(i) for i in record]
    record = record[0:750]
    pointnet_loss.append(record[0::3])
    pointnet_acc.append(record[1::3])
    pointnet_mean_acc.append(record[2::3])

x=np.arange(0,210,1)
x=np.array(x)

fig, ax = plt.subplots(figsize=(13,7))
ax.grid()
ax.xaxis.set_ticks(x[::25])
ax.set_ylabel('Loss', fontweight='bold',fontsize=22)
ax.set_xlabel('Epoch', fontweight='bold', fontsize=22)

ax.plot(x,pointnet_loss[0][0:210],linestyle = '-',color='red', markersize=4,linewidth = 3,label = 'PointNet')
for i in pointnet_loss[1:]:
    ax.plot(x,i[0:210],linestyle = '-',color='red', markersize=4,linewidth = 3)
  

def load_log_data(folder, filenames):
    loss_record = []
    mean_class_acc_record = []
    mean_acc_record = []
    matchers = ['the average acc among 40 class','===========average loss and acc for this epoch']
    for filename in filenames:
        with open(folder+filename,'r') as f:
            data = f.readlines()
            
        record = [s for s in data if any(xs in s for xs in matchers)]
        mean_acc = record[0::2]
        acc_loss = record[1::2]
        
        mean_class_acc = [float(re.findall(r"[-+]?\d*\.\d+|\d+",i)[1]) for i in mean_acc]
        loss =[float(re.findall(r"[-+]?\d*\.\d+|\d+",i)[0]) for i in acc_loss]
        acc =[float(re.findall(r"[-+]?\d*\.\d+|\d+",i)[1]) for i in acc_loss]
        loss_record.append(loss[0:210])
        mean_acc_record.append(acc[0:210])
        mean_class_acc_record.append(mean_class_acc[0:210])
    
    return loss_record, mean_acc_record, mean_class_acc_record


#=====================================load PointGCN (global pooling) log==========================    
folder = 'gcn_log/'
filenames = ['gcn_basd_pc.sh.o3749','gcn_basd_pc.sh.o3774','gcn_basd_pc.sh.o3799','gcn_basd_pc.sh.o3816','gcn_basd_pc.sh.o3831']
gcn_loss, gcn_acc, gcn_mean_acc = load_log_data(folder, filenames)

#=====================================load PointGCN (multi-res pooling) log==========================
folder = 'multi_res_log/'
filenames = ['multi_res_gcn.sh.o3814','multi_res_gcn.sh.o3891','multi_res_gcn.sh.o3947','multi_res_gcn.sh.o3956','multi_res_gcn.sh.o4015']
multi_res_gcn_loss, multi_res_gcn_acc, multi_res_gcn_mean_acc = load_log_data(folder, filenames)

#=====================================load PointGCN log(global pooling with weighting scheme)  ==========================
folder = 'gcn_w_log/'
filenames = ['gcn_basd_pc.sh.o4201','gcn_basd_pc.sh.o4196','gcn_basd_pc.sh.o4190','gcn_basd_pc.sh.o4183','gcn_basd_pc.sh.o4180']
w_gcn_loss, w_gcn_acc, w_gcn_mean_acc = load_log_data(folder, filenames)

#=====================================load PointGCN log (multi-res pooling with weighting scheme)==========================
folder = 'w_multi_res_log/'
filenames = ['multi_res_gcn.sh.o3919','multi_res_gcn.sh.o3946','multi_res_gcn.sh.o3982','multi_res_gcn.sh.o4530','multi_res_gcn.sh.o5678']
w_multi_res_gcn_loss, w_multi_res_gcn_acc, w_multi_res_gcn_mean_acc = load_log_data(folder, filenames)
#=================================plot_loss_curve===========================================
x=np.arange(0,210,1)
x=np.array(x)
ax.plot(x, gcn_loss[0],linestyle = '-',color='blue', markersize=4,linewidth = 3,label='PointGCN with global pooling')
for i in gcn_loss[1:]:
    ax.plot(x, i,linestyle = '-',color='blue', markersize=4,linewidth = 3)

ax.plot(x, multi_res_gcn_loss[0],linestyle = '-',color='green', markersize=4,linewidth = 3,label='PointGCN with multi-res pooling')
for i in multi_res_gcn_loss[1:]:
    ax.plot(x, i,linestyle = '-',color='green', markersize=4,linewidth = 3)

for label in ax.get_yticklabels():
    label.set_fontsize(22)  
for label in ax.get_xticklabels():
    label.set_fontsize(22)  
ax.legend(loc=0, prop={'weight': 'bold', 'size': 22})
plt.show()

#======================================plot average acc curve==================================
fig, ax = plt.subplots(figsize=(13,7))
ax.grid()
ax.xaxis.set_ticks(x[::25])
ax.set_ylabel('Accuracy', fontweight='bold',fontsize=22)
ax.set_xlabel('Epoch', fontweight='bold', fontsize=22)

x=np.arange(0,210,5)
x=np.array(x)

#point net
average_pointnet_acc = np.mean(np.asarray(pointnet_acc), axis = 0)
average_pointnet_mean_acc = np.mean(np.asarray(pointnet_mean_acc), axis = 0)
ax.plot(x, average_pointnet_acc[:210:5],linestyle = '-',color='red', markersize=4,linewidth = 3,label='PointNet (overall acc)')
ax.plot(x, average_pointnet_mean_acc[:210:5],linestyle = ':',marker='>',color='red', markersize=4,linewidth = 3,label='PointNet (mean class)')

#global pool
average_global_gcn_acc = np.mean(np.asarray(gcn_acc), axis = 0)
average_global_gcn_mean_acc = np.mean(np.asarray(gcn_mean_acc), axis = 0)
w_average_global_gcn_acc = np.mean(np.asarray(w_gcn_acc), axis = 0)
w_average_global_gcn_mean_acc = np.mean(np.asarray(w_gcn_mean_acc), axis = 0)
ax.plot(x, w_average_global_gcn_acc[::5],linestyle = '-',color='blue', markersize=4,linewidth = 3,label='Proposed model with global pooling (overall acc)')
ax.plot(x, w_average_global_gcn_mean_acc[::5],linestyle = ':',marker='>',color='blue', markersize=4,linewidth = 3,label='Proposed model with global pooling (mean class)')

#multi-res pool
average_multi_res_gcn_acc = np.mean(np.asarray(multi_res_gcn_acc), axis = 0)
average_multi_res_gcn_mean_acc = np.mean(np.asarray(multi_res_gcn_mean_acc), axis = 0)
w_average_multi_res_gcn_acc = np.mean(np.asarray(w_multi_res_gcn_acc), axis = 0)
w_average_multi_res_gcn_mean_acc = np.mean(np.asarray(w_multi_res_gcn_mean_acc), axis = 0)
ax.plot(x, w_average_multi_res_gcn_acc[::5],linestyle = '-',color='green', markersize=4,linewidth = 3,label='Proposed model with multi-res pooling (overall)')
ax.plot(x, w_average_multi_res_gcn_mean_acc[::5],linestyle = ':',marker='>',color='green', markersize=4,linewidth = 3,label='Proposed model with multi-res pooling (mean class)')

for label in ax.get_yticklabels():
    label.set_fontsize(22)  
for label in ax.get_xticklabels():
    label.set_fontsize(22)  
ax.legend(loc=0, prop={'weight': 'bold', 'size': 20})


#===================================overall acc================================
global_acc = []
multi_res_acc = []
pointnet_gcn_acc = []
w_global_acc = []
w_multi_res_acc = []

for i in multi_res_gcn_acc:
    multi_res_acc.append(i[200:210])
multi_res_acc = [item for sublist in multi_res_acc for item in sublist]

for i in w_multi_res_gcn_acc:
    w_multi_res_acc.append(i[200:210])
w_multi_res_acc = [item for sublist in w_multi_res_acc for item in sublist]

for i in gcn_acc:
    global_acc.append(i[200:210])
global_acc = [item for sublist in global_acc for item in sublist]

for i in w_gcn_acc:
    w_global_acc.append(i[200:210])
w_global_acc = [item for sublist in w_global_acc for item in sublist]

for i in pointnet_acc:
    pointnet_gcn_acc.append(i[240:250])
pointnet_gcn_acc = [item for sublist in pointnet_gcn_acc for item in sublist]

from matplotlib import rcParams
data=[global_acc, w_global_acc, multi_res_acc, w_multi_res_acc, pointnet_gcn_acc]
plt.figure()
plt.grid()
plt.boxplot(data)
labelsize = 20
ax.set_ylabel('Accuracy', fontweight='bold',fontsize=22)
ax.set_xlabel('Epoch', fontweight='bold', fontsize=22)
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20 
plt.ylabel('Accuracy (mean instance)',fontsize=20)
plt.xticks([1,2,3,4,5],['PointGCN'+'\n'+'global pooling','PointGCN'+'\n'+'global pooling'+'\n'+'(weighted)','PointGCN'+'\n'+'multi-res pooling', 'PointGCN'+'\n'+'multi-res pooling''\n'+'(weighted)','PointNet'])

plt.xticks(fontsize=16)

print "============ Results from overall accuracy ============="
print 'mean and std for PointGCN (multi-res pooling) is {} and {}'.format(np.mean(w_multi_res_acc), np.std(w_multi_res_acc))
print 'mean and std for wPointGCN (global pooling) is {} and {}'.format(np.mean(w_global_acc), np.std(w_global_acc))
print 'mean and std for PointNet is {} and {}'.format(np.mean(pointnet_gcn_acc), np.std(pointnet_gcn_acc))


#===================================mean class acc================================
converge_pointnet_mean_acc = []
converge_w_multi_res_mean_acc = []
converge_w_global_mean_acc = []

for i in w_multi_res_gcn_mean_acc:
    converge_w_multi_res_mean_acc.append(i[200:210])
converge_w_multi_res_mean_acc = [item for sublist in converge_w_multi_res_mean_acc for item in sublist]

for i in w_gcn_mean_acc:
    converge_w_global_mean_acc.append(i[200:210])
converge_w_global_mean_acc = [item for sublist in converge_w_global_mean_acc for item in sublist]

for i in pointnet_mean_acc:
    converge_pointnet_mean_acc.append(i[240:250])
converge_pointnet_mean_acc = [item for sublist in converge_pointnet_mean_acc for item in sublist]

print "============ Results from mean acc accuracy ============="
print 'mean and std for PointGCN (multi-res pooling) is {} and {}'.format(np.mean(converge_w_multi_res_mean_acc), np.std(converge_w_multi_res_mean_acc))
print 'mean and std for PointGCN (global pooling) is {} and {}'.format(np.mean(converge_w_global_mean_acc), np.std(converge_w_global_mean_acc))
print 'mean and std for PointNet is {} and {}'.format(np.mean(converge_pointnet_mean_acc), np.std(converge_pointnet_mean_acc))



