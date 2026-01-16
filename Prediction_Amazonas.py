
"""
@author: Administrator 

"""

import sys
sys.path.insert(0, '../../Utilities/')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import pandas        
import math          
from math import gamma               
from scipy.integrate import odeint   
import matplotlib.dates as mdates    
import tensorflow as tf              
import numpy as np                   
from numpy import *                   
import matplotlib.pyplot as plt       
import scipy.io                       
from scipy.interpolate import griddata             
import time                                        
from itertools import product, combinations                   
from mpl_toolkits.mplot3d import Axes3D                      
from mpl_toolkits.mplot3d.art3d import Poly3DCollection       
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec                          
import datetime                                                 
from pyDOE import lhs                                           
start_time = time.time()                                        


from datetime import datetime  
now = datetime.now()    


# Load Data
data_frame = pandas.read_csv('Data/dengue_Amazonas_24250525.csv')
Ih_new_star = data_frame['Ihnew']  
Ih_sum_star = data_frame['Ihcum']  
Ih_sum_star = Ih_sum_star.to_numpy(dtype=np.float64)
Ih_new_star = Ih_new_star.to_numpy(dtype=np.float64) 
Ih_new_star = Ih_new_star.reshape([len(Ih_new_star), 1]) 
Ih_sum_star = Ih_sum_star.reshape([len(Ih_sum_star), 1])
t_star = np.arange(len(Ih_new_star))
t_star = t_star.reshape([len(t_star), 1])
Nh = 3952262
Nm=8000000
eta=0.988


data_frame_ture = pandas.read_csv('Data/Amazonas_pre.csv') 
Ih_new_ture = data_frame_ture['Ihnew']  
Ih_sum_ture = data_frame_ture['Ihcum'] 
Ih_new_ture = Ih_new_ture.to_numpy(dtype=np.float64)
Ih_sum_ture = Ih_sum_ture.to_numpy(dtype=np.float64)
Ih_new_ture = Ih_new_ture.reshape([len(Ih_new_ture), 1])
Ih_sum_ture = Ih_sum_ture.reshape([len(Ih_sum_ture), 1])


first_date = np.datetime64('2024-01-07')  
last_date = np.datetime64('2025-05-25') + np.timedelta64(7, 'D')  
first_date_pred = np.datetime64('2025-05-25') 
last_date_pred = np.datetime64('2025-08-24') + np.timedelta64(7, 'D') 

date_total = np.arange(first_date, last_date, np.timedelta64(7, 'D'))[:,None] 
data_mean = np.arange(first_date, last_date, np.timedelta64(7, 'D'))[:,None] 
data_pred = np.arange(first_date_pred, last_date_pred, np.timedelta64(7, 'D'))[:,None] 


sf = 1e-5

# load data
Beta1_PINN = np.loadtxt('dengue_Amazonas31/Train-Results-10-04-set1/Beta1.txt')
Beta2_PINN = np.loadtxt('dengue_Amazonas31/Train-Results-10-04-set1/Beta2.txt')
Mu_PINN = np.loadtxt('dengue_Amazonas31/Train-Results-10-04-set1/Mu.txt')

t_mean = np.arange(len(Beta1_PINN))  


Sh_PINN = np.loadtxt('dengue_Amazonas31/Train-Results-10-04-set1/Sh.txt') 
Sh_PINN = Sh_PINN/sf
Ih_PINN = np.loadtxt('dengue_Amazonas31/Train-Results-10-04-set1/Ih.txt')
Ih_PINN = Ih_PINN/sf
Im_PINN = np.loadtxt('dengue_Amazonas31/Train-Results-10-04-set1/Im.txt')
Im_PINN = Im_PINN/sf
Ih_sum_PINN = np.loadtxt('dengue_Amazonas31/Train-Results-10-04-set1/Ih_sum.txt')
Ih_sum_PINN = Ih_sum_PINN/sf
Ih_new_PINN = np.loadtxt('dengue_Amazonas31/Train-Results-10-04-set1/Ih_new.txt')
Ih_new_PINN = Ih_new_PINN/sf


######################################################################
################ Predicting by sloving forward problem ###############
######################################################################  

Sh_init = float(Sh_PINN[-1])  
Ih_init = float(Ih_PINN[-1])
Im_init = float(Im_PINN[-1])
Ih_sum_init = float(Ih_sum_PINN[-1])
U_init = [Sh_init, Ih_init, Im_init, Ih_sum_init]  


Beta1_pre = np.loadtxt('dengue_Amazonas31/Beta1_pre_12.txt') 
Beta1_pre = np.array(Beta1_pre)  

Beta2_pre = np.loadtxt('dengue_Amazonas31/Beta2_pre_18.txt') 
Beta2_pre = np.array(Beta2_pre)  

Mu_pre = np.loadtxt('dengue_Amazonas31/Mu_pre_18.txt') 
Mu_pre = np.array(Mu_pre)  


t_pred = np.arange(0, len(Beta1_pre))  
t_pred = t_pred.reshape([len(t_pred), 1])  



def ODEs_mean(X, t, xi, Pert):
    Sh, Ih, Im, sumIh = X
    time_index = int(round(t))  
    if time_index < 0 or time_index >= len(Beta1_pre):
        beta1_val = Beta1_pre[-1]
    else:
        beta1_val = Beta1_pre[time_index]
    if time_index < 0 or time_index >= len(Beta2_pre):
        beta2_val = Beta2_pre[-1]
    else:
        beta2_val = Beta2_pre[time_index]
    if time_index < 0 or time_index >= len(Mu_pre):
        mu_val = Mu_pre[-1]
    else:
        mu_val = Mu_pre[time_index]
    
    dShdt = -(beta1_val * (1 + xi * Pert)) * Sh * Im / Nh
    dIhdt = (beta1_val * (1 + xi * Pert)) * Sh * Im / Nh - eta * Ih
    dImdt = (beta2_val) * (Nm - Im) * Ih / Nh - (mu_val) * Im
    dsumIhdt = (beta1_val * (1 + xi * Pert)) * Sh * Im / Nh
    return [float(dShdt), float(dIhdt), float(dImdt), float(dsumIhdt)]


t_pred_flat = t_pred.flatten()

Pert0 = 0.1   
Sol_ub_d0 = odeint(ODEs_mean, U_init, t_pred_flat, args = (1,Pert0))  
Sh_ub_d0 = Sol_ub_d0[:,0]
Ih_ub_d0 = Sol_ub_d0[:,1]
Im_ub_d0 = Sol_ub_d0[:,2]
sumIh_ub_d0 = Sol_ub_d0[:,3]
newIh_ub_d0 = sumIh_ub_d0[1:] - sumIh_ub_d0[:-1]  


Sol_lb_d0 = odeint(ODEs_mean, U_init, t_pred_flat, args = (-1,Pert0))    
Sh_lb_d0 = Sol_lb_d0[:,0]
Ih_lb_d0 = Sol_lb_d0[:,1]
Im_lb_d0 = Sol_lb_d0[:,2]
sumIh_lb_d0 = Sol_lb_d0[:,3]
newIh_lb_d0 = sumIh_lb_d0[1:] - sumIh_lb_d0[:-1]


Pert1 = 0.2   
Sol_ub_d1 = odeint(ODEs_mean, U_init, t_pred_flat, args = (1,Pert1))
Sh_ub_d1 = Sol_ub_d1[:,0]
Ih_ub_d1 = Sol_ub_d1[:,1]
Im_ub_d1 = Sol_ub_d1[:,2]
sumIh_ub_d1 = Sol_ub_d1[:,3]
newIh_ub_d1 = sumIh_ub_d1[1:] - sumIh_ub_d1[:-1]


Sol_lb_d1 = odeint(ODEs_mean, U_init, t_pred_flat, args = (-1,Pert1))
Sh_lb_d1 = Sol_lb_d1[:,0]
Ih_lb_d1 = Sol_lb_d1[:,1]
Im_lb_d1 = Sol_lb_d1[:,2]
sumIh_lb_d1 = Sol_lb_d1[:,3]
newIh_lb_d1 = sumIh_lb_d1[1:] - sumIh_lb_d1[:-1]


Sol_mean = odeint(ODEs_mean, U_init, t_pred_flat, args = (0,0))  
Sh_mean = Sol_mean[:,0]
Ih_mean = Sol_mean[:,1]
Im_mean = Sol_mean[:,2]
sumIh_mean = Sol_mean[:,3]
newIh_mean = sumIh_mean[1:] - sumIh_mean[:-1]


######################################################################
############################# Save the results #######################
######################################################################

current_directory = os.getcwd()
relative_path = '/dengue_Amazonas31/Prediction-Results2/'
save_results_to = current_directory + relative_path
if not os.path.exists(save_results_to):
    os.makedirs(save_results_to)    

np.savetxt(save_results_to + 'Sh_mean.txt', Sh_mean.reshape((-1,1)))
np.savetxt(save_results_to + 'Ih_mean.txt', Ih_mean.reshape((-1,1)))
np.savetxt(save_results_to + 'Im_mean.txt', Im_mean.reshape((-1,1)))
np.savetxt(save_results_to + 'newIh_mean.txt', newIh_mean.reshape((-1,1)))
np.savetxt(save_results_to + 'sumIh_mean.txt', sumIh_mean.reshape((-1,1)))


np.savetxt(save_results_to + 'Sh_ub_d0.txt', Sh_ub_d0.reshape((-1,1)))
np.savetxt(save_results_to + 'Ih_ub_d0.txt', Ih_ub_d0.reshape((-1,1)))
np.savetxt(save_results_to + 'Im_ub_d0.txt', Im_ub_d0.reshape((-1,1)))
np.savetxt(save_results_to + 'newIh_ub_d0.txt', newIh_ub_d0.reshape((-1,1)))
np.savetxt(save_results_to + 'sumIh_ub_d0.txt', sumIh_ub_d0.reshape((-1,1)))



np.savetxt(save_results_to + 'Sh_lb_d0.txt', Sh_lb_d0.reshape((-1,1)))
np.savetxt(save_results_to + 'Ih_lb_d0.txt', Ih_lb_d0.reshape((-1,1)))
np.savetxt(save_results_to + 'Im_lb_d0.txt', Im_lb_d0.reshape((-1,1)))
np.savetxt(save_results_to + 'newIh_lb_d0.txt', newIh_lb_d0.reshape((-1,1)))
np.savetxt(save_results_to + 'sumIh_lb_d0.txt', sumIh_lb_d0.reshape((-1,1)))



np.savetxt(save_results_to + 'Sh_ub_d1.txt', Sh_ub_d1.reshape((-1,1)))
np.savetxt(save_results_to + 'Ih_ub_d1.txt', Ih_ub_d1.reshape((-1,1)))
np.savetxt(save_results_to + 'Im_ub_d1.txt', Im_ub_d1.reshape((-1,1)))
np.savetxt(save_results_to + 'newIh_ub_d1.txt', newIh_ub_d1.reshape((-1,1)))
np.savetxt(save_results_to + 'sumIh_ub_d1.txt', sumIh_ub_d1.reshape((-1,1)))


np.savetxt(save_results_to + 'Sh_lb_d1.txt', Sh_lb_d1.reshape((-1,1)))
np.savetxt(save_results_to + 'Ih_lb_d1.txt', Ih_lb_d1.reshape((-1,1)))
np.savetxt(save_results_to + 'Im_lb_d1.txt', Im_lb_d1.reshape((-1,1)))
np.savetxt(save_results_to + 'newIh_lb_d1.txt', newIh_lb_d1.reshape((-1,1)))
np.savetxt(save_results_to + 'sumIh_lb_d1.txt', sumIh_lb_d1.reshape((-1,1)))



######################################################################
############################# Plotting ###############################
######################################################################  

plt.rc('font', size=40)         


fig, ax = plt.subplots()
ax.plot(data_mean, Beta1_PINN, 'k-', lw=4, label='PINN-Training')
ax.plot(data_pred[:-1].flatten(), Beta1_pre, 'm--', lw=4, label='Prediction-mean')
plt.fill_between(data_pred[:-1].flatten(), \
                 Beta1_pre*(1.1), \
                 Beta1_pre*(0.9), \
                 facecolor=(0.1,0.2,0.5,0.3), interpolate=True, label='Prediction-std-(10%)')

plt.fill_between(data_pred[:-1].flatten(), \
                 Beta1_pre*(1.2), \
                 Beta1_pre*(0.8), \
                 facecolor=(0.1,0.5,0.8,0.3), interpolate=True, label='Prediction-std-(20%)')


ax.tick_params(direction='out', axis='x', labelsize = 40)
ax.tick_params(direction='out', axis='y', labelsize = 40)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=35, ncol = 1, loc = 'best')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 40)
plt.rc('font', size=40)
ax.grid(True)
ax.set_ylabel(r'$\beta_{1}$', fontsize = 80)
fig.set_size_inches(w=25, h=12.5)
plt.savefig(save_results_to +'Beta1.pdf', dpi=300)
plt.savefig(save_results_to +'Beta1.png', dpi=300)



for i in [1,2]:
    fig, ax = plt.subplots()
    plt.fill_between(data_pred[:-2].flatten(), \
                      newIh_lb_d0.flatten(), newIh_ub_d0.flatten(), \
                      facecolor=(0.9, 0.6, 0.2, 0.3), interpolate=True, label='Prediction-std-(10%)')
    plt.fill_between(data_pred[:-2].flatten(), \
                      newIh_lb_d1.flatten(), newIh_ub_d1.flatten(), \
                      facecolor=(0.95, 0.7, 0.3, 0.3), interpolate=True, label='Prediction-std-(20%)')


    if i==1:
        ax.plot(data_pred[:-2], newIh_mean, 'm--', lw=4, label='Prediction-mean')
        ax.plot(date_total, Ih_new_star, 'ro', lw=4, markersize=8, label='Data')
        ax.plot(data_pred[1:-2], Ih_new_ture[:-2], 'ro', lw=4, markersize=8) 
        ax.plot(data_mean[1:], Ih_new_PINN, 'k-', lw=4, label='PINN-Training')

    if i==2:
        ax.plot(data_pred[:-2], newIh_mean, 'm--', lw=7, label='Prediction-mean')
        ax.plot(data_pred[1:-2], Ih_new_ture[:-2], 'ro', lw=4, markersize=8, label='Data')


    ax.tick_params(direction='out', axis='x', labelsize = 40)
    ax.tick_params(direction='out', axis='y', labelsize = 40)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 40)
    ax.tick_params(axis='y', labelsize = 40)
    plt.rc('font', size=40)
    ax.grid(True)
    ax.set_ylabel('Weekly infectious cases', fontsize = 40)
    fig.set_size_inches(w=25, h=12.5)
    if i==1:
        plt.savefig(save_results_to + 'new_cases.pdf', dpi=300)
        plt.savefig(save_results_to + 'new_cases.png', dpi=300)
    if i==2:
        plt.savefig(save_results_to + 'new_cases_zoom.pdf', dpi=300)
        plt.savefig(save_results_to + 'new_cases_zoom.png', dpi=300)



for i in [1,2]:
    fig, ax = plt.subplots()
    plt.fill_between(data_pred[:-1].flatten(), \
                      sumIh_lb_d0.flatten(), sumIh_ub_d0.flatten(), \
                      facecolor=(0.9, 0.6, 0.2, 0.3), interpolate=True, label='Prediction-std-(10%)')
    plt.fill_between(data_pred[:-1].flatten(), \
                      sumIh_lb_d1.flatten(), sumIh_ub_d1.flatten(), \
                      facecolor=(0.95, 0.7, 0.3, 0.3), interpolate=True, label='Prediction-std-(20%)')


    if i==1:
        ax.plot(data_pred[:-1], sumIh_mean, 'm--', lw=4, label='Prediction-mean')
        ax.plot(date_total, Ih_sum_star, 'ro', lw=4, markersize=8, label='Data')
        ax.plot(data_pred[1:-1], Ih_sum_ture[:-1], 'ro', lw=4, markersize=8, label='Data')
        ax.plot(data_mean, Ih_sum_PINN, 'k-', lw=4, label='VOCs-INN-Training')

    if i==2:
        ax.plot(data_pred[:-1], sumIh_mean, 'm--', lw=7, label='Prediction-mean')
        ax.plot(data_pred[1:-1], Ih_sum_ture[:-1], 'ro', lw=4, markersize=8, label='Data')


    ax.tick_params(direction='out', axis='x', labelsize = 40)
    ax.tick_params(direction='out', axis='y', labelsize = 40)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=30)
    ax.legend(fontsize=35, ncol = 1, loc = 'best')
    ax.tick_params(axis='x', labelsize = 40)
    ax.tick_params(axis='y', labelsize = 40)
    plt.rc('font', size=40)
    ax.grid(True)
    ax.set_ylabel('($\mathbf{Ih}^{cum}$)', fontsize = 40)
    fig.set_size_inches(w=25, h=12.5)
    if i==1:
        plt.savefig(save_results_to + 'sum_cases.pdf', dpi=300)
        plt.savefig(save_results_to + 'sum_cases.png', dpi=300)
    if i==2:
        plt.savefig(save_results_to + 'sum_cases_zoom.pdf', dpi=300)
        plt.savefig(save_results_to + 'sum_cases_zoom.png', dpi=300)

