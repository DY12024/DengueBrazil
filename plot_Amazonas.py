"""
Created on Mon Oct  6 19:49:37 2025
@author: 
"""

import sys
sys.path.insert(0, '../../Utilities/')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import pandas       
import matplotlib.dates as mdates    
import numpy as np                   
import matplotlib.pyplot as plt  
from matplotlib.dates import date2num
from matplotlib.ticker import FormatStrFormatter

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



Beta1_pre = np.loadtxt('dengue_Amazonas31/Beta1_pre_12.txt') 
Beta1_pre = np.array(Beta1_pre)


Sh_mean = np.loadtxt('dengue_Amazonas31/Prediction-Results2/Sh_mean.txt') 
Ih_mean=np.loadtxt('dengue_Amazonas31/Prediction-Results2/Ih_mean.txt')
Im_mean=np.loadtxt('dengue_Amazonas31/Prediction-Results2/Im_mean.txt')
newIh_mean=np.loadtxt('dengue_Amazonas31/Prediction-Results2/newIh_mean.txt')
sumIh_mean=np.loadtxt('dengue_Amazonas31/Prediction-Results2/sumIh_mean.txt')


Sh_ub_d0=np.loadtxt('dengue_Amazonas31/Prediction-Results2/Sh_ub_d0.txt')
Ih_ub_d0=np.loadtxt('dengue_Amazonas31/Prediction-Results2/Ih_ub_d0.txt')
Im_ub_d0=np.loadtxt('dengue_Amazonas31/Prediction-Results2/Im_ub_d0.txt')
newIh_ub_d0=np.loadtxt('dengue_Amazonas31/Prediction-Results2/newIh_ub_d0.txt')
sumIh_ub_d0=np.loadtxt('dengue_Amazonas31/Prediction-Results2/sumIh_ub_d0.txt')


Sh_lb_d0=np.loadtxt('dengue_Amazonas31/Prediction-Results2/Sh_lb_d0.txt')
Ih_lb_d0=np.loadtxt('dengue_Amazonas31/Prediction-Results2/Ih_lb_d0.txt')
Im_lb_d0=np.loadtxt('dengue_Amazonas31/Prediction-Results2/Im_lb_d0.txt')
newIh_lb_d0=np.loadtxt('dengue_Amazonas31/Prediction-Results2/newIh_lb_d0.txt')
sumIh_lb_d0=np.loadtxt('dengue_Amazonas31/Prediction-Results2/sumIh_lb_d0.txt')


Sh_ub_d1=np.loadtxt('dengue_Amazonas31/Prediction-Results2/Sh_ub_d1.txt')
Ih_ub_d1=np.loadtxt('dengue_Amazonas31/Prediction-Results2/Ih_ub_d1.txt')
Im_ub_d1=np.loadtxt('dengue_Amazonas31/Prediction-Results2/Im_ub_d1.txt')
newIh_ub_d1=np.loadtxt('dengue_Amazonas31/Prediction-Results2/newIh_ub_d1.txt')
sumIh_ub_d1=np.loadtxt('dengue_Amazonas31/Prediction-Results2/sumIh_ub_d1.txt')


Sh_lb_d1=np.loadtxt('dengue_Amazonas31/Prediction-Results2/Sh_lb_d1.txt')
Ih_lb_d1=np.loadtxt('dengue_Amazonas31/Prediction-Results2/Ih_lb_d1.txt')
Im_lb_d1=np.loadtxt('dengue_Amazonas31/Prediction-Results2/Im_lb_d1.txt')
newIh_lb_d1=np.loadtxt('dengue_Amazonas31/Prediction-Results2/newIh_lb_d1.txt')
sumIh_lb_d1=np.loadtxt('dengue_Amazonas31/Prediction-Results2/sumIh_lb_d1.txt')


##############################################################################
current_directory = os.getcwd()
relative_path = '/dengue_Amazonas31/Prediction-Results2/'
save_results_to = current_directory + relative_path
if not os.path.exists(save_results_to):
    os.makedirs(save_results_to)    


SAVE_FIG = True


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 40  


fig, ax1 = plt.subplots(figsize=(25, 12.5))


ax1.fill_between(date_total.flatten(), Ih_new_star.flatten(), 
                 color='blue', alpha=0.2, label='True Data')  

ax1.plot(date_total[1:].flatten(), Ih_new_PINN.flatten(), 
         'b-', lw=6, label='PINN-Training')  


ax1.set_ylabel('Weekly infected cases', fontsize=40)  
ax1.tick_params(axis='both', labelsize=40)  


target_dates_str = [
    '24/01/07', '24/03/10', '24/05/12', '24/07/14',
    '24/09/15', '24/11/17', '25/01/19', '25/03/23', '25/05/25'
]



def convert_date_str(dstr):
    yy = '20' + dstr[:2]      
    mm = dstr[3:5]             
    dd = dstr[6:8]             
    return np.datetime64(f'{yy}-{mm}-{dd}')


tick_dates = np.array([convert_date_str(d) for d in target_dates_str], dtype='datetime64')


ax1.set_xticks(tick_dates)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m/%d'))
ax1.set_xticks(tick_dates)
plt.setp(ax1.get_xticklabels(), rotation=30, ha='right',fontsize=30,y=-0.01)


xmin = date_total.min()
xmax = date_total.max()
ax1.set_xlim(xmin, xmax)  
ax1.set_ylim(0, 2500)  


ax2 = ax1.twinx()

ax2.plot(data_mean.flatten(), Beta1_PINN.flatten(), 
         'r--', lw=6, label='$\\beta_1(t)$')  


ax2.set_ylabel('$\\beta_1(t)$', fontsize=40)  
ax2.set_xlabel('Date', fontsize=40)  
ax2.tick_params(axis='y', labelsize=40)  


target_dates_str = [
    '24/01/07', '24/03/10', '24/05/12', '24/07/14',
    '24/09/15', '24/11/17', '25/01/19', '25/03/23', '25/05/25'
]


def convert_date_str(dstr):
    yy = '20' + dstr[:2]       
    mm = dstr[3:5]             
    dd = dstr[6:8]             
    return np.datetime64(f'{yy}-{mm}-{dd}')


tick_dates = np.array([convert_date_str(d) for d in target_dates_str], dtype='datetime64')


ax2.set_xticks(tick_dates)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m/%d'))
ax2.set_xticks(tick_dates)
plt.setp(ax2.get_xticklabels(), rotation=30, ha='right',fontsize=30,y=-0.01)


xmin = date_total.min()
xmax = date_total.max()
ax2.set_xlim(xmin, xmax)  


handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()


handles = handles1 + handles2
labels = labels1 + labels2


ax1.legend(handles, labels, 
           fontsize=30, 
           ncol=1,                     
           loc='upper right',            
           frameon=True,                
           fancybox=True,               
           shadow=True,                 
           borderpad=1,                 
           title_fontsize=30
          )

if SAVE_FIG:
    
    save_path = save_results_to + 'Amazonas.png'  
    plt.savefig(save_path, dpi=300, bbox_inches='tight') 

plt.show()



####################################################################################
Sh_ode = np.loadtxt('dengue_Amazonas31/Train-Results-10-04-set1/Sh_ode.txt') 
Ih_ode = np.loadtxt('dengue_Amazonas31/Train-Results-10-04-set1/Ih_ode.txt')
Im_ode = np.loadtxt('dengue_Amazonas31/Train-Results-10-04-set1/Im_ode.txt')
Ih_new_ode = np.loadtxt('dengue_Amazonas31/Train-Results-10-04-set1/Ih_new_ode.txt')
Ih_sum_ode = np.loadtxt('dengue_Amazonas31/Train-Results-10-04-set1/Ih_sum_ode.txt')



SAVE_FIG = True

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 40  


font = 40
fig, ax = plt.subplots()
ax.plot(date_total.flatten(), Ih_new_star.flatten(), 'ko', marker='o', lw=10, markersize=10, label='Data')
ax.plot(date_total[1:].flatten(), Ih_new_ode.flatten(), 'r-', lw=6, label='Ode solver')

ax.tick_params(direction='out', axis='x', labelsize = 40)
ax.tick_params(direction='out', axis='y', labelsize = 40)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=35, ncol = 1, loc = 'upper right')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 40)
ax.set_ylabel('Weekly infectious cases', fontsize = 40)
fig.set_size_inches(w=25, h=12.5)
    
    
target_dates_str = [
        '24/01/07', '24/03/10', '24/05/12', '24/07/14',
        '24/09/15', '24/11/17', '25/01/19', '25/03/23', '25/05/25'
    ]


def convert_date_str(dstr):
        yy = '20' + dstr[:2]       
        mm = dstr[3:5]             
        dd = dstr[6:8]             
        return np.datetime64(f'{yy}-{mm}-{dd}')

    
tick_dates = np.array([convert_date_str(d) for d in target_dates_str], dtype='datetime64')

   
ax.set_xticks(tick_dates)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m/%d'))
plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=30,y=-0.01)
ax.set_ylim(0, 2500)  

if SAVE_FIG:
    plt.savefig(save_results_to + 'New cases_ode.png', dpi=300)

plt.show()

#########################################################

SAVE_FIG = True

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 40  


font = 40
fig, ax = plt.subplots()
ax.plot(date_total.flatten(), Ih_sum_star.flatten(), 'ko', marker='o', lw=10, markersize=10, label='Data')
ax.plot(date_total.flatten(), Ih_sum_ode.flatten(), 'r-', lw=6, label='Ode solver')
ax.tick_params(direction='out', axis='x', labelsize = 40)
ax.tick_params(direction='out', axis='y', labelsize = 40)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=35, ncol = 1, loc = 'best')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 40)
ax.set_ylabel('Cumulative infected cases', fontsize = 40)
fig.set_size_inches(w=25, h=12.5)
    
    
target_dates_str = [
        '24/01/07', '24/03/10', '24/05/12', '24/07/14',
        '24/09/15', '24/11/17', '25/01/19', '25/03/23', '25/05/25'
    ]


def convert_date_str(dstr):
        yy = '20' + dstr[:2]       
        mm = dstr[3:5]             
        dd = dstr[6:8]             
        return np.datetime64(f'{yy}-{mm}-{dd}')

tick_dates = np.array([convert_date_str(d) for d in target_dates_str], dtype='datetime64')

ax.set_xticks(tick_dates)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m/%d'))
plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=30,y=-0.01)
ax.set_ylim(0, 50000) 

if SAVE_FIG:
    plt.savefig(save_results_to + 'Cumulative_cases_ode.png', dpi=300)


plt.show()


############################################################################################
current_directory = os.getcwd()
relative_path = '/dengue_Amazonas31/Prediction-Results2/'
save_results_to = current_directory + relative_path
if not os.path.exists(save_results_to):
    os.makedirs(save_results_to)    
    


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 40  


fig, ax = plt.subplots()
ax.plot(data_mean, Beta1_PINN, 'k-', lw=6, label='PINN-Training')
ax.plot(data_pred[:-1].flatten(), Beta1_pre, 'm--', lw=6, label='Prediction')
plt.fill_between(data_pred[:-1].flatten(), \
                 Beta1_pre*(1.1), \
                 Beta1_pre*(0.9), \
                 facecolor=(0.9, 0.6, 0.2, 0.3), interpolate=True, label='Prediction-std-(10%)') 

plt.fill_between(data_pred[:-1].flatten(), \
                 Beta1_pre*(1.2), \
                 Beta1_pre*(0.8), \
                 facecolor=(0.95, 0.7, 0.3, 0.3), interpolate=True, label='Prediction-std-(20%)')


ax.tick_params(direction='out', axis='x', labelsize = 40)
ax.tick_params(direction='out', axis='y', labelsize = 40)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=30, ncol = 1, loc = 'upper right')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 40)
plt.rc('font', size=40)
ax.set_ylabel(r'$\beta_{1}(t)$', fontsize = 40)
fig.set_size_inches(w=25, h=12.5)

target_dates_str = [
    '24/01/07', '24/03/10', '24/05/12', '24/07/14',
    '24/09/15', '24/11/17', '25/01/19', '25/03/23', '25/05/25','25/07/27'
]


def convert_date_str(dstr):
    yy = '20' + dstr[:2]       
    mm = dstr[3:5]             
    dd = dstr[6:8]             
    return np.datetime64(f'{yy}-{mm}-{dd}')


tick_dates = np.array([convert_date_str(d) for d in target_dates_str], dtype='datetime64')
ax.set_xticks(tick_dates)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m/%d'))
ax.set_xticks(tick_dates)
plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=30,y=-0.01)
ax.set_ylim(0, 0.4) 

vert_line_x = date2num(np.datetime64('2025-05-25'))
ax.axvline(x=vert_line_x, color='gold', linestyle='-', linewidth=5)   

plt.savefig(save_results_to +'Beta1.pdf', dpi=300)
plt.savefig(save_results_to +'Beta1.png', dpi=300)

#########################################################################################
#New infectious
current_directory = os.getcwd()
relative_path = '/dengue_Amazonas31/Prediction-Results2/'
save_results_to = current_directory + relative_path
if not os.path.exists(save_results_to):
    os.makedirs(save_results_to)    
    
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 40  

fig, ax = plt.subplots()
plt.fill_between(data_pred[:-2].flatten(), \
                      newIh_lb_d0.flatten(), newIh_ub_d0.flatten(), \
                      facecolor=(0.9, 0.6, 0.2, 0.3), interpolate=True, label='Prediction-std-(10%)')
plt.fill_between(data_pred[:-2].flatten(), \
                      newIh_lb_d1.flatten(), newIh_ub_d1.flatten(), \
                      facecolor=(0.95, 0.7, 0.3, 0.3), interpolate=True, label='Prediction-std-(20%)')

ax.plot(data_pred[:-2], newIh_mean, 'm-', lw=6, label='Prediction')
ax.plot(date_total, Ih_new_star, 'ro', lw=6, markersize=8, label='Data')
ax.plot(data_pred[1:-1], Ih_new_ture[:-1], 'ro', lw=6, markersize=8) 
ax.plot(data_mean[1:], Ih_new_PINN, 'k-', lw=6, label='PINN-Training')

ax.tick_params(direction='out', axis='x', labelsize = 40)
ax.tick_params(direction='out', axis='y', labelsize = 40)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=35, ncol = 1, loc = 'upper right')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 40)
plt.rc('font', size=40)
ax.set_ylabel('Weekly infectious cases', fontsize = 40)
fig.set_size_inches(w=25, h=12.5)
ax.set_ylim(0, 2500)    
    
target_dates_str = [
        '24/01/07', '24/03/10', '24/05/12', '24/07/14',
        '24/09/15', '24/11/17', '25/01/19', '25/03/23', '25/05/25','25/07/27'
    ]


def convert_date_str(dstr):
        yy = '20' + dstr[:2]       
        mm = dstr[3:5]             
        dd = dstr[6:8]             
        return np.datetime64(f'{yy}-{mm}-{dd}')

 
tick_dates = np.array([convert_date_str(d) for d in target_dates_str], dtype='datetime64')


ax.set_xticks(tick_dates)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m/%d'))
ax.set_xticks(tick_dates)
plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=30,y=-0.01)


vert_line_x = date2num(np.datetime64('2025-05-25'))
ax.axvline(x=vert_line_x, color='gold', linestyle='-', linewidth=5)
plt.savefig(save_results_to + 'new_cases.pdf', dpi=300)
plt.savefig(save_results_to + 'new_cases.png', dpi=300)


#######################################################################################

#Cumulative Infectious

fig, ax = plt.subplots()
plt.fill_between(data_pred[:-1].flatten(), \
                      sumIh_lb_d0.flatten(), sumIh_ub_d0.flatten(), \
                      facecolor=(0.9, 0.6, 0.2, 0.3), interpolate=True, label='Prediction-std-(10%)')
plt.fill_between(data_pred[:-1].flatten(), \
                      sumIh_lb_d1.flatten(), sumIh_ub_d1.flatten(), \
                      facecolor=(0.95, 0.7, 0.3, 0.3), interpolate=True, label='Prediction-std-(20%)')

ax.plot(data_pred[:-1], sumIh_mean, 'm-', lw=6, label='Prediction')
ax.plot(date_total, Ih_sum_star, 'ro', lw=6, markersize=8, label='Data')
ax.plot(data_pred[1:-1], Ih_sum_ture[:-1], 'ro', lw=6, markersize=8)
ax.plot(data_mean, Ih_sum_PINN, 'k-', lw=6, label='PINN-Training')
        
ax.tick_params(direction='out', axis='x', labelsize = 40)
ax.tick_params(direction='out', axis='y', labelsize = 40)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=30)
ax.legend(fontsize=35, ncol = 1, loc = 'lower right')
ax.tick_params(axis='x', labelsize = 40)
ax.tick_params(axis='y', labelsize = 40)
plt.rc('font', size=40)
ax.set_ylabel('Cumulative infected cases', fontsize = 40)
fig.set_size_inches(w=25, h=12.5)
target_dates_str = [
        '24/01/07', '24/03/10', '24/05/12', '24/07/14',
        '24/09/15', '24/11/17', '25/01/19', '25/03/23', '25/05/25','25/07/27'
    ]


def convert_date_str(dstr):
        yy = '20' + dstr[:2]       
        mm = dstr[3:5]             
        dd = dstr[6:8]             
        return np.datetime64(f'{yy}-{mm}-{dd}')

 
tick_dates = np.array([convert_date_str(d) for d in target_dates_str], dtype='datetime64')
ax.set_xticks(tick_dates)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m/%d'))
ax.set_xticks(tick_dates)

plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=30,y=-0.01)


vert_line_x = date2num(np.datetime64('2025-05-25'))
ax.axvline(x=vert_line_x, color='gold', linestyle='-', linewidth=5)
plt.savefig(save_results_to + 'sum_cases.pdf', dpi=300)
plt.savefig(save_results_to + 'sum_cases.png', dpi=300)
    