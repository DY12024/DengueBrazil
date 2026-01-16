# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 09:27:55 2025

@author: 
"""

import sys 
sys.path.insert(0, '../../Utilities/') 
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import pandas 
import math   
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
from scipy.integrate import odeint  
start_time = time.time()   


class PhysicsInformedNN:
    
    def __init__(self, t_train, Ih_new_train,                   
                 Ih_sum_train,  U0, t_f, lb, ub, Nh, Nm, eta,
                 layers, layers_beta1, layers_beta2, layers_mu,  sf):

        self.Nh = Nh     
        self.Nm = Nm
        self.eta = eta 
        self.sf = sf

        # Data for training
        self.t_train = t_train
        self.Ih_new_train = Ih_new_train
        self.Ih_sum_train = Ih_sum_train
        self.Sh0 = U0[0]
        self.Ih0 = U0[1]
        self.Im0 = U0[2]
        self.t_f = t_f

        # Time division s
        self.M = len(t_f) - 1             
        self.tau = t_f[1] - t_f[0]        

        # Bounds
        self.lb = lb
        self.ub = ub

        # initialize NN
        self.weights, self.biases = self.initialize_NN(layers)
        self.weights_beta1, self.biases_beta1 = self.initialize_NN(layers_beta1)
        self.weights_beta2, self.biases_beta2 = self.initialize_NN(layers_beta2)
        self.weights_mu, self.biases_mu = self.initialize_NN(layers_mu)
        

        # Fixed parameters
        self.Nh = Nh
        self.Nm = Nm
        self.eta = eta

        # tf placeholders and graph
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,log_device_placement=True))
        self.saver = tf.compat.v1.train.Saver()

        # placeholders for inputs
        self.t_u = tf.compat.v1.placeholder(tf.float64, shape=[None, self.t_train.shape[1]])  
        self.Ih_new_u = tf.compat.v1.placeholder(tf.float64, shape=[None, self.Ih_new_train.shape[1]]) 
        self.Ih_sum_u = tf.compat.v1.placeholder(tf.float64, shape=[None, self.Ih_sum_train.shape[1]])
        self.Sh0_u = tf.compat.v1.placeholder(tf.float64, shape=[None, self.Sh0.shape[1]])
        self.Ih0_u = tf.compat.v1.placeholder(tf.float64, shape=[None, self.Ih0.shape[1]]) 
        self.Im0_u = tf.compat.v1.placeholder(tf.float64, shape=[None, self.Im0.shape[1]])
        self.t_tf = tf.compat.v1.placeholder(tf.float64, shape=[None, self.t_f.shape[1]]) 

        # physics informed neural networks
        self.Sh_pred, self.Ih_pred, self.Im_pred, self.Ih_sum_pred= self.net_u(
            self.t_u)

        self.Beta1_pred = self.net_Beta1(self.t_u)
        self.Beta2_pred = self.net_Beta2(self.t_u)
        self.Mu_pred = self.net_Mu(self.t_u)

       
        self.Ih_new_pred = self.Ih_sum_pred[1:, :] - self.Ih_sum_pred[0:-1, :]

        
        self.Sh0_pred = self.Sh_pred[0]
        self.Ih0_pred = self.Ih_pred[0]
        self.Im0_pred = self.Im_pred[0]

        
        # self.S_f, self.I1_f, self.I2_f, self.D_f, self.R_f, self.I1_sum_f, self.I2_sum_f = self.net_f(self.t_u)
        self.Sh_f, self.Ih_f, self.Im_f, self.Ih_sum_f = self.net_f(self.t_tf)
        # self.I1_f, self.I2_f, self.D_f, self.R_f, self.I1_sum_f, self.I2_sum_f, self.con_f = self.net_f(self.t_u)

        
        self.lossU0 = tf.reduce_mean(tf.square(self.Sh0_u - self.Sh0_pred)) + \
                     tf.reduce_mean(tf.square(self.Ih0_u - self.Ih0_pred)) + \
                     tf.reduce_mean(tf.square(self.Im0_u - self.Im0_pred))

        self.lossU =  100*tf.reduce_mean(tf.square(self.Ih_new_u[1:, :] - self.Ih_new_pred)) + \
                      tf.reduce_mean(tf.square(self.Ih_sum_u - self.Ih_sum_pred))

        
        self.lossF = tf.reduce_mean(tf.square(self.Sh_f)) + \
                     tf.reduce_mean(tf.square(self.Ih_f)) + \
                     tf.reduce_mean(tf.square(self.Im_f)) + \
                     tf.reduce_mean(tf.square(self.Ih_sum_f))


        self.loss = 1*self.lossU0 + 20*self.lossU + 50*self.lossF

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method='L-BFGS-B',  
                                                                options={'maxiter': 10000, 
                                                                         'maxfun': 50000, 
                                                                         'maxcor': 100, 
                                                                         'maxls': 100, 
                                                                         'gtol': 1e-8, 
                                                                         'ftol': 1.0 * np.finfo(float).eps}) 

        self.optimizer_Adam = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4) 
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss) 

        
        init = tf.compat.v1.global_variables_initializer() 
        self.sess.run(init)  

        
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):  
            W = self.xavier_init(size=[layers[l], layers[l + 1]])  
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float64), 
                            dtype=tf.float64)  
            weights.append(W)    
            biases.append(b)  
        return weights, biases

    # generating weights
    def xavier_init(self, size):  
        in_dim = size[0]    
        out_dim = size[1]   
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))  
        return tf.Variable(tf.random.truncated_normal(shape =(in_dim,out_dim),stddev=xavier_stddev, dtype=tf.float64),
                           dtype=tf.float64) 

    
    def neural_net(self, t, weights, biases):
        num_layers = len(weights) + 1  

        H = 2.0 * (t - self.lb) / (self.ub - self.lb) - 1.0  
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))  
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)  
        return Y


    def net_u(self, t):
        ShIhSm = self.neural_net(t, self.weights, self.biases) 
        Sh = ShIhSm[:, 0:1] 
        Ih = ShIhSm[:, 1:2]
        Im = ShIhSm[:, 2:3]
        Ih_sum = ShIhSm[:, 3:4]
        return Sh, Ih, Im, Ih_sum

    def net_Beta1(self, t):    
        Beta1 = self.neural_net(t, self.weights_beta1, self.biases_beta1)   
        bound_b = [tf.constant(0.0, dtype=tf.float64), tf.constant(1, dtype=tf.float64)]  
        return bound_b[0] + (bound_b[1] - bound_b[0]) * tf.sigmoid(Beta1)  
        

    def net_Beta2(self, t):  
        Beta2 = self.neural_net(t, self.weights_beta2, self.biases_beta2)
        bound_b = [tf.constant(0.0, dtype=tf.float64), tf.constant(0.1, dtype=tf.float64)]
        return bound_b[0] + (bound_b[1] - bound_b[0]) * tf.sigmoid(Beta2)
        

    def net_Mu(self, t):
        mu = self.neural_net(t, self.weights_mu, self.biases_mu)
        bound_b = [tf.constant(0.0, dtype=tf.float64), tf.constant(0.1, dtype=tf.float64)]
        return bound_b[0] + (bound_b[1] - bound_b[0]) * tf.sigmoid(mu)  
        
        
    
    def net_f(self, t):    
        
        
        beta1 = self.net_Beta1(t)
        beta2 = self.net_Beta2(t)
        mu = self.net_Mu(t)
        

        Sh, Ih, Im, Ih_sum = self.net_u(t)   

          
        Sh_t = tf.gradients(Sh, t, unconnected_gradients='zero')[0]
        Ih_t = tf.gradients(Ih, t, unconnected_gradients='zero')[0]
        Im_t = tf.gradients(Im, t, unconnected_gradients='zero')[0]
        Ih_sum_t = tf.gradients(Ih_sum, t, unconnected_gradients='zero')[0]

      
        f_Sh = Sh_t + (beta1 * Sh * Im) / self.Nh  
        f_Ih = Ih_t -(beta1 * Sh * Im) / self.Nh +  self.eta * Ih
        f_Im = Im_t - (beta2*(self.Nm-Im)*Ih)/ self.Nh+mu*Im
        f_Ih_sum=Ih_sum_t-(beta1 * Sh * Im) / self.Nh

        return f_Sh, f_Ih, f_Im, f_Ih_sum
        



    def callback(self, loss, lossU0, lossU, lossF):  
        total_records_LBFGS.append(np.array([loss, lossU0, lossU, lossF]))  
        print('Loss: %.3e, LossU0: %.3e, LossU: %.3e, LossF: %.3e'
              % (loss, lossU0, lossU, lossF))   

    def train(self, nIter):  

        tf_dict = {self.t_u: self.t_train, self.t_tf: self.t_f,
                   self.Ih_new_u: self.Ih_new_train, 
                   self.Ih_sum_u: self.Ih_sum_train, 
                   self.Sh0_u: self.Sh0, self.Ih0_u: self.Ih0, 
                   self.Im0_u: self.Im0}

        start_time = time.time()  
        for it in range(nIter + 1):  
            self.sess.run(self.train_op_Adam, tf_dict)  

            
            if it % 100 == 0:
                elapsed = time.time() - start_time  
                loss_value = self.sess.run(self.loss, tf_dict)  
                lossU0_value = self.sess.run(self.lossU0, tf_dict)
                lossU_value = self.sess.run(self.lossU, tf_dict)
                lossF_value = self.sess.run(self.lossF, tf_dict)
                total_records.append(np.array([it, loss_value, lossU0_value, lossU_value, lossF_value])) 
                print('It: %d, Loss: %.3e, LossU0: %.3e, LossU: %.3e, LossF: %.3e, Time: %.2f' %
                      (it, loss_value, lossU0_value, lossU_value, lossF_value, elapsed))  
                start_time = time.time() 

        if LBFGS:  
            self.optimizer.minimize(self.sess, 
                                    feed_dict=tf_dict,  
                                    fetches=[self.loss, self.lossU0, self.lossU, self.lossF],    
                                    loss_callback=self.callback)  

    def predict_data(self, t_star):  

        tf_dict = {self.t_u: t_star}  
        
        Sh = self.sess.run(self.Sh_pred, tf_dict)  
        Ih = self.sess.run(self.Ih_pred, tf_dict)
        Im = self.sess.run(self.Im_pred, tf_dict)
        Ih_sum = self.sess.run(self.Ih_sum_pred, tf_dict)
        Ih_new = self.sess.run(self.Ih_new_pred, tf_dict)
        return Sh, Ih, Im, Ih_new, Ih_sum  

    def predict_par(self, t_star):   
        
        tf_dict = {self.t_u: t_star}   
        Beta1 = self.sess.run(self.Beta1_pred, tf_dict) 
        Beta2 = self.sess.run(self.Beta2_pred, tf_dict)
        Mu = self.sess.run(self.Mu_pred, tf_dict)
        return Beta1, Beta2, Mu   


############################################################
if __name__ == "__main__":   

    
    layers = [1] + 5 * [32] + [4]  
    layers_beta1 = [1] + 5 * [64] + [1]
    layers_beta2 = [1] + 5 * [64] + [1]
    layers_mu = [1] + 5 * [64] + [1]

    # Load data
    data_frame = pandas.read_csv('Data\dengue_Amazonas_24250525.csv')  
    Ih_new_star = data_frame['Ihnew']  
    Ih_sum_star = data_frame['Ihcum']

    
    Ih_new_star = Ih_new_star.to_numpy(dtype=np.float64)  
    Ih_sum_star = Ih_sum_star.to_numpy(dtype=np.float64)
    Ih_new_star = Ih_new_star.reshape([len(Ih_new_star), 1])  
    Ih_sum_star = Ih_sum_star.reshape([len(Ih_sum_star), 1])
    t_star = np.arange(len(Ih_new_star))   
    t_star = t_star.reshape([len(t_star), 1])  
    Nh = 3952262
    Nm=8000000
    eta=0.988
    X0 = [3e6, 1910, 1e4, 1910]
    X0 = np.array(X0)  
    

    
    lb = t_star.min(0) 
    ub = t_star.max(0)  

    # Initial conditions
    Ih0_new = Ih_new_star[0:1, :]  
    Ih0_sum = Ih_sum_star[0:1, :]

    # Scaling
    sf = 1e-5 
    Nh = Nh * sf  
    Nm=Nm* sf
    
    Ih_new_star = Ih_new_star * sf
    Ih_sum_star = Ih_sum_star * sf
    X0 = X0*sf

    # Initial conditions
    Sh0 = np.array([[3e6]]) * sf
    Ih0 = Ih_new_star[0:1, :]
    Im0 = np.array([[1e4]]) * sf
    U0 = [Sh0, Ih0, Im0]  
    N_f = 500   
    t_f = lb + (ub - lb) * lhs(1, N_f)   

    ######################################################################
    ######################## Training and Predicting #####################
    ######################################################################
    t_train = t_star  
    Ih_new_train = Ih_new_star
    Ih_sum_train = Ih_sum_star

    from datetime import datetime
    
    now = datetime.now()  
    dt_string = now.strftime("%m-%d")   

    # save results
    current_directory = os.getcwd()  
    for j in range(1):    
        casenumber = 'set' + str(j + 1)   

        relative_path_results = '/dengue_Amazonas31/Train-Results-' + dt_string + '-' + casenumber + '/'  
        save_results_to = current_directory + relative_path_results  
        if not os.path.exists(save_results_to):  
            os.makedirs(save_results_to)

        relative_path = '/dengue_Amazonas31/Train-model-' + dt_string + '-' + casenumber + '/' 
        save_models_to = current_directory + relative_path  
        if not os.path.exists(save_models_to):
            os.makedirs(save_models_to)


        ####model
        total_records = []  
        total_records_LBFGS = []   
        model = PhysicsInformedNN(t_train, Ih_new_train, 
                                  Ih_sum_train,  U0, t_f, lb, ub, Nh, Nm, eta,
                                  layers, layers_beta1, layers_beta2, layers_mu, sf) 
        ####Training
        LBFGS=True  
        # LBFGS = False
        model.train(10000)  

        ####save model
        model.saver.save(model.sess, save_models_to + "model.ckpt")  

        ####Predicting
        Sh, Ih, Im, Ih_new, Ih_sum= model.predict_data(t_star) 
        Beta1, Beta2, Mu = model.predict_par(t_star) 
        import datetime

        end_time = time.time()
        print(datetime.timedelta(seconds=int(end_time - start_time))) 

        ##################save data and plot

        ####save data
        np.savetxt(save_results_to + 'Sh.txt', Sh.reshape((-1, 1))) 
        np.savetxt(save_results_to + 'Ih.txt', Ih.reshape((-1, 1)))
        np.savetxt(save_results_to + 'Im.txt', Im.reshape((-1, 1)))
        np.savetxt(save_results_to + 'Ih_new.txt', Ih_new.reshape((-1, 1)))
        np.savetxt(save_results_to + 'Ih_sum.txt', Ih_sum.reshape((-1, 1)))

        
        np.savetxt(save_results_to + 't_star.txt', t_star.reshape((-1, 1)))
        np.savetxt(save_results_to + 'Beta1.txt', Beta1.reshape((-1, 1)))
        np.savetxt(save_results_to + 'Beta2.txt', Beta2.reshape((-1, 1)))
        np.savetxt(save_results_to + 'Mu.txt', Mu.reshape((-1, 1)))

        
        N_Iter = len(total_records)  
        iteration = np.asarray(total_records)[:, 0]  
        loss_his = np.asarray(total_records)[:, 1]    
        loss_his_u0 = np.asarray(total_records)[:, 2]  
        loss_his_u = np.asarray(total_records)[:, 3]   
        loss_his_f = np.asarray(total_records)[:, 4] 

       
        if LBFGS:   
            N_Iter_LBFGS = len(total_records_LBFGS)   
            iteration_LBFGS = np.arange(N_Iter_LBFGS) + N_Iter * 100  
            loss_his_LBFGS = np.asarray(total_records_LBFGS)[:, 0]   
            loss_his_u0_LBFGS = np.asarray(total_records_LBFGS)[:, 1]
            loss_his_u_LBFGS = np.asarray(total_records_LBFGS)[:, 2]
            loss_his_f_LBFGS = np.asarray(total_records_LBFGS)[:, 3]

        
        np.savetxt(save_results_to + 'iteration.txt', iteration.reshape((-1, 1)))
        np.savetxt(save_results_to + 'loss_his.txt', loss_his.reshape((-1, 1)))
        np.savetxt(save_results_to + 'loss_his_u0.txt', loss_his_u0.reshape((-1, 1)))
        np.savetxt(save_results_to + 'loss_his_u.txt', loss_his_u.reshape((-1, 1)))
        np.savetxt(save_results_to + 'loss_his_f.txt', loss_his_f.reshape((-1, 1)))

        if LBFGS:  
            np.savetxt(save_results_to + 'iteration_LBFGS.txt', iteration_LBFGS.reshape((-1, 1)))
            np.savetxt(save_results_to + 'loss_his_LBFGS.txt', loss_his_LBFGS.reshape((-1, 1)))
            np.savetxt(save_results_to + 'loss_his_u0_LBFGS.txt', loss_his_u0_LBFGS.reshape((-1, 1)))
            np.savetxt(save_results_to + 'loss_his_u_LBFGS.txt', loss_his_u_LBFGS.reshape((-1, 1)))
            np.savetxt(save_results_to + 'loss_his_f_LBFGS.txt', loss_his_f_LBFGS.reshape((-1, 1)))


        ############################# Plotting ###############################
        ######################################################################
        SAVE_FIG = True  

        # History of loss
        font = 24  
        fig, ax = plt.subplots()   
        plt.locator_params(axis='x', nbins=6)  
        plt.locator_params(axis='y', nbins=6)
        plt.tick_params(axis='y', which='both', labelleft='on', labelright='off') 
        plt.xlabel('$iteration$', fontsize=font)
        plt.ylabel('$loss values$', fontsize=font)
        plt.yscale('log')  
        plt.grid(True)  
        plt.plot(iteration, loss_his, label='$loss$')   
        plt.plot(iteration, loss_his_u0, label='$loss_{u0}$')
        plt.plot(iteration, loss_his_u, label='$loss_u$')
        plt.plot(iteration, loss_his_f, label='$loss_f$')
        if LBFGS:
            plt.plot(iteration_LBFGS, loss_his_LBFGS, label='$loss-LBFGS$')
            plt.plot(iteration_LBFGS, loss_his_u0_LBFGS, label='$loss_{u0}-LBFGS$')
            plt.plot(iteration_LBFGS, loss_his_u_LBFGS, label='$loss_u-LBFGS$')
            plt.plot(iteration_LBFGS, loss_his_f_LBFGS, label='$loss_f-LBFGS$')
        plt.legend(loc="upper right", fontsize=24, ncol=4)  
        plt.legend()    
        ax.tick_params(axis='both', labelsize=24)  
        fig.set_size_inches(w=13, h=6.5)  
        if SAVE_FIG:   
            plt.savefig(save_results_to + 'History_loss.png', dpi=300)   


        
        font = 24   
        fig, ax = plt.subplots()   
        ax.plot(t_star, Ih / sf, 'r-', lw=2, label='VOCs-INN')  
        ax.legend(fontsize=22)   
        ax.tick_params(axis='both', labelsize=24)  
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3, 3))  
        ax.grid(True)
        ax.set_xlabel('Weeks', fontsize=font)
        ax.set_ylabel('$I_{h}$', fontsize=font)
        fig.set_size_inches(w=13, h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to + 'Ih.png', dpi=300)

       

        # New cases
        font = 24
        fig, ax = plt.subplots()
        ax.plot(t_star, Ih_new_star / sf, 'k--', marker='o', lw=2, markersize=5, label='Data')
        ax.plot(t_star[1:], Ih_new / sf, 'r-', lw=2, label='PINN')
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize=24)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3, 3))
        ax.grid(True)
        ax.set_xlabel('Weeks', fontsize=font)
        ax.set_ylabel('Weekly number of cases', fontsize=font)
        fig.set_size_inches(w=13, h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to + 'fitting Ih.png', dpi=300)



        # Cumulative  cases
        font = 24
        fig, ax = plt.subplots()
        ax.plot(t_star, Ih_sum_star / sf, 'k--', marker='o', lw=2, markersize=5, label='Data')
        ax.plot(t_star, Ih_sum / sf, 'r-', lw=2, label='PINN')
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize=24)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3, 3))
        ax.grid(True)
        ax.set_xlabel('Weeks', fontsize=font)
        ax.set_ylabel('Cumulative cases', fontsize=font)
        fig.set_size_inches(w=13, h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to + 'Cumulative_cases.png', dpi=300)

       

        # Beta1 curve
        font = 24
        fig, ax = plt.subplots()
        ax.plot(t_star, Beta1, 'r-', lw=2, label='PINN')
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize=24)
        ax.grid(True)
        ax.set_xlabel('Weeks', fontsize=font)
        ax.set_ylabel('$beta_{1}$', fontsize=font)
        fig.set_size_inches(w=13, h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to + 'beta1.png', dpi=300)

        # Beta2 curve
        font = 24
        fig, ax = plt.subplots()
        ax.plot(t_star, Beta2, 'r-', lw=2, label='PINN')
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize=24)
        ax.grid(True)
        ax.set_xlabel('Weeks', fontsize=font)
        ax.set_ylabel('$beta_{2}$', fontsize=font)
        fig.set_size_inches(w=13, h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to + 'beta2.png', dpi=300)

        
        font = 24
        fig, ax = plt.subplots()
        ax.plot(t_star, Mu, 'r-', lw=2, label='PINN')
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize=24)
        ax.grid(True)
        ax.set_xlabel('Weeks', fontsize=font)
        ax.set_ylabel('$mu$', fontsize=font)
        fig.set_size_inches(w=13, h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to + 'mu.png', dpi=300)
            
            
           
        def Par_fun(t):   
               t = np.array(t)  
               t = t.reshape([1, 1])    
               Beta1, Beta2, Mu = model.predict_par(t)  
               return Beta1, Beta2, Mu  


        def ODEs_mean(X, t):  
               Beta1_NN,Beta2_NN,Mu_NN=Par_fun(t)  
               Sh, Ih, Im, sumIh = X    
               dShdt = -((Beta1_NN) * Sh * Im )/ Nh 
               dIhdt = ((Beta1_NN) * Sh * Im )/ Nh  - eta * Ih
               dImdt = ((Beta2_NN) * (Nm-Im) * Ih) / Nh - (Mu_NN) * Im
               dsumIhdt = ((Beta1_NN) * Sh * Im) / Nh
               return [float(dShdt), float(dIhdt), float(dImdt), float(dsumIhdt)]  



           
        Sol = odeint(ODEs_mean, X0, t_star.flatten())  
        Sh = Sol[:, 0]  
        Ih = Sol[:, 1]
        Im = Sol[:, 2]
        SUMh = Sol[:,3]
        NEWh = np.diff(SUMh) 
        
        
        Sh = Sh.reshape([len(Sh), 1])/sf  
        Ih = Ih.reshape([len(Ih), 1])/sf
        Im = Im.reshape([len(Im), 1])/sf
        SUMh = SUMh.reshape([len(SUMh), 1])/sf
        NEWh = NEWh.reshape([len(NEWh), 1])/sf

        np.savetxt(save_results_to + 'Sh_ode.txt', Sh.reshape((-1, 1)))  
        np.savetxt(save_results_to + 'Ih_ode.txt', Ih.reshape((-1, 1)))
        np.savetxt(save_results_to + 'Im_ode.txt', Im.reshape((-1, 1)))
        np.savetxt(save_results_to + 'Ih_new_ode.txt', NEWh.reshape((-1, 1)))
        np.savetxt(save_results_to + 'Ih_sum_ode.txt', SUMh.reshape((-1, 1)))
        
        
         # New cases
        font = 24
        fig, ax = plt.subplots()
        ax.plot(t_star, Ih_new_star/sf, 'k--', marker='o', lw=2, markersize=5, label='Data')
        ax.plot(t_star[1:], NEWh, 'r-', lw=2, label='Odeslover')
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize=24)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3, 3))
        ax.grid(True)
        ax.set_xlabel('Weeks', fontsize=font)
        ax.set_ylabel('Weekly number of cases', fontsize=font)
        fig.set_size_inches(w=13, h=6.5)
        if SAVE_FIG:
             plt.savefig(save_results_to + 'New cases_ode.png', dpi=300)



         # Cumulative cases
        font = 24
        fig, ax = plt.subplots()
        ax.plot(t_star, Ih_sum_star/ sf, 'k--', marker='o', lw=2, markersize=5, label='Data')
        ax.plot(t_star, SUMh, 'r-', lw=2, label='Odeslover')
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize=24)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3, 3))
        ax.grid(True)
        ax.set_xlabel('Weeks', fontsize=font)
        ax.set_ylabel('Cumulative cases', fontsize=font)
        fig.set_size_inches(w=13, h=6.5)
        if SAVE_FIG:
             plt.savefig(save_results_to + 'Cumulative_cases_ode.png', dpi=300)

