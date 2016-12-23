# -*- coding: utf-8 -*-
"""
Created on Sat Dec 03 18:17:37 2016

@author: Josh
"""
import pickle

p_1_names = ["nb_bernoulli_roc_auc_mi", "nb_multinomial_roc_auc_mi","nb_bernoulli_roc_auc_mi_labels","nb_multinomial_roc_auc_mi_labels"]
p_2_names = ["nb_bernoulli_roc_auc", "nb_bernoulli_active_roc_auc","nb_bernoulli_label_roc_auc","nb_bernoulli_active_label_roc_auc"]

p_1_results = [pickle.load(open( "./" + name + ".p", "rb" )) for name in p_1_names]
p_2_results = [pickle.load(open( "./" + name + ".p", "rb" )) for name in p_2_names]
               
nb_all_labels = pickle.load(open( "./nb_bernoulli_label_roc_auc.p", "rb" ))
svm_all_labels = pickle.load(open( "./linear_svm_roc_auc.p", "rb" ))

nb_best = pickle.load(open( "./nb_bernoulli_roc_auc.p", "rb" ))
