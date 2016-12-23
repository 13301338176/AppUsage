# -*- coding: utf-8 -*-
"""
Created on Sat Dec 03 14:49:13 2016

@author: Josh
"""

import pandas
import matplotlib.pyplot as plt
import numpy
import pickle
import datetime
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

print datetime.datetime.now()

events_path = "../csv_files/events.csv"
gender_age_path = "../csv_files/gender_age_train.csv"

genders = ["F", "M"]

df_gender_age = pandas.read_csv(gender_age_path)

df_events = pandas.read_csv(events_path)

device_ids = df_events["device_id"].unique()
del df_events
df_device_id = pandas.DataFrame(device_ids,columns=["device_id"])

df_gender_age = df_gender_age.merge(df_device_id,on="device_id")

#load saved data structures

#the two dictionaries below have the form:
#   device_id -> label_id -> label counts
#the first dicionary counts the number of times a given label was involved in
#an app event for that device_id and the second counts the number of times a
#label was associated with an app that was active for that device_id
device_label_count_dict = pickle.load(open( "../data_structures/device_label_count_dict.p", "rb" ) )
device_active_label_count_dict = pickle.load(open( "../data_structures/device_label_count_dict.p", "rb" ))

#list of all labels that were associated with an app that was involved in at
#least one app event
label_list = pickle.load( open( "../data_structures/label_list.p", "rb" ) )

active_label_list = pickle.load( open( "../data_structures/active_label_list.p", "rb" ) )

#recodes genders from "M" and "F" to 0 and 1
def binarize(row):
    if row[0] == "F":
        return 1
    else:
        return 0

#divides a dataframe into two mutually exclusive subsets for use as training and
#test sets.  The train set size is passed as a parameter and the test set is
#always size 1000
def get_train_test_sets(df, train_set_size):
    train_set = df.sample(n=train_set_size)
    train_set_ids = train_set["device_id"].tolist()
    train_set_complement = df[df.apply(lambda row: row["device_id"] not in train_set_ids, axis=1)]
    test_set = train_set_complement.sample(n=1000)
    return train_set, test_set

#constructs the X matrix used to fit the Bernoulli NB model.  Each row in the 
#matrix represents one user and there is one column for each item.  Items are
#either apps or app labels depending on the model being fitted.  The 
#item_count_dict is either:
#           device_ids -> app_ids -> app counts
#           device_ids -> label_ids -> label counts
#Note that the counts themselves are ignored in Bernoulli NB and so a 1 in a 
#given column indicates the presence of an item for that users and a 0 indicates
#it's absence.
#The item list is a list of all possible apps or app_labels.  
def construct_x(train_set, item_count_dict, item_list):
    #initialize a matrix of the correct size
    x = numpy.zeros((len(train_set),len(item_list)))
    devices = train_set["device_id"].tolist()
    
    #item_index_dict is used to find the index of a given item in each row
    item_index_dict = dict(zip(item_list,range(len(item_list))))
    for i in range(len(devices)):
        if item_count_dict[devices[i]] is not None:
            items = item_count_dict[devices[i]].keys()
            for item in items:
                #in some cases a subset of all possible items is being used.
                #For example we may be using only the 1000 apps with the highest
                #MI scores so make sure an app is on the list
                if item in item_list:
                    x[i][item_index_dict[item]] = 1
    return x

#tests Bernoulli NB model.  item_count_dict and item_list 
#have same form as arguments to construct_x function above
def test_model(model, test_set, item_count_dict, item_list):
    device_ids = test_set["device_id"].tolist()
    #initialize matrix of correct size
    x = numpy.zeros((len(device_ids),len(item_list)))
    y = numpy.array(test_set["gender"].tolist())
    item_index_dict = dict(zip(item_list,range(len(item_list))))
    for i in range(len(device_ids)):
        if item_count_dict[device_ids[i]] is not None:
            item_dict = item_count_dict[device_ids[i]]
            for item in item_dict.keys():
                
                #in some cases we may be using a subset of all possible items.
                #For example we may be using only the 1000 apps with the highest
                #MI scores so make sure an app is on the list
                if item in item_list:
                    x[i][item_index_dict[item]] = 1
    predictions = model.predict(x)
    accuracy = model.score(x, y)

    #convert predictions and true labels to binary form
    bin_predictions = pandas.Series(predictions).apply(binarize)
    bin_labels = pandas.Series(y).apply(binarize)
    roc_auc = roc_auc_score(bin_labels, bin_predictions)
    return accuracy, roc_auc
    

#constructs the X matrix used to fit the multinomial NB model.  Each row in the 
#matrix represents one user and there is one column for each item.  Items are
#either apps or app labels depending on the model being fitted.  The 
#item_count_dict is either:
#           device_ids -> app_ids -> app counts
#           device_ids -> label_ids -> label counts
#The item list is a list of all possible apps or app_labels.  
def construct_x_multinomial(train_set, item_count_dict, item_list):
    #initialize a matrix of the correct size
    x = numpy.zeros((len(train_set),len(item_list)))
    devices = train_set["device_id"].tolist()
    
    #item_index_dict is used to find the index of a given item in each row
    item_index_dict = dict(zip(item_list,range(len(item_list))))
    for i in range(len(devices)):
        if item_count_dict[devices[i]] is not None:
            items = item_count_dict[devices[i]].keys()
            for item in items:
                #in some cases a subset of all possible items is being used.
                #For example we may be using only the 1000 apps with the highest
                #MI scores so make sure an app is on the list
                if item in item_list:
                    x[i][item_index_dict[item]] = item_count_dict[devices[i]][item]
    return x
    
#tests Bernoulli NB model.  item_count_dict and item_list 
#have same form as arguments to construct_x function above
def test_model_multinomial(model, test_set, item_count_dict, item_list):
    device_ids = test_set["device_id"].tolist()
    #initialize matrix of correct size
    x = numpy.zeros((len(device_ids),len(item_list)))
    y = numpy.array(test_set["gender"].tolist())
    item_index_dict = dict(zip(item_list,range(len(item_list))))
    for i in range(len(device_ids)):
        if item_count_dict[device_ids[i]] is not None:
            item_dict = item_count_dict[device_ids[i]]
            for item in item_dict.keys():
                
                #in some cases we may be using a subset of all possible items.
                #For example we may be using only the 1000 apps with the highest
                #MI scores so make sure an app is on the list
                if item in item_list:
                    x[i][item_index_dict[item]] = item_dict[item]
    predictions = model.predict(x)
    accuracy = model.score(x, y)

    #convert predictions and true labels to binary form
    bin_predictions = pandas.Series(predictions).apply(binarize)
    bin_labels = pandas.Series(y).apply(binarize)
    roc_auc = roc_auc_score(bin_labels, bin_predictions)
    return accuracy, roc_auc

def get_all_items_for_users(df_user,dictionary):
    device_ids = df_user["device_id"].tolist()
    list_of_lists = []
    for device in device_ids:
        if dictionary[device] is not None:
            list_of_lists.append(dictionary[device].keys())
    flattened_list = [item for sublist in list_of_lists for item in sublist]
    return pandas.Series(flattened_list).unique()
    
def mutual_information(df_users, dictionary, item_list):
    users_by_gender = [df_users[df_users["gender"] == gender] for gender in genders]

    #priors is a list of the prior probabilities for each class
    priors = numpy.array([float(len(gender)) / len(df_users) for gender in users_by_gender]) 
    
    #class conditional densities is a list of dictionaries where each dictionary
    #defines the probabilities of each app occuring in each class.  For example,
    #the dicionary in class_conditional_densities[0] will map to the frequencies
    #of each app for all female users in df_users
    class_conditional_densities = []
    for gender in users_by_gender:
        devices = gender["device_id"]
        
        #to compute the frequencies for each class, we create a list of all the 
        #app_ids or label_ids for users in a  given class and then use the 
        #series value_counts method to find the total number of occurances of 
        #each app_id or label_id in that class.  This is then divided by the size
        #of the class to get class conditional probabilities
        list_of_lists = []
        for device in devices:
            if dictionary[device] is not None:
                list_of_lists.append(dictionary[device].keys())
        flattened_list = [item for sublist in list_of_lists for item in sublist]
        vals = pandas.Series(flattened_list).value_counts()
        class_dict = dict(zip(vals.index, vals.values/float((len(gender)))))
        class_conditional_densities.append(class_dict)
    
    #densities is a mapping from items (app_ids or label_ids) to the probability
    #of that item occuring.  Defined on Murphy pg. 87 to be product of the class
    #prior times the items class conditional density summed over all classes
    densities = dict.fromkeys(item_list,0)

    #the following list is created to avoid repeated calls to dicionary.keys()
    #in the next section
    class_keys = [class_conditional_densities[i].keys() for i in range(len(class_conditional_densities))]
    for item in item_list:
        result = 0
        for i in range(len(class_conditional_densities)):
            if item in class_keys[i]:
                result += priors[i] * class_conditional_densities[i][item]
        densities[item] = result

    #compute the mutual information for each item
    mutual_info_dict = dict.fromkeys(item_list,0)
    for item in item_list:
        mi = 0
        for i in range(len(genders)):
            if item in class_keys[i]:
                d_mi = class_conditional_densities[i][item] * priors[i] * numpy.log(class_conditional_densities[i][item] / densities[item]) + (1 - class_conditional_densities[i][item]) * priors[i] * numpy.log((1 - class_conditional_densities[i][item]) / (1 - densities[item]))
            else:
                d_mi = priors[i] * numpy.log(1 / (1 - densities[item]))
            mi += d_mi
        mutual_info_dict[item] = mi
    
    mi_list = [(k, v) for k, v in mutual_info_dict.iteritems()]
    sorted_mi_list = sorted(mi_list, key=lambda x: x[1], reverse=True)
    return sorted_mi_list

def save_plot(x,y,xlabel,ylabel,title,file_name):
    plt.plot(x,y)
    plt.ylim([.47,.75])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
#    plt.title(title)
    plt.savefig("../plots/" + file_name + ".png")
    plt.show()

train_set_size = [100,200,300,400,500,750,1000]

print "Contruct models for linear kernel SVM with all labels"
svm_linear_accuracy_list = []
svm_linear_roc_auc_list = []
for i in train_set_size:
    print "Training linear SVM model with training set size " , i
    train_set, test_set = get_train_test_sets(df_gender_age,i)
    x = construct_x(train_set, device_label_count_dict, label_list)
    y = numpy.array(train_set["gender"].tolist())
    svm = SVC(kernel="linear")
    svm.fit(x,y)
    accuracy, roc_auc = test_model(svm, test_set, device_label_count_dict, label_list)
    svm_linear_accuracy_list.append(accuracy)
    svm_linear_roc_auc_list.append(roc_auc)
    
save_plot(train_set_size,svm_linear_accuracy_list,'Number of users in training set',"Accuracy rate","Accuracy for linear kernel SVM with all labels","linear_svm")    
save_plot(train_set_size,svm_linear_roc_auc_list,'Number of users in training set',"ROC AUC score","ROC AUC for linear kernel SVM with all labels","linear_svm_roc")

pickle.dump(svm_linear_accuracy_list, open( "../results/linear_svm_accuracy.p", "wb" ))
pickle.dump(svm_linear_roc_auc_list, open( "../results/linear_svm_roc_auc.p", "wb" ))

print "Contruct models for polynomial kernel SVM with all labels"
svm_poly_accuracy_list = []
svm_poly_roc_auc_list = []
for i in train_set_size:
    print "Training polynomial SVM with training set size " , i
    train_set, test_set = get_train_test_sets(df_gender_age,i)
    x = construct_x(train_set, device_label_count_dict, label_list)
    y = numpy.array(train_set["gender"].tolist())
    svm = SVC(kernel="poly",degree=2)
    svm.fit(x,y)
    accuracy, roc_auc = test_model(svm, test_set, device_label_count_dict, label_list)
    svm_poly_accuracy_list.append(accuracy)
    svm_poly_roc_auc_list.append(roc_auc)
 
save_plot(train_set_size,svm_poly_accuracy_list,'Number of users in training set',"Accuracy rate","Accuracy for polynomial kernel SVM with all labels","poly_svm")    
save_plot(train_set_size,svm_poly_roc_auc_list,'Number of users in training set',"ROC AUC score","ROC AUC for polynomial kernel SVM with all labels","poly_svm_roc")

pickle.dump(svm_poly_accuracy_list, open( "../results/linear_svm_accuracy.p", "wb" ))
pickle.dump(svm_poly_roc_auc_list, open( "../results/linear_svm_roc_auc.p", "wb" ))

######################################
#dimensionality reduction

number_of_labels = [10,20,30,50,100,200,400]

print "Contruct models for linear SVM with greatest entropy reduction labels"
svm_accuracy_list_mi_linear = []
svm_roc_auc_list_mi_linear = []
for i in number_of_labels:
    print "Training linear SVM model with the top" , i , " apps by reduction of entropy scores"
    train_set, test_set = get_train_test_sets(df_gender_age,1000)
    label_list = get_all_items_for_users(train_set, device_label_count_dict)
    label_info = mutual_information(train_set, device_label_count_dict, label_list)
    sorted_label_list = [j[0] for j in label_info]
    x = construct_x(train_set, device_label_count_dict, sorted_label_list[0:i])
    y = numpy.array(train_set["gender"].tolist())
    svm = SVC(kernel="linear")
    svm.fit(x,y)
    accuracy, roc_auc = test_model(svm, test_set, device_label_count_dict, sorted_label_list[0:i])
    svm_accuracy_list_mi_linear.append(accuracy)
    svm_roc_auc_list_mi_linear.append(roc_auc)
    
save_plot(number_of_labels,svm_accuracy_list_mi_linear,'n highest MI scores',"Accuracy rate","Accuracy for linear SVM with highest MI labels","svm_linear_mi")    
save_plot(number_of_labels,svm_roc_auc_list_mi_linear,'n highest MI scores',"ROC AUC score","ROC AUC for linear SVM with highest MI apps","svm_linear_mi_roc")

pickle.dump(svm_accuracy_list_mi_linear, open( "../results/svm_linear_accuracy_mi.p", "wb" ))
pickle.dump(svm_roc_auc_list_mi_linear, open( "../results/svm_linear_roc_auc_mi.p", "wb" ))

print "Contruct models for polynomial SVM with greatest entropy reduction labels"
svm_accuracy_list_mi_poly = []
svm_roc_auc_list_mi_poly = []
for i in number_of_labels:
    print "Training linear SVM model with the top" , i , " labels by reduction of entropy scores"
    train_set, test_set = get_train_test_sets(df_gender_age,1000)
    label_list = get_all_items_for_users(train_set, device_label_count_dict)
    label_info = mutual_information(train_set, device_label_count_dict, label_list)
    sorted_label_list = [j[0] for j in label_info]
    x = construct_x(train_set, device_label_count_dict, sorted_label_list[0:i])
    y = numpy.array(train_set["gender"].tolist())
    svm = SVC(kernel="poly",degree=2)
    svm.fit(x,y)
    accuracy, roc_auc = test_model(svm, test_set, device_label_count_dict, sorted_label_list[0:i])
    svm_accuracy_list_mi_poly.append(accuracy)
    svm_roc_auc_list_mi_poly.append(roc_auc)
    
save_plot(number_of_labels,svm_accuracy_list_mi_poly,'n highest MI scores',"Accuracy rate","Accuracy for polynomial SVM with highest MI labels","svm_poly_mi")    
save_plot(number_of_labels,svm_roc_auc_list_mi_poly,'n highest MI scores',"ROC AUC score","ROC AUC for polynomial SVM with highest MI apps","svm_poly_mi_roc")

pickle.dump(svm_accuracy_list_mi_poly, open( "../results/svm_poly_accuracy_mi.p", "wb" ))
pickle.dump(svm_roc_auc_list_mi_poly, open( "../results/svm_poly_roc_auc_mi.p", "wb" ))

########################################################
#investigate soft margin penalty

c_roc_auc = []
c = [.1, .5, .75, 1, 2, 5, 10, 20, 50]
for i in c:
    print "Training model with soft margin penalty : ", i
    train_set, test_set = get_train_test_sets(df_gender_age,1000)
    x = construct_x(train_set, device_label_count_dict, label_list)
    y = numpy.array(train_set["gender"].tolist())
    svm = SVC(kernel="linear", C=i)
    svm.fit(x,y)
    accuracy, roc_auc = test_model(svm, test_set, device_label_count_dict, label_list)
    c_roc_auc.append(roc_auc)
    
save_plot(c,c_roc_auc,'C',"ROC AUC Score","Accuracy for polynomial SVM with highest MI labels","svm_c_roc_auc")
pickle.dump(c_roc_auc, open( "../results/svm_c_roc_auc.p", "wb" ))
    