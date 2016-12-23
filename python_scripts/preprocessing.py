# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 09:28:37 2016

@author: Josh
"""
import pandas
import pickle
import datetime
import numpy

print datetime.datetime.now()

events_path = "../csv_files/events.csv"
gender_age_path = "../csv_files/gender_age_train.csv"
app_labels_path = "../csv_files/app_labels.csv"

genders = ["F", "M"]

#read in csv containing a mapping from app_ids to label_ids.  Each unique app_id
#can correspond to multiple lable_ids
df_app_labels = pandas.read_csv(app_labels_path)

#read in csv containing a mapping from device_id's to events.  Each event
#corresponds to multiple app events, one for each app involved in the event
df_events = pandas.read_csv(events_path)

#read in csv containg device_id's and age/gender data for the user of that device
df_gender_age = pandas.read_csv(gender_age_path)
df_gender_age = df_gender_age[["device_id","group"]]

device_ids = df_events["device_id"].unique()
df_device_id = pandas.DataFrame(device_ids,columns=["device_id"])

#there are some device id's for which there aren't any associated events.  These
#users are ignored in this analysis.  Merging the dataframes below removes them
#from df_gender_age
df_gender_age = df_gender_age.merge(df_device_id,on="device_id")

#dictionary that maps for each device id to another dictionary that maps from apps
#installed on that device to the count of app events involving that given app
#For example, device_app_count_dict[1] maps to a dictionary for device_id 1. 
#This dictionary might look something like {"Google" : 10 , "Facebook" : 25} 
#indicating that Google was involved in 10 app events for that user and Facebook
#was involved in 25 (The acutal names of apps aren't actually given in the dataset)
device_app_count_dict = dict.fromkeys(df_gender_age["device_id"])

#dictionary that maps for each device id to another dictionary that maps from apps
#installed on that device to the count of app events involving that given app
#where the given app was active, not just installed.
#For example, device_active_app_count_dict[1] maps to a dictionary for device_id 1. 
#This dictionary might look something like {"Google" : 10 , "Facebook" : 25} 
#indicating that Google was active in 10 app events for that user and Facebook
#was active in 25 (The acutal names of apps aren't actually given in the dataset)
device_active_app_count_dict = dict.fromkeys(df_gender_age["device_id"])

device_label_count_dict = dict.fromkeys(df_gender_age["device_id"])
device_active_label_count_dict = dict.fromkeys(df_gender_age["device_id"])

#merge so df_gender age has age/gender info associated with each device_id
df_events = df_events.merge(df_gender_age, on="device_id")

app_list = []
active_app_list = []
label_list = []
active_label_list = []

def construct_app_dictionary(row):
    device_id = row["device_id"]
    app_id = row["app_id"]
    #labels = df_app_labels[df_app_labels["app_id"] == app_id]["label_id"].tolist()
    if app_id not in app_list:
        app_list.append(app_id)
    current_dict = device_app_count_dict[device_id]
    if current_dict is None:
        current_dict = {app_id : 1}
    elif app_id in current_dict.keys():
        current_dict[app_id] = current_dict[app_id] + 1
    else:
        current_dict[app_id] = 1
    device_app_count_dict[device_id] = current_dict

    if row["is_active"] == 1:
    
        if app_id not in active_app_list:
            active_app_list.append(app_id)
        current_active_dict = device_active_app_count_dict[device_id]
        if current_active_dict is None:
            current_active_dict = {app_id : 1}
        elif app_id in current_active_dict.keys():
            current_active_dict[app_id] = current_active_dict[app_id] + 1
        else:
            current_active_dict[app_id] = 1
        device_active_app_count_dict[device_id] = current_active_dict
    
print "Constructing data structures"
for i in range(1,34):
    print "Processing app_events csv file ", i , " of 35"
    path = "../csv_files/app_events" + str(i) + ".csv"
    df_app_events = pandas.read_csv(path)
    merged_df = df_app_events.merge(df_events, on="event_id")
    merged_df.apply(construct_app_dictionary,axis=1)
    del merged_df

print "Constructing label dictionaries from app dicionaries"
for device in device_label_count_dict.keys():
    app_count_dict = device_app_count_dict[device]
    app_label_count_dict = device_label_count_dict[device]
    if app_count_dict is not None:
        apps = app_count_dict.keys()
        for app in apps:
            labels = df_app_labels[df_app_labels["app_id"] == app]["label_id"].tolist()
            for label in labels:
                if app_label_count_dict is None:
                    app_label_count_dict = {label : app_count_dict[app]}
                elif label not in app_label_count_dict.keys():
                    app_label_count_dict[label] = app_count_dict[app]
                else:
                    app_label_count_dict[label] += app_count_dict[app]
    device_label_count_dict[device] = app_label_count_dict

for device in device_active_label_count_dict.keys():
    active_app_count_dict = device_active_app_count_dict[device]
    active_app_label_count_dict = device_active_label_count_dict[device]
    if active_app_count_dict is not None:
        apps = active_app_count_dict.keys()
        for app in apps:
            labels = df_app_labels[df_app_labels["app_id"] == app]["label_id"].tolist()
            for label in labels:
                if active_app_label_count_dict is None:
                    active_app_label_count_dict = {label : active_app_count_dict[app]}
                elif label not in active_app_label_count_dict.keys():
                    active_app_label_count_dict[label] = active_app_count_dict[app]
                else:
                    active_app_label_count_dict[label] += active_app_count_dict[app]
    device_active_label_count_dict[device] = active_app_label_count_dict




pickle.dump(app_list, open( "../data_structures/app_list.p", "wb" )) 
pickle.dump(active_app_list, open( "../data_structures/active_app_list.p", "wb" ))
pickle.dump(label_list, open( "../data_structures/label_list.p", "wb" ))
pickle.dump(active_label_list, open( "../data_structures/active_label_list.p", "wb" ))
pickle.dump(device_app_count_dict, open( "../data_structures/device_app_count_dict.p", "wb" ))
pickle.dump(device_active_app_count_dict, open( "../data_structures/device_active_app_count_dict.p", "wb" ))
pickle.dump(device_label_count_dict, open( "../data_structures/device_label_count_dict.p", "wb" ))
pickle.dump(device_active_label_count_dict, open( "../data_structures/device_active_label_count_dict.p", "wb" ))

print datetime.datetime.now()

