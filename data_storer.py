__author__ = 'shidaiting01'

import csv

def save_data(labels, file_name):
    with open(file_name, 'wb') as my_file:
        my_writer=csv.writer(my_file)
        for label in labels:
            tmp=[]
            tmp.append(label)
            my_writer.writerow(tmp)
