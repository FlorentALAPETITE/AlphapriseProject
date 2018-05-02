import pandas
from sklearn import linear_model
from shutil import copyfile
import csv


df = pandas.read_csv("../data/cleaned_learning.csv", sep=";")
#import ipdb; ipdb.set_trace()

classe_emprunt=df.iloc[:,5]
classe_previsionnel=df.iloc[:,6]

df = df.iloc[:, [x for x in range(df.shape[1]) if x not in [0,1,2,3,5,6]] ]


classifiers = [
    linear_model.BayesianRidge,
    linear_model.LassoLars,
    linear_model.LinearRegression]


test = pandas.read_csv("../data/cleaned_test.csv", sep=";")
#import ipdb; ipdb.set_trace()
test = test.iloc[:,:-5]



def prediction_emprunt(classifier,filename):
    clf = item()
    clf.fit(df, classe_emprunt)
    prediction = clf.predict(test)  

    return prediction


def prediction_previsionnel(classifier,filename):
    clf = item()
    clf.fit(df, classe_previsionnel)
    prediction = clf.predict(test)  

    return prediction




for item in classifiers:    
    filename = "../data/predicted_"+item.__name__+".csv"  
    pred_emprunt = prediction_emprunt(item, filename)
    pred_previsionnel = prediction_previsionnel(item,filename)


    output_file = open(filename, 'wt')    
    output_writer = csv.writer(output_file)


    with open("../data/cleaned_test.csv", 'rt') as csvfile:        
        result_reader = csv.reader(csvfile, delimiter=';', quotechar='"')
        i=0
        for line in result_reader:
            if i!=0:   
                line[31] = pred_emprunt[i-1]
                line[32] = pred_previsionnel[i-1]
            i=i+1 
            output_writer.writerow(line)


