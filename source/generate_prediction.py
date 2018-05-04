import csv
from model import *
# from sklearn import linear_model
# from sklearn.neighbors import KNeighborsClassifier
# import pandas


best_classifiers = { 'CapacitéEmprunt' : linear_model.LassoLars, 
                     'PrévisionnelAnnuel' : linear_model.BayesianRidge,
                     'Secteur1' : KNeighborsClassifier,
                     'Secteur2' : KNeighborsClassifier,
                     'SecteurParticulier' : KNeighborsClassifier}



def generate_results():
    full_training_dataframe = pandas.read_csv("../data/cleaned_learning.csv", sep=";")
    training_dataframe = full_training_dataframe.iloc[:, [x for x in range(full_training_dataframe.shape[1]) if x not in [0, 1, 2, 3, 5, 6]]]

    test = pandas.read_csv("../data/cleaned_test.csv", sep=";")
    test = test.iloc[:, 1:-5]

    filename = "../data/predictions.csv"
    pred_emprunt = prediction(best_classifiers['CapacitéEmprunt'],training_dataframe,test,full_training_dataframe.iloc[:, 5])
    pred_previsionnel = prediction(best_classifiers['PrévisionnelAnnuel'],training_dataframe,test, full_training_dataframe.iloc[:, 6])

    
    (pred_scoring_secteur1,score_sect1) = scoring_prediction(best_classifiers['Secteur1'],training_dataframe,test,full_training_dataframe,1)
    (pred_scoring_secteur2,score_sect2) = scoring_prediction(best_classifiers['Secteur2'],training_dataframe,test,full_training_dataframe,2)
    (pred_scoring_secteur_parti,score_sect_parti) = scoring_prediction(best_classifiers['SecteurParticulier'],training_dataframe,test,full_training_dataframe,3)


    output_file = open(filename, 'wt')
    output_writer = csv.writer(output_file)

    with open("../data/cleaned_test.csv", 'rt') as csvfile:
        result_reader = csv.reader(csvfile, delimiter=';', quotechar='"')
        i = 0
        for line in result_reader:
            if i != 0:
                line[COL_EMPRUNT] = int(round(pred_emprunt[i - 1]))
                line[COL_PREVISIO] = int(round(pred_previsionnel[i - 1]))

                line[COL_SECT1] = pred_scoring_secteur1[i - 1]
                line[COL_SECT2] = pred_scoring_secteur2[i - 1]
                line[COL_SECT_PART] = pred_scoring_secteur_parti[i - 1]   
            i = i + 1
            output_writer.writerow(line)


generate_results()