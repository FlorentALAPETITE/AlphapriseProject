import pandas
from sklearn import linear_model
import csv

COL_EMPRUNT = 31
COL_PREVISIO = 32
CLASSIFIERS = [
    linear_model.BayesianRidge,
    linear_model.LassoLars,
    linear_model.LinearRegression
]


def prediction_emprunt(classifier, df_app, df_test, class_index=5):
    classe_emprunt = df.iloc[:, class_index]
    classifier.fit(df_app, classe_emprunt)
    prediction = classifier.predict(df_test)

    return prediction


def prediction_previsionnel(classifier, df_app, df_test, class_index=6):
    classe_previsionnel = df.iloc[:, class_index]
    classifier.fit(df_app, classe_previsionnel)
    prediction = classifier.predict(df_test)

    return prediction

def emprunt_previsionnel_prediction():
    df = pandas.read_csv("../data/cleaned_learning.csv", sep=";")
    df = df.iloc[:, [x for x in range(df.shape[1]) if x not in [0, 1, 2, 3, 5, 6]]]

    test = pandas.read_csv("../data/cleaned_test.csv", sep=";")
    test = test.iloc[:, 1:-5]

    for item in CLASSIFIERS:
        filename = "../data/predicted_" + item.__name__ + ".csv"
        pred_emprunt = prediction_emprunt(item)
        pred_previsionnel = prediction_previsionnel(item)

        output_file = open(filename, 'wt')
        output_writer = csv.writer(output_file)

        with open("../data/cleaned_test.csv", 'rt') as csvfile:
            result_reader = csv.reader(csvfile, delimiter=';', quotechar='"')
            i = 0
            for line in result_reader:
                if i != 0:
                    line[COL_EMPRUNT] = pred_emprunt[i - 1]
                    line[COL_PREVISIO] = pred_previsionnel[i - 1]
                i = i + 1
                output_writer.writerow(line)
