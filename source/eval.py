import csv
import random

import pandas
import sklearn

import model


FILENAME = "../data/cleaned_learning.csv"
NB_REPETITION = 20  # validation repetee
CLASS_INDEXES = {
    "emprunt": 5,
    "prevision": 6,
}


def cut_data(filename, percent_test=20):
    app = []
    test = []
    with open(filename, 'rt') as csvfile:
        input_file = csv.reader(csvfile, delimiter=';', quotechar='"')
        next(input_file)  # skip the headline
        for line in input_file:
            if random.randint(0, 100) < percent_test:
                test.append(line)
            else:
                app.append(line)
    return app, test


def make_dataframes(app, test):
    df_app_full = pandas.DataFrame.from_dict(app)
    df_test_full = pandas.DataFrame.from_dict(test)

    df_app = df_app_full.iloc[:, [x for x in range(df_app_full.shape[1]) if x not in [0, 1, 2, 3, 5, 6]]]  # remove useless variables
    df_test = df_test_full.iloc[:, [x for x in range(df_test_full.shape[1]) if x not in [0, 1, 2, 3, 5, 6]]]  # remove useless variables
    return df_app_full, df_test_full, df_app, df_test


def make_class(df_app_full, class_index):
    return df_app_full.iloc[:, class_index]


def make_prediction(df_app_full, df_test_full, df_app, df_test):
    pred = {}
    true = {}
    for clss in model.CLASSIFIERS:
        pred[clss.__name__] = {}
        pred[clss.__name__]["emprunt"] = model.prediction_emprunt(clss, df_app, df_test, df_app_full)
        pred[clss.__name__]["prevision"] = model.prediction_previsionnel(clss, df_app, df_test, df_app_full)
    true["emprunt"] = make_class(df_test, CLASS_INDEXES["emprunt"])
    true["prevision"] = make_class(df_test, CLASS_INDEXES["prevision"])
    return pred, true


def give_precision(pred, true):
    prec = {'mse' : {}, 'r2' : {}}
    for clss in model.CLASSIFIERS:
        prec['mse'][clss.__name__] = {}
        prec['r2'][clss.__name__] = {}
        for col in CLASS_INDEXES.keys():
            prec['mse'][clss.__name__][col] = sklearn.metrics.mean_squared_error(true[col], pred[clss.__name__][col])
            prec['r2'][clss.__name__][col] = sklearn.metrics.r2_score(true[col], pred[clss.__name__][col])
    return prec


def do_report(prec):
    print(" -- Rapport de precision --")
    print(" -- Validation repétée %s fois" % (NB_REPETITION, ))
    print(" -- Avec MSE")
    for col in CLASS_INDEXES.keys():
        mini_mse = min([v[col] for k, v in prec['mse'].items()])  # min mse
        mini_cls = [k for k, v in prec['mse'].items() if v[col] == mini_mse]
        mini_cls = ", ".join(mini_cls)
        print("    - %s : %s" % (col.capitalize(), mini_cls))
    print(" -- Avec R²")
    for col in CLASS_INDEXES.keys():
        max_r2 = max([v[col] for k, v in prec['r2'].items()])  # max r2
        mini_cls = [k for k, v in prec['r2'].items() if v[col] == max_r2]
        mini_cls = ", ".join(mini_cls)
        print("    - %s : %s" % (col.capitalize(), mini_cls))


def aggregate_iteration(big_prec):
    prec = {}
    for metric in big_prec[0].keys():
        prec[metric] = {}
        for clss in model.CLASSIFIERS:
            prec[metric][clss.__name__] = {}
            for col in CLASS_INDEXES.keys():
                l = [p[metric][clss.__name__][col] for p in big_prec]
                prec[metric][clss.__name__][col] = sum(l) / len(l)
    return prec


def eval():
    big_prec = []
    for i in range(NB_REPETITION):
        app, test = cut_data(FILENAME)
        df_app_full, df_test_full, df_app, df_test = make_dataframes(app, test)
        pred, true = make_prediction(df_app_full, df_test_full, df_app, df_test)
        big_prec.append(give_precision(pred, true))
    do_report(aggregate_iteration(big_prec))

eval()