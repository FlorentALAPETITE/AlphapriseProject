import csv
import random

import pandas
import sklearn

import model


FILENAME = "../data/cleaned_learning.csv"
NB_REPETITION = 20  # validation repetee

LINEAR_CLASS_INDEXES = {
    "emprunt": 5,
    "prevision": 6,
}
SCORING_CLASS_INDEXES = {
    "secteur1": 1,
    "secteur2": 2,
    "secteurParticulier": 3,
}
METRIC = {
    "LINEAR": {
        'mse': sklearn.metrics.mean_squared_error,
        'r2': sklearn.metrics.r2_score,
    },
    "SCORING": {
        'matrix': sklearn.metrics.confusion_matrix
    },
}
METRIC["NAMES"] = [k for k in METRIC["LINEAR"].keys()] + [k for k in METRIC["SCORING"].keys()]


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

    for col in LINEAR_CLASS_INDEXES:
        true[col] = make_class(df_test_full, LINEAR_CLASS_INDEXES[col])
    for col in SCORING_CLASS_INDEXES:
        true[col] = make_class(df_test_full, SCORING_CLASS_INDEXES[col])

    # model 1
    for clss in model.LINEAR_CLASSIFIERS:
        pred[clss.__name__] = {}
        for col in LINEAR_CLASS_INDEXES.keys():
            pred[clss.__name__][col] = model.prediction(clss, df_app, df_test, make_class(df_app_full, LINEAR_CLASS_INDEXES[col]))

    # model scoring
    for clss in model.SCORING_CLASSIFIERS:
        pred[clss.__name__] = {}
        for col in SCORING_CLASS_INDEXES.keys():
            pred[clss.__name__][col] = model.scoring_prediction(clss, df_app, df_test, df_app_full, SCORING_CLASS_INDEXES[col])

    return pred, true


def give_precision(pred, true):
    prec = {k: {} for k in METRIC["NAMES"]}

    for metric, func in METRIC["LINEAR"].items():
        prec[metric] = {}
        for clss in model.LINEAR_CLASSIFIERS:
            prec[metric][clss.__name__] = {}
            for col in LINEAR_CLASS_INDEXES.keys():
                prec[metric][clss.__name__][col] = func(true[col], pred[clss.__name__][col])

    for metric, func in METRIC["SCORING"].items():
        prec[metric] = {}
        for clss in model.SCORING_CLASSIFIERS:
            prec[metric][clss.__name__] = {}
            for col in SCORING_CLASS_INDEXES.keys():
                prec[metric][clss.__name__][col] = func(true[col], pred[clss.__name__][col])

    return prec


def do_report(prec):
    print(" == Rapport de precision ==")
    print(" == Validation repétée %s fois" % (NB_REPETITION, ))
    print(" -- Modele 1 : CapaciteEmprunt et PrevisionnelAnnuel")
    for metric in METRIC["LINEAR"].keys():
        print(" -- Avec %s" % (metric.capitalize()))
        for col in LINEAR_CLASS_INDEXES.keys():
                vals = [v[col] for k, v in prec[metric].items()]
                opti_val = min(vals) if metric == 'mse' else max(vals)  # min mse, max r2
                opti_cls = [k for k, v in prec[metric].items() if v[col] == opti_val]
                opti_cls = ", ".join(opti_cls)
                print("    - %s : %s" % (col.capitalize(), opti_cls))


def aggregate_iteration(big_prec):
    prec = {}
    for metric in METRIC["LINEAR"].keys():
        prec[metric] = {}
        for clss in model.LINEAR_CLASSIFIERS:
            prec[metric][clss.__name__] = {}
            for col in LINEAR_CLASS_INDEXES.keys():
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

if __name__ == '__main__':
    eval()
