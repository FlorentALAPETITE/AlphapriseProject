import csv
import random

from matplotlib import pyplot
import pandas
import sklearn

import model


FILENAME = "../data/cleaned_learning.csv"
NB_REPETITION = 2  # validation repetee

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
        'accuracy': sklearn.metrics.accuracy_score, # tp + tn / total
        'precision': sklearn.metrics.precision_score,  # tp / (tp + fp)
        'recall': sklearn.metrics.recall_score,  # tp / (tp + fn)
        'f1 score': sklearn.metrics.f1_score,
        'ROC': sklearn.metrics.roc_auc_score,
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
    score = {}

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
        score[clss.__name__] = {}
        for col in SCORING_CLASS_INDEXES.keys():
            pred[clss.__name__][col], score_pred = model.scoring_prediction(clss, df_app, df_test, df_app_full, SCORING_CLASS_INDEXES[col])
            score[clss.__name__][col] = [pos for (neg, pos) in score_pred]

    return pred, true, score


def give_precision(pred, true, score):
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
                if metric == "accuracy":
                    prec[metric][clss.__name__][col] = func(true[col], pred[clss.__name__][col])
                elif metric == "ROC":
                    prec[metric][clss.__name__][col] = func([int(cl) for cl in true[col]], score[clss.__name__][col])
                else:
                    prec[metric][clss.__name__][col] = func(true[col], pred[clss.__name__][col], pos_label='1')

    return prec


def do_report(prec):
    print(" == Rapport de precision ==")
    print(" == Validation repétée %s fois" % (NB_REPETITION, ))
    print("\n -- Modele 1 : CapaciteEmprunt et PrevisionnelAnnuel")

    for col in LINEAR_CLASS_INDEXES.keys():
        print("\n -- %s :" % (col.capitalize()))
        for metric in METRIC["LINEAR"].keys():
            vals = [v[col] for k, v in prec[metric].items()]
            opti_val = min(vals) if metric == 'mse' else max(vals)  # min mse, max r2
            opti_cls = [k for k, v in prec[metric].items() if v[col] == opti_val]
            opti_cls = ", ".join(opti_cls)
            print("    - %s : %s" % (metric.capitalize(), opti_cls))

    print("\n\n -- Modele 2 : Secteurs")
    for col in SCORING_CLASS_INDEXES.keys():
        print("\n -- %s :" % (col.capitalize()))
        for metric in METRIC["SCORING"].keys():
            vals = [v[col] for k, v in prec[metric].items()]
            opti_val = max(vals)
            opti_cls = [k for k, v in prec[metric].items() if v[col] == opti_val]
            opti_cls = ", ".join(opti_cls)
            print("    - %s : %s avec %s" % (metric.capitalize(), opti_cls, str(opti_val)))


def aggregate_iteration(big_prec):
    prec = {}
    for metric in METRIC["LINEAR"].keys():
        prec[metric] = {}
        for clss in model.LINEAR_CLASSIFIERS:
            prec[metric][clss.__name__] = {}
            for col in LINEAR_CLASS_INDEXES.keys():
                l = [p[metric][clss.__name__][col] for p in big_prec]
                prec[metric][clss.__name__][col] = sum(l) / len(l)

    for metric in METRIC["SCORING"].keys():
        prec[metric] = {}
        for clss in model.SCORING_CLASSIFIERS:
            prec[metric][clss.__name__] = {}

            for col in SCORING_CLASS_INDEXES.keys():
                l = [p[metric][clss.__name__][col] for p in big_prec]
                prec[metric][clss.__name__][col] = sum(l) / len(l)
    return prec


def tracer_courbes_roc(big_true, big_score):
    colors = ['red', 'green', 'blue']
    curves = {v: {k.__name__ : ([], []) for k in model.SCORING_CLASSIFIERS} for v in SCORING_CLASS_INDEXES.keys()}
    i = 1
    for k in SCORING_CLASS_INDEXES.keys():
        sub = pyplot.subplot(2, 2, i)
        sub.set_title(k.capitalize())
        # get the curve
        for clss in model.SCORING_CLASSIFIERS:
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(big_true[k], big_score[clss.__name__][k], pos_label='1')
            curves[k][clss.__name__] = (fpr, tpr)

        # display
        c = 0
        for k, (fpr, tpr) in curves[k].items():
            sub.plot(fpr, tpr, color=colors[c], label=k)
            sub.legend()
            c += 1
        i += 1
    pyplot.suptitle("Courbes ROC")
    pyplot.legend()
    pyplot.show()
    return


def eval():
    big_prec = []
    big_score = {v.__name__: {k : [] for k in SCORING_CLASS_INDEXES.keys()} for v in model.SCORING_CLASSIFIERS}
    big_true = {k: [] for k in SCORING_CLASS_INDEXES.keys()}
    for i in range(NB_REPETITION):
        app, test = cut_data(FILENAME)
        df_app_full, df_test_full, df_app, df_test = make_dataframes(app, test)
        pred, true, score = make_prediction(df_app_full, df_test_full, df_app, df_test)
        big_prec.append(give_precision(pred, true, score))
        for k in  SCORING_CLASS_INDEXES.keys():
            big_true[k].extend(true[k])
            for clss in model.SCORING_CLASSIFIERS:
                big_score[clss.__name__][k].extend(score[clss.__name__][k])
    prec = aggregate_iteration(big_prec)
    do_report(prec)
    tracer_courbes_roc(big_true, big_score)

if __name__ == '__main__':
    eval()
