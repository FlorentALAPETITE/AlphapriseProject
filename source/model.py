import pandas
from sklearn import linear_model
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import csv



# ================= Modèles CapacitéEmprunt et PrevisionnelAnnuel =================



COL_EMPRUNT = 32
COL_PREVISIO = 33

LINEAR_CLASSIFIERS = [
    linear_model.BayesianRidge,
    linear_model.LassoLars,
    linear_model.LinearRegression
]


def prediction(classifier, training_dataframe, test_dataframe, pred_class):
    classifier = classifier()  
    classifier.fit(training_dataframe, pred_class)
    prediction = classifier.predict(test_dataframe)

    return prediction
    

def emprunt_previsionnel_prediction(full_training_dataframe, training_dataframe, test_dataframe):
    for classifier in LINEAR_CLASSIFIERS:
        print("     ====== Mise en place du modèle avec le classifieur : "+ classifier.__name__ +" ======")
        filename = "../data/predicted_" + classifier.__name__ + ".csv"
        pred_emprunt = prediction(classifier,training_dataframe,test_dataframe,full_training_dataframe.iloc[:, 5])
        pred_previsionnel = prediction(classifier,training_dataframe,test_dataframe, full_training_dataframe.iloc[:, 6])

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




# ================= Modèles Scoring Secteur1, Secteur2 et SecteurParticulier =================


COL_SECT1 = 34
COL_SECT2 = 35
COL_SECT_PART = 36

SCORING_CLASSIFIERS = [
    tree.DecisionTreeClassifier,
    KNeighborsClassifier,
    SVC    
]


# Valeurs trouvées en executant la fonction best_KNeighborsClassifier
BEST_KNN_SECTEUR1 = 18
BEST_KNN_SECTEUR2 = 30
BEST_KNN_SECTEUR_PARTI = 4


# Fonction qui retourne pour chaque classe le meilleur choix du nombre de voisins pour le KNeighborsClassifier
# Le temps d'execution étant très grand, nous avons fixé les valeurs trouvées pour chaque classe
def best_KNeighborsClassifier(training, classe):
    print("           --- Choix du meilleur nombre de voisin ---")
    parameters = {'n_neighbors':list(range(1,31))}
    algorithm = KNeighborsClassifier()
    clf = GridSearchCV(algorithm,parameters)
    best_model_choice = clf.fit(training, classe)


    print("           --- Meilleur nombre de voisin pour la classe : " + str(best_model_choice.best_params_["n_neighbors"])+" ---")
    return best_model_choice.best_params_["n_neighbors"]



def scoring_prediction(classifier, training_dataframe, test_dataframe, full_training_dataframe, class_index):
    classe_scoring_prediction = full_training_dataframe.iloc[:, class_index]  

    if classifier == KNeighborsClassifier :
        # Utilisation directe de la méthode pour trouver le meilleur choix de nombre de voisins (très lent)

        # best_knn = best_KNeighborsClassifier(training_dataframe, classe_scoring_prediction)
        # classifier = classifier(n_neighbors=best_knn)


        # Utilisation des valeurs trouvées
        if class_index == 1:
            classifier = classifier(n_neighbors=BEST_KNN_SECTEUR1)
        elif class_index == 2:
            classifier = classifier(n_neighbors=BEST_KNN_SECTEUR2)
        else:
            classifier = classifier(n_neighbors=BEST_KNN_SECTEUR_PARTI)
    elif classifier == SVC:
        classifier = classifier(probability=True)
    else :
        classifier = classifier()  

    classifier.fit(training_dataframe, classe_scoring_prediction)    
    #score = classifier.predict_proba(test_dataframe)
    score=2
    prediction = classifier.predict(test_dataframe)

    return (prediction,score)


def secteur_scoring_prediction(full_training_dataframe, training_dataframe, test_dataframe):
    for classifier in SCORING_CLASSIFIERS:
        print("     ====== Mise en place du modèle de scoring avec le classifieur : "+ classifier.__name__ +" ======")
        filename = "../data/predicted_" + classifier.__name__ + ".csv"
        (pred_scoring_secteur1,score_sect1) = scoring_prediction(classifier,training_dataframe,test_dataframe,full_training_dataframe,1)
        (pred_scoring_secteur2,score_sect2) = scoring_prediction(classifier,training_dataframe,test_dataframe,full_training_dataframe,2)
        (pred_scoring_secteur_parti,score_sect_parti) = scoring_prediction(classifier,training_dataframe,test_dataframe,full_training_dataframe,3)

        output_file = open(filename, 'wt')
        output_writer = csv.writer(output_file)

        with open("../data/cleaned_test.csv", 'rt') as csvfile:
            result_reader = csv.reader(csvfile, delimiter=';', quotechar='"')
            i = 0
            for line in result_reader:
                if i != 0:
                    line[COL_SECT1] = pred_scoring_secteur1[i - 1]
                    line[COL_SECT2] = pred_scoring_secteur2[i - 1]
                    line[COL_SECT_PART] = pred_scoring_secteur_parti[i - 1]                    
                i = i + 1
                output_writer.writerow(line)



# ================= Appel aux prédictions sur les modèles =================



def make_prediction():
    full_training_dataframe = pandas.read_csv("../data/cleaned_learning.csv", sep=";")
    training_dataframe = full_training_dataframe.iloc[:, [x for x in range(full_training_dataframe.shape[1]) if x not in [0, 1, 2, 3, 5, 6]]]

    test = pandas.read_csv("../data/cleaned_test.csv", sep=";")
    test = test.iloc[:, 1:-5]

    print("====== Estimation des variables CapacitéEmprunt et PrévisionnelAnnuel ======")
    emprunt_previsionnel_prediction(full_training_dataframe,training_dataframe, test)

    print("====== Estimation des variables Secteur1, Secteur2 et SecteurParticulier avec des modèles de scoring ======")
    secteur_scoring_prediction(full_training_dataframe,training_dataframe, test)


if __name__ == '__main__':
    make_prediction()
