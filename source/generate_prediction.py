from model import *


best_classifiers = { 'CapacitéEmprunt' : linear_model.BayesianRidge, 
                     'PrévisionnelAnnuel' : linear_model.BayesianRidge,
                     'Secteur1' : KNeighborsClassifier,
                     'Secteur2' : KNeighborsClassifier,
                     'SecteurParticulier' : tree.DecisionTreeClassifier}




def keep_results(tab, predictions, index):
    tab[0] += int(round(predictions[0][index - 1]))
    tab[1] += int(round(predictions[1][index - 1]))

    tab[2] += predictions[2][index - 1]
    tab[3] += predictions[3][index - 1]
    tab[4] += predictions[4][index - 1]   

    return tab


def reset_results(tab, predictions, index):
    tab[0] = int(round(predictions[0][index - 1]))
    tab[1] = int(round(predictions[1][index - 1]))

    tab[2] = predictions[2][index - 1]
    tab[3] = predictions[3][index - 1]
    tab[4] = predictions[4][index - 1]   

    return tab


def divide_result(tab, count):
    tab[0] = int(round(tab[0]/count))
    tab[1] = int(round(tab[1]/count))

    tab[2] = int(round(tab[2]/count))
    tab[3] = int(round(tab[3]/count))
    tab[4] = int(round(tab[4]/count))

    return tab



def write_last(tab, line, file, count_same):
    line[COL_EMPRUNT] = tab[0]
    line[COL_PREVISIO] = tab[1]

    line[COL_SECT1] = tab[2]
    line[COL_SECT2] = tab[3]
    line[COL_SECT_PART] = tab[4]

    if count_same > 1:
        line[1] = ''

    file.writerow(line)


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


    predictions_tuple = (pred_emprunt, pred_previsionnel, pred_scoring_secteur1, pred_scoring_secteur2, pred_scoring_secteur_parti)


    output_file = open(filename, 'wt')
    output_writer = csv.writer(output_file)

    with open("../data/cleaned_test.csv", 'rt') as csvfile:
        result_reader = csv.reader(csvfile, delimiter=';', quotechar='"')
        i = 0

        last_id = None
        last_line = None
        sum_tab = [None, None, None, None, None]
        count_same = 1

        for line in result_reader:

            if i == 0:
                output_writer.writerow(line)

            else:

                if last_id == None:
                    last_id = line[0]
                    last_line = line
                    count_same = 1                    
                    tab = reset_results(sum_tab, predictions_tuple,i)

                elif line[0] == last_id :
                    tab = keep_results(sum_tab, predictions_tuple,i)
                    count_same += 1
 
                else :                    
                    tab = divide_result(sum_tab, count_same)

                    write_last(sum_tab, last_line, output_writer, count_same)
                    
                    last_id = line[0]
                    last_line = line
                    count_same = 1                    
                    tab = reset_results(sum_tab, predictions_tuple,i)

            i = i + 1

        # Write the last line
        tab = divide_result(sum_tab, count_same)
        write_last(sum_tab, last_line, output_writer, count_same)
                    
          


generate_results()