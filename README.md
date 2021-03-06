# AlphapriseProject
Data mining project for the ALMA Master
Uses python3

## Dependencies :
`$ pip3 install -r requirements.txt`

## Lauch the project :
`./exec`

## In details

### Pre-treatment of the data :
`$ python3 treatement.py`

### Analysis of the data distribution :
`$ python3 analyse_distribution.py`

Display the data distribution plot + the winsorized data distribution.


### Models and preliminary predictions : 
`$ python3 model.py`

Produce a folder named **prediction/** in the **data/** folder of the project. Fill the folder with :
+ prediction_linear_*classifier_name*.csv => Contains the prediction for **CapacitéEmprunt** and **PrévisionnelAnnuel**  (*Secteur1*, *Secteur2* and *SecteurParticulier* are filled with zeros).
+ prediction_scoring_*classifier_name*.csv => Contains the prediction for **Secteur1**, **Secteur2** and **SecteurParticulier**  (*CapacitéEmprunt* and *PrévisionnelAnnuel* are filled with zeros).

### Evaluation :
`$ python3 eval.py`


### Final predictions :
`$ python3 generate_prediction.py`

Generate final predictions in **data/predictions.csv*.