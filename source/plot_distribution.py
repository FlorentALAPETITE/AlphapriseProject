import pandas
from sklearn.neighbors.kde import KernelDensity
import numpy as np
from matplotlib import pyplot
import scipy.stats as stats
from scipy.stats import norm
from scipy.stats.mstats import winsorize
from collections import Counter


EMPRUNT = 5
PREVISION = 6
SECTEUR1 = 1
SECTEUR2 = 2
SECTEUR_PARTICULIER = 3


def plot_normal_distribution(numeric_data):

	i=1
	for data in numeric_data:	
		mu, std = norm.fit(data)
		
		pyplot.hist(data, bins=25, normed=True, alpha=0.6, color='b')

		xmin, xmax = pyplot.xlim()
		x = np.linspace(xmin, xmax, 100)
		p = norm.pdf(x, mu, std)
		pyplot.plot(x, p, 'k', linewidth=2)

		if i==1:
			pyplot.title("Distribution normale de la variable CapaciteEmprunt")
		else:
			pyplot.title("Distribution normale de la variable PrevisionnelAnnuel")

		pyplot.legend()
		pyplot.show()
		i+=1



def plot_binary_distribution(full_dataset):		
	count_secteur = []
	colors = ['red', 'green', 'blue']

	count_secteur.append(Counter(full_dataset.iloc[:, 1]))
	count_secteur.append(Counter(full_dataset.iloc[:, 2]))
	count_secteur.append(Counter(full_dataset.iloc[:, 3]))

	i = 1

	for count in count_secteur:
		sub = pyplot.subplot(1, 3, i)

		if i==1:
			sub.set_title("Distribution de Secteur1")
		elif i==2:
			sub.set_title("Distribution de Secteur2")
		elif i==3:
			sub.set_title("Distribution de SecteurParticulier")
		sub.pie(list(count.values()), labels=list(count.keys()))
		sub.legend()
		i+=1


	pyplot.suptitle("Distribution des variables binaires")
	pyplot.legend()
	pyplot.show()


def apply_winsorisation(full_dataset):
	numeric_data = [sorted(full_dataset.iloc[:, EMPRUNT]),sorted(full_dataset.iloc[:, PREVISION])]
	winorized_data = [winsorize(numeric_data[0],limits=(0,0.05)), winsorize(numeric_data[1],limits=(0,0.05))]	

	plot_normal_distribution(winorized_data)



def plot_data_distribution():
	np.random.seed(1)
	full_training_dataframe = pandas.read_csv("../data/cleaned_learning.csv", sep=";")

	numeric_data = [sorted(full_training_dataframe.iloc[:, EMPRUNT]),sorted(full_training_dataframe.iloc[:, PREVISION])]

	plot_normal_distribution(numeric_data)
	plot_binary_distribution(full_training_dataframe)
	apply_winsorisation(full_training_dataframe)


plot_data_distribution()

