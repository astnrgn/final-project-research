# Following https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)


# shape (how many colums (i.e. attributes) the data contains)[(instances, attributes)]
print(dataset.shape)


# head (look at the first 20 rows of data)
print(dataset.head(20))


# descriptions (includes more info on the data, lik mean and min)
print(dataset.describe())


# class distribution (number of instances (datapoints or rows) in each class)
print(dataset.groupby('class').size())


# box and whisker plots (for total dataset)
dataset.plot(kind='box', subplots=True, layout=(
    2, 2), sharex=False, sharey=False)
plt.show()


# histograms (of data)
dataset.hist()
plt.show()


# scatter plot matrix
scatter_matrix(dataset)
plt.show()
