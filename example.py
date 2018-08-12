import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from darwin.being import CategoricalBeing
from darwin.slaughterhouse import BasicSlaughterhouse


# read data
titanic = pd.read_csv("titanic.csv")
x = np.array(titanic.drop(['Survived'], axis=1))
y = np.array(titanic['Survived'])

# one hot encode y
y2 = [[0, 1] if elem == 1 else [1, 0] for elem in y]
y2 = np.array(y2)

x_train, x_test, y_train, y_test = train_test_split(x, y2, test_size=.2)

# create template
template = CategoricalBeing(x.shape[1], classes=2, lr=1, variable_power=6, weight_sigma=5, metric='acc')

# create slaughterhouse
house = BasicSlaughterhouse(template, population=1000, batch_size=32, decay=.99)

# do machine learning
final = house.go(x_train, y_train, generations=10)

# score the best being with test data
test_accuracy = final[0].evaluate(x_test, y_test, metric='acc')
print(test_accuracy)

# for data lovers, print out the actual regression in a very ugly form
print(final[0].variables, final[0].weights, final[0].e, final[0].eW, final[0].ln, final[0].lnW)
