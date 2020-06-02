import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import pandas as pd


def future_predictor(game_file, testing_size):
    '''
    A linear regression model which will predict how
    many users a game will have at some point in the
    future
    '''
    data = pd.read_csv(game_file)
    total = len(data['Month'])
    data['Month Index'] = pd.DataFrame([x for x in range(total, 0, -1)])
    features = data['Month Index']
    labels = data['Avg. Players']
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=testing_size)
    #features_train = features_train.reshape(-1, 1)
    model = LinearRegression()
    print("month index:", data['Month Index'])
    print("features", features_train)
    print("lables", labels_train)
    model.fit(features_train, labels_train)




def main():
    future_predictor('data/csgo.csv', .5)


if __name__ == '__main__':
    main()
