import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import clean


def predictor(game_file, game_data):
    '''
    A regression model which will predict how
    many users a game will have at some point
    '''
    total = len(game_data['Month'])
    game_data['Month Index'] = pd.DataFrame([x for x in range(total, 0, -1)])
    features = np.array(game_data['Month Index'].tolist()).reshape(-1, 1)
    labels = np.array(game_data['Avg. Players'].tolist())
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=.75)
    model = RandomForestRegressor(n_estimators=500, oob_score=True,
                                  random_state=100)
    model.fit(features_train, labels_train)
    prediction = model.predict(features_test)
    ac = plt.scatter(features_test, labels_test, c='b', marker='o')
    pred = plt.scatter(features_test, prediction, c='r', marker='x')
    title = "Predicting # Of Concurrent Players for {}"
    plt.title(title.format(game_file[:-4]))
    plt.legend((ac, pred), ('Actual Values', 'Predicted Values'))
    plt.show()
    print("Random Regressor Tree Score:", r2_score(labels_test, prediction))


def dead_or_not(game_file, game_data):
    '''
    A regression model which will predict at what point
    in the future will a game be considered "dead" in terms
    of number of players
    '''
    total = len(game_data['Month'])
    game_data['Month Index'] = pd.DataFrame([x for x in range(total, 0, -1)])
    most_player = np.max(game_data['Avg. Players'])
    death_level = most_player / 4
    features = np.array(game_data['Month Index'].tolist()).reshape(-1, 1)
    labels = np.array(game_data['Avg. Players'].tolist())
    future = np.arange(total+1, total+20, 1).reshape(-1, 1)
    model = LinearRegression()
    model.fit(features, labels)
    prediction_2 = model.predict(future)
    prediction = list(prediction_2)
    prediction = [y if y >= 0 else 0 for y in prediction]
    pst = plt.scatter(features, labels, c='r', marker='x')
    ftr = plt.scatter(future, prediction, c='g', marker='o')
    title = "Predicting # of Concurrent Players In The Future for {}"
    plt.title(title.format(game_file[:-4]))
    plt.legend((pst, ftr), ('Past Data', 'Expected Future Data'))
    plt.show()
    future_months = 1
    for x in prediction:
        if x < death_level:
            return 'In {} months, {} playerbase will be considered' \
                'dead'.format(future_months, game_file[:-4])
        else:
            future_months += 1

    return 'In the next {} months, we predict that this game {} will' \
        ' still have an active player base'.format(future_months,
                                                   game_file[:-4])


def average_predictor(game_file, game_data):
    '''
    We will determine if a game will be popular by
    doing a comparison between the avg # of players
    a game has in the first 6  months following its
    release compared to the avg # of players in the
    following 6 months.
    '''
    total = len(game_data['Month'])
    game_data['Month Index'] = pd.DataFrame([x for x in range(total, 0, -1)])
    first_6 = game_data[-6:]
    average_f6 = np.mean(first_6['Avg. Players'])
    next_6 = game_data[-12:-6]
    avg_next_6 = np.mean(next_6['Avg. Players'])
    if average_f6 < avg_next_6:
        return "This game, {}, is increasing in popularity"\
            " by {}%".format(game_file[:-4], (avg_next_6/average_f6)-1)
    else:
        return "This game, {}, appears not to be increasing"\
            " in popularity".format(game_file[:-4])


def linear_predictor(game_file, game_data):
    '''
    Add description here
    '''
    total = len(game_data['Month'])
    game_data['Month Index'] = pd.DataFrame([x for x in range(total, 0, -1)])
    actual_fy = game_data[-12:-7]
    first_6 = game_data[-6:]
    features = np.array(first_6['Month Index'].tolist()).reshape(-1, 1)
    labels = np.array(first_6['Avg. Players'].tolist())
    future = np.arange(7, 12, 1).reshape(-1, 1)
    model = LinearRegression()
    model.fit(features, labels)
    prediction = model.predict(future)
    prediction = list(prediction)
    prediction = [y if y >= 0 else 0 for y in prediction]
    pst = plt.scatter(features, labels, c='r', marker='x')
    expect = plt.scatter(future, prediction, c='g', marker='o')
    act = plt.scatter(future, actual_fy['Avg. Players'], c='r', marker='^')
    plt.title("Popularity Predictor for {}".format(game_file[:-4]))
    plt.legend((pst, expect, act), ('First 6 Months', 'Expected Next 6',
                                    'Actual Next 6'))
    plt.show()
    average_fy = np.mean(actual_fy['Avg. Players'])
    average_predicted = np.mean(prediction)
    if average_fy < average_predicted:
        return "This game, {}, is increasing in popularity"\
            " by {}%".format(game_file[:-4], (average_predicted/average_fy)-1)
    else:
        return "This game, {}, appears not to be increasing in"\
            " popularity".format(game_file[:-4])


def main():
    data = clean.load_directory_data('data')
    csgo = 'csgo.csv'
    predictor(csgo, data[csgo])
    print(dead_or_not(csgo, data[csgo]))
    print(average_predictor(csgo, data[csgo]))
    print(linear_predictor(csgo, data[csgo]))


if __name__ == '__main__':
    main()
