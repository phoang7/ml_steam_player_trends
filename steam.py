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
    Takes a game_file (name of the game) and
    game_data (a DataFrame composed of two columns:
    months and average players at a specific month)
    and saves a graph of our model predicting how
    many users a game will have at some point for our test
    dataset and outputs the regressor score for our test
    dataset.
    '''
    total = len(game_data['Month'])
    game_data['Month Index'] = pd.DataFrame([x for x in range(total, 0, -1)])
    features = np.array(game_data['Month Index'].tolist()).reshape(-1, 1)
    labels = np.array(game_data['Avg. Players'].tolist())
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=.50)
    model = RandomForestRegressor(n_estimators=10)
    model.fit(features_train, labels_train)
    prediction = model.predict(features_test)
    ac = plt.scatter(features_test, labels_test, c='b', marker='o')
    pred = plt.scatter(features_test, prediction, c='r', marker='x')
    title = "Predicting # Of Concurrent Players for {}"
    plt.title(title.format(game_file[:-4]))
    plt.legend((ac, pred), ('Actual Values', 'Predicted Values'))
    plt.xlabel("Months Since Release Data")
    plt.ylabel("Number of Average Players")
    plt.savefig('graphs/' + game_file[:-4] + '.png')
    plt.clf()
    score = r2_score(labels_test, prediction)
    return 'The R Squared Score for {} is {}'.format(game_file[:-4], score)


def dead_or_not(game_file, game_data):
    '''
    Takes a game_file (name of the game) and
    game_data (a DataFrame composed of two columns:
    months and average players at a specific month) and saves
    a graph of our model which will predict at what point
    in the future (months following our datset, e.g. if our data has 30 months
    in the future is 31 months, 32 months, etc.) will a game be considered
    "dead" in terms of average number of players (a game is considered dead
    if at any point the number of average players is less than the peak
    average number of players divided by 4). Returns whether a game is
    considered "dead" at what month in the future or not in 20 future
    months interval.
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
    plt.xlabel("Months Since Release Data")
    plt.ylabel("Number of Average Players")
    plt.savefig('graphs/' + game_file[:-4] + '.png')
    plt.clf()
    future_months = 1
    for x in prediction:
        if x < death_level:
            return 'In {} month(s), the {} player base will be considered' \
                ' dead'.format(future_months, game_file[:-4])
        else:
            future_months += 1

    return 'In the next {} months, we predict that this game {} will' \
        ' still have an active player base'.format(future_months,
                                                   game_file[:-4])


def average_predictor(game_file, game_data):
    '''
    Takes a game_file (name of the game) and
    game_data (a DataFrame composed of two columns:
    months and average players at a specific month)
    and determines if a game will be popular  by
    doing a comparison between the avg # of players
    a game has in the first 6  months following its
    release compared to the avg # of players in the
    following 6 months (returns whether the game is
    predicted to increase in popularity at what percentage or not).
    A game is increasing in popularity if the avg # of players
    from 7-12 months is greater than the avg # of players from 1-6 months.
    '''
    total = len(game_data['Month'])
    game_data['Month Index'] = pd.DataFrame([x for x in range(total, 0, -1)])
    first_6 = game_data[-6:]
    average_f6 = np.mean(first_6['Avg. Players'])
    next_6 = game_data[-12:-6]
    avg_next_6 = np.mean(next_6['Avg. Players'])
    if average_f6 < avg_next_6:
        return "This game, {}, is increasing in popularity (average)"\
            " by {}%".format(game_file[:-4], ((avg_next_6/average_f6)-1) * 100)
    else:
        return "This game, {}, appears not to be increasing"\
            " in popularity (average)".format(game_file[:-4])


def linear_predictor(game_file, game_data):
    '''
    Takes a game_file (name of the game) and
    game_data (a DataFrame composed of two columns:
    months and average players at a specific month) and
    saves a graph of our model predicting the linear trend of
    the number of average players for 7-12 months based on
    the first 6 months (1-6 months). Returns whether the game is
    predicted to increase in popularity at what percentage or not
    by comparing the avg number of players from 1-6 months to
    the predicted avg number of players from 7-12 months.
    A game is increasing in popularity if the predicted avg # of players
    from 7-12 months is greater than the avg # of players from 1-6 months.
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
    title = "Popularity Predictor for {} (linear regression)"
    plt.title(title.format(game_file[:-4]))
    plt.legend((pst, expect, act), ('First 6 Months', 'Expected Next 6',
                                    'Actual Next 6'))
    plt.xlabel("Months Since Release Data")
    plt.ylabel("Number of Average Players")
    plt.savefig('graphs/' + game_file[:-4] + '.png')
    plt.clf()
    average_fy = np.mean(actual_fy['Avg. Players'])
    average_predicted = np.mean(prediction)
    percentage = ((average_predicted/average_fy)-1)*100
    if average_fy < average_predicted:
        return "This game, {}, is increasing in popularity"\
            " by {}% (linear)".format(game_file[:-4], percentage)
    else:
        return "This game, {}, appears not to be increasing in"\
            " popularity (linear)".format(game_file[:-4])


def part1(data):
    '''
    Takes data (a dictionary that has the game_file
    (the name of the data) and a reference to the data)
    and does our analysis for our 1st research question.
    '''
    nba20 = 'nba20.csv'
    sims = 'sims.csv'
    csgo = 'csgo.csv'
    dsiii = 'dsiii.csv'
    print(predictor(nba20, data[nba20]))
    print(predictor(sims, data[sims]))
    print(predictor(csgo, data[csgo]))
    print(predictor(dsiii, data[dsiii]))


def part2(data):
    '''
    Takes data (a dictionary that has the game_file
    (name of our data) and a reference to the data)
    and does our analysis for our 2nd research question.
    '''
    cod = 'cod.csv'
    undertale = 'undertale.csv'
    tf2 = 'tf2.csv'
    nba19 = 'nba19.csv'
    print(dead_or_not(cod, data[cod]))
    print(dead_or_not(undertale, data[undertale]))
    print(dead_or_not(tf2, data[tf2]))
    print(dead_or_not(nba19, data[nba19]))


def part3(data):
    '''
    Takes data (a dictionary that has the game_file
    (name of our data)and a reference to the data)
    and does our analysis for our 2nd research question.
    '''
    terraria = 'terraria.csv'
    rb6 = 'rb6.csv'
    gta = 'gta.csv'
    portal = 'portal.csv'
    print(average_predictor(terraria, data[terraria]))
    print(linear_predictor(terraria, data[terraria]))
    print(average_predictor(rb6, data[rb6]))
    print(linear_predictor(rb6, data[rb6]))
    print(average_predictor(gta, data[gta]))
    print(linear_predictor(gta, data[gta]))
    print(average_predictor(portal, data[portal]))
    print(linear_predictor(portal, data[portal]))


def main():
    data = clean.load_directory_data('data')
    part1(data)
    part2(data)
    part3(data)


if __name__ == '__main__':
    main()
