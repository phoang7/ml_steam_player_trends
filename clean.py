import pandas as pd
import os

# Note: We use macbooks and hate .DS_Store so we want to ignore if it pops up
ignore = '.DS_Store'


def load_game_data(game_file):
    '''
    Takes a game path as data and returns the data
    that is loaded and filters out the columns
    we do not need. We only need the month timestamp
    and the average
    amount of players at that particular month
    '''
    df = pd.read_csv(game_file)
    df = df[['Month', 'Avg. Players']]
    return df


def load_directory_data(dir_name):
    '''
    Takes a directory path and returns a dictionary
    of the game file name with its data for each game file
    in the passed directory
    '''
    files = os.listdir(dir_name)
    game_data = {}
    if ignore in files:
        files.remove(ignore)

    for file in files:
        game_data[file] = load_game_data(dir_name + '/' + file)

    return game_data


def main():
    load_game_data('data/csgo.csv')
    load_directory_data('data')


if __name__ == '__main__':
    main()
