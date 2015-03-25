#!/usr/bin/env python

import sys
import json

from matplotlib import pyplot as plt

from util import joke

def get_data():
    with open('./data/jokes_sickipedia.json') as f:
        contents = json.load(f)
    return [joke.Joke(**c) for c in contents]

def main(args):
    data = get_data()
    scores = [d.rating for d in data if d.rating < 100]
    plt.figure()
    plt.hist(scores, 20)
    plt.title('Score histogram')
    plt.show()


if __name__ == '__main__':
    sys.exit(main(sys.argv))
