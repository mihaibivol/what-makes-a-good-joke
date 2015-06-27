#!/usr/bin/env python


import sys
import json
import math
import numpy

from matplotlib import pyplot as plt
from matplotlib import mlab

from util import joke

def get_data():
    with open('./data/jokes_sickipedia.json') as f:
        contents = json.load(f)
    return [joke.Joke(**c) for c in contents]

def main(args):
    data = get_data()
    scores = [math.log(d.rating) for d in data if d.rating]
    maxs = float(max(scores))
    mins = float(min(scores))
    scores = [(s - mins) / (maxs - mins) for s in scores]

    mean = numpy.mean(scores)
    variance = numpy.var(scores)
    sigma = numpy.sqrt(variance)
    x = numpy.linspace(min(scores), max(scores),100)
    plt.plot(x,mlab.normpdf(x,mean,sigma))

    plt.hist(scores, 20, normed=True)
    plt.title('Score histogram')
    plt.show()


if __name__ == '__main__':
    sys.exit(main(sys.argv))
