from src.train import *

train('reddit', 470, [1, 0, 1, 0], 602, 128, 41, prune=None, multi_class=False)
train('yelp', 2359, [1, 1, 0], 300, 512, 100, prune=None, multi_class=True)
train('ppi', 624, [1, 0, 1, 0], 50, 512, 121, prune=None, multi_class=True)
