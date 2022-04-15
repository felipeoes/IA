import pandas as pd


class MultilayerPerceptron(object):
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame):
        self.X = X
        self.y = y
