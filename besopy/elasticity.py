import numpy as np


class ElasticityMatrix:
    def __init__(self, poisson=1, young=1):
        self.poisson = poisson
        self.young = young

    def get_matrix(self):
        elasticity_factor = (self.young / ((1 + self.poisson) * (1 - 2 * self.poisson)))

        return elasticity_factor * np.matrix([
            [1 - self.poisson, self.poisson, self.poisson, 0, 0, 0],
            [self.poisson, 1 - self.poisson, self.poisson, 0, 0, 0],
            [self.poisson, self.poisson, 1 - self.poisson, 0, 0, 0],
            [0, 0, 0, (1 - 2 * self.poisson) / 2, 0, 0],
            [0, 0, 0, 0, (1 - 2 * self.poisson) / 2, 0],
            [0, 0, 0, 0, 0, (1 - 2 * self.poisson) / 2]])

