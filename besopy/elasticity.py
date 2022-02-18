import numpy as np

class ElasticityMatrix:

    """ ---------------------------------------------
        Elastic Matrix for Linear Isotropic Material

        ν: Poisson's Ratio
        λ: Young's Modulus  
    ---------------------------------------------- """   

    def __init__(self, ν: float = 1.0, λ: float = 1.0):
        self.ν = ν
        self.λ = λ

    def get_matrix(self, ν: float = 1.0, λ: float = 1.0):
  
        elasticity_factor = (λ / ((1 + ν) * (1 - 2 * ν)))

        elasticity_matrix = np.matrix([
            [1-ν,  ν,  ν,  0,  0,  0],
            [ν,  1-ν,  ν,  0,  0,  0],
            [ν,  ν,  1-ν,  0,  0,  0],
            [0,  0,  0,  1-2*ν, 0, 0],
            [0,  0,  0,  0, 1-2*ν, 0],
            [0,  0,  0,  0, 0, 1-2*ν]
        ])

        return elasticity_factor * elasticity_matrix

print(ElasticityMatrix().get_matrix())