import numpy as np

class LinearElastic:
    def __init__(self, E, nu):
        self.E = E
        self.nu = nu

    def get_matrix(self, plane_type="strain"):
        E = self.E
        nu = self.nu

        if plane_type == "strain":
            factor = E / ((1.0 + nu) * (1.0 -2.0 * nu))
            return factor * np.array([[1.0 - nu, nu, 0.0], [nu, 1.0 - nu, 0.0], [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0]])
        else:
            factor = E / (1.0 - nu**2)
            return factor * np.array([[1.0, nu, 0.0],[nu, 1.0, 0.0],[0.0, 0.0, (1.0 - nu) / 2.0]])
