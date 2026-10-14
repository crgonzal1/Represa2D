import numpy


class DenseMatrix:
    def __init__(self, number_of_dofs):
        self.number_of_dofs = number_of_dofs
        self.values = numpy.zeros((number_of_dofs, number_of_dofs))

    def add(self, i, j, value):
        self.values[i][j] = self.values[i][j] + value

    def set(self, i, j, value):
        self.values[i][j] = value

    def scatter_local_to_global(self, element):
        Ke = element.stiffness_matrix()
        dofs = element.global_dofs()

        for i in range(len(dofs)):
            for j in range(len(dofs)):
                self.add(dofs[i], dofs[j], Ke[i][j])
