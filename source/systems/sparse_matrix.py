import numpy as np
from scipy.sparse import coo_matrix

class SparseMatrix:
    def __init__(self, number_of_dofs):
        self.number_of_dofs = number_of_dofs
        self.row_indices = []
        self.col_indices = []
        self.data_values = []
        self.matrix_csr = None
        self.is_assembled = False

    def scatter_local_to_global(self, element):
        # Inserta la matriz de rigidez elemental en las listas de coordenadas globales
        if self.is_assembled:
            raise ValueError("La matriz ya fue ensamblada. No se pueden agregar más elementos.")

        K_e = element.stiffness_matrix()
        dofs = element.global_dofs()

        for i, global_i in enumerate(dofs):
            for j, global_j in enumerate(dofs):
                self.row_indices.append(global_i)
                self.col_indices.append(global_j)
                self.data_values.append(K_e[i, j])

    def assemble(self):
        # Construye la matriz dispersa CSR a partir de las listas de coordenadas
        if not self.is_assembled:
            coo = coo_matrix((self.data_values, (self.row_indices, self.col_indices)), 
                             shape=(self.number_of_dofs, self.number_of_dofs))
            self.matrix_csr = coo.tocsr()
            self.is_assembled = True

    def apply_dirichlet(self, dof, value, f):
        # Aplica condiciones de contorno de Dirichlet modificando la matriz CSR
        if not self.is_assembled:
            self.assemble()

        # Forzar ceros en la fila 'dof'
        row_start = self.matrix_csr.indptr[dof]
        row_end = self.matrix_csr.indptr[dof+1]
        for j in range(row_start, row_end):
            self.matrix_csr.data[j] = 0.0

        # Forzar ceros en la columna 'dof' y ajustar vector de fuerzas
        for i in range(self.number_of_dofs):
            # Extraer el valor de la matriz antes de modificar
            val = self.matrix_csr[i, dof] 
            if val != 0.0:
                self.matrix_csr[i, dof] = 0.0
                f[i] -= val * value

        # Poner 1 en la diagonal principal
        self.matrix_csr[dof, dof] = 1.0
        f[dof] = value

    def solve(self, f):
        # Resuelve el sistema lineal K * u = f usando un solver disperso
        from scipy.sparse.linalg import spsolve
        if not self.is_assembled:
            self.assemble()
        return spsolve(self.matrix_csr, f)