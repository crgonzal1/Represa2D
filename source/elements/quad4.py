import numpy as np

class Quad4:
    def __init__(self, element_id, nodes, material, thickness=1.0):
        # Inicializa el elemento Q4 isoparamétrico.
        # nodes: Lista de 4 objetos Node esperados en orden antihorario.

        self.id = element_id
        self.nodes = nodes
        self.material = material
        self.thickness = thickness
        self.dofs_per_node = 2

    def global_dofs(self):
        dofs = []
        for node in self.nodes:
            dofs.append(node.global_id * self.dofs_per_node)
            dofs.append(node.global_id * self.dofs_per_node + 1)
        return dofs

    def get_natural_derivatives(self, xi, eta):
        # Calcula las derivadas parciales de las funciones de forma respecto a xi y eta.
        # Retorna una matriz de dimensiones 2x4.

        # Derivadas de Ni respecto a xi
        dN_dxi = np.array([-(1.0 - eta) / 4.0, (1.0 - eta) / 4.0, (1.0 + eta) / 4.0, -(1.0 + eta) / 4.0])

        # Derivadas de Ni respecto a eta
        dN_deta = np.array([-(1.0 - xi) / 4.0, -(1.0 + xi) / 4.0, (1.0 + xi) / 4.0, (1.0 - xi) / 4.0])

        # Apila las filas para formar la matriz 2x4
        return np.vstack((dN_dxi, dN_deta))

    def get_jacobian(self, natural_derivatives):

        # Calcula la Matriz Jacobiana (2x2), su inversa y su determinante.

        # Matriz de coordenadas físicas de los nodos (4x2)
        coords = np.array([[node.x, node.y] for node in self.nodes])

        # El Jacobiano es el producto de las derivadas naturales (2x4) por las coordenadas (4x2)
        jacobian = natural_derivatives @ coords

        # Calcular determinante e inversa mediante álgebra lineal
        det_jacobian = np.linalg.det(jacobian)
        inv_jacobian = np.linalg.inv(jacobian)

        return jacobian, inv_jacobian, det_jacobian

    def get_b_matrix(self, natural_derivatives, inv_jacobian):

        # Construye la matriz cinemática B (3x8) relacionando deformaciones y desplazamientos.

        # Transformar derivadas naturales a cartesianas mediante el Jacobiano inverso (2x4)
        cartesian_derivatives = inv_jacobian @ natural_derivatives

        B = np.zeros((3, 8))
        for i in range(4):
            # Deformación normal X: dNi/dx
            B[0, 2*i] = cartesian_derivatives[0, i]
            # Deformación normal Y: dNi/dy
            B[1, 2*i + 1] = cartesian_derivatives[1, i]
            # Deformación por corte XY: dNi/dy y dNi/dx
            B[2, 2*i] = cartesian_derivatives[1, i]
            B[2, 2*i + 1] = cartesian_derivatives[0, i]

        return B

    def stiffness_matrix(self):

        # Calcula la matriz de rigidez K (8x8) usando integración de Gauss 2x2.

        k_matrix = np.zeros((8, 8))

        # Puntos y pesos de Gauss para regla 2x2
        a = 1.0 / np.sqrt(3)
        gauss_points = [-a, a]
        weights = [1.0, 1.0]

        # Matriz constitutiva del material (3x3)
        C = self.material.get_matrix()

        # Integración numérica (doble sumatoria)
        for i, xi in enumerate(gauss_points):
            for j, eta in enumerate(gauss_points):
                weight = weights[i] * weights[j]

                # 1. Derivadas naturales en el punto de Gauss actual
                nat_derivs = self.get_natural_derivatives(xi, eta)

                # 2. Jacobiano (se descarta la matriz original con _, solo se usa inversa y determinante)
                _, inv_J, det_J = self.get_jacobian(nat_derivs)

                # 3. Matriz B
                B = self.get_b_matrix(nat_derivs, inv_J)

                # 4. Acumulación iterativa de K
                k_matrix += B.T @ C @ B * det_J * self.thickness * weight

        return k_matrix