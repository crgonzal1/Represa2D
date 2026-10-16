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

        # Garantiza ordenamiento antihorario
        self._check_and_fix_winding()

    def _check_and_fix_winding(self):
        
        x = [node.x for node in self.nodes]
        y = [node.y for node in self.nodes]
        
        area = 0.5 * (
            (x[0]*y[1] - x[1]*y[0]) +
            (x[1]*y[2] - x[2]*y[1]) +
            (x[2]*y[3] - x[3]*y[2]) +
            (x[3]*y[0] - x[0]*y[3])
        )
        
        # Si el área es negativa, se invierte secuencia.
        if area < 0:
            self.nodes = [self.nodes[0], self.nodes[3], self.nodes[2], self.nodes[1]]

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

        # Validación de seguridad (identifica elementos incorrectamente nombrados)
        if det_jacobian <= 0:
            raise ValueError(f"Error topológico fatal en Elemento {self.id}: Jacobiano no positivo ({det_jacobian}). Posible elemento degenerado.")

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

    def compute_stress(self, u_global):
        # Calcula el tensor de tensiones y tensiones principales en el centroide
        # 1. Extraer desplazamientos locales de los 4 nodos (8 grados de libertad)
        u_local = np.zeros(8)
        dofs = self.global_dofs()
        for i, dof in enumerate(dofs):
            u_local[i] = u_global[dof]
            
        # 2. Evaluar matriz B en el centroide (xi=0.0, eta=0.0)
        nat_derivs = self.get_natural_derivatives(0.0, 0.0)
        _, inv_J, _ = self.get_jacobian(nat_derivs)
        B = self.get_b_matrix(nat_derivs, inv_J)
        
        # 3. Calcular deformación (B * u) y tensión (C * deformación)
        strain = B @ u_local
        C = self.material.get_matrix()
        stress = C @ strain  # [sigma_x, sigma_y, tau_xy]
        
        # 4. Calcular Tensiones Principales y Von Mises (Estado Plano de Deformación)
        sx, sy, txy = stress[0], stress[1], stress[2]
        center = (sx + sy) / 2.0
        radius = np.sqrt(((sx - sy) / 2.0)**2 + txy**2)
        sigma_1 = center + radius
        sigma_2 = center - radius
        
        # Tensión en Z
        sz = self.material.nu * (sx + sy)
        von_mises = np.sqrt(0.5 * ((sx - sy)**2 + (sy - sz)**2 + (sz - sx)**2 + 6 * txy**2))
        
        return stress, sigma_1, sigma_2, von_mises
