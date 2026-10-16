import numpy as np

class Tri6:
    def __init__(self, global_id, nodes, material, thickness=1.0):
        self.global_id = global_id
        self.nodes = nodes
        self.material = material
        self.thickness = thickness
        self.num_nodes = 6
        self.dofs_per_node = 2

        # Verificación y corrección de orientación topológica
        self._check_and_fix_winding()

    def _check_and_fix_winding(self):
        # Determina la orientación a partir del producto vectorial de las esquinas
        x0, y0 = self.nodes[0].x, self.nodes[0].y
        x1, y1 = self.nodes[1].x, self.nodes[1].y
        x2, y2 = self.nodes[2].x, self.nodes[2].y
        
        det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        
        if det < 0:
            # Invierte secuencia a sentido antihorario (esquinas y nodos medios)
            self.nodes = [
                self.nodes[0], self.nodes[2], self.nodes[1],
                self.nodes[5], self.nodes[4], self.nodes[3]
            ]

    def global_dofs(self):
        dofs = []
        for node in self.nodes:
            dofs.extend([node.global_id * self.dofs_per_node, node.global_id * self.dofs_per_node + 1])
        return dofs

    def get_shape_functions(self, xi, eta):
        L1 = 1.0 - xi - eta
        L2 = xi
        L3 = eta
        N = np.zeros(6)
        N[0] = L1 * (2 * L1 - 1)
        N[1] = L2 * (2 * L2 - 1)
        N[2] = L3 * (2 * L3 - 1)
        N[3] = 4 * L1 * L2
        N[4] = 4 * L2 * L3
        N[5] = 4 * L3 * L1
        return N

    def get_natural_derivatives(self, xi, eta):
        L1 = 1.0 - xi - eta
        dN = np.zeros((2, 6))
        # dN/dxi
        dN[0, 0] = -4 * L1 + 1
        dN[0, 1] = 4 * xi - 1
        dN[0, 2] = 0
        dN[0, 3] = 4 * (L1 - xi)
        dN[0, 4] = 4 * eta
        dN[0, 5] = -4 * eta
        # dN/deta
        dN[1, 0] = -4 * L1 + 1
        dN[1, 1] = 0
        dN[1, 2] = 4 * eta - 1
        dN[1, 3] = -4 * xi
        dN[1, 4] = 4 * xi
        dN[1, 5] = 4 * (L1 - eta)
        return dN

    def get_jacobian(self, dN):
        J = np.zeros((2, 2))
        for i in range(self.num_nodes):
            J[0, 0] += dN[0, i] * self.nodes[i].x
            J[0, 1] += dN[0, i] * self.nodes[i].y
            J[1, 0] += dN[1, i] * self.nodes[i].x
            J[1, 1] += dN[1, i] * self.nodes[i].y
        
        signed_det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        det_J = abs(signed_det)
        inv_J = np.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]]) / signed_det
        return J, inv_J, det_J

    def get_b_matrix(self, dN, inv_J):
        B = np.zeros((3, self.num_nodes * 2))
        dN_dx_dy = inv_J @ dN
        for i in range(self.num_nodes):
            B[0, 2*i]     = dN_dx_dy[0, i]
            B[1, 2*i + 1] = dN_dx_dy[1, i]
            B[2, 2*i]     = dN_dx_dy[1, i]
            B[2, 2*i + 1] = dN_dx_dy[0, i]
        return B

    def stiffness_matrix(self):
        K_e = np.zeros((12, 12))
        C = self.material.get_matrix()
        pts = [(1/6, 1/6), (2/3, 1/6), (1/6, 2/3)]
        wts = [1/6, 1/6, 1/6] 
        
        for i in range(3):
            xi, eta = pts[i]
            weight = wts[i]
            dN = self.get_natural_derivatives(xi, eta)
            _, inv_J, det_J = self.get_jacobian(dN)
            B = self.get_b_matrix(dN, inv_J)
            K_e += B.T @ C @ B * det_J * weight * self.thickness
        return K_e