import numpy as np

class Quad8:
    def __init__(self, global_id, nodes, material, thickness=1.0):
        self.global_id = global_id
        self.nodes = nodes
        self.material = material
        self.thickness = thickness
        self.num_nodes = 8
        self.dofs_per_node = 2

    def global_dofs(self):
        dofs = []
        for node in self.nodes:
            dofs.extend([node.global_id * self.dofs_per_node, node.global_id * self.dofs_per_node + 1])
        return dofs

    def get_shape_functions(self, xi, eta):
        # Funciones de forma
        N = np.zeros(8)
        N[0] = 0.25 * (1 - xi) * (1 - eta) * (-xi - eta - 1)
        N[1] = 0.25 * (1 + xi) * (1 - eta) * ( xi - eta - 1)
        N[2] = 0.25 * (1 + xi) * (1 + eta) * ( xi + eta - 1)
        N[3] = 0.25 * (1 - xi) * (1 + eta) * (-xi + eta - 1)
        N[4] = 0.5 * (1 - xi**2) * (1 - eta)
        N[5] = 0.5 * (1 + xi) * (1 - eta**2)
        N[6] = 0.5 * (1 - xi**2) * (1 + eta)
        N[7] = 0.5 * (1 - xi) * (1 - eta**2)
        return N

    def get_natural_derivatives(self, xi, eta):
        # Derivadas naturales
        dN = np.zeros((2, 8))
        
        # dN / dxi
        dN[0, 0] = 0.25 * (1 - eta) * (2 * xi + eta)
        dN[0, 1] = 0.25 * (1 - eta) * (2 * xi - eta)
        dN[0, 2] = 0.25 * (1 + eta) * (2 * xi + eta)
        dN[0, 3] = 0.25 * (1 + eta) * (2 * xi - eta)
        dN[0, 4] = -xi * (1 - eta)
        dN[0, 5] = 0.5 * (1 - eta**2)
        dN[0, 6] = -xi * (1 + eta)
        dN[0, 7] = -0.5 * (1 - eta**2)

        # dN / deta
        dN[1, 0] = 0.25 * (1 - xi) * (xi + 2 * eta)
        dN[1, 1] = 0.25 * (1 + xi) * (-xi + 2 * eta)
        dN[1, 2] = 0.25 * (1 + xi) * (xi + 2 * eta)
        dN[1, 3] = 0.25 * (1 - xi) * (-xi + 2 * eta)
        dN[1, 4] = -0.5 * (1 - xi**2)
        dN[1, 5] = -eta * (1 + xi)
        dN[1, 6] = 0.5 * (1 - xi**2)
        dN[1, 7] = -eta * (1 - xi)
        
        return dN

    def get_jacobian(self, dN):
        J = np.zeros((2, 2))
        for i in range(self.num_nodes):
            J[0, 0] += dN[0, i] * self.nodes[i].x
            J[0, 1] += dN[0, i] * self.nodes[i].y
            J[1, 0] += dN[1, i] * self.nodes[i].x
            J[1, 1] += dN[1, i] * self.nodes[i].y
        
        # Determinante signado para la inversa, valor absoluto para el volumen
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
        K_e = np.zeros((16, 16))
        C = self.material.get_matrix()
        
        pts = [-np.sqrt(0.6), 0.0, np.sqrt(0.6)]
        wts = [5.0/9.0, 8.0/9.0, 5.0/9.0]
        
        for i, xi in enumerate(pts):
            for j, eta in enumerate(pts):
                weight = wts[i] * wts[j]
                dN = self.get_natural_derivatives(xi, eta)
                _, inv_J, det_J = self.get_jacobian(dN)
                B = self.get_b_matrix(dN, inv_J)
                
                K_e += B.T @ C @ B * det_J * weight * self.thickness
        return K_e

    def compute_stress(self, u_global):
        u_local = np.zeros(16)
        dofs = self.global_dofs()
        for i, dof in enumerate(dofs):
            u_local[i] = u_global[dof]
            
        nat_derivs = self.get_natural_derivatives(0.0, 0.0)
        _, inv_J, _ = self.get_jacobian(nat_derivs)
        B = self.get_b_matrix(nat_derivs, inv_J)
        
        strain = B @ u_local
        C = self.material.get_matrix()
        stress = C @ strain
        
        sx, sy, txy = stress[0], stress[1], stress[2]
        center = (sx + sy) / 2.0
        radius = np.sqrt(((sx - sy) / 2.0)**2 + txy**2)
        sigma_1 = center + radius
        sigma_2 = center - radius
        
        sz = self.material.nu * (sx + sy)
        von_mises = np.sqrt(0.5 * ((sx - sy)**2 + (sy - sz)**2 + (sz - sx)**2 + 6 * txy**2))
        
        return stress, sigma_1, sigma_2, von_mises