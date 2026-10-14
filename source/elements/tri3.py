import numpy as np

class Tri3:
    def __init__(self, element_id, nodes, material, thickness=1.0):
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

    def stiffness_matrix(self):
        x1, y1 = self.nodes[0].x, self.nodes[0].y
        x2, y2 = self.nodes[1].x, self.nodes[1].y
        x3, y3 = self.nodes[2].x, self.nodes[2].y

        # Cálculo del Área mediante el determinante jacobiano
        det_J = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        area = abs(det_J) / 2.0

        # Coeficientes geométricos (diferencias de coordenadas)
        b1, b2, b3 = y2 - y3, y3 - y1, y1 - y2
        c1, c2, c3 = x3 - x2, x1 - x3, x2 - x1

        # Matriz Cinemática B (3x6) para deformación constante
        B = (1.0 / (2.0 * area)) * np.array([
            [ b1, 0.0,  b2, 0.0,  b3, 0.0],
            [0.0,  c1, 0.0,  c2, 0.0,  c3],
            [ c1,  b1,  c2,  b2,  c3,  b3]
        ])

        C = self.material.get_matrix()

        # Integración exacta (valores constantes multiplicados por el área)
        K = B.T @ C @ B * area * self.thickness
        return K
