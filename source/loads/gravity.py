import numpy as np

def apply_self_weight(elements, f, density=2400.0, g=9.81, thickness=1.0):
    # Calcula el peso propio de cada elemento mediante el área del polígono y distribuye la fuerza vertical (-Y) equitativamente en sus nodos.

    total_weight = 0.0
    
    for element in elements:
        # Extraer coordenadas
        x = [node.x for node in element.nodes]
        y = [node.y for node in element.nodes]
        n_nodes = len(element.nodes)
        
        # Calcular área usando la fórmula del polígono
        area = 0.0
        for i in range(n_nodes):
            j = (i + 1) % n_nodes
            area += x[i] * y[j] - x[j] * y[i]
        area = abs(area) / 2.0
        
        # Peso total del elemento = Área * espesor * densidad * gravedad
        weight = area * thickness * density * g
        total_weight += weight
        
        # Fuerza nodal equivalente (distribuida o 'lumped')
        nodal_force = weight / n_nodes
        
        for node in element.nodes:
            dof_y = node.global_id * 2 + 1  # Grado de libertad en Y
            f[dof_y] -= nodal_force         # Dirección descendente negativa
            
    print(f"Peso propio aplicado. Fuerza vertical total: {-total_weight:.4e} N")
