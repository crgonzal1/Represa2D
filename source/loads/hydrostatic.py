import numpy as np

def apply_hydrostatic_pressure(mesh, f, rho_water=1000.0, g=9.81, h_water=28.0):
    # 1. Determinar automáticamente la coordenada X del paramento (mínimo X de la malla)
    x_min = min(node.x for node in mesh.nodes)
    
    # 2. Seleccionar nodos en el paramento sumergido (X próximo a x_min e Y <= h_water)
    upstream_nodes = []
    for i in range(mesh.number_of_nodes()):
        node = mesh.node(i)
        if abs(node.x - x_min) < 1e-3 and node.y <= h_water + 1e-3:
            upstream_nodes.append(node)

    # Ordenar los nodos verticalmente de abajo hacia arriba (por coordenada Y)
    upstream_nodes.sort(key=lambda n: n.y)

    if len(upstream_nodes) < 2:
        print(f"ADVERTENCIA: No se encontraron suficientes nodos en X = {x_min}")
        return

    # 3. Integrar la presión tramo por tramo entre nodos contiguos
    total_force_x = 0.0
    for i in range(len(upstream_nodes) - 1):
        n1 = upstream_nodes[i]
        n2 = upstream_nodes[i+1]

        y1, y2 = n1.y, n2.y
        L = abs(y2 - y1)
        
        if L < 1e-8:
            continue

        # Presión hidrostática en la posición de cada nodo: P = rho * g * h
        p1 = rho_water * g * max(0.0, h_water - y1)
        p2 = rho_water * g * max(0.0, h_water - y2)

        # Integración de carga trapezoidal en 1D a fuerzas nodales equivalentes
        f1_x = (2.0 * p1 + p2) * L / 6.0
        f2_x = (p1 + 2.0 * p2) * L / 6.0

        # Asignación a los grados de libertad globales X (DOF = 2 * global_id)
        dof1_x = n1.global_id * 2
        dof2_x = n2.global_id * 2

        f[dof1_x] += f1_x
        f[dof2_x] += f2_x

        total_force_x += (f1_x + f2_x)


    print(f"Fuerza horizontal resultante en la represa: {total_force_x:.4e} N")
