import numpy as np

def apply_hydrostatic_pressure(mesh, elements, f, rho_water=1000.0, g=9.81, h_water=28.0):
    total_force_x = 0.0
    total_force_y = 0.0
    
    # 1. Extraer todas las aristas para identificar los contornos del dominio
    edge_counts = {}
    element_edges = []
    
    for element in elements:
        n_nodes = len(element.nodes)
        
        # Mapeo de índices de las aristas (nodos de esquina y nodos medios si existen)
        if n_nodes == 3:
            edges_idx = [(0, 1), (1, 2), (2, 0)]
        elif n_nodes == 4:
            edges_idx = [(0, 1), (1, 2), (2, 3), (3, 0)]
        elif n_nodes == 6:
            edges_idx = [(0, 3, 1), (1, 4, 2), (2, 5, 0)]
        elif n_nodes == 8:
            edges_idx = [(0, 4, 1), (1, 5, 2), (2, 6, 3), (3, 7, 0)]
        else:
            edges_idx = []
            
        el_edges = []
        for e_idx in edges_idx:
            n1 = element.nodes[e_idx[0]]
            n2 = element.nodes[e_idx[-1]]
            edge_key = tuple(sorted((n1.global_id, n2.global_id)))
            edge_counts[edge_key] = edge_counts.get(edge_key, 0) + 1
            el_edges.append((edge_key, e_idx))
        element_edges.append(el_edges)
        
    # 2. Integración de Gauss 1D a lo largo de las caras mojadas
    pts = [-np.sqrt(0.6), 0.0, np.sqrt(0.6)]
    wts = [5.0/9.0, 8.0/9.0, 5.0/9.0]
    
    for i, element in enumerate(elements):
        # Evaluar el sentido de los nodos del elemento (Horario o Antihorario) usando el Teorema de Green
        area = 0.0
        n_nodes = len(element.nodes)
        n_corners = 3 if n_nodes in [3, 6] else 4
        for j in range(n_corners):
            nj = element.nodes[j]
            nj_next = element.nodes[(j + 1) % n_corners]
            area += nj.x * nj_next.y - nj_next.x * nj.y
        is_ccw = area > 0
        
        for edge_key, e_idx in element_edges[i]:
            if edge_counts[edge_key] == 1:  # Es arista de contorno
                
                n1 = element.nodes[e_idx[0]]
                n2 = element.nodes[e_idx[-1]]
                
                # Definir normal exterior
                dx = n2.x - n1.x
                dy = n2.y - n1.y
                
                if is_ccw:
                    nx, ny = dy, -dx
                else:
                    nx, ny = -dy, dx
                    
                length = np.hypot(nx, ny)
                if length == 0: continue
                nx /= length
                ny /= length
                
                # Restringir a cara "Aguas Arriba": 
                # (nx < -1e-3) localiza fronteras orientadas hacia el agua.
                if nx < -1e-3 and min(n1.y, n2.y) <= h_water:
                    n_edge_nodes = len(e_idx)
                    is_linear = (n_edge_nodes == 2)
                    
                    for xi, w in zip(pts, wts):
                        # Funciones de interpolación 1D
                        if is_linear:
                            N1D = [0.5 * (1 - xi), 0.5 * (1 + xi)]
                        else:
                            N1D = [-0.5 * xi * (1 - xi), 1.0 - xi**2, 0.5 * xi * (1 + xi)]
                            
                        # Y interpolada en el punto de Gauss
                        y_gauss = sum(N1D[k] * element.nodes[e_idx[k]].y for k in range(n_edge_nodes))
                        
                        if y_gauss > h_water:
                            continue  # Descartar aportes por encima del nivel hidrostático
                            
                        p_gauss = rho_water * g * (h_water - y_gauss)
                        det_J_1D = length / 2.0
                        
                        for k in range(n_edge_nodes):
                            node = element.nodes[e_idx[k]]
                            # Fuerzas nodales equivalentes proyectadas (-n)
                            fx = -nx * p_gauss * det_J_1D * w * N1D[k]
                            fy = -ny * p_gauss * det_J_1D * w * N1D[k]
                            
                            f[node.global_id * 2] += fx
                            f[node.global_id * 2 + 1] += fy
                            
                            total_force_x += fx
                            total_force_y += fy
                            
    print(f"Fuerza hidrostática - Fuerza horizontal: {total_force_x:.4e} N")
    print(f"Fuerza hidrostática - Fuerza vertical: {total_force_y:.4e} N")