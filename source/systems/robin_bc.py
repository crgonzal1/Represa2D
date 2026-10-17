import numpy as np

def apply_robin_boundary_conditions(mesh, K_global, k_n=1.0e9, k_t=5.0e8, y_tol=0.1):
    # Agrega la rigidez elástica del suelo (Robin BC) a la matriz global.
    
    y_min = min(node.y for node in mesh.nodes)
    base_nodes = [node for node in mesh.nodes if abs(node.y - y_min) < y_tol]
    base_nodes_sorted = sorted(base_nodes, key=lambda n: n.x)
    
    # Integración de Gauss-Legendre (2 puntos)
    gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
    gauss_wts = [1.0, 1.0]
    
    for i in range(len(base_nodes_sorted) - 1):
        n1 = base_nodes_sorted[i]
        n2 = base_nodes_sorted[i+1]
        
        L = abs(n2.x - n1.x)
        if L < 1e-6:
            continue
            
        k_elem = np.zeros((4, 4))
        for xi, w in zip(gauss_pts, gauss_wts):
            N1 = 0.5 * (1.0 - xi)
            N2 = 0.5 * (1.0 + xi)
            N = np.array([N1, N2])
            J = L / 2.0
            
            for r in range(2):
                for c in range(2):
                    # Rigidez horizontal (corte)
                    k_elem[2*r, 2*c] += k_t * N[r] * N[c] * J * w
                    # Rigidez vertical (normal)
                    k_elem[2*r+1, 2*c+1] += k_n * N[r] * N[c] * J * w
        
        dofs_n1 = mesh.node_global_dofs(n1.global_id)
        dofs_n2 = mesh.node_global_dofs(n2.global_id)
        elem_dofs = [dofs_n1[0], dofs_n1[1], dofs_n2[0], dofs_n2[1]]
        
        # Sumar la rigidez del resorte a la matriz global
        for r_idx, g_r in enumerate(elem_dofs):
            for c_idx, g_c in enumerate(elem_dofs):
                if hasattr(K_global, 'add_val'):
                    # Si es SparseMatrix
                    K_global.add_val(g_r, g_c, k_elem[r_idx, c_idx])
                elif hasattr(K_global, 'values'):
                    # Si es DenseMatrix
                    K_global.values[g_r][g_c] += k_elem[r_idx, c_idx]
                else:
                    # Si es un ndarray directo de NumPy
                    K_global[g_r, g_c] += k_elem[r_idx, c_idx]