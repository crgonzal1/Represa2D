import numpy as np

def apply_hydrostatic_pressure(mesh, elements, f, rho_water=1000.0, g=9.81, h_water=28.0):
    # Calcula el vector de fuerzas consistentes para la presión hidrostática integrando numéricamente sobre los bordes mojados de cualquier elemento (Q4, Q8, Tri3, Tri6).
    x_min = min(node.x for node in mesh.nodes)
    total_force_x = 0.0
    
    # Puntos y pesos de Gauss 1D (3 puntos integran polinomios de hasta grado 5 exactamente)
    pts = [-np.sqrt(0.6), 0.0, np.sqrt(0.6)]
    wts = [5.0/9.0, 8.0/9.0, 5.0/9.0]
    
    for element in elements:
        wet_nodes = []
        for node in element.nodes:
            if abs(node.x - x_min) < 1e-3 and node.y <= h_water + 1e-3:
                wet_nodes.append(node)
                
        n_tot = len(element.nodes)
        n_wet = len(wet_nodes)
        
        # Detecta si es un borde lineal (Q4, Tri3) o cuadrático (Q8, Tri6)
        is_linear_edge = (n_tot in [3, 4] and n_wet == 2)
        is_quadratic_edge = (n_tot in [6, 8] and n_wet == 3)
        
        if is_linear_edge or is_quadratic_edge:
            
            # Ordenar los nodos mojados por su coordenada Y (de abajo hacia arriba)
            wet_nodes.sort(key=lambda n: n.y)
            y_bottom = wet_nodes[0].y
            y_top = wet_nodes[-1].y
            L = y_top - y_bottom
            
            if L <= 0: continue
            
            f_edge = np.zeros(n_wet)
            
            # Integración de Gauss 1D a lo largo de la cara
            for xi, w in zip(pts, wts):
                # Mapeo de xi (-1 a 1) a coordenada y real
                y_gauss = y_bottom + (xi + 1.0) * L / 2.0
                
                # Presión hidrostática en el punto de Gauss
                p_gauss = rho_water * g * max(0.0, h_water - y_gauss)
                
                # Funciones de forma 1D
                if is_linear_edge:
                    N1D = [0.5 * (1 - xi), 0.5 * (1 + xi)]
                else:
                    N1D = [-0.5 * xi * (1 - xi), 1.0 - xi**2, 0.5 * xi * (1 + xi)]
                
                # Jacobiano 1D (dx/dxi = L/2)
                det_J_1D = L / 2.0
                
                # Acumular fuerza en los nodos locales del borde
                for k in range(n_wet):
                    f_edge[k] += N1D[k] * p_gauss * det_J_1D * w
            
            # Ensamblar en el vector global
            for k, node in enumerate(wet_nodes):
                dof_x = node.global_id * 2
                f[dof_x] += f_edge[k]
                total_force_x += f_edge[k]

    print(f"Fuerza hidrostática - Fuerza horizontal: {total_force_x:.4e} N")