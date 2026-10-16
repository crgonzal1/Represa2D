import numpy as np

def apply_self_weight(elements, f, density=2400.0, g=9.81, thickness=1.0):
    # Calcula el vector de fuerzas consistentes o concentradas según la capacidad del elemento.
    total_weight = 0.0
    
    for element in elements:
        # Obtener cantidad de nodos dinámicamente para soportar cualquier clase
        n_nodes = len(element.nodes)
        f_e = np.zeros(n_nodes * 2)
        weight_e = 0.0
        
        # 1. Verificación de capacidad de integración numérica (Q8, Tri6)
        if hasattr(element, 'get_shape_functions') and hasattr(element, 'get_jacobian'):
            if n_nodes in [4, 8]:
                p = [-np.sqrt(0.6), 0.0, np.sqrt(0.6)]
                w = [5.0/9.0, 8.0/9.0, 5.0/9.0]
                g_pts = [(xi, eta, wi*wj) for xi, wi in zip(p, w) for eta, wj in zip(p, w)]
            elif n_nodes in [3, 6]:
                g_pts = [
                    (1.0/6.0, 1.0/6.0, 1.0/6.0),
                    (2.0/3.0, 1.0/6.0, 1.0/6.0),
                    (1.0/6.0, 2.0/3.0, 1.0/6.0)
                ]
            else:
                continue
                
            for xi, eta, weight in g_pts:
                N = element.get_shape_functions(xi, eta)
                dN = element.get_natural_derivatives(xi, eta)
                _, _, det_J = element.get_jacobian(dN)
                
                dV = abs(det_J) * weight * thickness
                weight_e += density * g * dV
                
                for k in range(n_nodes):
                    f_e[2*k + 1] -= N[k] * density * g * dV
                    
       
        else:
            x = [node.x for node in element.nodes]
            y = [node.y for node in element.nodes]
            
            # Fórmula del polígono (Shoelace) universal para n nodos
            area = 0.0
            for i in range(n_nodes):
                j = (i + 1) % n_nodes
                area += x[i] * y[j] - x[j] * y[i]
            area = abs(area) / 2.0
            
            weight = area * thickness * density * g
            weight_e += weight
            nodal_force = weight / n_nodes
            
            # Distribución equitativa de masa (Lumped mass)
            for k in range(n_nodes):
                f_e[2*k + 1] -= nodal_force
        
        total_weight += weight_e
        
        # Ensamblaje en el vector global
        dofs = element.global_dofs()
        for k, dof in enumerate(dofs):
            f[dof] += f_e[k]
            
    print(f"Peso propio - Fuerza vertical total: {-total_weight:.4e} N")