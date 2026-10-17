import numpy as np

from source.mesh.mesh import Mesh
from source.materials.linear_elastic import LinearElastic
from source.elements.quad4 import Quad4
from source.elements.tri3 import Tri3
from source.elements.quad8 import Quad8
from source.elements.tri6 import Tri6
from source.systems.dense_matrix import DenseMatrix
from source.systems.boundary_conditions import apply_dirichlet_bc
from source.systems.robin_bc import apply_robin_boundary_conditions
from source.loads.hydrostatic import apply_hydrostatic_pressure
from source.loads.gravity import apply_self_weight
from source.solvers.linear import solve
from source.output.paraview import export_to_paraview


# 1. Cargar Malla
mesh = Mesh()
mesh.read("meshes/500Q8.mesh") 
mesh.dofs_per_node = 2

# 2. Propiedades del Material (Hormigón: E=21 GPa, nu=0.2)
material = LinearElastic(21.0e9, 0.2)
thickness = 1.0

# 3. Instanciación Híbrida de Elementos
elements = []
for e in range(mesh.number_of_elements()):
    node_ids = mesh.connectivity[e]
    element_nodes = [mesh.node(nid) for nid in node_ids]
    
    if len(node_ids) == 4:
        element = Quad4(mesh.element_global_ids[e], element_nodes, material, thickness)
    elif len(node_ids) == 8:
        element = Quad8(mesh.element_global_ids[e], element_nodes, material, thickness)
    elif len(node_ids) == 3:
        element = Tri3(mesh.element_global_ids[e], element_nodes, material, thickness)
    elif len(node_ids) == 6:
        element = Tri6(mesh.element_global_ids[e], element_nodes, material, thickness)
    else:
        continue
        
    elements.append(element)

print(f"Total de elementos discretizados y ensamblados: {len(elements)}")

# 4. Ensamble Global de la Estructura (Sin condiciones de borde)
number_of_dofs = mesh.number_of_dofs()
K_struct = DenseMatrix(number_of_dofs)

for element in elements:
    K_struct.scatter_local_to_global(element)

# 5. Vector de Cargas Externas (Hidrostática + Peso Propio)
f_ext = np.zeros(number_of_dofs)
apply_hydrostatic_pressure(mesh, elements, f_ext, rho_water=1000.0, g=9.81, h_water=28.0)
apply_self_weight(elements, f_ext, density=2400.0, g=9.81, thickness=thickness)

# Identificar nodos de la base
y_min = min(node.y for node in mesh.nodes)
base_nodes = [mesh.node(i) for i in range(mesh.number_of_nodes()) if abs(mesh.node(i).y - y_min) < 0.1]
base_nodes_sorted = sorted(base_nodes, key=lambda n: n.x)
puntera_node = max(base_nodes, key=lambda n: n.x)
x_puntera = puntera_node.x
puntera_dof_x = mesh.node_global_dofs(puntera_node.global_id)[0]



# MODELO 1: DIRICHLET (APOYO MÓVIL + FIJO)

K_dir = DenseMatrix(number_of_dofs)
K_dir.values = np.copy(K_struct.values)
f_dir = np.copy(f_ext)

for node in base_nodes:
    dofs = mesh.node_global_dofs(node.global_id)
    # Apoyo movil (ux libre, uy fijo)
    apply_dirichlet_bc(K_dir, f_dir, dofs[1], 0.0)
    # Apoyo fijo (ux fijo, uy fijo)
    apply_dirichlet_bc(K_dir, f_dir, puntera_dof_x, 0.0)

# Estabilización de nodos huérfanos
for i in range(K_dir.number_of_dofs):
    if K_dir.values[i][i] == 0.0:
        K_dir.values[i][i] = 1.0
        f_dir[i] = 0.0

u_dirichlet = solve(K_dir, f_dir)

# Reacciones en Dirichlet (Fuerzas nodales de apoyo)
R_dir = K_struct.values @ u_dirichlet - f_ext
Rx_dir = sum(float(R_dir[mesh.node_global_dofs(n.global_id)[0]]) for n in base_nodes)
Ry_dir = sum(float(R_dir[mesh.node_global_dofs(n.global_id)[1]]) for n in base_nodes)
M_dir = sum(float(R_dir[mesh.node_global_dofs(n.global_id)[1]]) * (x_puntera - n.x) for n in base_nodes)



# MODELO 2: ROBIN (APOYO ELÁSTICO EN BASE)

K_rob = DenseMatrix(number_of_dofs)
K_rob.values = np.copy(K_struct.values)
f_rob = np.copy(f_ext)

k_n_val = 1.0e9  # Rigidez vertical [N/m^3]
k_t_val = 5.0e8  # Rigidez horizontal [N/m^3]

apply_robin_boundary_conditions(mesh, K_rob, k_n=k_n_val, k_t=k_t_val)
apply_dirichlet_bc(K_rob, f_rob, puntera_dof_x, 0.0)  # Apoyo isostático lateral

# Estabilización de nodos huérfanos
for i in range(K_rob.number_of_dofs):
    if K_rob.values[i][i] == 0.0:
        K_rob.values[i][i] = 1.0
        f_rob[i] = 0.0

u_robin = solve(K_rob, f_rob)

# Reacciones en Robin (Integración de tensiones elásticas del suelo)
gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
gauss_wts = [1.0, 1.0]

Rx_rob = 0.0
Ry_rob = 0.0
M_rob = 0.0

for i in range(len(base_nodes_sorted) - 1):
    n1 = base_nodes_sorted[i]
    n2 = base_nodes_sorted[i+1]
    L_seg = abs(n2.x - n1.x)
    if L_seg < 1e-6:
        continue
        
    dofs_n1 = mesh.node_global_dofs(n1.global_id)
    dofs_n2 = mesh.node_global_dofs(n2.global_id)
    
    u1, v1 = u_robin[dofs_n1[0]], u_robin[dofs_n1[1]]
    u2, v2 = u_robin[dofs_n2[0]], u_robin[dofs_n2[1]]
    
    for xi, w in zip(gauss_pts, gauss_wts):
        N1 = 0.5 * (1.0 - xi)
        N2 = 0.5 * (1.0 + xi)
        
        # Desplazamientos locales interpolados en la base
        u_base = N1 * u1 + N2 * u2
        v_base = N1 * v1 + N2 * v2
        x_base = N1 * n1.x + N2 * n2.x
        
        # Tensiones del suelo
        sigma_y = -k_n_val * v_base
        tau_xy = -k_t_val * u_base
        
        # Integración de Fuerzas y Momento respecto a la puntera
        d_Rx = tau_xy * (L_seg / 2.0) * w
        d_Ry = sigma_y * (L_seg / 2.0) * w
        
        Rx_rob += d_Rx
        Ry_rob += d_Ry
        M_rob += d_Ry * (x_puntera - x_base)



# CÁLCULO DE TENSIONES Y EXPORTACIÓN A PARAVIEW

def calcular_stresses(u_vector):
    stresses_list = []
    C = material.get_matrix()
    nu = material.nu

    for element in elements:
        n_nodes = len(element.nodes)
        if n_nodes in [4, 8]:
            xi, eta = 0.0, 0.0
        elif n_nodes in [3, 6]:
            xi, eta = 1.0/3.0, 1.0/3.0
        
        if hasattr(element, 'get_b_matrix'):
            dN = element.get_natural_derivatives(xi, eta)
            _, inv_J, _ = element.get_jacobian(dN)
            B_elem = element.get_b_matrix(dN, inv_J)
        else:
            x1, y1 = element.nodes[0].x, element.nodes[0].y
            x2, y2 = element.nodes[1].x, element.nodes[1].y
            x3, y3 = element.nodes[2].x, element.nodes[2].y
            area = ((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1)) / 2.0
            B_elem = (1.0 / (2.0 * area)) * np.array([
                [y2-y3, 0.0, y3-y1, 0.0, y1-y2, 0.0],
                [0.0, x3-x2, 0.0, x1-x3, 0.0, x2-x1],
                [x3-x2, y2-y3, x1-x3, y3-y1, x2-x1, y1-y2]
            ])

        u_e = u_vector[element.global_dofs()]
        strain = B_elem @ u_e
        stress = C @ strain
        sx, sy, txy = stress[0], stress[1], stress[2]
        
        center = (sx + sy) / 2.0
        radius = np.sqrt(((sx - sy) / 2.0)**2 + txy**2)
        sig_1 = center + radius
        sig_2 = center - radius
        sz = nu * (sx + sy) 
        von_mises = np.sqrt(0.5 * ((sx - sy)**2 + (sy - sz)**2 + (sz - sx)**2 + 6 * txy**2))
        
        stresses_list.append({
            'sig_x': sx, 'sig_y': sy, 'tau_xy': txy, 
            'sig_1': sig_1, 'sig_2': sig_2, 'von_mises': von_mises
        })
    return stresses_list

stresses_dirichlet = calcular_stresses(u_dirichlet)
stresses_robin = calcular_stresses(u_robin)

export_to_paraview("results/500Q8_presa_dirichlet.vtu", mesh, elements, u_dirichlet, stresses_dirichlet)
export_to_paraview("results/500Q8_presa_robin.vtu", mesh, elements, u_robin, stresses_robin)


# CÁLCULO DE FACTORES DE SEGURIDAD Y VERIFICACIÓN NORMATIVA

mu = 0.65
B_base = 18.0
limite_e = B_base / 6.0

# Geometría y Teóricos para Vuelco
rho_w, rho_c, g, h_w, H, c = 1000.0, 2400.0, 9.81, 28.0, 30.0, 3.0
F_h = 0.5 * rho_w * g * (h_w ** 2)
M_vuelco_teorico = F_h * (h_w / 3.0)
W_rect = (c * H) * rho_c * g
W_tri = (0.5 * (B_base - c) * H) * rho_c * g
M_estabilizante_teorico = (W_rect * (B_base - c/2.0)) + (W_tri * (B_base - c) * (2.0/3.0))

# 1. Indicadores Dirichlet
FS_d_dir = (mu * Ry_dir) / abs(Rx_dir)
FS_v_dir = M_estabilizante_teorico / M_vuelco_teorico
x_R_dir = M_dir / Ry_dir
e_dir = abs((B_base / 2.0) - x_R_dir)

# 2. Indicadores Robin
FS_d_rob = (mu * Ry_rob) / abs(Rx_rob)
FS_v_rob = M_estabilizante_teorico / M_vuelco_teorico
x_R_rob = M_rob / Ry_rob
e_rob = abs((B_base / 2.0) - x_R_rob)


# TABLA COMPARATIVA DE RESULTADOS Y SEGURIDAD

u_max_dir = np.max(np.abs(u_dirichlet))
u_max_rob = np.max(np.abs(u_robin))

print("\n" + "=" * 75)
print("             COMPARATIVA COMPLETA DE CONDICIONES DE BORDE")
print("=" * 75)
print(f"{'Métrica / Verificación':<38} | {'Dirichlet (Rígido)':<15} | {'Robin (Elástico)':<15}")
print("-" * 75)
print(f"{'Desplazamiento Máximo [m]':<38} | {u_max_dir:<15.6e} | {u_max_rob:<15.6e}")
print(f"{'Reacción Horizontal Rx [N]':<38} | {Rx_dir:<15.2e} | {Rx_rob:<15.2e}")
print(f"{'Reacción Vertical Ry [N]':<38} | {Ry_dir:<15.2e} | {Ry_rob:<15.2e}")
print(f"{'Momento Estabilizante Reacción [Nm]':<38} | {M_dir:<15.2e} | {M_rob:<15.2e}")
print("-" * 75)
print(f"{'FS Deslizamiento (>= 1.5)':<38} | {FS_d_dir:<15.2f} | {FS_d_rob:<15.2f}")
print(f"{'  -> Estado Deslizamiento':<38} | {'Cumple' if FS_d_dir>=1.5 else 'No Cumple':<15} | {'Cumple' if FS_d_rob>=1.5 else 'No Cumple':<15}")
print(f"{'FS Vuelco (>= 1.5)':<38} | {FS_v_dir:<15.2f} | {FS_v_rob:<15.2f}")
print(f"{'  -> Estado Vuelco':<38} | {'Cumple' if FS_v_dir>=1.5 else 'No Cumple':<15} | {'Cumple' if FS_v_rob>=1.5 else 'No Cumple':<15}")
print(f"{'Posición Resultante x_R [m]':<38} | {x_R_dir:<15.2f} | {x_R_rob:<15.2f}")
print(f"Excentricidad e [m] (Límite: {limite_e:.2f}m)".ljust(38) + f" | {e_dir:<15.2f} | {e_rob:<15.2f}")
print(f"{'  -> Tercio Medio':<38} | {'Cumple' if e_dir<=limite_e else 'No Cumple':<15} | {'Cumple' if e_rob<=limite_e else 'No Cumple':<15}")
print("=" * 75 + "\n")
