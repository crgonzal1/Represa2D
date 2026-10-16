import numpy as np

from source.mesh.mesh import Mesh
from source.materials.linear_elastic import LinearElastic
from source.elements.quad4 import Quad4
from source.elements.tri3 import Tri3
from source.elements.quad8 import Quad8
from source.elements.tri6 import Tri6
from source.systems.dense_matrix import DenseMatrix
from source.systems.sparse_matrix import SparseMatrix
from source.systems.boundary_conditions import apply_dirichlet_bc
from source.loads.hydrostatic import apply_hydrostatic_pressure
from source.loads.gravity import apply_self_weight
from source.solvers.linear import solve
from source.post.visualizer import plot_deformed_mesh
from source.output.paraview import export_to_paraview


# 1. Cargar Malla
mesh = Mesh()
mesh.read("meshes/250Q4.mesh") 
mesh.dofs_per_node = 2

# 2. Propiedades del Material (Hormigón: E=21 GPa, nu=0.2)
material = LinearElastic(21.0e9, 0.2)
thickness = 1.0

# 3. Instanciación Híbrida de Elementos
elements = []
for e in range(mesh.number_of_elements()):
    node_ids = mesh.connectivity[e]
    element_nodes = [mesh.node(nid) for nid in node_ids]
    
    # Ruteo topológico según cantidad de nodos
    if len(node_ids) == 4:
        element = Quad4(mesh.element_global_ids[e], element_nodes, material, thickness)
    elif len(node_ids) == 8:
        element = Quad8(mesh.element_global_ids[e], element_nodes, material, thickness)
    elif len(node_ids) == 3:
        element = Tri3(mesh.element_global_ids[e], element_nodes, material, thickness)
    else:
        continue
        
    elements.append(element)

print(f"Total de elementos discretizados y ensamblados: {len(elements)}")

# 4. Ensamblaje Global
number_of_dofs = mesh.number_of_dofs()
K = DenseMatrix(number_of_dofs)

for element in elements:
    K.scatter_local_to_global(element)

# 5. Vector de Cargas (Fuerzas Externas)
f = np.zeros(number_of_dofs)

apply_hydrostatic_pressure(mesh, elements, f, rho_water=1000.0, g=9.81, h_water=28.0)

apply_self_weight(elements, f, density=2400.0, g=9.81, thickness=thickness)

K_orig = np.copy(K.values)
f_orig = np.copy(f)

# 6. Condiciones de Borde (Empotramiento en base)
y_min = min(node.y for node in mesh.nodes)
fixed_nodes = 0

for i in range(mesh.number_of_nodes()):
    node = mesh.node(i)
    if abs(node.y - y_min) < 0.1:
        fixed_nodes += 1
        dofs = mesh.node_global_dofs(node.global_id)
        apply_dirichlet_bc(K, f, dofs[0], 0.0) 
        apply_dirichlet_bc(K, f, dofs[1], 0.0)

print(f"Nodos empotrados en la base: {fixed_nodes}")

# 7. Filtro de Nodos Huérfanos

orphan_dofs = 0
for i in range(K.number_of_dofs):
    if K.values[i][i] == 0.0:
        K.values[i][i] = 1.0  # Rigidez ficticia unitaria
        f[i] = 0.0            # Desplazamiento nulo forzado
        orphan_dofs += 1

if orphan_dofs > 0:
    print(f"Se estabilizaron {orphan_dofs} grados de libertad huérfanos del CAD.")

# 8. Resolución del Sistema
u = solve(K, f)
print(f"Desplazamiento máximo: {np.max(np.abs(u)):.6e} m")

# 9. Recupero fuerzas nodales globales
R = K_orig @ u - f_orig

# 10. Identifica el nodo base con mayor coordenada X como punto de pivote
base_nodes = [mesh.node(i) for i in range(mesh.number_of_nodes()) if abs(mesh.node(i).y - y_min) < 0.1]
x_puntera = max(node.x for node in base_nodes)

# 11. Suma reacciones y calcula momento respecto a la puntera
Rx_total = 0.0
Ry_total = 0.0
M_reacciones_fem = 0.0

for node in base_nodes:
    dofs = mesh.node_global_dofs(node.global_id)
    rx = R[dofs[0]]
    ry = R[dofs[1]]
    
    Rx_total += rx
    Ry_total += ry
    
    # Fuerza vertical por brazo de palanca
    M_reacciones_fem += ry * (x_puntera - node.x) 

# 12. Cálculo Analítico Teórico (Geometría de la represa)
rho_w = 1000.0
rho_c = 2400.0
g = 9.81
h_w = 28.0
H = 30.0
B = 18.0
c = 3.0

# Empuje hidrostático y momento de vuelco
F_h = 0.5 * rho_w * g * (h_w ** 2)
M_vuelco_teorico = F_h * (h_w / 3.0)

# Peso propio y momento estabilizante (dividido en rectángulo y triángulo)
W_rect = (c * H) * rho_c * g
brazo_rect = B - (c / 2.0) 

W_tri = (0.5 * (B - c) * H) * rho_c * g
brazo_tri = (B - c) * (2.0 / 3.0)

M_estabilizante_teorico = (W_rect * brazo_rect) + (W_tri * brazo_tri)
M_neto_teorico = M_estabilizante_teorico - M_vuelco_teorico

print(f"Rx Total: {Rx_total:.2e} N  (Fuerza Hidrostática Teórica: {-F_h:.2e} N)")
print(f"Ry Total: {Ry_total:.2e} N  (Peso Propio Teórico: {W_rect+W_tri:.2e} N)")

print(f"Momento de Vuelco (Teórico):       {M_vuelco_teorico:.2e} Nm")
print(f"Momento Estabilizante (Teórico):   {M_estabilizante_teorico:.2e} Nm")

print(f"Momento Neto Resultante (Teórico): {M_neto_teorico:.2e} Nm")
print(f"Momento de Reacción (FEM):         {M_reacciones_fem:.2e} Nm")

# Calcular error porcentual
error = abs(M_neto_teorico - M_reacciones_fem) / M_neto_teorico * 100
print(f"Error Relativo:                    {error:.4f} %\n")

# 13. Exportar resultados
export_to_paraview("results/presa_q4_250.vtu", mesh, elements, u)

# 14. Post-proceso: Visualización
# plot_deformed_mesh(mesh, elements, u, scale_factor=5000.0)