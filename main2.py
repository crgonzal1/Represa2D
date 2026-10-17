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
mesh.read("meshes/750Q8.mesh") 
mesh.dofs_per_node = 2

# 2. Propiedades del Material (Hormigón: E=21 GPa, nu=0.2)
material = LinearElastic(21.0e9, 0.2)
thickness = 1.0

# 3. Instanciación Híbrida de Elementos
elements = []
for e in range(mesh.number_of_elements()):
    node_ids = mesh.connectivity[e]
    element_nodes = [mesh.node(nid) for nid in node_ids]
    
    # Ruteo topológico directo según cantidad de nodos
    if len(node_ids) == 4:
        element = Quad4(mesh.element_global_ids[e], element_nodes, material, thickness)
    elif len(node_ids) == 8:
        element = Quad8(mesh.element_global_ids[e], element_nodes, material, thickness)
    elif len(node_ids) == 6:
        element = Tri6(mesh.element_global_ids[e], element_nodes, material, thickness)
    elif len(node_ids) == 3:
        element = Tri3(mesh.element_global_ids[e], element_nodes, material, thickness)
    else:
        continue
        
    elements.append(element)

print(f"Total de elementos discretizados y ensamblados: {len(elements)}")

# 4. Ensamblaje Global con Matrices Dispersas (Sparse)
from source.systems.sparse_matrix import SparseMatrix
number_of_dofs = mesh.number_of_dofs()
K = SparseMatrix(number_of_dofs)

for element in elements:
    K.scatter_local_to_global(element)

K.assemble() # Convierte las listas en formato CSR comprimido

# 5. Vector de Cargas (Fuerzas Externas)
f = np.zeros(number_of_dofs)

apply_hydrostatic_pressure(mesh, elements, f, rho_water=1000.0, g=9.81, h_water=28.0)

apply_self_weight(elements, f, density=2400.0, g=9.81, thickness=thickness)

K_orig = K.matrix_csr.copy()
f_orig = np.copy(f)

# 6. Condiciones de Borde (Empotramiento en base)
y_min = min(node.y for node in mesh.nodes)
fixed_nodes = 0
for i in range(mesh.number_of_nodes()):
    node = mesh.node(i)
    if abs(node.y - y_min) < 0.1:
        fixed_nodes += 1
        dofs = mesh.node_global_dofs(node.global_id)
        K.apply_dirichlet(dofs[0], 0.0, f)
        K.apply_dirichlet(dofs[1], 0.0, f)

print(f"Nodos empotrados en la base: {fixed_nodes}")

# 7. Filtro de Nodos Huérfanos

orphan_dofs = 0
# Extraemos la diagonal de la matriz dispersa
diag = K.matrix_csr.diagonal()

for i in range(K.number_of_dofs):
    if diag[i] == 0.0:
        K.matrix_csr[i, i] = 1.0  # Rigidez ficticia unitaria
        f[i] = 0.0                # Desplazamiento nulo forzado
        orphan_dofs += 1

# 8. Resolución del Sistema
u = K.solve(f)
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

R_1d = np.array(R).flatten()

for node in base_nodes:
    dofs = mesh.node_global_dofs(node.global_id)
    rx = float(R_1d[dofs[0]])
    ry = float(R_1d[dofs[1]])
    
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
B_base = 18.0
c = 3.0

# Empuje hidrostático y momento de vuelco
F_h = 0.5 * rho_w * g * (h_w ** 2)
M_vuelco_teorico = F_h * (h_w / 3.0)

# Peso propio y momento estabilizante (dividido en rectángulo y triángulo)
W_rect = (c * H) * rho_c * g
brazo_rect = B_base - (c / 2.0) 

W_tri = (0.5 * (B_base - c) * H) * rho_c * g
brazo_tri = (B_base - c) * (2.0 / 3.0)

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

# 13. Cálculo de Tensores de Tensión (Centroides)
stresses = []
C = material.get_matrix()
nu = material.nu

for element in elements:
    n_nodes = len(element.nodes)
    
    # Seleccionar punto de integracion (Centroide)
    if n_nodes in [4, 8]:
        xi, eta = 0.0, 0.0
    elif n_nodes in [3, 6]:
        xi, eta = 1.0/3.0, 1.0/3.0
    
    # Extraer matriz B
    if hasattr(element, 'get_b_matrix'):
        dN = element.get_natural_derivatives(xi, eta)
        _, inv_J, _ = element.get_jacobian(dN)
        B = element.get_b_matrix(dN, inv_J)
    else:
        
        x1, y1 = element.nodes[0].x, element.nodes[0].y
        x2, y2 = element.nodes[1].x, element.nodes[1].y
        x3, y3 = element.nodes[2].x, element.nodes[2].y
        area = ((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1)) / 2.0
        B = (1.0 / (2.0 * area)) * np.array([
            [y2-y3, 0.0, y3-y1, 0.0, y1-y2, 0.0],
            [0.0, x3-x2, 0.0, x1-x3, 0.0, x2-x1],
            [x3-x2, y2-y3, x1-x3, y3-y1, x2-x1, y1-y2]
        ])

    # Extraer desplazamientos nodales
    u_e = u[element.global_dofs()]
    
    # Hooke
    strain = B @ u_e
    stress = C @ strain
    sx, sy, txy = stress[0], stress[1], stress[2]
    
    # Tensiones Principales y Von Mises (tensión plana / deformación plana)
    center = (sx + sy) / 2.0
    radius = np.sqrt(((sx - sy) / 2.0)**2 + txy**2)
    sig_1 = center + radius
    sig_2 = center - radius
    
    # Tension fuera del plano (asumiendo Deformacion Plana para presas)
    sz = nu * (sx + sy) 
    
    von_mises = np.sqrt(0.5 * ((sx - sy)**2 + (sy - sz)**2 + (sz - sx)**2 + 6 * txy**2))
    
    stresses.append({
        'sig_x': sx, 'sig_y': sy, 'tau_xy': txy, 
        'sig_1': sig_1, 'sig_2': sig_2, 'von_mises': von_mises
    })

# 14. Exportar resultados

export_to_paraview("results/presa_q8_750.vtu", mesh, elements, u, stresses)

# 15. Verificación

# Coeficiente de fricción hormigón-roca (típico)
mu = 0.65

# Extraer escalares puros
Rx_tot = np.sum(Rx_total)
Ry_tot = np.sum(Ry_total)
M_reac = np.sum(M_reacciones_fem)

# Factor de Seguridad al Deslizamiento (FS_d)

FS_d = (mu * Ry_total) / abs(Rx_total)
estado_deslizamiento = "Cumple" if FS_d >= 1.5 else "No Cumple"
print(f"FS Deslizamiento:                  {FS_d:.2f}  ({estado_deslizamiento})")

# Factor de Seguridad al Vuelco (FS_v)

FS_v = M_estabilizante_teorico / M_vuelco_teorico
estado_vuelco = "Cumple" if FS_v >= 1.5 else "No Cumple"
print(f"FS Vuelco:                         {FS_v:.2f}  ({estado_vuelco})")

# Excentricidad

x_R = M_reac / Ry_tot

# Excentricidad respecto al centro de la base (B/2)

e = abs((B_base / 2.0) - x_R)
limite_e = B_base / 6.0

estado_excentricidad = "Cumple (Compresión en toda la base)" if e <= limite_e else "No Cumple (Tracción en el talón)"
print(f"Posición de la resultante (x_R):   {x_R:.2f} m desde la puntera")
print(f"Excentricidad (e):                 {e:.2f} m  (Límite normativo: {limite_e:.2f} m)")
print(f"Regla del tercio medio:            {estado_excentricidad}")