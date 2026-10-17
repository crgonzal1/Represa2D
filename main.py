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
    
    # Ruteo topológico según cantidad de nodos
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
base_nodes = [mesh.node(i) for i in range(mesh.number_of_nodes()) if abs(mesh.node(i).y - y_min) < 0.1]
puntera_node = max(base_nodes, key=lambda n: n.x)
puntera_dof_x = mesh.node_global_dofs(puntera_node.global_id)[0]
fixed_nodes_y = 0

for i in range(mesh.number_of_nodes()):
    node = mesh.node(i)
    if abs(node.y - y_min) < 0.1:
        fixed_nodes_y += 1
        dofs = mesh.node_global_dofs(node.global_id)
        # Apoyo movil (ux libre, uy fijo)
        apply_dirichlet_bc(K, f, dofs[1], 0.0)
        # Apoyo fijo (ux fijo, uy fijo)
        apply_dirichlet_bc(K, f, puntera_dof_x, 0.0)

print(f"Nodos empotrados en la base: {fixed_nodes_y}")
print(f"Nodo fijo (puntera) en X: Global ID {puntera_node.global_id}")

# 7. Filtro de Nodos Huérfanos

orphan_dofs = 0
for i in range(K.number_of_dofs):
    if K.values[i][i] == 0.0:
        K.values[i][i] = 1.0  # Rigidez ficticia unitaria
        f[i] = 0.0            # Desplazamiento nulo forzado
        orphan_dofs += 1

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
thickness = 1.0

# 12.1 Extraer dimensiones dinámicamente
y_min = min(node.y for node in mesh.nodes)
y_max = max(node.y for node in mesh.nodes)
base_nodes = [node for node in mesh.nodes if abs(node.y - y_min) < 1e-3]
x_puntera = max(node.x for node in base_nodes)
x_min = min(node.x for node in base_nodes)
B_base_total = x_puntera - x_min

# 12.2 Fuerzas Hidrostáticas (Empuje horizontal de vuelco)
F_h = 0.5 * rho_w * g * (h_w ** 2) * thickness
M_vuelco_teorico = F_h * (h_w / 3.0)

# 12.3 Peso Propio Teórico (Hormigón) usando los elementos
area_total = 0.0
M_estabilizante_hormigon = 0.0

for element in elements:
    n_nodes = len(element.nodes)
    if n_nodes in [3, 4]:
        x = [node.x for node in element.nodes]
        y = [node.y for node in element.nodes]
    elif n_nodes == 6:
        x = [element.nodes[i].x for i in [0, 1, 2]]
        y = [element.nodes[i].y for i in [0, 1, 2]]
    elif n_nodes == 8:
        x = [element.nodes[i].x for i in [0, 1, 2, 3]]
        y = [element.nodes[i].y for i in [0, 1, 2, 3]]
    
    n = len(x)
    area_elem = 0.0
    cx_elem = 0.0
    for i in range(n):
        j = (i + 1) % n
        cross = x[i]*y[j] - x[j]*y[i]
        area_elem += cross
        cx_elem += (x[i] + x[j]) * cross
        
    area_elem = abs(area_elem) / 2.0
    cx_elem = cx_elem / (6.0 * (area_elem if area_elem != 0 else 1.0))
    if area_elem > 0:
        peso_elem = area_elem * thickness * rho_c * g
        area_total += area_elem
        # Momento estabilizante respecto a la puntera
        M_estabilizante_hormigon += peso_elem * (x_puntera - abs(cx_elem))

W_hormigon = area_total * thickness * rho_c * g

# 12.4 Peso del Agua sobre caras horizontales/inclinadas
W_agua = 0.0
M_estabilizante_agua = 0.0

# Suma vertical de la fuerza hidrostática calculada mediante FEM para el teórico
W_agua = abs(np.sum(f[1::2]))  # Fuerzas en Y del vector de cargas hidrostáticas + gravedad
W_agua_pura = abs(np.sum(f_orig[1::2])) - W_hormigon

# Aproximación del centroide de esa masa de agua (asumiendo que apoya mayormente en el talón)
L_talon = min(node.x for node in mesh.nodes if abs(node.y - y_max) < 1e-3) - x_min
if L_talon > 0.5 and W_agua_pura > 0:
    brazo_agua = x_puntera - (x_min + L_talon / 2.0)
    M_estabilizante_agua = W_agua_pura * brazo_agua

Ry_teorico = W_hormigon + W_agua_pura
M_estabilizante_teorico = M_estabilizante_hormigon + M_estabilizante_agua
M_neto_teorico = M_estabilizante_teorico - M_vuelco_teorico

# 12.5 Impresión
print(f"Rx Total: {Rx_total:.2e} N  (Fuerza Hidrostática Teórica: {-F_h:.2e} N)")
print(f"Ry Total: {Ry_total:.2e} N  (Peso Propio + Agua Teórico: {Ry_teorico:.2e} N)")
print(f"Momento de Vuelco (Teórico):       {M_vuelco_teorico:.2e} Nm")
print(f"Momento Estabilizante (Teórico):   {M_estabilizante_teorico:.2e} Nm")
print(f"Momento Neto Resultante (Teórico): {M_neto_teorico:.2e} Nm")
print(f"Momento de Reacción (FEM):         {M_reacciones_fem:.2e} Nm")

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

export_to_paraview("results/presa_Q8_750.vtu", mesh, elements, u, stresses)

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

e = abs((B_base_total / 2.0) - x_R)
limite_e = B_base_total / 6.0

estado_excentricidad = "Cumple (Compresión en toda la base)" if e <= limite_e else "No Cumple (Tracción en el talón)"
print(f"Posición de la resultante (x_R):   {x_R:.2f} m desde la puntera")
print(f"Excentricidad (e):                 {e:.2f} m  (Límite normativo: {limite_e:.2f} m)")
print(f"Regla del tercio medio:            {estado_excentricidad}")