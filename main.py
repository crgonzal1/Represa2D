import numpy as np

from source.mesh.mesh import Mesh
from source.materials.linear_elastic import LinearElastic
from source.elements.quad4 import Quad4
from source.elements.tri3 import Tri3
from source.systems.dense_matrix import DenseMatrix
from source.systems.boundary_conditions import apply_dirichlet_bc
from source.loads.hydrostatic import apply_hydrostatic_pressure
from source.loads.gravity import apply_self_weight
from source.solvers.linear import solve


# 1. Cargar Malla
mesh = Mesh()
mesh.read("meshes/q4_250.mesh") 
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
    elif len(node_ids) == 3:
        element = Tri3(mesh.element_global_ids[e], element_nodes, material, thickness)
    else:
        continue # Ignorar nodos sueltos o líneas 1D si los hubiera
        
    elements.append(element)

print(f"Total de elementos discretizados y ensamblados: {len(elements)}")

# 4. Ensamblaje Global
number_of_dofs = mesh.number_of_dofs()
K = DenseMatrix(number_of_dofs)

for element in elements:
    K.scatter_local_to_global(element)

# 5. Vector de Cargas (Fuerzas Externas)
f = np.zeros(number_of_dofs)

apply_hydrostatic_pressure(mesh, f, rho_water=1000.0, g=9.81, h_water=28.0)

apply_self_weight(elements, f, density=2400.0, g=9.81, thickness=thickness)

# Calcular y aplicar fuerzas nodales equivalentes

# 6. Condiciones de Borde (Empotramiento en base Y=0)
for i in range(mesh.number_of_nodes()):
    node = mesh.node(i)
    if abs(node.y - 0.0) < 1e-6:
        dofs = mesh.node_global_dofs(node.global_id)
        apply_dirichlet_bc(K, f, dofs[0], 0.0) 
        apply_dirichlet_bc(K, f, dofs[1], 0.0) 

# 7. Resolución del Sistema
u = solve(K, f)
print(f"Desplazamiento máximo: {np.max(np.abs(u)):.6e} m")