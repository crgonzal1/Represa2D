import meshio
import numpy as np

def export_to_paraview(filename, mesh, elements, u):
    # Extraer coordenadas de los nodos
    points = np.array([[node.x, node.y, 0.0] for node in mesh.nodes])
    
    # Inicializar listas para topologías
    quad4_cells = []
    quad8_cells = []
    tri3_cells = []
    tri6_cells = []
    
    # Clasificar cada elemento según su cantidad de nodos
    for e in elements:
        n_nodes = len(e.nodes)
        ids = [n.global_id for n in e.nodes]
        
        if n_nodes == 4:
            quad4_cells.append(ids)
        elif n_nodes == 8:
            quad8_cells.append(ids)
        elif n_nodes == 6:
            tri6_cells.append(ids)
        elif n_nodes == 3:
            tri3_cells.append(ids)
            
    # Armar estructura requerida por meshio con las etiquetas VTK estándar
    cells = []
    if quad4_cells:
        cells.append(("quad", quad4_cells))          # Etiqueta estricta 'quad'
    if quad8_cells:
        cells.append(("quad8", quad8_cells))
    if tri6_cells:
        cells.append(("triangle6", tri6_cells))
    if tri3_cells:
        cells.append(("triangle", tri3_cells))       # Etiqueta estricta 'triangle'
        
    # Agregar desplazamientos como datos vectoriales 3D
    point_data = {
        "Desplazamiento": np.column_stack((u[0::2], u[1::2], np.zeros(len(u)//2)))
    }
    
    # Guardar archivo VTU
    meshio.write_points_cells(filename, points, cells, point_data=point_data)