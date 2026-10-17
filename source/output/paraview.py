import meshio
import numpy as np

def export_to_paraview(filename, mesh, elements, u, stresses=None):
    points = np.array([[node.x, node.y, 0.0] for node in mesh.nodes])
    
    quad4_cells, quad8_cells, tri3_cells, tri6_cells = [], [], [], []
    
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
            
    cells = []
    if quad4_cells: cells.append(("quad", quad4_cells))
    if quad8_cells: cells.append(("quad8", quad8_cells))
    if tri6_cells:  cells.append(("triangle6", tri6_cells))
    if tri3_cells:  cells.append(("triangle", tri3_cells))
        
    point_data = {
        "Desplazamiento": np.column_stack((u[0::2], u[1::2], np.zeros(len(u)//2)))
    }
    
    cell_data = {}
    if stresses is not None:
        # Extraer campos tensoriales para todas las celdas concatenadas
        sig_x = np.array([s['sig_x'] for s in stresses])
        sig_y = np.array([s['sig_y'] for s in stresses])
        tau_xy = np.array([s['tau_xy'] for s in stresses])
        sig_1 = np.array([s['sig_1'] for s in stresses])
        sig_2 = np.array([s['sig_2'] for s in stresses])
        von_mises = np.array([s['von_mises'] for s in stresses])
        
        # Agrupar los datos según el tipo de celda para meshio
        
        blocks_sig_x, blocks_sig_y, blocks_tau_xy = [], [], []
        blocks_sig_1, blocks_sig_2, blocks_vm = [], [], []
        
        idx = 0
        for cell_type, cell_list in cells:
            n_c = len(cell_list)
            blocks_sig_x.append(sig_x[idx:idx+n_c])
            blocks_sig_y.append(sig_y[idx:idx+n_c])
            blocks_tau_xy.append(tau_xy[idx:idx+n_c])
            blocks_sig_1.append(sig_1[idx:idx+n_c])
            blocks_sig_2.append(sig_2[idx:idx+n_c])
            blocks_vm.append(von_mises[idx:idx+n_c])
            idx += n_c
            
        cell_data = {
            "Sigma_X": blocks_sig_x,
            "Sigma_Y": blocks_sig_y,
            "Tau_XY": blocks_tau_xy,
            "Sigma_1": blocks_sig_1,
            "Sigma_2": blocks_sig_2,
            "Von_Mises": blocks_vm
        }
    
    meshio.write_points_cells(filename, points, cells, point_data=point_data, cell_data=cell_data)