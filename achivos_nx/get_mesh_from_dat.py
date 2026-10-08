from pyNastran.bdf.bdf import BDF
import pandas as pd
import numpy as np

# Se requieren los paquetes pyNastran, pandas y numpy para ejecutar este script :)

def correct_comment(BDF_FILE):
    with open(BDF_FILE, 'r') as f:
        lines = f.readlines()

        for i, line in enumerate(lines):
            if not line.startswith('$'):
                if line.strip().upper() == 'BEGIN BULK':
                    lines[i] = '$' + line
                break

        with open(BDF_FILE, 'w') as f:
            f.writelines(lines)

def nastran_mesh_to_csv(bdf_file, output_dir, file_prefix,
                        export_nodes=True, export_elements=True,
                        export_properties=False, export_materials=False):
    correct_comment(bdf_file)
    model = BDF()
    model.read_bdf(bdf_file, punch=True, xref=True, validate=True, encoding='latin1') # se leen todos los contenidos

    if export_nodes:
        nids = list(model.nodes.keys())
        xyz  = np.array([model.nodes[nid].xyz for nid in nids])
        df_nodes = pd.DataFrame({'nid': nids, 'x': xyz[:, 0], 'y': xyz[:, 1], 'z': xyz[:, 2]})
        df_nodes.to_csv(f'{output_dir}/{file_prefix}_nodes.csv', index=False)

    if export_elements:
        df_eles = pd.DataFrame(
            [{'eid': eid, 'type': elem.type, 'pid': elem.pid, 'nodes': elem.nodes} for eid, elem in model.elements.items()]
        )
        df_eles.to_csv(f'{output_dir}/{file_prefix}_elements.csv', index=False)

    if export_properties:
        df_props = pd.DataFrame(
            [{'pid': pid, 'mid': prop.mid} for pid, prop in model.properties.items()]
        )
        df_props.to_csv(f'{output_dir}/{file_prefix}_properties.csv', index=False)

    if export_materials:
        df_mats = pd.DataFrame(
            [{'mid': mid, 'E': mat.e, 'G': mat.g, 'nu': mat.nu} for mid, mat in model.materials.items()]
        )
        df_mats.to_csv(f'{output_dir}/{file_prefix}_materials.csv', index=False)

nastran_mesh_to_csv(
    bdf_file='q8_250.dat',
    output_dir='./mesh_dat_files',
 file_prefix='q8_250',
export_nodes=True, export_elements=True,
export_properties=True, export_materials=True 
)
print('done')
