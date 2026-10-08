import pandas as pd
import ast
import os

# Lista con todas las mallas que organizaste
MALLAS = [
    'q4_250', 'q4_500', 'q4_750',
    'q8_250', 'q8_500', 'q8_750'
]

BASE_DIR = 'mesh_dat_files'
OUTPUT_DIR = 'meshes'

def convertir_mallas():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for nombre_malla in MALLAS:
        # Detectar si es Q4 o Q8 para escribir el tipo correcto en el .mesh
        elem_type_str = "QUAD8" if "q8" in nombre_malla else "QUAD4"
        
        # Buscar los CSV dentro de cada subcarpeta
        nodes_csv = os.path.join(BASE_DIR, nombre_malla, f"{nombre_malla}_nodes.csv")
        elem_csv = os.path.join(BASE_DIR, nombre_malla, f"{nombre_malla}_elements.csv")
        output_mesh = os.path.join(OUTPUT_DIR, f"{nombre_malla}.mesh")
        
        if not os.path.exists(nodes_csv) or not os.path.exists(elem_csv):
            print(f"error con {nombre_malla}: No se encontraron los CSV en {os.path.dirname(nodes_csv)}")
            continue
            
        print(f"Procesando {nombre_malla} ({elem_type_str})...")
        
        df_nodes = pd.read_csv(nodes_csv)
        df_elem = pd.read_csv(elem_csv)
        df_elem['nodes'] = df_elem['nodes'].apply(ast.literal_eval)

        # Mapeo de IDs de Nastran a secuencia 1..N

        node_map = {old_id: new_id for new_id, old_id in enumerate(df_nodes['nid'], 1)}

        with open(output_mesh, 'w') as f:
            # Cabecera
            f.write("# n_nodes n_elements dim\n")
            f.write(f"{len(df_nodes)} {len(df_elem)} 2\n\n")

            # Nodos
            f.write("# node_number x y\n")
            for i, row in df_nodes.iterrows():
                f.write(f"{i+1} {row['x']} {row['y']}\n")

            # Elementos
            f.write("\n# element_number element_type connectivity\n")
            for i, row in df_elem.iterrows():
                conn = [str(node_map[n]) for n in row['nodes']]
                f.write(f"{i+1} {elem_type_str} {' '.join(conn)}\n")

        print(f"  Guardado: {output_mesh}")

if __name__ == "__main__":
    convertir_mallas()