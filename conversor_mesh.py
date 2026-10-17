import pandas as pd
import ast
import os

BASE_DIR = 'mesh_dat_files' # Asegurate de que todas las carpetas de la imagen estén acá adentro
OUTPUT_DIR = 'meshes'

def determinar_tipo_elemento(nombre_malla):
    nombre = nombre_malla.upper()
    if "Q8" in nombre:
        return "QUAD8"
    elif "Q4" in nombre:
        return "QUAD4"
    elif "LST" in nombre:
        return "TRI6"
    elif "CST" in nombre:
        return "TRI3"
    else:
        return None

def convertir_mallas():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Escanear automáticamente todas las carpetas dentro de BASE_DIR
    if not os.path.exists(BASE_DIR):
        print(f"Error: No existe el directorio base '{BASE_DIR}'.")
        return
        
    mallas = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
    
    for nombre_malla in mallas:
        # 2. Identificar tipo de elemento correctamente
        elem_type_str = determinar_tipo_elemento(nombre_malla)
        
        if not elem_type_str:
            print(f"Saltando {nombre_malla}: Tipo de elemento desconocido.")
            continue
            
        # Buscar los CSV dentro de cada subcarpeta
        nodes_csv = os.path.join(BASE_DIR, nombre_malla, f"{nombre_malla}_nodes.csv")
        elem_csv = os.path.join(BASE_DIR, nombre_malla, f"{nombre_malla}_elements.csv")
        output_mesh = os.path.join(OUTPUT_DIR, f"{nombre_malla}.mesh")
        
        if not os.path.exists(nodes_csv) or not os.path.exists(elem_csv):
            print(f"Error con {nombre_malla}: No se encontraron los CSV en {os.path.join(BASE_DIR, nombre_malla)}")
            continue
            
                
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