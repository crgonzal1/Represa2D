import matplotlib.pyplot as plt

def plot_deformed_mesh(mesh, elements, u, scale_factor=1.0):
    
    # Grafica la configuración original y la configuración deformada de la malla.
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    print(f"Generando gráfico con factor de escala: {scale_factor}x...")

    # Dibujar elementos
    for element in elements:
        # Extraer coordenadas originales
        x_orig = [node.x for node in element.nodes]
        y_orig = [node.y for node in element.nodes]
        
        # Extraer desplazamientos y calcular coordenadas deformadas
        x_def = []
        y_def = []
        for node in element.nodes:
            dof_x = node.global_id * 2
            dof_y = node.global_id * 2 + 1
            
            x_def.append(node.x + u[dof_x] * scale_factor)
            y_def.append(node.y + u[dof_y] * scale_factor)
            
        # Cerrar los polígonos repitiendo el primer nodo al final
        x_orig.append(x_orig[0])
        y_orig.append(y_orig[0])
        x_def.append(x_def[0])
        y_def.append(y_def[0])
        
        # Graficar original (gris punteado)
        ax.plot(x_orig, y_orig, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        
        # Graficar deformada (azul sólido)
        ax.plot(x_def, y_def, color='blue', linestyle='-', linewidth=1.0)
        
    ax.set_aspect('equal', 'box')
    plt.title(f"Configuración Deformada - Presa de Gravedad\nFactor de escala: {scale_factor}")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Mostrar la figura
    plt.show()
