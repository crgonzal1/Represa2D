def apply_dirichlet_bc(K, f, dof, value):
    for i in range(K.number_of_dofs):
        f[i] = f[i] - K.values[i][dof] * value

    for i in range(K.number_of_dofs):
        K.values[dof][i] = 0.0
        K.values[i][dof] = 0.0

    K.values[dof][dof] = 1.0
    f[dof] = value
