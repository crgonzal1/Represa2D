import numpy as np

from source.materials.linear_elastic import LinearElastic
from source.elements.quad4 import Quad4
from source.elements.tri3 import Tri3
from source.elements.quad8 import Quad8
from source.elements.tri6 import Tri6


class Node:
    def __init__(self, global_id, x, y):
        self.global_id = global_id
        self.x = float(x)
        self.y = float(y)


def run_patch_test():
    material = LinearElastic(21.0e9, 0.2)
    C = material.get_matrix()

    c1, c2, c3 = 1.0e-3, 2.0e-3, 1.5e-3
    d1, d2, d3 = 2.0e-3, -1.0e-3, 3.0e-3

    def u_exact(x, y):
        return c1 + c2 * x + c3 * y

    def v_exact(x, y):
        return d1 + d2 * x + d3 * y

    eps_exact = np.array([c2, d3, c3 + d2])
    sig_exact = C @ eps_exact

    # Resolutor generico
    def solve_patch(elements, nodes_dict, int_ids, bnd_ids):
        n_dofs = max(nodes_dict.keys()) * 2 + 2
        K = np.zeros((n_dofs, n_dofs))

        for elem in elements:
            ke = elem.stiffness_matrix()
            dofs = elem.global_dofs()
            for i, gi in enumerate(dofs):
                for j, gj in enumerate(dofs):
                    K[gi, gj] += ke[i, j]

        int_dofs = [d for nid in int_ids for d in (nid * 2, nid * 2 + 1)]
        bnd_dofs = [d for nid in bnd_ids for d in (nid * 2, nid * 2 + 1)]

        u_total = np.zeros(n_dofs)
        for n in nodes_dict.values():
            u_total[2 * n.global_id] = u_exact(n.x, n.y)
            u_total[2 * n.global_id + 1] = v_exact(n.x, n.y)

        u_b = u_total[bnd_dofs]
        K_ii = K[np.ix_(int_dofs, int_dofs)]
        K_ib = K[np.ix_(int_dofs, bnd_dofs)]
        u_i_calc = np.linalg.solve(K_ii, -K_ib @ u_b)

        u_fem = np.copy(u_total)
        u_fem[int_dofs] = u_i_calc

        err_u = np.max(np.abs(u_i_calc - u_total[int_dofs]))
        return u_fem, err_u

    results = {}

    # Generacion de malla base (4 elementos cuadrilateros)
    coords = [
        (0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (2.0, 1.0),
        (2.0, 2.0), (1.0, 2.0), (0.0, 2.0), (0.0, 1.0),
        (1.15, 0.85)  # Nodo central distorsionado
    ]
    
    quad4_conn = [
        [0, 1, 8, 7],
        [1, 2, 3, 8],
        [8, 3, 4, 5],
        [7, 8, 5, 6]
    ]

    tri3_conn = []
    for q in quad4_conn:
        tri3_conn.append([q[0], q[1], q[2]])
        tri3_conn.append([q[0], q[2], q[3]])

    #  QUAD4 

    nodes_q4 = {i: Node(i, x, y) for i, (x, y) in enumerate(coords)}
    elems_q4 = [Quad4(i, [nodes_q4[n] for n in conn], material) for i, conn in enumerate(quad4_conn)]
    u_q4, err_u_q4 = solve_patch(elems_q4, nodes_q4, [8], [0,1,2,3,4,5,6,7])

    err_eps_q4 = [np.max(np.abs((e.get_b_matrix(e.get_natural_derivatives(0.0, 0.0), e.get_jacobian(e.get_natural_derivatives(0.0, 0.0))[1]) @ u_q4[e.global_dofs()]) - eps_exact)) for e in elems_q4]
    err_sig_q4 = [np.max(np.abs((C @ (e.get_b_matrix(e.get_natural_derivatives(0.0, 0.0), e.get_jacobian(e.get_natural_derivatives(0.0, 0.0))[1]) @ u_q4[e.global_dofs()])) - sig_exact)) for e in elems_q4]
    results["Quad4"] = (err_u_q4, max(err_eps_q4), max(err_sig_q4))

    #  TRI3 

    elems_tri3 = [Tri3(i, [nodes_q4[n] for n in conn], material) for i, conn in enumerate(tri3_conn)]
    u_tri3, err_u_tri3 = solve_patch(elems_tri3, nodes_q4, [8], [0,1,2,3,4,5,6,7])
    
    err_eps_tri3, err_sig_tri3 = [], []
    for e in elems_tri3:
        x1, y1 = e.nodes[0].x, e.nodes[0].y
        x2, y2 = e.nodes[1].x, e.nodes[1].y
        x3, y3 = e.nodes[2].x, e.nodes[2].y
        area = ((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1)) / 2.0
        B = (1.0 / (2.0 * area)) * np.array([
            [y2-y3, 0.0, y3-y1, 0.0, y1-y2, 0.0],
            [0.0, x3-x2, 0.0, x1-x3, 0.0, x2-x1],
            [x3-x2, y2-y3, x1-x3, y3-y1, x2-x1, y1-y2]
        ])
        strain = B @ u_tri3[e.global_dofs()]
        err_eps_tri3.append(np.max(np.abs(strain - eps_exact)))
        err_sig_tri3.append(np.max(np.abs(C @ strain - sig_exact)))
    results["Tri3"] = (err_u_tri3, max(err_eps_tri3), max(err_sig_tri3))

    #  QUAD8 

    nodes_q8 = {i: Node(i, x, y) for i, (x, y) in enumerate(coords)}
    next_id = 9
    edge_map = {}
    quad8_conn = []
    
    for q in quad4_conn:
        mids = []
        for edge in [(q[0],q[1]), (q[1],q[2]), (q[2],q[3]), (q[3],q[0])]:
            ekey = tuple(sorted(edge))
            if ekey not in edge_map:
                nodes_q8[next_id] = Node(next_id, (nodes_q8[edge[0]].x + nodes_q8[edge[1]].x)/2, (nodes_q8[edge[0]].y + nodes_q8[edge[1]].y)/2)
                edge_map[ekey] = next_id
                next_id += 1
            mids.append(edge_map[ekey])
        quad8_conn.append(q + mids)

    int_q8 = [i for i, n in nodes_q8.items() if not (abs(n.x) < 1e-3 or abs(n.x - 2.0) < 1e-3 or abs(n.y) < 1e-3 or abs(n.y - 2.0) < 1e-3)]
    bnd_q8 = [i for i, n in nodes_q8.items() if (abs(n.x) < 1e-3 or abs(n.x - 2.0) < 1e-3 or abs(n.y) < 1e-3 or abs(n.y - 2.0) < 1e-3)]

    elems_q8 = [Quad8(i, [nodes_q8[n] for n in conn], material) for i, conn in enumerate(quad8_conn)]
    u_q8, err_u_q8 = solve_patch(elems_q8, nodes_q8, int_q8, bnd_q8)

    err_eps_q8 = [np.max(np.abs((e.get_b_matrix(e.get_natural_derivatives(0.0, 0.0), e.get_jacobian(e.get_natural_derivatives(0.0, 0.0))[1]) @ u_q8[e.global_dofs()]) - eps_exact)) for e in elems_q8]
    err_sig_q8 = [np.max(np.abs((C @ (e.get_b_matrix(e.get_natural_derivatives(0.0, 0.0), e.get_jacobian(e.get_natural_derivatives(0.0, 0.0))[1]) @ u_q8[e.global_dofs()])) - sig_exact)) for e in elems_q8]
    results["Quad8"] = (err_u_q8, max(err_eps_q8), max(err_sig_q8))

    # TRI6

    nodes_tri6 = {i: Node(i, x, y) for i, (x, y) in enumerate(coords)}
    next_id = 9
    edge_map = {}
    tri6_conn = []
    
    for t in tri3_conn:
        mids = []
        for edge in [(t[0],t[1]), (t[1],t[2]), (t[2],t[0])]:
            ekey = tuple(sorted(edge))
            if ekey not in edge_map:
                nodes_tri6[next_id] = Node(next_id, (nodes_tri6[edge[0]].x + nodes_tri6[edge[1]].x)/2, (nodes_tri6[edge[0]].y + nodes_tri6[edge[1]].y)/2)
                edge_map[ekey] = next_id
                next_id += 1
            mids.append(edge_map[ekey])
        tri6_conn.append(t + mids)

    int_tri6 = [i for i, n in nodes_tri6.items() if not (abs(n.x) < 1e-3 or abs(n.x - 2.0) < 1e-3 or abs(n.y) < 1e-3 or abs(n.y - 2.0) < 1e-3)]
    bnd_tri6 = [i for i, n in nodes_tri6.items() if (abs(n.x) < 1e-3 or abs(n.x - 2.0) < 1e-3 or abs(n.y) < 1e-3 or abs(n.y - 2.0) < 1e-3)]

    elems_tri6 = [Tri6(i, [nodes_tri6[n] for n in conn], material) for i, conn in enumerate(tri6_conn)]
    u_tri6, err_u_tri6 = solve_patch(elems_tri6, nodes_tri6, int_tri6, bnd_tri6)

    err_eps_tri6 = [np.max(np.abs((e.get_b_matrix(e.get_natural_derivatives(1/3, 1/3), e.get_jacobian(e.get_natural_derivatives(1/3, 1/3))[1]) @ u_tri6[e.global_dofs()]) - eps_exact)) for e in elems_tri6]
    err_sig_tri6 = [np.max(np.abs((C @ (e.get_b_matrix(e.get_natural_derivatives(1/3, 1/3), e.get_jacobian(e.get_natural_derivatives(1/3, 1/3))[1]) @ u_tri6[e.global_dofs()])) - sig_exact)) for e in elems_tri6]
    results["Tri6"] = (err_u_tri6, max(err_eps_tri6), max(err_sig_tri6))

    # RESULTADOS

    print("=" * 82)
    print("         VALIDACION DEL PATCH TEST (DEFORMACION CONSTANTE Y CUERPO RIGIDO)       ")
    print("=" * 82)
    print(f"{'Elemento':<10} | {'Error Max u [m]':<18} | {'Error Max eps':<18} | {'Error Max sig [Pa]':<18} | {'Estado':<6}")
    print("-" * 82)

    all_passed = True
    for name, (eu, e_eps, e_sig) in results.items():
        st = "PASO" if (eu < 1e-12 and e_eps < 1e-12 and e_sig < 1e-4) else "FALLO"
        if st == "FALLO": all_passed = False
        print(f"{name:<10} | {eu:<18.2e} | {e_eps:<18.2e} | {e_sig:<18.2e} | {st:<6}")

    print("=" * 82)
    print(">> VERIFICACION EXITOSA." if all_passed else ">> ERROR: Falla de convergencia.")
    print("=" * 82)

if __name__ == "__main__":
    run_patch_test()