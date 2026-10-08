from source.mesh.node import Node


class Mesh:
    def __init__(self):
        self.dim = 0
        self.dofs_per_node = 0
        self.nodes = []
        self.element_global_ids = []
        self.element_types = []
        self.connectivity = []

    def read(self, filename):
        file = open(filename, "r")

        lines = []

        for raw_line in file:
            line = raw_line.split("#")[0]
            line = line.strip()

            if line != "":
                lines.append(line)

        file.close()

        header = lines[0].split()

        n_nodes = int(header[0])
        n_elements = int(header[1])
        dim = int(header[2])

        self.dim = dim

        line_number = 1

        for i in range(n_nodes):
            values = lines[line_number].split()
            line_number = line_number + 1

            global_id = int(values[0]) - 1

            if global_id != i:
                raise ValueError("Unexpected node numbering")

            x = float(values[1])
            y = float(values[2])

            self.nodes.append(Node(global_id, x, y))

        for i in range(n_elements):
            values = lines[line_number].split()
            line_number = line_number + 1

            element_global_id = int(values[0]) - 1
            element_type = values[1]

            element_conn = []

            for j in range(2, len(values)):
                element_conn.append(int(values[j]) - 1)

            self.element_global_ids.append(element_global_id)
            self.element_types.append(element_type)
            self.connectivity.append(element_conn)

    def number_of_nodes(self):
        return len(self.nodes)

    def number_of_elements(self):
        return len(self.connectivity)

    def node(self, node_global_id):
        return self.nodes[node_global_id]

    def number_of_dofs(self):
        return self.number_of_nodes() * self.dofs_per_node

    def node_global_dofs(self, node_global_id):
        dofs = []

        for i in range(self.dofs_per_node):
            dofs.append(node_global_id * self.dofs_per_node + i)

        return dofs
