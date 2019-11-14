import numpy as np

DEGREES_OF_FREEDOM = 3
NODES_PER_ELEMENT = 8


class Topology:

    """ Summary: The topology class is used to set up the initial
        design space for the optimization process. This consists of
        a three dimensional design space with length, width, and height.
        Within this space you can define how many elements you want in
        each dimension for the initial design.

        length:argument - Length of design space
        width:argument - Width of design space
        height:argument - Height of design space
        x:argument - Number of x elements in initial design
        y:argument - Number of y elements in initial design
        z:argument - Number of z elements in initial design """

    def __init__(self, length, width, height, x, y, z):
        self.length = length
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.z = z

        self.num_elements = x * y * z
        self.num_nodes = (x+1) * (y+1) * (z+1)

    def get_num_elements(self): return self.num_elements

    def get_num_nodes(self): return self.num_nodes

    def brick_coordinates(self):
        coordinates = np.zeros((self.num_elements, DEGREES_OF_FREEDOM))

        dx, dy, dz = self.length / self.x, self.width / self.y, self.height / self.z

        for node in range(self.num_nodes):
            nx = node / ((self.y + 1) * (self.z + 1)) + 1
            ny = (node - (nx - 1) * (self.y + 1) * (self.z + 1)) / (self.z + 1) + 1
            nz = self.z + 1 if np.mod(node, (self.z + 1)) == 0 else np.mod(node, (self.z + 1)) + 1
            nz = 1 if node == 0 else nz
            coordinates[node] = [(nx - 1) * dx, (ny - 1) * dy, (nz - 1) * dz]

        return coordinates.astype(int)

    def brick_connections(self):
        connections = np.zeros((self.num_elements, NODES_PER_ELEMENT))

        for element in range(self.num_elements):
            elx = element / (self.y * self.z) + 1
            ely = (element - (elx - 1) * self.y * self.z) / self.z + 1
            elz = z if np.mod(element, self.z) == 0 else np.mod(element, self.z)
            elz = 0 if element == 0 else elz

            connections[element][0] = self.y * self.z * (elx - 1) + (ely - 1) * self.z + elz
            connections[element][1] = connections[element][0] + (self.y + 1) * (self.z + 1)
            connections[element][2] = connections[element][1] + (self.z + 1)
            connections[element][3] = connections[element][0] + (self.z + 1)
            connections[element][4:8] = connections[element][0:4] + 1

        return connections.astype(int)
