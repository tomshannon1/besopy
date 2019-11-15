from besopy.topology import Topology
import unittest

DEGREES_OF_FREEDOM = 3
NODES_PER_ELEMENT = 8


class TestTopology(unittest.TestCase):

    def setUp(self):
        self.length = 40
        self.width = 10
        self.height = 20
        self.x = 20
        self.y = 8
        self.z = 10

        self.topology = Topology(self.length, self.width, self.height, self.x, self.y, self.z)
        self.num_nodes = ((self.x+1) * (self.y+1) * (self.z+1))
        self.num_elements = (self.x * self.y * self.z)

    def test_topology_brick_coordinates(self):
        self.assertTrue(self.topology.brick_coordinates().shape == (self.num_nodes, DEGREES_OF_FREEDOM))

    def test_topology_brick_connections(self):
        self.assertTrue(self.topology.brick_connections().shape == (self.num_elements, NODES_PER_ELEMENT))

    def test_get_num_connections(self):
        self.assertTrue(self.topology.get_num_elements() == self.num_elements)

    def test_get_num_coordinates(self):
        self.assertTrue(self.topology.get_num_nodes() == self.num_nodes)
