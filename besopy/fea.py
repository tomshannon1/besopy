from besopy.elasticity import ElasticityMatrix
from scipy.sparse import coo_matrix
import numpy as np

DEGREES_OF_FREEDOM = 3
DEGREES_OF_FREEDOM_PER_ELEMENT = 24

class FEA:

    """ Summary: The FEA class computes the finite element
        analysis of topology designs generated by the
        Topology class.

        topology:argument - Used to get the topology connections,
        coordinates, number of elements, and number of nodes

        youngs:argument - Young's modulus of material

        NU:argument - Poisson's ratio """

    def __init__(self, topology, young=1, poisson=1):
        self.topology = topology
        self.elasticity_matrix = ElasticityMatrix(young, poisson)

    def brick_stiffness(self, ke, x):
        num_nodes = self.topology.get_num_nodes()
        num_elements = self.topology.get_num_elements()

        K = np.zeros((DEGREES_OF_FREEDOM * num_nodes, DEGREES_OF_FREEDOM * num_nodes))

        for element in range(num_elements):
            element_dof = np.empty(DEGREES_OF_FREEDOM_PER_ELEMENT)
            for connection in self.topology.brick_connections()[element]:
                np.append(element_dof, (DEGREES_OF_FREEDOM * connection - 3, DEGREES_OF_FREEDOM * connection - 2, DEGREES_OF_FREEDOM * connection - 1))

            K[element_dof, element_dof[:, np.newaxis]] += ke[:, :, element] * x[element]
        return K

    def get_integration_points(self):
        g = 1 / np.sqrt(3)
        return np.array([[-g, -g, -g], [g, -g, -g], [g, g, -g], [-g, g, -g], [-g, -g, g], [g, -g, g], [g, g, g], [-g, g, g]])


    def elementstiffness(self, element):
        elasticity_matrix = ElasticityMatrix.get_matrix()
        integration_points = self.get_integration_points()

        i, j, k, l, m, n, o, p = self.topology.brick_connections()[element][0:8].astype(int)

        # Interpolation of element coordinates
        x1, y1, z1 = coord[i][0:3]
        x2, y2, z2 = coord[j][0:3]
        x3, y3, z3 = coord[k][0:3]
        x4, y4, z4 = coord[l][0:3]
        x5, y5, z5 = coord[m][0:3]
        x6, y6, z6 = coord[n][0:3]
        x7, y7, z7 = coord[o][0:3]
        x8, y8, z8 = coord[p][0:3]

        # Create space for elemental stiffness matrix
        ke = np.zeros((24, 24))

        for point in range(8):
            # Get the nodal integration points for natural coordinates
            s = round(GP[point][0], 5)
            t = round(GP[point][1], 5)
            u = round(GP[point][2], 5)

            # Compute Jacobian derivates for eight-node hexahedron element
            xs = (x2 * (t - 1.0) * (u - 1.0)) / 8.0 - (x1 * (t - 1.0) * (u - 1.0)) / 8.0 - (
                        x3 * (t + 1.0) * (u - 1.0)) / 8.0 + (x4 * (t + 1.0) * (u - 1.0)) / 8.0 + (
                             x5 * (t - 1.0) * (u + 1.0)) / 8.0 - (x6 * (t - 1.0) * (u + 1.0)) / 8 + (
                             x7 * (t + 1.0) * (u + 1.0)) / 8.0 - (x8 * (t + 1.0) * (u + 1.0)) / 8.0
            xt = x2 * (s / 8.0 + 1.0 / 8.0) * (u - 1.0) - x1 * (s / 8.0 - 1.0 / 8.0) * (u - 1.0) - x3 * (
                        s / 8.0 + 1.0 / 8.0) * (u - 1.0) + x4 * (s / 8.0 - 1.0 / 8.0) * (u - 1.0) + x5 * (
                             s / 8.0 - 1.0 / 8.0) * (u + 1.0) - x6 * (s / 8.0 + 1.0 / 8.0) * (u + 1.0) + x7 * (
                             s / 8.0 + 1.0 / 8.0) * (u + 1.0) - x8 * (s / 8.0 - 1 / 8.0) * (u + 1.0)
            xu = x2 * (s / 8.0 + 1.0 / 8.0) * (t - 1.0) - x1 * (s / 8.0 - 1.0 / 8.0) * (t - 1.0) - x3 * (
                        s / 8.0 + 1.0 / 8.0) * (t + 1.0) + x4 * (s / 8.0 - 1.0 / 8.0) * (t + 1.0) + x5 * (
                             s / 8.0 - 1.0 / 8.0) * (t - 1.0) - x6 * (s / 8.0 + 1.0 / 8.0) * (t - 1.0) + x7 * (
                             s / 8.0 + 1.0 / 8.0) * (t + 1.0) - x8 * (s / 8.0 - 1.0 / 8.0) * (t + 1.0)
            ys = (y2 * (t - 1.0) * (u - 1.0)) / 8.0 - (y1 * (t - 1.0) * (u - 1.0)) / 8.0 - (
                        y3 * (t + 1.0) * (u - 1.0)) / 8.0 + (y4 * (t + 1.0) * (u - 1.0)) / 8.0 + (
                             y5 * (t - 1.0) * (u + 1.0)) / 8.0 - (y6 * (t - 1.0) * (u + 1.0)) / 8.0 + (
                             y7 * (t + 1.0) * (u + 1.0)) / 8.0 - (y8 * (t + 1.0) * (u + 1.0)) / 8.0
            yt = y2 * (s / 8.0 + 1.0 / 8.0) * (u - 1.0) - y1 * (s / 8.0 - 1.0 / 8.0) * (u - 1.0) - y3 * (
                        s / 8.0 + 1.0 / 8.0) * (u - 1.0) + y4 * (s / 8.0 - 1.0 / 8.0) * (u - 1.0) + y5 * (
                             s / 8.0 - 1.0 / 8.0) * (u + 1.0) - y6 * (s / 8.0 + 1.0 / 8.0) * (u + 1.0) + y7 * (
                             s / 8.0 + 1.0 / 8.0) * (u + 1.0) - y8 * (s / 8.0 - 1.0 / 8.0) * (u + 1.0)
            yu = y2 * (s / 8.0 + 1.0 / 8.0) * (t - 1.0) - y1 * (s / 8.0 - 1.0 / 8.0) * (t - 1.0) - y3 * (
                        s / 8.0 + 1.0 / 8.0) * (t + 1.0) + y4 * (s / 8.0 - 1.0 / 8.0) * (t + 1.0) + y5 * (
                             s / 8.0 - 1.0 / 8.0) * (t - 1.0) - y6 * (s / 8.0 + 1.0 / 8.0) * (t - 1.0) + y7 * (
                             s / 8.0 + 1.0 / 8.0) * (t + 1.0) - y8 * (s / 8.0 - 1.0 / 8.0) * (t + 1.0)
            zs = (z2 * (t - 1.0) * (u - 1.0)) / 8.0 - (z1 * (t - 1.0) * (u - 1.0)) / 8.0 - (
                        z3 * (t + 1.0) * (u - 1.0)) / 8.0 + (z4 * (t + 1.0) * (u - 1.0)) / 8.0 + (
                             z5 * (t - 1.0) * (u + 1.0)) / 8.0 - (z6 * (t - 1.0) * (u + 1.0)) / 8.0 + (
                             z7 * (t + 1.0) * (u + 1.0)) / 8.0 - (z8 * (t + 1.0) * (u + 1.0)) / 8.0
            zt = z2 * (s / 8.0 + 1.0 / 8.0) * (u - 1.0) - z1 * (s / 8.0 - 1.0 / 8.0) * (u - 1.0) - z3 * (
                        s / 8.0 + 1.0 / 8.0) * (u - 1.0) + z4 * (s / 8.0 - 1.0 / 8.0) * (u - 1.0) + z5 * (
                             s / 8.0 - 1.0 / 8.0) * (u + 1.0) - z6 * (s / 8.0 + 1.0 / 8.0) * (u + 1.0) + z7 * (
                             s / 8.0 + 1.0 / 8.0) * (u + 1.0) - z8 * (s / 8.0 - 1.0 / 8.0) * (u + 1.0)
            zu = z2 * (s / 8.0 + 1.0 / 8.0) * (t - 1.0) - z1 * (s / 8.0 - 1.0 / 8.0) * (t - 1.0) - z3 * (
                        s / 8.0 + 1.0 / 8.0) * (t + 1.0) + z4 * (s / 8.0 - 1.0 / 8.0) * (t + 1.0) + z5 * (
                             s / 8.0 - 1.0 / 8.0) * (t - 1.0) - z6 * (s / 8.0 + 1.0 / 8.0) * (t - 1.0) + z7 * (
                             s / 8.0 + 1.0 / 8.0) * (t + 1.0) - z8 * (s / 8.0 - 1.0 / 8.0) * (t + 1.0)

            # Make values that are close to zero, zero
            coordlist = np.array([xs, xt, xu, ys, yt, yu, zs, zt, zu])
            xs, xt, xu, ys, yt, yu, zs, zt, zu = [0.0 if nc > 1e-10 for nc in coordlist]

            # Calculate the Jacobian and the determinant
            J = np.array([[xs, ys, zs],
                          [xt, yt, zt],
                          [xu, yu, zu]])

            detJ = xs * (yt * zu - zt * yu) - ys * (xt * zu - zt * xu) + zs * (xt * yu - yt * xu)

            # Compute shape functions for eight-node hexahedron element
            N1s = -((t - 1.0) * (u - 1.0)) / 8.0
            N2s = ((t - 1.0) * (u - 1.0)) / 8.0
            N3s = -((t + 1.0) * (u - 1.0)) / 8.0
            N4s = ((t + 1.0) * (u - 1.0)) / 8.0
            N5s = ((t - 1.0) * (u + 1.0)) / 8.0
            N6s = -((t - 1.0) * (u + 1.0)) / 8.0
            N7s = ((t + 1.0) * (u + 1.0)) / 8.0
            N8s = -((t + 1.0) * (u + 1.0)) / 8.0

            N1t = -(s / 8.0 - 1.0 / 8.0) * (u - 1.0)
            N2t = (s / 8.0 + 1.0 / 8.0) * (u - 1.0)
            N3t = -(s / 8.0 + 1.0 / 8.0) * (u - 1.0)
            N4t = (s / 8.0 - 1.0 / 8.0) * (u - 1.0)
            N5t = (s / 8.0 - 1.0 / 8.0) * (u + 1.0)
            N6t = -(s / 8.0 + 1.0 / 8.0) * (u + 1.0)
            N7t = (s / 8.0 + 1.0 / 8.0) * (u + 1.0)
            N8t = -(s / 8.0 - 1.0 / 8.0) * (u + 1.0)

            N1u = -(s / 8.0 - 1.0 / 8.0) * (t - 1.0)
            N2u = (s / 8.0 + 1.0 / 8.0) * (t - 1.0)
            N3u = -(s / 8.0 + 1.0 / 8.0) * (t + 1.0)
            N4u = (s / 8.0 - 1.0 / 8.0) * (t + 1.0)
            N5u = (s / 8.0 - 1.0 / 8.0) * (t - 1.0)
            N6u = -(s / 8.0 + 1.0 / 8.0) * (t - 1.0)
            N7u = (s / 8.0 + 1.0 / 8.0) * (t + 1.0)
            N8u = -(s / 8.0 - 1.0 / 8.0) * (t + 1.0)

            # Expand Jacobian with shape functions for eight-node hexahedron element
            rightmatrix = np.array([[N1s, N2s, N3s, N4s, N5s, N6s, N7s, N8s], [N1t, N2t, N3t, N4t, N5t, N6t, N7t, N8t],
                                    [N1u, N2u, N3u, N4u, N5u, N6u, N7u, N8u]])
            Nxyz = np.linalg.lstsq(J, rightmatrix)[0]

            # Extrapolate solutions to shape functions
            N1x, N2x, N3x, N4x, N5x, N6x, N7x, N8x = Nxyz[0][0:8]
            N1y, N2y, N3y, N4y, N5y, N6y, N7y, N8y = Nxyz[1][0:8]
            N1z, N2z, N3z, N4z, N5z, N6z, N7z, N8z = Nxyz[2][0:8]

            # Calculate the strain matrix
            B = np.array([[N1x, 0.0, 0.0, N2x, 0.0, 0.0, N3x, 0.0, 0.0, N4x, 0.0, 0.0, N5x, 0.0, 0.0, N6x, 0.0, 0.0,
                           N7x, 0.0, 0.0, N8x, 0.0, 0.0],
                          [0.0, N1y, 0.0, 0.0, N2y, 0.0, 0.0, N3y, 0.0, 0.0, N4y, 0.0, 0.0, N5y, 0.0, 0.0, N6y, 0.0,
                           0.0, N7y, 0.0, 0.0, N8y, 0.0],
                          [0.0, 0.0, N1z, 0.0, 0.0, N2z, 0.0, 0.0, N3z, 0.0, 0.0, N4z, 0.0, 0.0, N5z, 0.0, 0.0, N6z,
                           0.0, 0.0, N7z, 0.0, 0.0, N8z],
                          [N1y, N1x, 0.0, N2y, N2x, 0.0, N3y, N3x, 0.0, N4y, N4x, 0.0, N5y, N5x, 0.0, N6y, N6x, 0.0,
                           N7y, N7x, 0.0, N8y, N8x, 0.0],
                          [0.0, N1z, N1y, 0.0, N2z, N2y, 0.0, N3z, N3y, 0.0, N4z, N4y, 0.0, N5z, N5y, 0.0, N6z, N6y,
                           0.0, N7z, N7y, 0.0, N8z, N8y],
                          [N1z, 0.0, N1x, N2z, 0.0, N2x, N3z, 0.0, N3x, N4z, 0.0, N4x, N5z, 0.0, N5x, N6z, 0.0, N6x,
                           N7z, 0.0, N7x, N8z, 0.0, N8x]])

            # Solve for elemental stiffness matrix
            BE_D = np.dot(np.transpose(B), D)
            BEB_D = np.dot(BE_D, B) * detJ

            ke = ke + BEB_D

        return ke






