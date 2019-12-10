import numpy as np
from ross.fluid_flow import fluid_flow as flow
from ross.fluid_flow import fluid_flow_graphics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# The square root of 3 over 3, applied in Gaussian quadrature.
gq3 = (3**0.5)/float(3)

# Parameters to create phi_1, phi_2, phi_3 and phi_4.
phiOrd = [[-1, -1], [1, -1], [1, 1], [-1, 1]]


def phi_a(ksa, na, ks, n):
    """This is a basis function phi_a. As such, it must return one over node_a and zero over
    any other node.
    Parameters
    ----------
    ksa, na: int
        Locate the phi_a function along the finite element square.
        The value of ksa and na per phi_a function must be, respectively:
            phi_1: -1, -1
            phi_2: 1, -1
            phi_3: 1, 1
            phi_4: -1, 1
    ks, n: float
        The parameters of the phi_a function.
    Returns
    -------
    float
    Examples
    --------
    >>> # Function phi_2, measured over node 3
    >>> phi_a(1, -1, 1, 1)
    0.0
    >>> # Function phi_2, measured over node 2
    >>> phi_a(1, -1, 1, -1)
    1.0
    """
    return (1 + ksa * ks) * (1 + na * n) / float(4)


def d_phi_a_d_ks(ksa, na, ks, n):
    """The partial derivative of a basis function phi_a with respect to ks.
    Parameters
    ----------
    ksa, na: int
        Locate the phi_a function along the finite element square.
        The value of ksa and na per phi_a function must be, respectively:
            phi_1: -1, -1
            phi_2: 1, -1
            phi_3: 1, 1
            phi_4: -1, 1
    ks, n: float
        The parameters of the function.
    Returns
    -------
    float
    """
    return ksa * (1 + na * n) / float(4)


def d_phi_a_d_n(ksa, na, ks, n):
    """The partial derivative of a basis function phi_a with respect to n.
    Parameters
    ----------
    ksa, na: int
        Locate the phi_a function along the finite element square.
        The value of ksa and na per phi_a function must be, respectively:
            phi_1: -1, -1
            phi_2: 1, -1
            phi_3: 1, 1
            phi_4: -1, 1
    ks, n: float
        The parameters of the function.
    Returns
    -------
    float
    """
    return na * (1 + ksa * ks) / float(4)


class Element:
    """This class calculates every attribute of one square finite element.

    Parameters
    ----------
    list_x, list_y: list of size 4, giving the xs and ys of the finite element.
    list_q: list of floats of size 4, giving the known values of the nodes (0 if unknown).
    list_f: list of floats of size 4, giving the values of the right side of the Poisson equation.
    c_matrix: a 2x2 matrix of floats, giving the constants of the Poisson equation.

    Returns
    -------
    A finite element and its data.

    Attributes
    ----------
    Ke: 4x4 matrix of floats.
        The local K matrix.
    Fe: 4x1 matrix of floats.
        The local F matrix.

    Examples
    --------
    >>> list_x = [0, 0.333, 0.333, 0]
    >>> list_y = [0, 0, 0.333, 0.333]
    >>> list_q = [0, 0.866, 0, 0]
    >>> list_f = [0, 34.189, 25.64198, 17.09]
    >>> c_matrix = [[2, 1], [1, 2]]
    >>> my_finite_element = Element(list_x, list_y, list_q, list_f, c_matrix)
    """
    def __init__(self, list_x, list_y, list_q, list_f, c_matrix):
        self.Ke = np.zeros(shape=(4, 4))
        self.M = np.zeros(shape=(2, 4))
        self.q = np.zeros(4)
        self.Q = np.zeros(shape=(2, 2))
        for i in range(0, 2):
            for j in range(0, 2):
                self.Q[i][j] = c_matrix[i][j]
        self.f_vector = np.zeros(4)
        for j in range(0, 4):
            self.M[0][j] = list_x[j]
            self.M[1][j] = list_y[j]
            self.q[j] = list_q[j]
            self.f_vector[j] = list_f[j]
        self.Ke += self.gaussian_quadrature_matrix_term_k(-gq3, -gq3)
        self.Ke += self.gaussian_quadrature_matrix_term_k(gq3, -gq3)
        self.Ke += self.gaussian_quadrature_matrix_term_k(-gq3, gq3)
        self.Ke += self.gaussian_quadrature_matrix_term_k(gq3, gq3)
        self.F_Matrix = np.zeros(shape=(4, 4))
        for i in range(0, 4):
            for j in range(0, 4):
                self.F_Matrix[i][j] += self.gaussian_quadrature_term_f(-gq3, -gq3, i, j)
                self.F_Matrix[i][j] += self.gaussian_quadrature_term_f(gq3, -gq3, i, j)
                self.F_Matrix[i][j] += self.gaussian_quadrature_term_f(-gq3, gq3, i, j)
                self.F_Matrix[i][j] += self.gaussian_quadrature_term_f(gq3, gq3, i, j)
        self.Fe = np.dot(self.F_Matrix, self.f_vector)
        self.q_vector = np.dot(self.Ke, self.q)
        self.Fe -= self.q_vector

    def gaussian_quadrature_term_f(self, st, nd, i, j):
        """Calculates one of the four terms that will be added to F_Matrix in position (i, j).
        This matrix is the integral of phi_i * phi_j calculated using Gaussian quadrature,
        multiplied by J (transform matrix).
        F_Matrix will later be multiplied by f_vector to return Fe matrix.
        Parameters
        ----------
        st, nd: float
            The Gaussian points.
        i, j: int
            Position in the square finite element.
        Returns
        -------
            float
        Examples
        --------
        >>> st = (3**0.5)/3.0
        >>> nd = -st
        >>> my_finite_element = finite_element_example()
        >>> F_Matrix = np.zeros(shape=(4, 4))
        >>> F_Matrix[0][0] += my_finite_element.gaussian_quadrature_term_f(st, nd, 0, 0)
        """
        derivatives_matrix = np.zeros(shape=(2, 4))
        for k in range(0, 4):
            derivatives_matrix[0][k] = d_phi_a_d_ks(phiOrd[k][0], phiOrd[k][1], st, nd)
            derivatives_matrix[1][k] = d_phi_a_d_n(phiOrd[k][0], phiOrd[k][1], st, nd)
        transform_matrix = np.dot(self.M, np.transpose(derivatives_matrix))
        term1 = phi_a(phiOrd[i][0], phiOrd[i][1], st, nd)
        term2 = phi_a(phiOrd[j][0], phiOrd[j][1], st, nd)
        return term1 * term2 * np.linalg.det(transform_matrix)

    def gaussian_quadrature_matrix_term_k(self, st, nd):
        """Calculates one of the four matrices that will be added to Ke matrix.
        Ke matrix is the integral of (nabla phi_a)*Q*(nabla phi_b), calculated using Gaussian quadrature,
        multiplied by J (transform matrix).
        Parameters
        ----------
        st, nd: float
            The Gaussian points.
        Returns
        -------
            4x4 matrix of float
        Examples
        --------
        >>> st = (3**0.5)/3.0
        >>> nd = -st
        >>> my_finite_element = finite_element_example()
        >>> Ke = np.zeros(shape=(4, 4))
        >>> Ke += my_finite_element.gaussian_quadrature_matrix_term_k(st, nd)
        """
        derivatives_matrix = np.zeros(shape=(2, 4))
        for k in range(0, 4):
            derivatives_matrix[0][k] = d_phi_a_d_ks(phiOrd[k][0], phiOrd[k][1], st, nd)
            derivatives_matrix[1][k] = d_phi_a_d_n(phiOrd[k][0], phiOrd[k][1], st, nd)
        jacob_matrix = np.dot(self.M, np.transpose(derivatives_matrix))
        inv_jacob_matrix = np.transpose(np.linalg.inv(jacob_matrix))
        term1 = np.linalg.det(jacob_matrix) * np.dot(np.transpose(derivatives_matrix), np.transpose(inv_jacob_matrix))
        term2 = np.dot(self.Q, np.dot(inv_jacob_matrix, derivatives_matrix))
        return np.dot(term1, term2)


class FiniteElementMethod:
    """This class runs the finite element method, taking a FluidFlow object from ross, and
    calculates its pressure matrix.

    Parameters
    ----------
    fluid_flow_object: a FluidFlow object.

    Returns
    -------
    An object containing finite element method attributes.

    Attributes
    ----------
    pressure_matrix: matrix of floats.
        The product of the finite element method.
    K, C, F: matrices of floats.
        A n x n linear system, in which n is the number of unknowns.

    Examples
    --------
    >>> from ross.fluid_flow.fluid_flow import fluid_flow_example
    >>> my_fluid_flow = fluid_flow_example()
    >>> my_fem = FiniteElementMethod(my_fluid_flow)
    """
    def __init__(self, fluid_flow_object):
        fluid_flow_object.calculate_coefficients()
        self.fluid_flow_object = fluid_flow_object
        self.n_elements_y = fluid_flow_object.nz - 1
        self.n_elements_x = fluid_flow_object.ntheta - 1
        self.matrix_y = np.arange(0, fluid_flow_object.length + (fluid_flow_object.length /
                                                                 self.n_elements_y) / 2.0,
                                  fluid_flow_object.length / self.n_elements_y)
        self.matrix_x = np.arange(0, 2 * np.pi + ((2 * np.pi) / self.n_elements_x) / 2.0,
                                  (2 * np.pi) / self.n_elements_x)
        self.matrix_x, self.matrix_y = np.meshgrid(self.matrix_x, self.matrix_y)
        self.n_nodes = fluid_flow_object.nz * fluid_flow_object.ntheta
        self.n_known_nodes = fluid_flow_object.ntheta * 2
        self.n_elements = self.n_elements_y * self.n_elements_x
        self.equation_vector = np.zeros(self.n_nodes, dtype=int)
        self.q_function_in_node = np.zeros(self.n_nodes, dtype=float)
        self.f_function_in_node = np.zeros(self.n_nodes, dtype=float)
        self.i_in_node = np.zeros(self.n_nodes, dtype=int)
        self.j_in_node = np.zeros(self.n_nodes, dtype=int)
        self.calculate_i_and_j_in_node()
        self.z_in_node = np.zeros(self.n_nodes, dtype=float)
        self.theta_in_node = np.zeros(self.n_nodes, dtype=float)
        self.calculate_nodes_positions()
        self.init_vectors()
        self.local_nodes_matrix = np.zeros(dtype=int, shape=(self.n_elements, 4))
        self.build_local_nodes_matrix()
        self.n_equations = self.n_nodes - self.n_known_nodes

        # Adding periodicity to the problem
        eq_theta_zero = 0
        k = -2
        m = 0
        while self.equation_vector[k] != -1:
            k -= 1
            m += 1
        eq_theta_2pi = self.equation_vector[-2] + 1
        self.K = np.zeros(shape=(self.n_equations + m, self.n_equations + m))
        self.build_k_matrix(m, eq_theta_zero, eq_theta_2pi)
        self.n_equations += m
        self.F = np.zeros(self.n_equations)
        self.build_matrices()
        self.answer_vector = np.zeros(self.n_nodes)
        self.C = self.solve()
        self.build_answer_vector()
        self.pressure_matrix = np.copy(fluid_flow_object.p_mat_numerical)
        self.build_pressure_matrix()

    def calculate_i_and_j_in_node(self):
        """Calculates matching i and j for each node.
        """
        m = 0
        for i in range(0, self.fluid_flow_object.nz):
            for j in range(0, self.fluid_flow_object.ntheta):
                self.i_in_node[m] = i
                self.j_in_node[m] = j
                m += 1

    def init_vectors(self):
        """Inits vectors f_function_in_node (containing the values for the right side of equation in each
        node), q_function_in_node (containing the values of known nodes or 0),
        and equation_vector (containing the number of each equation per node or -1, in case of
        known nodes).
        """
        k = 0
        m = 0
        for i in range(0, self.fluid_flow_object.nz):
            for j in range(0, self.fluid_flow_object.ntheta):
                self.f_function_in_node[m] = (
                        (self.fluid_flow_object.c0w[i][j] -
                         self.fluid_flow_object.c0w[i][j - 1]) /
                        self.fluid_flow_object.dtheta
                    )
                if i == 0 or i == self.n_elements_y:
                    self.equation_vector[m] = -1
                    if i == 0:
                        self.q_function_in_node[m] = self.fluid_flow_object.p_in
                    else:
                        self.q_function_in_node[m] = self.fluid_flow_object.p_out
                else:
                    self.equation_vector[m] = k
                    k += 1
                m += 1

    def build_local_nodes_matrix(self):
        """The local_nodes_matrix holds the actual number of the four nodes of each element.
        Each element has four nodes, numbered counterclockwise, starting with the lowest (theta, z).
        """
        k = 0
        j = 1
        m = self.n_elements_x
        for i in range(0, self.n_elements):
            self.local_nodes_matrix[i][0] = k
            self.local_nodes_matrix[i][1] = k + 1
            k += 1
            if k >= m:
                k = (self.n_elements_x + 1) * j
                j += 1
                m = k + self.n_elements_x
        k = self.n_elements_x + 1
        j = 2
        m = k + self.n_elements_x
        for i in range(0, self.n_elements):
            self.local_nodes_matrix[i][3] = k
            self.local_nodes_matrix[i][2] = k + 1
            k += 1
            if k >= m:
                k = (self.n_elements_x + 1) * j
                j += 1
                m = k + self.n_elements_x

    def calculate_nodes_positions(self):
        """For each node, calculates the values of theta and z in that node.
        """
        for i in range(0, self.n_nodes):
            self.z_in_node[i] = self.matrix_y[self.i_in_node[i]][self.j_in_node[i]]
            self.theta_in_node[i] = self.matrix_x[self.i_in_node[i]][self.j_in_node[i]]

    def build_k_matrix(self, m, eq_theta_zero, eq_theta_2pi):
        """Adds the periodicity to the K matrix.
        """
        for i in range(0, m):
            self.K[self.n_equations + i][eq_theta_zero] = 1
            self.K[self.n_equations + i][eq_theta_2pi] = -1
            eq_theta_zero += 1
            eq_theta_2pi += 1

    def build_matrices(self):
        """Build the global matrices F and K.
        """
        for e in range(0, self.n_elements):
            list_z = np.zeros(4)
            list_theta = np.zeros(4)
            list_q = np.zeros(4)
            list_f = np.zeros(4)
            for a in range(0, 4):
                list_z[a] = self.z_in_node[self.local_nodes_matrix[e][a]]
                list_theta[a] = self.theta_in_node[self.local_nodes_matrix[e][a]]
                list_q[a] = self.q_function_in_node[self.local_nodes_matrix[e][a]]
                list_f[a] = self.f_function_in_node[self.local_nodes_matrix[e][a]]
            c_matrix = np.zeros(shape=(2, 2))
            c_matrix[0][0] = -self.fluid_flow_object.c1[self.i_in_node[e]][self.j_in_node[e]]
            c_matrix[1][1] = -self.fluid_flow_object.c2[self.i_in_node[e]][self.j_in_node[e]]
            this_element = Element(list_theta, list_z, list_q, list_f, c_matrix)
            for a in range(0, 4):
                for b in range(0, 4):
                    i = self.equation_vector[self.local_nodes_matrix[e][a]]
                    j = self.equation_vector[self.local_nodes_matrix[e][b]]
                    if i != -1 and j != -1:
                        self.K[i][j] += this_element.Ke[a][b]
                if self.equation_vector[self.local_nodes_matrix[e][a]] != -1:
                    self.F[self.equation_vector[self.local_nodes_matrix[e][a]]] += this_element.Fe[a]

    def solve(self):
        """Solves the linear system.
        """
        return np.linalg.solve(self.K, self.F)

    def build_answer_vector(self):
        """Build the answer_vector, that holds the pressure value in each node.
        """
        k = 0
        for i in range(0, self.n_nodes):
            if self.equation_vector[i] == -1:
                self.answer_vector[i] = self.q_function_in_node[i]
            else:
                self.answer_vector[i] = self.C[k]
                k += 1

    def build_pressure_matrix(self):
        """Builds the pressure matrix.
        """
        k = 0
        for i in range(0, self.fluid_flow_object.nz):
            for j in range(0, self.fluid_flow_object.ntheta):
                self.pressure_matrix[i][j] = self.answer_vector[k]
                k += 1

    def matplot_3d_graphic(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(self.matrix_y, self.matrix_x, self.pressure_matrix,
                               cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.7)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.scatter(np.append(self.z_in_node, self.matrix_y[-1]),
                   np.append(self.theta_in_node, self.matrix_x[-1]), self.answer_vector, c='r',
                   marker='^')
        plt.show()


def ross_finite_element_solution(nz, ntheta):
    """Executes the finite element method for a given grid.
    """
    nradius = 11
    omega = 100. * 2 * np.pi / 60
    p_in = 0.
    p_out = 0.
    radius_rotor = 0.1999996
    radius_stator = 0.1999996 + 0.000194564
    length = (1 / 10) * (2 * radius_stator)
    eccentricity = 0.0001
    visc = 0.015
    rho = 860.
    my_fluid_flow = flow.PressureMatrix(nz, ntheta, nradius, length, omega, p_in,
                                        p_out, radius_rotor, radius_stator,
                                        visc, rho, eccentricity=eccentricity)

    my_finite_element = FiniteElementMethod(my_fluid_flow)
    my_fluid_flow.p_mat_numerical = np.copy(my_finite_element.pressure_matrix)
    my_fluid_flow.numerical_pressure_matrix_available = True
    my_fluid_flow.calculate_pressure_matrix_analytical()
    ax = fluid_flow_graphics.matplot_pressure_theta(my_fluid_flow, z=int(my_fluid_flow.nz / 2))
    # my_fluid_flow.calculate_pressure_matrix_numerical()
    # ax = fluid_flow_graphics.matplot_pressure_theta(my_fluid_flow, z=int(my_fluid_flow.nz/2), ax=ax,
    #                                                 color="black")
    plt.show()


def finite_element_example():
    """An example of one finite element.
    """
    list_x = [0, 0.333, 0.333, 0]
    list_y = [0, 0, 0.333, 0.333]
    list_q = [0, 0.866, 0, 0]
    list_f = [0, 34.189, 25.64198, 17.09]
    c_matrix = [[2, 1], [1, 2]]
    my_finite_element = Element(list_x, list_y, list_q, list_f, c_matrix)
    return my_finite_element


if __name__ == "__main__":
    ross_finite_element_solution(4, 128)




