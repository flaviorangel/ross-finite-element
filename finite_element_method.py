import numpy as np
from ross.fluid_flow import fluid_flow as flow
from ross.fluid_flow import fluid_flow_graphics
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import time

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
        inv_jacob_matrix = np.linalg.inv(jacob_matrix)
        term1 = np.linalg.det(jacob_matrix) * np.dot(inv_jacob_matrix, derivatives_matrix)
        term2 = np.dot(self.Q, np.dot(inv_jacob_matrix, derivatives_matrix))
        return np.dot(np.transpose(term1), term2)


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
        self.K = np.zeros(shape=(self.n_equations, self.n_equations))
        self.F = np.zeros(self.n_equations)

        # Adding periodicity to the problem
        m = fluid_flow_object.nz - 2
        theta_zero = 0
        theta_2pi = fluid_flow_object.ntheta - 1
        self.build_matrices()
        for i in range(0, m):
            periodicity_line = np.zeros(self.n_equations)
            periodicity_line[theta_zero] = 1
            periodicity_line[theta_2pi] = -1
            self.K[theta_2pi] = periodicity_line
            theta_zero += fluid_flow_object.ntheta
            theta_2pi += fluid_flow_object.ntheta

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
                i = self.i_in_node[self.local_nodes_matrix[e][a]]
                j = self.j_in_node[self.local_nodes_matrix[e][a]]
                list_theta[a] = self.fluid_flow_object.gama[i][j]
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
                if self.pressure_matrix[i][j] < 0:
                    self.pressure_matrix[i][j] = 0
                k += 1

    def matplot_3d_graphic(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(self.matrix_y, self.matrix_x, self.pressure_matrix,
                               cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.7)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()


def ross_finite_element_solution(my_fluid_flow):
    """Executes the finite element method (FEM) for a given FluidFlow (object defining the grid).
    Also executes the finite difference method (FDM) and return comparison results.
    Parameters
    ----------
    my_fluid_flow: a FluidFlow object.
    Returns
    -------
    List of floats containing: the relative error for the FEM, the relative error for the FDM,
    time spent for the FEM, time spent for the FDM.
    # """
    start_time = time.time()
    my_finite_element = FiniteElementMethod(my_fluid_flow)
    time_fem = time.time() - start_time
    my_fluid_flow.p_mat_numerical = np.copy(my_finite_element.pressure_matrix)
    my_fluid_flow.numerical_pressure_matrix_available = True
    my_fluid_flow.calculate_pressure_matrix_analytical()
    ax = fluid_flow_graphics.matplot_pressure_theta(my_fluid_flow, z=int(my_fluid_flow.nz / 2))
    plt.show()
    # my_finite_element.matplot_3d_graphic()
    max_reference_value = max(my_fluid_flow.p_mat_analytical[int(my_fluid_flow.nz / 2)])
    position = 0
    for position in range(0, my_fluid_flow.ntheta):
        if my_fluid_flow.p_mat_analytical[int(my_fluid_flow.nz / 2)][position] == max_reference_value:
            break
    r_error_fem = abs((my_fluid_flow.p_mat_numerical[int(my_fluid_flow.nz / 2)][position] - max_reference_value) /
                      max_reference_value)
    start_time = time.time()
    my_fluid_flow.calculate_pressure_matrix_numerical()
    time_fdm = time.time() - start_time
    r_error_fdm = abs((my_fluid_flow.p_mat_numerical[int(my_fluid_flow.nz / 2)][position] - max_reference_value) /
                      max_reference_value)

    return [r_error_fem, r_error_fdm, time_fem, time_fdm]


def instantiate_fluid_flow_object(number_of_theta_nodes, number_of_z_nodes, length_type):
    """Instantiates a FluidFlow object, either short or long, using the same parameters in ROSS tests.

    Parameters
    ----------
    number_of_theta_nodes: int
    number_of_z_nodes: int
    length_type: bool
        Defines if short (True) or long (False) bearing.

    Returns
    -------
    A FluidFlow object
    """
    nradius = 11
    omega = 100.0 * 2 * np.pi / 60
    p_in = 0.0
    p_out = 0.0
    eccentricity = 0.0001
    visc = 0.015
    rho = 860.0
    if length_type:
        radius_rotor = 0.1999996
        radius_stator = 0.1999996 + 0.000194564
        length = (1 / 10) * (2 * radius_stator)
    else:
        p_in = 1
        p_out = 0
        radius_rotor = 1
        h = 0.000194564
        radius_stator = radius_rotor + h
        length = 8 * 2 * radius_stator

    return flow.FluidFlow(
        number_of_z_nodes,
        number_of_theta_nodes,
        nradius,
        length,
        omega,
        p_in,
        p_out,
        radius_rotor,
        radius_stator,
        visc,
        rho,
        eccentricity=eccentricity,
        immediately_calculate_pressure_matrix_numerically=False
    )


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


def grid_convergence(theta_or_z, type_of_bearing):
    """Executes grid convergence and print error.

    Parameters
    ----------
    theta_or_z: bool
        Defines if either theta (True) or z (False) will be the focus along convergence.
    type_of_bearing: bool
        Defines if short (True) or long (False) bearing.

    Returns
    -------
    None
    """
    if theta_or_z:
        nz = 2
        while nz < 16:
            nz += 2
            ntheta = 4
            ntheta_list = []
            error_fem_list = []
            error_fdm_list = []
            time_fem_list = []
            time_fdm_list = []
            while ntheta < 512:
                ntheta = ntheta * 2
                ntheta_list.append(ntheta)
                fluid_flow_obj = instantiate_fluid_flow_object(ntheta, nz, type_of_bearing)
                results = ross_finite_element_solution(fluid_flow_obj)
                error_fem_list.append(results[0])
                error_fdm_list.append(results[1])
                time_fem_list.append(results[2])
                time_fdm_list.append(results[3])
            print("nthetas_z" + str(nz) + " =", ntheta_list)
            print("error_fem_z" + str(nz) + " =", error_fem_list)
            print("error_fdm_z" + str(nz) + " =", error_fdm_list)
            print("time_fem_z" + str(nz) + " =", time_fem_list)
            print("time_fdm_z" + str(nz) + " =", time_fdm_list)
            print()
    else:
        ntheta = 4
        while ntheta < 512:
            nz_list = []
            error_fem_list = []
            error_fdm_list = []
            time_fem_list = []
            time_fdm_list = []
            ntheta = ntheta * 2
            nz = 2
            while nz < 16:
                nz = nz + 2
                nz_list.append(nz)
                fluid_flow_obj = instantiate_fluid_flow_object(ntheta, nz, type_of_bearing)
                results = ross_finite_element_solution(fluid_flow_obj)
                error_fem_list.append(results[0])
                error_fdm_list.append(results[1])
                time_fem_list.append(results[2])
                time_fdm_list.append(results[3])
            print("nz_t" + str(ntheta) + " =", nz_list)
            print("error_fem_t" + str(ntheta) + " =", error_fem_list)
            print("error_fdm_t" + str(ntheta) + " =", error_fdm_list)
            print("time_fem_t" + str(ntheta) + " =", time_fem_list)
            print("time_fdm_t" + str(ntheta) + " =", time_fdm_list)
            print()


if __name__ == "__main__":
    grid_convergence(True, True)



