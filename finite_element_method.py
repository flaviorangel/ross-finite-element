import numpy as np
from ross.fluid_flow import fluid_flow as flow
from ross.fluid_flow import fluid_flow_graphics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

gq3 = (3**0.5)/float(3)
phiOrd = [[-1, -1], [1, -1], [1, 1], [-1, 1]]


def phi_a(ksa, na, ks, n):
    return (1 + ksa * ks) * (1 + na * n) / float(4)


def d_phi_a_d_ks(ksa, na, ks, n):
    return ksa * (1 + na * n) / float(4)


def d_phi_a_d_n(ksa, na, ks, n):
    return na * (1 + ksa * ks) / float(4)


class Element:
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
        self.Ke += self.g_function(-gq3, -gq3)
        self.Ke += self.g_function(gq3, -gq3)
        self.Ke += self.g_function(-gq3, gq3)
        self.Ke += self.g_function(gq3, gq3)
        self.F_Matrix = np.zeros(shape=(4, 4))
        for i in range(0, 4):
            for j in range(0, 4):
                self.F_Matrix[i][j] += self.g_for_f_function(-gq3, -gq3, i, j)
                self.F_Matrix[i][j] += self.g_for_f_function(gq3, -gq3, i, j)
                self.F_Matrix[i][j] += self.g_for_f_function(-gq3, gq3, i, j)
                self.F_Matrix[i][j] += self.g_for_f_function(gq3, gq3, i, j)
        self.Fe = np.dot(self.F_Matrix, self.f_vector)
        self.q_vector = np.dot(self.Ke, self.q)
        self.Fe -= self.q_vector
        for i in range(0, 4):
            for j in range(0, 4):
                if i > j:
                    self.Ke[i][j] = 0

    def g_for_f_function(self, st, nd, i, j):
        B = np.zeros(shape=(2, 4))
        for k in range(0, 4):
            B[0][k] = d_phi_a_d_ks(phiOrd[k][0], phiOrd[k][1], st, nd)
            B[1][k] = d_phi_a_d_n(phiOrd[k][0], phiOrd[k][1], st, nd)
        J = np.dot(self.M, np.transpose(B))
        term1 = phi_a(phiOrd[i][0], phiOrd[i][1], st, nd)
        term2 = phi_a(phiOrd[j][0], phiOrd[j][1], st, nd)
        return term1 * term2 * np.linalg.det(J)

    def g_function(self, st, nd):
        B = np.zeros(shape=(2, 4))
        for k in range(0, 4):
            B[0][k] = d_phi_a_d_ks(phiOrd[k][0], phiOrd[k][1], st, nd)
            B[1][k] = d_phi_a_d_n(phiOrd[k][0], phiOrd[k][1], st, nd)
        J = np.dot(self.M, np.transpose(B))
        R = np.transpose(np.linalg.inv(J))
        JBTRT = np.linalg.det(J) * np.dot(np.transpose(B), np.transpose(R))
        QRB = np.dot(self.Q, np.dot(R, B))
        return np.dot(JBTRT, QRB)


class FiniteElementMethod:
    def __init__(self, fluid_flow_object, theta_max=None,
                 book_task=False, f_function=None, q_function=None):
        fluid_flow_object.calculate_coefficients()
        self.fluid_flow_object = fluid_flow_object
        self.n_elements_z = fluid_flow_object.nz - 1
        self.n_elements_theta = fluid_flow_object.ntheta - 1
        self.matrix_z = np.arange(0, fluid_flow_object.length + (fluid_flow_object.length /
                                                                 self.n_elements_z) / 2.0,
                                  fluid_flow_object.length / self.n_elements_z)
        if theta_max is None:
            self.matrix_theta = np.arange(0, 2 * np.pi + ((2 * np.pi) / self.n_elements_theta) / 2.0,
                                          (2 * np.pi) / self.n_elements_theta)
        else:
            self.matrix_theta = np.arange(0, theta_max + (theta_max / self.n_elements_theta) / 2.0,
                                          theta_max / self.n_elements_theta)
        self.matrix_theta, self.matrix_z = np.meshgrid(self.matrix_theta, self.matrix_z)
        self.n_nodes = fluid_flow_object.nz * fluid_flow_object.ntheta
        if not book_task:
            self.n_known_nodes = fluid_flow_object.ntheta * 2
        else:
            self.n_known_nodes = self.fluid_flow_object.nz * 2 + 2 * (self.fluid_flow_object.ntheta - 2)
        self.n_elements = self.n_elements_z * self.n_elements_theta
        self.equation_vector = np.zeros(self.n_nodes, dtype=int)
        self.q_function_in_node = np.zeros(self.n_nodes, dtype=float)
        self.f_function_in_node = np.zeros(self.n_nodes, dtype=float)
        self.i_in_node = np.zeros(self.n_nodes, dtype=int)
        self.j_in_node = np.zeros(self.n_nodes, dtype=int)
        self.calculate_i_and_j_in_node()
        self.z_in_node = np.zeros(self.n_nodes, dtype=float)
        self.theta_in_node = np.zeros(self.n_nodes, dtype=float)
        self.calculate_nodes_positions()
        self.init_vectors(book_task, f_function, q_function)
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
        if not book_task:
            self.K = np.zeros(shape=(self.n_equations + m, self.n_equations + m))
            self.build_k_matrix(m, eq_theta_zero, eq_theta_2pi, book_task)
            self.n_equations += m
        else:
            self.K = np.zeros(shape=(self.n_equations, self.n_equations))
        self.F = np.zeros(self.n_equations)
        self.build_matrices(book_task)
        self.answer_vector = np.zeros(self.n_nodes)
        self.C = self.solve()
        self.build_answer_vector()
        self.pressure_matrix = np.copy(fluid_flow_object.p_mat_numerical)
        self.build_pressure_matrix()

    def calculate_i_and_j_in_node(self):
        m = 0
        for i in range(0, self.fluid_flow_object.nz):
            for j in range(0, self.fluid_flow_object.ntheta):
                self.i_in_node[m] = i
                self.j_in_node[m] = j
                m += 1

    def init_vectors(self, book_task, f_function, q_function):
        if book_task:
            k = 0
            m = 0
            for i in range(0, self.fluid_flow_object.nz):
                for j in range(0, self.fluid_flow_object.ntheta):
                    self.f_function_in_node[m] = f_function(self.theta_in_node[m], self.z_in_node[m])
                    if (j == 0) or (i == self.fluid_flow_object.nz - 1) or (j == self.fluid_flow_object.ntheta - 1) or (i == 0):
                        self.equation_vector[m] = -1
                        if i == 0:
                            self.q_function_in_node[m] = q_function(self.theta_in_node[m],
                                                                    self.z_in_node[m],
                                                                    0)
                        elif j == self.fluid_flow_object.nz - 1:
                            self.q_function_in_node[m] = q_function(self.theta_in_node[m],
                                                                    self.z_in_node[m],
                                                                    1)
                        elif i == self.fluid_flow_object.ntheta - 1:
                            self.q_function_in_node[m] = q_function(self.theta_in_node[m],
                                                                    self.z_in_node[m],
                                                                    2)
                        elif j == 0:
                            self.q_function_in_node[m] = q_function(self.theta_in_node[m],
                                                                    self.z_in_node[m],
                                                                    3)
                    else:
                        self.q_function_in_node[m] = 0
                        self.equation_vector[m] = k
                        k += 1
                    m += 1
        else:
            k = 0
            m = 0
            for i in range(0, self.fluid_flow_object.nz):
                for j in range(0, self.fluid_flow_object.ntheta):
                    self.i_in_node[m] = i
                    self.j_in_node[m] = j
                    if j == 0:
                        self.f_function_in_node[m] = (
                            (self.fluid_flow_object.c0w[i][j] -
                             self.fluid_flow_object.c0w[i][self.fluid_flow_object.ntheta - 1]) /
                            self.fluid_flow_object.dtheta
                        )
                    else:
                        self.f_function_in_node[m] = (
                                (self.fluid_flow_object.c0w[i][j] -
                                 self.fluid_flow_object.c0w[i][j - 1]) /
                                self.fluid_flow_object.dtheta
                        )
                    if i == 0 or i == self.n_elements_z:
                        self.equation_vector[m] = -1
                        if i == 0:
                            if book_task is None:
                                self.q_function_in_node[m] = self.fluid_flow_object.p_in
                        else:
                            if book_task is None:
                                self.q_function_in_node[m] = self.fluid_flow_object.p_out
                    else:
                        self.equation_vector[m] = k
                        k += 1
                    m += 1

    def build_local_nodes_matrix(self):
        k = 0
        j = 1
        m = self.n_elements_theta
        for i in range(0, self.n_elements):
            self.local_nodes_matrix[i][0] = k
            self.local_nodes_matrix[i][1] = k + 1
            k += 1
            if k >= m:
                k = (self.n_elements_theta + 1) * j
                j += 1
                m = k + self.n_elements_theta
        k = self.n_elements_theta + 1
        j = 2
        m = k + self.n_elements_theta
        for i in range(0, self.n_elements):
            self.local_nodes_matrix[i][3] = k
            self.local_nodes_matrix[i][2] = k + 1
            k += 1
            if k >= m:
                k = (self.n_elements_theta + 1) * j
                j += 1
                m = k + self.n_elements_theta

    def calculate_nodes_positions(self):
        for i in range(0, self.n_nodes):
            self.z_in_node[i] = self.matrix_z[self.i_in_node[i]][self.j_in_node[i]]
            self.theta_in_node[i] = self.matrix_theta[self.i_in_node[i]][self.j_in_node[i]]

    def build_k_matrix(self, m, eq_theta_zero, eq_theta_2pi, book_task):
        if not book_task:
            for i in range(0, m):
                self.K[self.n_equations + i][eq_theta_zero] = 1
                self.K[self.n_equations + i][eq_theta_2pi] = -1
                eq_theta_zero += 1
                eq_theta_2pi += 1

    def build_matrices(self, book_task):
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
            if not book_task:
                c_matrix[0][0] = self.fluid_flow_object.c2[self.i_in_node[e]][self.j_in_node[e]]
                c_matrix[1][1] = self.fluid_flow_object.c1[self.i_in_node[e]][self.j_in_node[e]]
                this_element = Element(list_z, list_theta, list_q, list_f, c_matrix)
            else:
                c_matrix[0][0] = 2
                c_matrix[0][1] = 1
                c_matrix[1][0] = 1
                c_matrix[1][1] = 2
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
        return np.linalg.solve(self.K, self.F)

    def build_answer_vector(self):
        k = 0
        for i in range(0, self.n_nodes):
            if self.equation_vector[i] == -1:
                self.answer_vector[i] = self.q_function_in_node[i]
            else:
                self.answer_vector[i] = self.C[k]
                k += 1

    def build_pressure_matrix(self):
        k = 0
        for i in range(0, self.fluid_flow_object.nz):
            for j in range(0, self.fluid_flow_object.ntheta):
                self.pressure_matrix[i][j] = self.answer_vector[k]
                k += 1

    def matplot_3d_graphic(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(self.matrix_z, self.matrix_theta, self.pressure_matrix,
                               cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.7)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.scatter(np.append(self.z_in_node, self.matrix_z[-1]),
                   np.append(self.theta_in_node, self.matrix_theta[-1]), self.answer_vector, c='r',
                   marker='^')
        plt.show()


def book_example1():
    def f_function(x, y):
        return (2*np.pi**2)*(2*np.sin(np.pi*x)*np.cos(np.pi*y) + np.cos(np.pi*x)*np.sin(np.pi*y))

    def solution(x, y):
        return np.sin(np.pi*x)*np.cos(np.pi*y)

    def q_function(x, y, boundary):
        if boundary == 0:
            return np.sin(np.pi*x)
        elif boundary == 1:
            return 0
        elif boundary == 2:
            return -np.sin(np.pi*x)
        else:
            return 0

    c_matrix = np.zeros(shape=(2, 2))
    c_matrix[0][0] = 2
    c_matrix[0][1] = 1
    c_matrix[1][0] = 1
    c_matrix[1][1] = 2
    nz = 5
    ntheta = 5
    nradius = 11
    omega = 100. * 2 * np.pi / 60
    p_in = 0.
    p_out = 0.
    radius_rotor = 0.1999996
    radius_stator = 0.1999996 + 0.000194564
    length = 1.0
    eccentricity = 0.0001
    visc = 0.015
    rho = 860.
    my_fluid_flow = flow.PressureMatrix(nz, ntheta, nradius, length, omega, p_in,
                                        p_out, radius_rotor, radius_stator,
                                        visc, rho, eccentricity=eccentricity)
    my_finite_element = FiniteElementMethod(my_fluid_flow, theta_max=1.0, f_function=f_function, q_function=q_function,
                                            book_task=True)
    print("F")
    print(my_finite_element.F)
    print("K:")
    print(my_finite_element.K)
    print("C")
    print(my_finite_element.C)


def ross_finite_element_solution():
    nz = 16
    ntheta = 128
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
    # my_fluid_flow.calculate_pressure_matrix_analytical()
    ax = fluid_flow_graphics.matplot_pressure_theta(my_fluid_flow, z=int(my_fluid_flow.nz / 2))
    # my_fluid_flow.calculate_pressure_matrix_numerical()
    # ax = fluid_flow_graphics.matplot_pressure_theta(my_fluid_flow, z=int(my_fluid_flow.nz/2), ax=ax,
    #                                                 color="black")
    plt.show()


if __name__ == "__main__":
    book_example1()
    # ross_finite_element_solution()




