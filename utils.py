import numpy as np

I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1.j], [1.j, 0]])
Z = np.array([[1, 0], [0, -1]])
H = np.sqrt(1 / 2) * np.array([[1, 1], [1, -1]])


def tensor(lst):
    out = np.array([[1]])
    for arg in lst:
            out = np.kron(out, arg)
    return out


def direct_sum(A, B, np_type=np.double):
    H_12 = np.zeros((A.shape[0], B.shape[1]))
    H_21 = np.zeros((B.shape[0], A.shape[1]))
    return np.block([[A, H_12], [H_21, B]]).astype(np_type)


def connect(i:int, j:int, N:int, val=None):
    """
    Returns the connection operator matrix, C(i,j), which contains zeros in all entries apart from $C_{ij}(i,j)$
    and $C_{ji}(i,j)$ which are one.
    :param N: the dimension of the
    """
    assert i < N and j < N, "Error. Element indices cannot be larger than Hamiltonian dimension (noting counting from 0)"
    if val is None:
        C__of_i_j = np.zeros((N, N))
        C__of_i_j[i, j] = 1
        C__of_i_j[j, i] = 1
    else:
        C__of_i_j = np.zeros((N, N), dtype=np.complex_)
        C__of_i_j[i, j] = val
        C__of_i_j[j, i] = np.conj(val)
    return C__of_i_j


def duplicate(H):
    """
    Given the NxN MATRIX ENCODING of a graph, this returns the matrix representing the graph duplicated as two
    unconnected components. The output matrix is of dimension 2Nx2N.
    """
    I = np.diag([1, 1])
    return np.kron(I, H)


def collect(H1, H2):
    """
    While this operation on matrices may be defined by the direct sum, in terms of operations with the underlying
    Paulis, you have H = 0.5 (I + Z) \otimes H1 + 0.5 (I + XZX) \otimes H2
    """
    assert H1.shape[0] == H2.shape[0] and H1.shape[1] == H2.shape[1],\
        "Error. The collection operator is only defined on inputs with the same dimension."
    return direct_sum(H1, H2)

