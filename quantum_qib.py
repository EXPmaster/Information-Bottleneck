import argparse
from functools import reduce

import numpy as np
import scipy.linalg
import torch
from torch.utils.tensorboard import SummaryWriter
from qiskit.quantum_info import random_density_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm


class Queue:

    def __init__(self, max_len=4):
        self.data = []
        self.max_len = max_len
    
    def push(self, x):
        if len(self.data) < self.max_len:
            self.data.append(x)
        else:
            self.data.pop(0)
            self.data.append(x)
    
    def getitem(self, idx):
        """ get which item, idx start from 1 """
        assert idx != 0, 'idx of queue equals to 0.'
        if idx > 0:
            return self.data[idx - 1]
        if idx < 0:
            return self.data[idx]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def is_hermitian(a, tol=1e-2):
    """Check if (batch of) matrices a are hermitian."""
    return torch.dist(a, a.mH) < tol


def is_positive(a):
    L, Q = torch.linalg.eigh(a)
    return (L.real > 0).all()


def batch_kron(a, b):
    """
    Kronecker product of matrices a and b with leading batch dimensions.
    Batch dimensions are broadcast.
    """
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    return res.reshape(siz0 + siz1)


def trace(a):
    return a.diagonal(dim1=-2, dim2=-1).sum(dim=-1)


def partial_trace(rho, trace_out, dims):
    """Calculate the partial trace for a batch of matrices.
    
    Args:
        trace_out: 
        dims: array
            An array of the dimensions of each space.
            For instance, if the space is A x B x C x D,
            dims = [dim_A, dim_B, dim_C, dim_D]
    Returns:
        Traced matrix
    """
    dims += dims
    rho_a = rho.reshape(rho.shape[0], *dims)
    if trace_out == 0:
        rho_a = torch.einsum('bijik->bjk', rho_a)
    else:
        rho_a = torch.einsum('bjiki->bjk', rho_a)
    return rho_a


def matrix_log2(matrix):
    """ Matrix log2 for a batch of hermitian matrices.
    """
    # assert is_hermitian(matrix), f'LOG: matrix {matrix} is not hermitian.'
    L, Q = torch.linalg.eigh(matrix)
    # assert (L.real > 0).all(), f'Non-positive values found in log. {L}'
    eigvals = torch.where(L.real > 0, L.real, 1e-9)
    return Q @ torch.diag_embed(torch.log2(eigvals)).cfloat() @ Q.mH
    # if len(matrix.shape) == 2:
    #     return torch.from_numpy(scipy.linalg.logm(matrix.cpu().numpy(), disp=False)[0]).to(matrix.device)

    # Am = torch.empty_like(matrix)
    # for i in range(len(matrix)):
    #     Am[i] = torch.from_numpy(scipy.linalg.logm(matrix[i].cpu().numpy(), disp=False)[0]).to(matrix.device)
    # return Am


def matrix_exp(matrix):
    assert is_hermitian(matrix), f'EXP: matrix {matrix} is not hermitian.'
    L, Q = torch.linalg.eigh(matrix)
    eigens = torch.diag_embed(torch.exp(L))
    # eigens = torch.nan_to_num(eigens.clamp(max=1e+9))
    return Q @ eigens.cfloat() @ Q.mH


def rho_Y(pX, rho_Y_x):
    """
    rho_Y = \sum_x pX(x) * rho_Y_x

    Args:
        pX: [n, 1]
        rho_Y_x: [n, m, m]
    Returns:
        rho_Y: [m, m]
    """
    return (pX[:, None, None] * rho_Y_x).sum(0)


def sigma_T(pX, sigma_T_x):
    """
    sigma_T[sigma_{T|X}] = \sum_x pX(x) * sigma_T_x

    Args:
        pX: [n, 1]
        sigma_T_x: [n, m', m']
    Returns:
        sigma_T: [m', m']
    """
    return (pX[:, None, None] * sigma_T_x).sum(0)


def sigma_YT(pX, sigma_T_x, rho_Y_x):
    """
    sigma_YT[sigma_{T|X}] = \sum_x pX(x) * sigma_T_x \otimes rho_Y_x

    Args:
        pX: [n, 1]
        sigma_T_x: [n, m', m']
        rho_Y_x: [n, m, m]
    Returns:
        sigma_YT: [m * m', m * m']
    """
    return (pX[:, None, None] * batch_kron(sigma_T_x, rho_Y_x)).sum(0)


def relative_entropy(a, b):
    return trace(a @ (matrix_log2(a) - matrix_log2(b)))


def kl_distance(pX, a, b):
    return (pX * relative_entropy(a, b)).sum()


def entropy(rho):
    """ Non-batched entropy of density operators. """
    return -torch.trace(rho @ matrix_log2(rho))


def F_alpha(alpha, beta, pX, sigma_T_X, rho_Y_X):
    """ F_alpha(x) for all x in a batch
    """
    sigma_t_TX = sigma_T(pX, sigma_T_X)
    dim = sigma_T_X.shape[1]
    eyes = torch.eye(dim).to(sigma_T_X.device)
    rho_y = rho_Y(pX, rho_Y_X)
    tmp_rho = batch_kron(eyes, rho_Y_X) @ (matrix_log2(batch_kron(sigma_t_TX, rho_y)) - \
                matrix_log2(sigma_YT(pX, sigma_T_X, rho_Y_X)))

    return -matrix_log2(sigma_t_TX) + alpha * matrix_log2(sigma_T_X) + \
            beta * partial_trace(tmp_rho, 1, dims=[sigma_T_X.shape[1], rho_Y_X.shape[1]])


def f_alpha(alpha, beta, pX, sigma_T_X, rho_Y_X):
    return (pX * trace(sigma_T_X @ F_alpha(alpha, beta, pX, sigma_T_X, rho_Y_X))).sum(0)


def J_gamma_alpha(gamma, alpha, beta, pX, sigma_T_X, sigma_T_X2, rho_Y_X):
    return gamma * (pX * relative_entropy(sigma_T_X, sigma_T_X2)).sum(0) + \
            f_alpha(alpha, beta, pX, sigma_T_X2, rho_Y_X)


def f_gamma(alpha, beta, pX, sigma_T_X1, sigma_T_X2, rho_Y_X):
    tmp_rho = sigma_T_X1 @ (F_alpha(alpha, beta, pX, sigma_T_X1, rho_Y_X) - \
                            F_alpha(alpha, beta, pX, sigma_T_X2, rho_Y_X))
    a0 = (pX * trace(tmp_rho)).sum(0)
    a1 = (pX * relative_entropy(sigma_T_X1, sigma_T_X2)).sum(0)
    return a0 / a1


def initialize_sigma_T_X(args):
    # sigma_T_X = torch.stack([torch.from_numpy(random_density_matrix(args.dim_T).data).cfloat() for _ in range(args.dim_X)])
    # sigma_T_X = torch.stack([torch.diag(torch.softmax(torch.rand(args.dim_T), -1)).cfloat()
    #                          for _ in range(args.dim_X)]).to(device)
    sigma_T_X = []
    for i in range(args.dim_X):
        rand_vec = torch.rand(args.dim_T)
        rand_vec[i] = 0
        rand_vec = rand_vec / rand_vec.sum() * 0.25
        rand_vec[i] = 0.75
        sigma_T_X.append(torch.diag(rand_vec).cfloat())
    return torch.stack(sigma_T_X).to(device)


def fixed_gamma_qib(args, pX, rho_Y_X, rho_X):
    sigma_T_X = initialize_sigma_T_X(args)
    assert is_hermitian(sigma_T_X), 'sigma_T|X is not hermitian.'
    assert is_positive(sigma_T_X), f'sigma_T|X is not positive.'
    previous_loss = float('inf')
    converged = False
    n = 0
    while not converged:
        sigma_T_X0 = sigma_T_X.clone()
        sigma_gamma_alpha_T = torch.matrix_exp(matrix_log2(sigma_T_X) - \
                                1 / args.gamma * F_alpha(args.alpha, args.beta, pX, sigma_T_X, rho_Y_X))
        sigma_T_X = sigma_gamma_alpha_T / trace(sigma_gamma_alpha_T)[:, None, None]
        # assert is_hermitian(sigma_T_X), f'sigma_T|X is not hermitian. dist: {torch.dist(sigma_T_X, sigma_T_X.mH)}'
        if not is_hermitian(sigma_T_X):
            return None, None, None
        loss = J_gamma_alpha(args.gamma, args.alpha, args.beta, pX, sigma_T_X, sigma_T_X0, rho_Y_X)
        # print('loss:', loss)
        f_alpha_value = f_alpha(args.alpha, args.beta, pX, sigma_T_X, rho_Y_X)
        # args.writer.add_scalar('f_alpha', f_alpha_value.real, n)
        # args.writer.add_scalar("J(a, a')", loss.real, n)
        # print('f_alpha:', f_alpha_value.item())
        if abs(loss.real - previous_loss.real) < args.cvg_thres:
            converged = True
        previous_loss = loss
        n += 1
    # print(f'Algorithm converged in {n} iterations.')

    rho_t = sigma_T(pX, sigma_T_X)
    rho_x = torch.diag(pX).cfloat()
    rho_xt = (pX[:, None, None] * batch_kron(rho_X, sigma_T_X)).sum(0)
    I_TX = entropy(rho_x) + entropy(rho_t) - entropy(rho_xt)

    rho_y = rho_Y(pX, rho_Y_X)
    rho_yt = sigma_YT(pX, sigma_T_X, rho_Y_X)
    I_YT = entropy(rho_y) + entropy(rho_t) - entropy(rho_yt)
    
    return I_TX, I_YT, n


def adaptive_qib(args, pX, rho_Y_X, rho_X):
    c = 1.0
    args.gamma = args.alpha
    sigma_T_X = Queue(max_len=4)
    sigma_T_X.push(initialize_sigma_T_X(args))
    sigma_gamma_alpha_T = matrix_exp(matrix_log2(sigma_T_X.getitem(-1)) - \
                          1 / args.gamma * F_alpha(args.alpha, args.beta, pX, sigma_T_X.getitem(-1), rho_Y_X))
    sigma_T_X.push(sigma_gamma_alpha_T / trace(sigma_gamma_alpha_T)[:, None, None])
    n = 2
    previous_loss = float('inf')
    converged = False
    lambda_n = 0.

    while not converged:
        gamma_star = f_gamma(args.alpha, args.beta, pX, sigma_T_X.getitem(1), sigma_T_X.getitem(2), rho_Y_X)
        if gamma_star.real > 0:
            if n == 2:
                lambda_n = 3 * c * kl_distance(pX, sigma_T_X.getitem(2), sigma_T_X.getitem(1))
            elif n == 3:
                lambda_n = 1.5 * c * kl_distance(pX, sigma_T_X.getitem(3), sigma_T_X.getitem(2)) + \
                           kl_distance(pX, sigma_T_X.getitem(2), sigma_T_X.getitem(1))
            else:
                lambda_n = c * kl_distance(pX, sigma_T_X.getitem(4), sigma_T_X.getitem(3)) + \
                           kl_distance(pX, sigma_T_X.getitem(3), sigma_T_X.getitem(2)) + \
                           kl_distance(pX, sigma_T_X.getitem(2), sigma_T_X.getitem(1))
            lambda_n = lambda_n.real
            args.gamma = min(args.alpha, lambda_n * (args.alpha - gamma_star.real) + gamma_star.real)

        sigma_gamma_alpha_T = matrix_exp(matrix_log2(sigma_T_X.getitem(-1)) - \
                                1 / args.gamma * F_alpha(args.alpha, args.beta, pX, sigma_T_X.getitem(-1), rho_Y_X))

        new_sigma_T_X = sigma_gamma_alpha_T / trace(sigma_gamma_alpha_T)[:, None, None]
        # assert is_hermitian(new_sigma_T_X), f'sigma_T|X is not hermitian. dist: {torch.dist(new_sigma_T_X, new_sigma_T_X.mH)}'
        if not is_hermitian(new_sigma_T_X):
            break
        sigma_T_X.push(new_sigma_T_X)
        loss = J_gamma_alpha(args.gamma, args.alpha, args.beta, pX, sigma_T_X.getitem(-1), sigma_T_X.getitem(-2), rho_Y_X)
        if abs(loss.real - previous_loss.real) < args.cvg_thres:
            converged = True
        previous_loss = loss
        n += 1

    rho_t = sigma_T(pX, sigma_T_X.getitem(-1))
    rho_x = torch.diag(pX).cfloat()
    rho_xt = (pX[:, None, None] * batch_kron(rho_X, sigma_T_X.getitem(-1))).sum(0)
    I_TX = entropy(rho_x) + entropy(rho_t) - entropy(rho_xt)

    rho_y = rho_Y(pX, rho_Y_X)
    rho_yt = sigma_YT(pX, sigma_T_X.getitem(-1), rho_Y_X)
    I_YT = entropy(rho_y) + entropy(rho_t) - entropy(rho_yt)
    
    return I_TX, I_YT, n


def rho_theta_lambda(th, la):
    def exp_i(A, theta):
        """assert A^2 = I"""
        theta = theta[:, None, None]
        return torch.cos(theta) * torch.eye(A.shape[1])[None] + 1j * torch.sin(theta) * A[None]
    pauli_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat, device=device)
    pauli_xs = reduce(torch.kron, [pauli_x for _ in range(5)])
    # tmp = 1j * th[:, None, None] * pauli_x[None]
    # return matrix_exp(tmp) @ torch.diag_embed(la).cfloat() @ matrix_exp(-tmp)
    return exp_i(pauli_xs, th) @ torch.diag_embed(la).cfloat() @ exp_i(pauli_xs, -th)


def gen_rho_y_x(args):
    """ Generate rho_{Y|X} """
    # thetas = torch.rand(args.dim_X) * 2 + 0.1
    # lambdas = torch.softmax(torch.rand(args.dim_X, args.dim_Y), -1)
    # thetas = torch.arange(1, args.dim_X + 1) * torch.pi / args.dim_X
    # lambdas = torch.arange(1, args.dim_X + 1) / (4 * args.dim_X)
    # return rho_theta_lambda(thetas, lambdas)
    alphas = torch.logspace(-1.3, 1.3, args.dim_X)
    rho_y = []
    for alpha in alphas:
        distr_y = torch.distributions.dirichlet.Dirichlet(torch.ones(args.dim_Y, dtype=torch.float) * alpha)
        py = distr_y.sample()
        rho_y.append(torch.diag(py).cfloat())
    return torch.stack(rho_y).to(device)


def gen_rho_x(args):
    """ rho_x: [dim_X, dim_X, dim_X], dim_x stacks of {|x><x|} """
    rho_x = []
    for i in range(args.dim_X):
        vec = torch.zeros(args.dim_X).cfloat()
        vec[i] = 1
        rho_x.append(torch.outer(vec, vec))
    return torch.stack(rho_x).to(device)


@torch.no_grad()
def main(args):
    # pX = torch.ones(args.dim_X) / args.dim_X
    distr_x = torch.distributions.dirichlet.Dirichlet(torch.ones(args.dim_X, dtype=torch.float) * 1000)
    pX = distr_x.sample().to(device)
    rho_X = gen_rho_x(args)
    rho_Y_X = gen_rho_y_x(args).to(device)
    assert is_positive(rho_Y_X), 'rho_Y|X is not positive.'
    assert is_hermitian(rho_Y_X), 'rho_Y|X is not hermitian.'

    x_coords = []
    y_coords = []
    betas = []
    iterations = []
    for beta in tqdm(torch.arange(0.001, 30, 0.1)):
        args.beta = beta
        if args.algorithm == 'deterministic':
            I_TX, I_YT, n = fixed_gamma_qib(args, pX, rho_Y_X, rho_X)
        else:
            I_TX, I_YT, n = adaptive_qib(args, pX, rho_Y_X, rho_X)
        if I_TX is not None:
            betas.append(beta)
            iterations.append(n)
            x_coords.append(I_TX.cpu())
            y_coords.append(I_YT.cpu())
    
    fig = plt.figure()
    plt.scatter(x_coords, y_coords)
    plt.xlabel('I(X;T)')
    plt.ylabel('I(T;Y)')
    plt.savefig(f'qib_plane_{args.algorithm}.png')

    fig = plt.figure()
    plt.plot(betas, iterations)
    plt.xlabel('beta')
    plt.ylabel('Iters')
    plt.savefig(f'qib_converge_itrs_{args.algorithm}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', default='adaptive', type=str, help='deterministic or adaptive QIB.')
    parser.add_argument('--gamma', default=0.8, type=float)
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--beta', default=20., type=float)
    parser.add_argument('--dim_X', default=2**4, type=int, help='dimension of X')
    parser.add_argument('--dim_Y', default=2**2, type=int, help='dimension of Y')
    parser.add_argument('--dim_T', default=2**4, type=int, help='dimension of T')
    parser.add_argument('--cvg_thres', default=1e-3, type=int, help='threshold for convergence of algorithm')
    args = parser.parse_args()

    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    print(f'==== Running quantum algorithm on {device}....')

    # set_seed(0)
    # args.writer = SummaryWriter(log_dir='./logs')
    main(args)
    # args.writer.flush()
    # args.writer.close()
    