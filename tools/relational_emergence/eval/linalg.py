"""The little dense linear algebra these evaluators need, in pure stdlib.

Kept here rather than pulled from numpy because the dependency-free CI job has no
third-party packages, and these evaluators are the phase's endpoints -- a
measurement that only runs where torch is installed is a measurement that is
rarely checked.

Only what is actually used: a symmetric eigensolver (cyclic Jacobi), and an SVD
built from it, because orthogonal Procrustes needs one and the role-equivariance
endpoint is defined by an orthogonal transform. Everything is dense and small --
the latent widths here are tens, not thousands -- so the O(n^3) routines are
adequate and the simple implementation is worth more than a fast one.
"""

from __future__ import annotations

import math

Matrix = list[list[float]]

JACOBI_SWEEPS = 60
JACOBI_TOLERANCE = 1e-12


def transpose(matrix: Matrix) -> Matrix:
    if not matrix:
        return []
    return [list(row) for row in zip(*matrix, strict=True)]


def matmul(left: Matrix, right: Matrix) -> Matrix:
    if not left or not right:
        return []
    inner = len(right)
    if len(left[0]) != inner:
        raise ValueError(f"cannot multiply {len(left)}x{len(left[0])} by {inner}x{len(right[0])}")
    right_t = transpose(right)
    return [
        [sum(a * b for a, b in zip(row, column, strict=True)) for column in right_t]
        for row in left
    ]


def symmetric_eigen(matrix: Matrix) -> tuple[list[float], Matrix]:
    """``(eigenvalues, eigenvectors)`` of a symmetric matrix, by cyclic Jacobi.

    Eigenvector ``k`` is column ``k`` of the returned matrix. Jacobi is chosen
    over a tridiagonal reduction because it is short enough to be read, and the
    matrices here are small enough that the constant factor does not matter.
    """
    size = len(matrix)
    if size == 0:
        return [], []
    if any(len(row) != size for row in matrix):
        raise ValueError("the matrix must be square")
    for i in range(size):
        for j in range(i + 1, size):
            if abs(matrix[i][j] - matrix[j][i]) > 1e-9 * max(1.0, abs(matrix[i][j])):
                raise ValueError("the matrix is not symmetric")

    a = [row[:] for row in matrix]
    vectors = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
    for _ in range(JACOBI_SWEEPS):
        off = math.sqrt(sum(a[i][j] ** 2 for i in range(size) for j in range(size) if i != j))
        if off < JACOBI_TOLERANCE:
            break
        for p in range(size):
            for q in range(p + 1, size):
                if abs(a[p][q]) < JACOBI_TOLERANCE:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                t = math.copysign(1.0, theta) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(size):
                    akp = a[k][p]
                    akq = a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(size):
                    apk = a[p][k]
                    aqk = a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
                for k in range(size):
                    vkp = vectors[k][p]
                    vkq = vectors[k][q]
                    vectors[k][p] = c * vkp - s * vkq
                    vectors[k][q] = s * vkp + c * vkq

    values = [a[i][i] for i in range(size)]
    order = sorted(range(size), key=lambda i: -values[i])
    return (
        [values[i] for i in order],
        [[vectors[row][i] for i in order] for row in range(size)],
    )


def svd(matrix: Matrix) -> tuple[Matrix, list[float], Matrix]:
    """``(U, singular_values, Vt)`` for a dense matrix, thin form.

    Built from the symmetric eigensolver: the right singular vectors are the
    eigenvectors of ``A^T A``, the singular values their square roots, and the
    left ones recovered as ``A v / sigma``. Columns whose singular value is
    negligible are completed by Gram-Schmidt, because a Procrustes rotation is
    only defined on a full basis.
    """
    if not matrix:
        raise ValueError("cannot decompose an empty matrix")
    rows = len(matrix)
    cols = len(matrix[0])
    gram = matmul(transpose(matrix), matrix)
    values, vectors = symmetric_eigen(gram)
    singular = [math.sqrt(max(value, 0.0)) for value in values]

    left_columns: list[list[float]] = []
    a_v = matmul(matrix, vectors)
    for k in range(cols):
        if singular[k] > 1e-10:
            left_columns.append([a_v[i][k] / singular[k] for i in range(rows)])
        else:
            left_columns.append([0.0] * rows)

    basis: list[list[float]] = []
    for k in range(cols):
        candidate = left_columns[k][:]
        for existing in basis:
            projection = sum(a * b for a, b in zip(candidate, existing, strict=True))
            candidate = [a - projection * b for a, b in zip(candidate, existing, strict=True)]
        norm = math.sqrt(sum(value * value for value in candidate))
        if norm < 1e-10:
            # Degenerate direction: complete the basis with a coordinate axis.
            for axis in range(rows):
                trial = [1.0 if index == axis else 0.0 for index in range(rows)]
                for existing in basis:
                    projection = sum(a * b for a, b in zip(trial, existing, strict=True))
                    trial = [a - projection * b for a, b in zip(trial, existing, strict=True)]
                norm = math.sqrt(sum(value * value for value in trial))
                if norm > 1e-10:
                    candidate = trial
                    break
        basis.append([value / norm for value in candidate])

    u = [[basis[column][row] for column in range(cols)] for row in range(rows)]
    vt = transpose(vectors)
    return u, singular, vt


def orthogonal_procrustes(source: Matrix, target: Matrix) -> Matrix:
    """The orthogonal ``T`` minimising ``||target - source @ T||``.

    This is the restricted fit the plan asks for: an orthogonal transform has
    ``d * (d - 1) / 2`` free parameters rather than the ``d^2`` of a general
    linear map, so it cannot absorb an arbitrary re-encoding. Its inverse is its
    transpose, which is what makes ``||T^2 - I||`` a meaningful quantity to
    report rather than an artefact of the parameterisation.
    """
    if len(source) != len(target):
        raise ValueError("source and target must have the same number of rows")
    u, singular, vt = svd(matmul(transpose(source), target))
    del singular
    return matmul(u, vt)
