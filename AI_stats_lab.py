import numpy as np

# -------------------------------------------------
# Joint PMF table
# -------------------------------------------------

_PMF = {
    (0, 0): 0.10, (0, 1): 0.05, (0, 2): 0.00, (0, 3): 0.00,
    (1, 0): 0.15, (1, 1): 0.20, (1, 2): 0.05, (1, 3): 0.00,
    (2, 0): 0.00, (2, 1): 0.10, (2, 2): 0.15, (2, 3): 0.05,
    (3, 0): 0.00, (3, 1): 0.00, (3, 2): 0.05, (3, 3): 0.10,
}

_VALS = [0, 1, 2, 3]


# -------------------------------------------------
# Joint PMF
# -------------------------------------------------

def joint_pmf(x, y):
    """
    Returns P(X=x, Y=y).
    """
    return _PMF.get((x, y), 0.0)


# -------------------------------------------------
# Marginal PMFs
# -------------------------------------------------

def marginal_px(x):
    """
    P_X(x) = sum over y of P(X=x, Y=y)
    """
    return sum(joint_pmf(x, y) for y in _VALS)


def marginal_py(y):
    """
    P_Y(y) = sum over x of P(X=x, Y=y)
    """
    return sum(joint_pmf(x, y) for x in _VALS)


# -------------------------------------------------
# Conditional PMFs
# -------------------------------------------------

def conditional_pmf_x_given_y(x, y):
    """
    P(X=x | Y=y)
    """
    py = marginal_py(y)

    if py == 0:
        return 0.0

    return joint_pmf(x, y) / py


def conditional_distribution_x_given_y(y):
    """
    Returns dictionary:
    {x: P(X=x | Y=y)}
    """
    return {
        x: conditional_pmf_x_given_y(x, y)
        for x in _VALS
    }


# -------------------------------------------------
# Probability Computation
# -------------------------------------------------

def probability_sum_greater_than_3():
    """
    Computes P(X + Y > 3)
    """
    return sum(
        joint_pmf(x, y)
        for x in _VALS
        for y in _VALS
        if x + y > 3
    )


# -------------------------------------------------
# Independence Check
# -------------------------------------------------

def independence_check():
    """
    Checks whether X and Y are independent.
    """
    return all(
        np.isclose(
            joint_pmf(x, y),
            marginal_px(x) * marginal_py(y)
        )
        for x in _VALS
        for y in _VALS
    )


# -------------------------------------------------
# Expectations
# -------------------------------------------------

def expected_x():
    """
    E[X]
    """
    return sum(
        x * marginal_px(x)
        for x in _VALS
    )


def expected_y():
    """
    E[Y]
    """
    return sum(
        y * marginal_py(y)
        for y in _VALS
    )


def expected_xy():
    """
    E[XY]
    """
    return sum(
        x * y * joint_pmf(x, y)
        for x in _VALS
        for y in _VALS
    )


# -------------------------------------------------
# Variances
# -------------------------------------------------

def variance_x():
    """
    Var(X) = E[X^2] - (E[X])^2
    """
    ex2 = sum(
        x**2 * marginal_px(x)
        for x in _VALS
    )

    ex = expected_x()

    return ex2 - ex**2


def variance_y():
    """
    Var(Y) = E[Y^2] - (E[Y])^2
    """
    ey2 = sum(
        y**2 * marginal_py(y)
        for y in _VALS
    )

    ey = expected_y()

    return ey2 - ey**2


# -------------------------------------------------
# Covariance & Correlation
# -------------------------------------------------

def covariance_xy():
    """
    Cov(X,Y) = E[XY] - E[X]E[Y]
    """
    return expected_xy() - expected_x() * expected_y()


def correlation_xy():
    """
    Corr(X,Y)
    """
    return covariance_xy() / np.sqrt(
        variance_x() * variance_y()
    )


# -------------------------------------------------
# Variance of Sum
# -------------------------------------------------

def variance_sum():
    """
    Var(X+Y)
    """
    exy = expected_x() + expected_y()

    exy2 = sum(
        (x + y) ** 2 * joint_pmf(x, y)
        for x in _VALS
        for y in _VALS
    )

    return exy2 - exy**2


def variance_identity_check():
    """
    Checks:
    Var(X+Y) = Var(X) + Var(Y) + 2Cov(X,Y)
    """
    lhs = variance_sum()

    rhs = (
        variance_x()
        + variance_y()
        + 2 * covariance_xy()
    )

    return bool(np.isclose(lhs, rhs))
