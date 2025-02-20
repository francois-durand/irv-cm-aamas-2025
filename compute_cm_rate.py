import numpy as np
from svvamp import GeneratorProfilePerturbedCulture, RuleTwoRound, RuleIRV

from fast_plurality import compute_plurality_scores_and_wmm, plurality_is_profile_cm


def plurality_cm_rate(n_v, n_c, theta, n_samples, rng=None):
    """Compute the CM rate of Plurality.

    Parameters
    ----------
    n_v: int
        Number of voters.
    n_c: int
        Number of candidates.
    theta: float
        Concentration parameter of the Perturbed Culture Model.
    n_samples: int
        Number of samples.
    rng: numpy.random.Generator
        Random number generator.

    Returns
    -------
    float
        CM rate of Plurality.

    Examples
    --------
        >>> n_v = 51
        >>> n_c = 4
        >>> theta = .2
        >>> n_samples = 100
        >>> rng = np.random.default_rng(seed=42)
        >>> plurality_cm_rate(n_v, n_c, theta, n_samples, rng)
        0.55
    """
    if rng is None:
        rng = np.random.default_rng()
    cm_count = 0
    for _ in range(n_samples):
        plurality_scores, wmm = compute_plurality_scores_and_wmm(n_v, n_c, theta, rng)
        if plurality_is_profile_cm(plurality_scores, wmm, n_c):
            cm_count += 1
    return cm_count / n_samples


def tr_cm_rate(n_v, n_c, theta, n_samples):
    """Compute the CM rate of the Two-Round system.

    Parameters
    ----------
    n_v: int
        Number of voters.
    n_c: int
        Number of candidates.
    theta: float
        Concentration parameter of the Perturbed Culture Model.
    n_samples: int
        Number of samples.

    Returns
    -------
    float
        CM rate of the Two-Round system.

    Examples
    --------
        >>> n_v = 51
        >>> n_c = 4
        >>> theta = .2
        >>> n_samples = 100
        >>> np.random.seed(42)
        >>> tr_cm_rate(n_v, n_c, theta, n_samples)
        0.13
    """
    n_profiles_cm = 0
    profile_generator = GeneratorProfilePerturbedCulture(n_v=n_v, theta=theta, n_c=n_c)
    for _ in range(n_samples):
        profile = profile_generator()
        if RuleTwoRound()(profile).is_cm_:
            n_profiles_cm += 1
    return n_profiles_cm / n_samples


def irv_cm_rate(n_v, n_c, theta, n_samples):
    """Compute the CM rate of IRV.

    Parameters
    ----------
    n_v: int
        Number of voters.
    n_c: int
        Number of candidates.
    theta: float
        Concentration parameter of the Perturbed Culture Model.
    n_samples: int
        Number of samples.

    Returns
    -------
    float
        CM rate of IRV.

    Examples
    --------
        >>> n_v = 51
        >>> n_c = 4
        >>> theta = .2
        >>> n_samples = 100
        >>> np.random.seed(42)
        >>> irv_cm_rate(n_v, n_c, theta, n_samples)
        0.08
    """
    n_profiles_cm = 0
    profile_generator = GeneratorProfilePerturbedCulture(n_v=n_v, theta=theta, n_c=n_c)
    for _ in range(n_samples):
        # The option 'precheck_heuristic=False' makes the code almost twice faster here, with the same results.
        if RuleIRV(cm_option='exact', precheck_heuristic=False)(profile_generator()).is_cm_:
            n_profiles_cm += 1
    return n_profiles_cm / n_samples


def plu_theta_c(n_c):
    """Compute theta_c for Plurality.

    Parameters
    ----------
    n_c: int
        Number of candidates.

    Returns
    -------
    float
        theta_c for Plurality.

    Examples
    --------
        >>> n_c = 4
        >>> plu_theta_c(n_c)
        0.2
    """
    return (n_c - 2) / (3 * n_c - 2)


def tr_theta_c(n_c):
    """Compute theta_c for Two-Round.

    Parameters
    ----------
    n_c: int
        Number of candidates.

    Returns
    -------
    float
        theta_c for Two-Round.

    Examples
    --------
        >>> n_c = 4
        >>> tr_theta_c(n_c)
        0.058823529411764705
    """
    return (n_c - 3) / (5 * n_c - 3)


def irv_theta_c(n_c):
    """Compute theta_c for IRV.

    Parameters
    ----------
    n_c: int
        Number of candidates.

    Returns
    -------
    float
        theta_c for IRV.

    Examples
    --------
        >>> n_c = 4
        >>> irv_theta_c(n_c)
        0
    """
    return 0
