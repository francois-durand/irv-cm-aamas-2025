import itertools
import math

import numpy as np
from svvamp.utils.misc import preferences_rk_to_preferences_borda_rk


d_n_c_i_j_who_prefer_i_to_j = dict()
d_n_c_upper_triangular = dict()
d_n_c_threshold_use_multinomial = {4: 16, 5: 55, 6: 294, 7: 2117}


def who_prefer_i_to_j(n_c, i, j):
    """Return the set of rankings who prefer i to j.

    Parameters
    ----------
    n_c: int
        Number of candidates.
    i: int
        Index of the first candidate.
    j: int
        Index of the second candidate.

    Returns
    -------
    np.ndarray of bool
        Which rankings prefer i to j (in the order of itertools.permutations(range(n_c))).

    Examples
    --------
        >>> who_prefer_i_to_j(n_c=3, i=0, j=1)
        array([ True,  True, False, False,  True, False])
        >>> for i, ranking in enumerate(itertools.permutations(range(3))):  # doctest: +NORMALIZE_WHITESPACE
        ...     print(ranking, who_prefer_i_to_j(3, 0, 1)[i])
        (0, 1, 2) True
        (0, 2, 1) True
        (1, 0, 2) False
        (1, 2, 0) False
        (2, 0, 1) True
        (2, 1, 0) False
    """
    try:
        return d_n_c_i_j_who_prefer_i_to_j[(n_c, i, j)]
    except KeyError:
        rankings = np.array(list(itertools.permutations(range(n_c))))
        borda_scores = preferences_rk_to_preferences_borda_rk(rankings)
        for ii in range(n_c):
            for jj in range(i + 1, n_c):
                d_n_c_i_j_who_prefer_i_to_j[(n_c, ii, jj)] = borda_scores[:, ii] > borda_scores[:, jj]
        return d_n_c_i_j_who_prefer_i_to_j[(n_c, i, j)]


def upper_triangular(n_c):
    """
    Upper triangular matrix of ones.

    This is equivalent but faster than `np.triu(np.ones((n_c, n_c), dtype=int), k=1)`, thanks to caching.

    Parameters
    ----------
    n_c: int
        Number of candidates.

    Returns
    -------
    np.ndarray
        Upper triangular matrix of ones.

    Examples
    --------
        >>> upper_triangular(3)
        array([[0, 1, 1],
               [0, 0, 1],
               [0, 0, 0]])
    """
    try:
        return d_n_c_upper_triangular[n_c]
    except KeyError:
        d_n_c_upper_triangular[n_c] = np.triu(np.ones((n_c, n_c), dtype=int), k=1)
        return d_n_c_upper_triangular[n_c]


def _compute_plurality_scores_and_wmm_using_multinomial(n_v, n_c, theta, rng):
    """Draw a random profile and return plurality scores and weighted majority matrix.

    Parameters
    ----------
    n_v: int
        Number of voters.
    n_c: int
        Number of candidates.
    theta: float
        Concentration parameter of the Perturbed Culture Model.
    rng: numpy.random.Generator
        Random number generator.

    Returns
    -------
    tuple
        Plurality scores, weighted majority matrix.

    Examples
    --------
        >>> n_v = 1000
        >>> n_c = 3
        >>> theta = 0.5
        >>> rng = np.random.default_rng(seed=42)
        >>> plurality_scores, wmm = _compute_plurality_scores_and_wmm_using_multinomial(n_v, n_c, theta, rng)
        >>> plurality_scores
        array([662, 163, 175])
        >>> wmm
        array([[  0, 756, 743],
               [244,   0, 756],
               [257, 244,   0]])
    """
    # Compute numbers of voters for each ranking
    fact_n_c = math.factorial(n_c)
    proba_each_due_to_ic = (1 - theta) / fact_n_c
    vector_of_probas = np.full(fact_n_c, proba_each_due_to_ic)
    vector_of_probas[0] += theta
    numbers_of_voters = rng.multinomial(n_v, vector_of_probas, size=1)[0]

    # Compute plurality scores and weighted majority matrix
    rankings = np.array(list(itertools.permutations(range(n_c))))
    plurality_scores = np.bincount(rankings[:, 0], numbers_of_voters, minlength=n_c).astype(int)
    weighted_majority_matrix = np.zeros((n_c, n_c), dtype=int)
    for i in range(n_c):
        for j in range(i + 1, n_c):
            weighted_majority_matrix[i, j] = numbers_of_voters[who_prefer_i_to_j(n_c, i, j)].sum()
            weighted_majority_matrix[j, i] = n_v - weighted_majority_matrix[i, j]
    return plurality_scores, weighted_majority_matrix


def _compute_plurality_scores_and_wmm_looping_on_voters(n_v, n_c, theta, rng):
    """Draw a random profile and return plurality scores and weighted majority matrix.

    Parameters
    ----------
    n_v: int
        Number of voters.
    n_c: int
        Number of candidates.
    theta: float
        Concentration parameter of the Perturbed Culture Model.
    rng: numpy.random.Generator
        Random number generator.

    Returns
    -------
    tuple
        Plurality scores, weighted majority matrix.

    Examples
    --------
        >>> n_v = 1000
        >>> n_c = 3
        >>> theta = 0.5
        >>> rng = np.random.default_rng(seed=42)
        >>> plurality_scores, wmm = _compute_plurality_scores_and_wmm_looping_on_voters(n_v, n_c, theta, rng)
        >>> plurality_scores
        array([651, 168, 181])
        >>> wmm
        array([[  0, 753, 735],
               [247,   0, 720],
               [265, 280,   0]])
    """
    # Numbers of Dirac voters and IC voters + Borda of IC voters
    n_dirac_voters = rng.binomial(n_v, theta)
    n_ic_voters = n_v - n_dirac_voters
    if n_ic_voters == 0:
        borda_scores_ic = np.zeros((0, n_c), dtype=int)
    else:
        borda_scores_ic = np.array([rng.permutation(n_c) for _ in range(n_ic_voters)])

    # Plurality scores
    plurality_scores = np.sum(borda_scores_ic == (n_c - 1), axis=0)
    plurality_scores[0] += n_dirac_voters

    # Weighted majority matrix
    weighted_majority_matrix = (
        np.sum(borda_scores_ic[:, :, np.newaxis] > borda_scores_ic[:, np.newaxis, :], axis = 0)
        + upper_triangular(n_c) * n_dirac_voters
    )

    return plurality_scores, weighted_majority_matrix


def _threshold_use_multinomial(n_c):
    """
    Threshold used to choose the subroutine for `compute_plurality_scores_and_wmm`.

    For values of `n_c` in the keys of `d_n_c_threshold_use_multinomial`, the threshold were optimized empirically
    to choose the fastest option. For higher values of `n_c`, this is an estimate order of magnitude.

    Parameters
    ----------
    n_c: int
        Number of candidates.

    Returns
    -------
    int
        Threshold for the number of voters.

    Examples
    --------
        >>> _threshold_use_multinomial(4)
        16
        >>> _threshold_use_multinomial(8)
        16128
    """
    try:
        return d_n_c_threshold_use_multinomial[n_c]
    except KeyError:
        d_n_c_threshold_use_multinomial[n_c] = int(0.4 * math.factorial(n_c))
        return d_n_c_threshold_use_multinomial[n_c]


def compute_plurality_scores_and_wmm(n_v, n_c, theta, rng):
    """Draw a random profile and return plurality scores and weighted majority matrix.

    Parameters
    ----------
    n_v: int
        Number of voters.
    n_c: int
        Number of candidates.
    theta: float
        Concentration parameter of the Perturbed Culture Model.
    rng: numpy.random.Generator
        Random number generator.

    Returns
    -------
    tuple
        Plurality scores, weighted majority matrix.

    Examples
    --------
        >>> n_c = 4
        >>> theta = 0.5
        >>> rng = np.random.default_rng(seed=42)

    If n_v is small, we loop on voters:

        >>> n_v = 10
        >>> plurality_scores, wmm = compute_plurality_scores_and_wmm(n_v, n_c, theta, rng)
        >>> plurality_scores
        array([7, 0, 2, 1])
        >>> wmm
        array([[0, 9, 8, 8],
               [1, 0, 6, 7],
               [2, 4, 0, 8],
               [2, 3, 2, 0]])

    If n_v is large, we use the multinomial distribution:

        >>> n_v = 100
        >>> plurality_scores, wmm = compute_plurality_scores_and_wmm(n_v, n_c, theta, rng)
        >>> plurality_scores
        array([62, 12, 16, 10])
        >>> wmm
        array([[ 0, 72, 71, 76],
               [28,  0, 68, 74],
               [29, 32,  0, 73],
               [24, 26, 27,  0]])
    """
    if n_v >= _threshold_use_multinomial(n_c):
        return _compute_plurality_scores_and_wmm_using_multinomial(n_v, n_c, theta, rng)
    else:
        return _compute_plurality_scores_and_wmm_looping_on_voters(n_v, n_c, theta, rng)


def plurality_is_profile_cm(plurality_scores, wmm, n_c):
    """Check if Plurality is coalitionally manipulable.

    Parameters
    ----------
    plurality_scores: np.ndarray
        Plurality scores.
    wmm: np.ndarray
        Weighted majority matrix.
    n_c: int
        Number of candidates.

    Returns
    -------
    bool
        True if the profile is CM, False otherwise.

    Examples
    --------
        >>> n_v = 1000
        >>> n_c = 3
        >>> theta = 0.5
        >>> rng = np.random.default_rng(seed=42)
        >>> plurality_scores, wmm = compute_plurality_scores_and_wmm(n_v, n_c, theta, rng)
        >>> plurality_scores
        array([662, 163, 175])
        >>> wmm
        array([[  0, 756, 743],
               [244,   0, 756],
               [257, 244,   0]])
        >>> plurality_is_profile_cm(plurality_scores, wmm, n_c)
        False
    """
    winner = np.argmax(plurality_scores)
    score_winner = plurality_scores[winner]
    for c in range(winner):
        if wmm[c, winner] >= score_winner:
            return True
    for c in range(winner + 1, n_c):
        if wmm[c, winner] > score_winner:
            return True
    return False
