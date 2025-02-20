import pickle
import re
import time
from math import sqrt
from pathlib import Path
from joblib import Parallel, delayed

import ipynbname
import numpy as np


# Workarounds to make tikzplotlib work despite the deprecation of the package
from matplotlib import pyplot as plt
import matplotlib.backends.backend_pgf
matplotlib.backends.backend_pgf.common_texification = matplotlib.backends.backend_pgf._tex_escape
np.float_ = np.float64
import matplotlib.legend
def get_legend_handles(legend):
    return legend.legend_handles
matplotlib.legend.Legend.legendHandles = property(get_legend_handles)
import webcolors

def integer_rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)

webcolors.CSS3_HEX_TO_NAMES = {integer_rgb_to_hex(webcolors.name_to_rgb(name)): name
                               for name in webcolors.names("css3")}

import tikzplotlib
# End of the workarounds


def tikzplotlib_fix_ncols(obj):
    """Workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib

    Cf. https://stackoverflow.com/questions/75900239/attributeerror-occurs-with-tikzplotlib-when-legend-is-plotted
    """
    if hasattr(obj, "_ncols"):
        # noinspection PyProtectedMember
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


def my_tikzplotlib_save(tikz_file_name, tikz_directory='sav', axis_width=r'\axisWidth', axis_height=r'\axisHeight',
                        invert_legend_order=False):
    """Save a figure in tikz.

    Parameters
    ----------
    tikz_file_name: str or Path
        Name of the tikz file.
    tikz_directory: str or Path
        Directory where to save the tikz file.
    axis_width: str
        Width of the axis in pgfplots.
    axis_height: str
        Height of the axis in pgfplots.
    invert_legend_order: bool
        If True, invert the order of the curves in the legend.
    """
    tikzplotlib_fix_ncols(plt.gcf())
    tikz_directory = Path(tikz_directory)
    if tikz_file_name is None:
        notebook_file_name = ipynbname.name()
        tikz_file_name = f'{notebook_file_name}.tex'
    tikz_directory.mkdir(parents=True, exist_ok=True)
    tikzplotlib.save(tikz_directory / tikz_file_name, axis_width=axis_width, axis_height=axis_height)
    with open(tikz_directory / tikz_file_name, 'r') as f:
        file_data = f.read()
    # Set 'fill opacity' of the legend to 1
    file_data = file_data.replace('fill opacity=0.8,', 'fill opacity=1,')
    # Set the font of the legend
    file_data = file_data.replace('legend style={', r'legend style={font=\legendFont, ')
    # Add yticks as they are in the matplotlib plot
    file_data = file_data.replace(
        'ytick style={',
        'ytick={' + ', '.join([str(y) for y in plt.yticks()[0]]) + '},\n'
        + 'ytick style={'
    )
    # Workaround in case of log scale
    if 'xmode=log' in file_data:
        file_data = re.sub(r'default{10\^{.*?}}', '', file_data)
        file_data = re.sub(r'(?s)xtick={.*?},.*?xticklabels={.*?}', 'xmode=log', file_data)
    if 'ymode=log' in file_data:
        file_data = re.sub(r'default{10\^{.*?}}', '', file_data)
        file_data = re.sub(r'(?s)ytick={.*?},.*?yticklabels={.*?}', 'ymode=log', file_data)
    # Prevent from scaling down the plt.text
    file_data = file_data.replace('scale=0.5,', 'scale=1.0,')
    # Invert legend order if asked
    if invert_legend_order:
        file_data = file_data.replace(r'\begin{axis}[', r'\begin{axis}[reverse legend,')
    with open(tikz_directory / tikz_file_name, 'w') as f:
        f.write(file_data)


def current_time():
    """Return the current time."""
    return time.time()


def elapsed_time(start_time):
    """Return the elapsed time since `t_start`.

    Parameters
    ----------
    start_time: float
        Start time.

    Returns
    -------
    tuple
        First element: elapsed time in seconds. Second element: string representation of the elapsed time.
    """
    duration_seconds = time.time() - start_time
    seconds = duration_seconds % 60
    minutes = (duration_seconds // 60) % 60
    hours = duration_seconds // 3600
    duration_str = f'{hours:.0f}h' if hours > 0 else ''
    duration_str += f'{minutes:.0f}m' if hours > 0 or minutes > 0 else ''
    duration_str += f'{seconds:.0f}s'
    return duration_seconds, duration_str


def my_magic_dump(obj, dmp_file_name=None, dmp_directory='sav'):
    """Dump an object in a file.

    Parameters
    ----------
    obj: any
        Object to dump.
    dmp_file_name: str
        Name of the file where to dump the object. If None, the name of the current notebook is used.
    dmp_directory: str or Path
        Directory where to dump the object.
    """
    dmp_directory = Path(dmp_directory)
    if dmp_file_name is None:
        notebook_file_name = ipynbname.name()
        dmp_file_name = f'{notebook_file_name}.dmp'
    dmp_directory.mkdir(parents=True, exist_ok=True)
    with open(dmp_directory / dmp_file_name, 'wb') as f:
        pickle.dump(obj, f)


def my_magic_load(d, dmp_file_name=None, dmp_directory='sav'):
    """Load an object from a file.

    Parameters
    ----------
    d: dict
        Dictionary where to load the object.
    dmp_file_name: str
        Name of the file where the object is dumped. If None, the name of the current notebook is used.
    dmp_directory: str or Path
        Directory where the object is dumped.

    Notes
    -----
    Using `my_magic_load(locals())` in a cell will update the local variables with the content of the dumped object.
    """
    dmp_directory = Path(dmp_directory)
    if dmp_file_name is None:
        notebook_file_name = ipynbname.name()
        dmp_file_name = f'{notebook_file_name}.dmp'
    with open(dmp_directory / dmp_file_name, 'rb') as f:
        d.update(pickle.load(f))


def compute_cm_rate_of_theta_and_n_v_joblib(compute_cm_rate, n_vs, n_c, thetas, n_samples, n_jobs=-2):
    """Emulate the function `compute_cm_rate_of_theta_and_n_v` using joblib.
    """
    start_time = current_time()
    cm_rates = np.array(
        Parallel(n_jobs=n_jobs)(
            delayed(compute_cm_rate)(n_v, n_c, theta, n_samples)
            for theta in thetas for n_v in n_vs
        )
    ).reshape(len(thetas), len(n_vs))
    run_time, run_time_str = elapsed_time(start_time)
    print(f'{run_time_str=}')
    my_magic_dump(locals())


def compute_cm_rate_of_theta_and_n_v(compute_cm_rate, n_vs, n_c, thetas, n_samples):
    """Compute the CM rate of Plurality for different values of theta and n_v, then dump the results.

    Dump the input variables and the output variables mentioned in the Notes section.

    Parameters
    ----------
    compute_cm_rate: callable
        Function that computes the CM rate. Arguments: `n_v`, `n_c`, `theta`, `n_samples`.
    n_vs: Iterable, Sized
        Values of n_v.
    n_c: int
        Number of candidates.
    thetas: Iterable, Sized
        Values of theta.
    n_samples: int
        Number of samples.

    Notes
    -----
    start_time: float
        Start time.
    cm_rates: np.ndarray
        CM rates.
    run_time: float
        Elapsed time in seconds.
    run_time_str: str
        String representation of the elapsed time.
    """
    start_time = current_time()
    cm_rates = np.zeros((len(thetas), len(n_vs)))
    for i, theta in enumerate(thetas):
        for j, n_v in enumerate(n_vs):
            cm_rates[i, j] = compute_cm_rate(n_v, n_c, theta, n_samples)
    run_time, run_time_str = elapsed_time(start_time)
    print(f'{run_time_str=}')
    my_magic_dump(locals())


def plot_cm_rate_of_theta_and_n_v(cm_rates, n_vs, n_c, thetas,
                                  rule_name, compute_theta_c,
                                  invert_legend_order=False,
                                  tikz_file=None):
    """Plot the CM rate of a rule for different values of theta and n_v, then save the plot in a tikz file.

    Parameters
    ----------
    cm_rates: np.ndarray
        CM rates obtained with `cm_rate_of_theta_and_n_v`.
    n_vs: Iterable, Sized
        Values of n_v.
    n_c: int
        Number of candidates.
    thetas: Iterable, Sized
        Values of theta.
    rule_name: str
        Name of the voting rule.
    compute_theta_c: callable
        Function that computes the critical value of theta. Argument: `n_c`.
    invert_legend_order: bool
        If True, invert the order of the curves in the legend.
    tikz_file: str or Path
        Tikz file where to save the plot.
    """
    theta_c = compute_theta_c(n_c)
    plt.figure()
    for i, n_v in enumerate(n_vs):
        plt.plot(thetas, cm_rates[:, i], label=f'$n = {n_v}$')
    if theta_c == 0:
        plt.plot([0, 1], [0, 0], label=r'$n \rightarrow \infty$')
    else:
        plt.plot([0, theta_c, theta_c, 1], [1, 1, 0, 0], label=r'$n \rightarrow \infty$')
    plt.xlim(min(thetas), max(thetas))
    plt.xlabel(r'$\theta$')
    plt.ylim(-0.05, 1.05)
    plt.ylabel(r'$\rho(\text{' + rule_name + '}, ' + f'{n_c}' + r', n, \theta)$')
    # Invert the order of the curves in the legend
    if invert_legend_order:
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles[::-1], labels[::-1])
    else:
        plt.legend()
    my_tikzplotlib_save(tikz_file_name=tikz_file, invert_legend_order=invert_legend_order)


def plot_cm_rate_of_n_v_and_theta_supercritical(cm_rates, n_vs, n_c, thetas, n_samples,
                                                rule_name, compute_theta_c,
                                                invert_legend_order=False,
                                                tikz_file=None):
    """Plot the CM rate of a rule for different values of n_v and theta > theta_c, then save the plot in a tikz file.

    Parameters
    ----------
    cm_rates: np.ndarray
        CM rates obtained with `cm_rate_of_theta_and_n_v`.
    n_vs: Iterable, Sized
        Values of n_v.
    n_c: int
        Number of candidates.
    thetas: Iterable, Sized
        Values of theta.
    n_samples: int
        Number of samples.
    rule_name: str
        Name of the voting rule.
    compute_theta_c: callable
        Function that computes the critical value of theta. Argument: `n_c`.
    invert_legend_order: bool
        If True, invert the order of the curves in the legend.
    tikz_file: str or Path
        Tikz file where to save the plot.
    """
    theta_c = compute_theta_c(n_c)
    plt.figure()
    for i, theta in enumerate(thetas):
        if theta > theta_c:
            indexes = [j for j, cm_rate in enumerate(cm_rates[i, :]) if cm_rate > 0]
            plt.plot(np.array(n_vs)[indexes], cm_rates[i, indexes], label=r'$\theta = ' + f'{theta}$')
    plt.yscale('log')
    plt.ylim(1 / sqrt(n_samples), 1)
    plt.xlabel(r'$n$')
    plt.ylabel(r'$\rho(\text{' + rule_name + '}, ' + f'{n_c}' + r', n, \theta)$')
    # Invert the order of the curves in the legend
    if invert_legend_order:
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles[::-1], labels[::-1])
    else:
        plt.legend()
    my_tikzplotlib_save(tikz_file_name=tikz_file, invert_legend_order=invert_legend_order)


def plot_ncm_rate_of_n_v_and_theta_subcritical(cm_rates, n_vs, n_c, thetas, n_samples,
                                                rule_name, compute_theta_c,
                                                invert_legend_order=False,
                                                tikz_file=None):
    """Plot the CM rate of a rule for different values of n_v and theta < theta_c, then save the plot in a tikz file.

    Parameters
    ----------
    cm_rates: np.ndarray
        CM rates obtained with `cm_rate_of_theta_and_n_v`.
    n_vs: Iterable, Sized
        Values of n_v.
    n_c: int
        Number of candidates.
    thetas: Iterable, Sized
        Values of theta.
    n_samples: int
        Number of samples.
    rule_name: str
        Name of the voting rule.
    compute_theta_c: callable
        Function that computes the critical value of theta. Argument: `n_c`.
    invert_legend_order: bool
        If True, invert the order of the curves in the legend.
    tikz_file: str or Path
        Tikz file where to save the plot.
    """
    theta_c = compute_theta_c(n_c)
    plt.figure()
    for i, theta in enumerate(thetas):
        if 0 < theta < theta_c:
            indexes = [j for j, cm_rate in enumerate(cm_rates[i, :]) if cm_rate < 1]
            plt.plot(np.array(n_vs)[indexes], 1 - cm_rates[i, indexes], label=r'$\theta = ' + f'{theta}$')
    plt.yscale('log')
    plt.ylim(1 / sqrt(n_samples), 1)
    plt.xlabel(r'$n$')
    plt.ylabel(r'$1 - \rho(\text{' + rule_name + '}, ' + f'{n_c}' + r', n, \theta)$')
    # Invert the order of the curves in the legend
    if invert_legend_order:
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles[::-1], labels[::-1])
    else:
        plt.legend()
    my_tikzplotlib_save(tikz_file_name=tikz_file, invert_legend_order=invert_legend_order)


def compute_cm_rate_at_theta_c_of_n_v_n_c(compute_cm_rate, compute_theta_c, n_vs, n_cs, n_samples):
    """Compute the CM rate at theta_c of a rule for different values of n_v and n_c, then dump the results.

    Dump the input variables and the output variables mentioned in the Notes section.

    Parameters
    ----------
    compute_cm_rate: callable
        Function that computes the CM rate. Arguments: `n_v`, `n_c`, `theta`, `n_samples`.
    compute_theta_c: callable
        Function that computes the critical value of theta. Argument: `n_c`.
    n_vs: Iterable, Sized
        Values of n_v.
    n_cs: Iterable, Sized
        Values of n_c.
    n_samples: int
        Number of samples.

    Notes
    -----
    start_time: float
        Start time.
    cm_rates: np.ndarray
        CM rates.
    theta_c: float
        Critical value of theta.
    run_time: float
        Elapsed time in seconds.
    run_time_str: str
        String representation of the elapsed time.
    """
    start_time = current_time()
    cm_rates = np.zeros((len(n_vs), len(n_cs)))
    for i, n_v in enumerate(n_vs):
        for j, n_c in enumerate(n_cs):
            theta_c = compute_theta_c(n_c)
            cm_rates[i, j] = compute_cm_rate(n_v, n_c, theta_c, n_samples)
    run_time, run_time_str = elapsed_time(start_time)
    print(f'{run_time_str=}')
    my_magic_dump(locals())


def compute_cm_rate_at_theta_c_of_n_v_n_c_joblib(compute_cm_rate, compute_theta_c, n_vs, n_cs, n_samples):
    """Emulate the function `compute_cm_rate_at_theta_c_of_n_v_n_c` using joblib.
    """
    start_time = current_time()
    cm_rates = np.array(
        Parallel(n_jobs=-2)(
            delayed(compute_cm_rate)(n_v, n_c, compute_theta_c(n_c), n_samples)
            for n_v in n_vs for n_c in n_cs
        )
    ).reshape(len(n_vs), len(n_cs))
    run_time, run_time_str = elapsed_time(start_time)
    print(f'{run_time_str=}')
    my_magic_dump(locals())


def plot_cm_rate_at_theta_c_of_n_v_n_c(cm_rates, n_vs, n_cs,
                                       rule_name, invert_legend_order=False,
                                       tikz_file=None):
    """Plot the CM rate at theta_c of a rule for different values of n_v and n_c, then save the plot in a tikz file.

    Parameters
    ----------
    cm_rates: np.ndarray
        CM rates obtained with `cm_rate_at_theta_c_of_n_v_n_c`.
    n_vs: Iterable, Sized
        Values of n_v.
    n_cs: Iterable, Sized
        Values of n_c.
    rule_name: str
        Name of the voting rule.
    invert_legend_order: bool
        If True, invert the order of the curves in the legend.
    tikz_file: str or Path
        Tikz file where to save the plot.
    """
    plt.figure()
    for i, n_c in enumerate(n_cs):
        plt.plot(n_vs, cm_rates[:, i], label=f'$m={n_c}$')
    plt.xscale('log')
    plt.xlabel('n')
    plt.ylabel(r'$\rho(\text{' + rule_name + r'}, m, n, \theta_c(\text{' + rule_name + r'}, m))$')
    plt.ylim(-0.05, 1.05)
    # Invert the order of the curves in the legend
    if invert_legend_order:
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles[::-1], labels[::-1])
    else:
        plt.legend()
    my_tikzplotlib_save(tikz_file_name=tikz_file, invert_legend_order=invert_legend_order)


def plot_slope_of_delta(deltas, minus_slopes, errors_on_slopes,
                        n_c, rule_name,
                        fit_values, fit_str, fit_values_2=None, fit_str_2=None,
                        tikz_file=None):
    """Plot the asymptotic negative slopes as a function of (theta-theta_c), then save the plot in a tikz file.

    Parameters
    ----------
    deltas: np.ndarray
        Values of (theta-theta_c).
    minus_slopes: np.ndarray
        Asymptotic negative slopes.
    errors_on_slopes: np.ndarray
        Errors on the asymptotic negative slopes.
    n_c: int
        Number of candidates.
    rule_name: str
        Name of the voting rule.
    fit_values: np.ndarray
        Values of the fit.
    fit_str: str
        String representation of the fit (used for the legend).
    fit_values_2: np.ndarray
        Values of the second fit.
    fit_str_2: str
        String representation of the second fit (used for the legend).
    tikz_file: str or Path
        Tikz file where to save the plot.
    """
    plt.figure()
    plt.errorbar(deltas, minus_slopes, yerr=errors_on_slopes, fmt='.', label='Measured slope')
    plt.plot(deltas, fit_values, label=fit_str)
    if fit_values_2 is not None:
        plt.plot(deltas, fit_values_2, label=fit_str_2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\theta - \theta_c(\text{' + rule_name + r'}, ' + f'{n_c}' + r')$')
    plt.ylabel(r'Asymptotic negative slope $C(' + f'{n_c}' + r', \theta)$')
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1])
    my_tikzplotlib_save(tikz_file_name=tikz_file, invert_legend_order=True)
