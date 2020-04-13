import numpy as np
import matplotlib.pyplot as plt

from sample import *

import time

import timeit
import cProfile

cols = plt.rcParams['axes.prop_cycle'].by_key()['color'] 


def aggregate_statistics(method, dim, n=1000):
    samples = []
    times = []

    num_samples = 0
    while num_samples < n:
        start = time.time()
        s = method(dim)
        end = time.time()

        times.append((end - start)*1000000)
        samples.append(s)
        if len(s.shape) > 1:
            num_samples += s.shape[1]
            times[-1] /= s.shape[1]
        else:
            num_samples += 1

    return samples, times

### for naming in the plots
def batch_normalized_gaussian(d, n=100):
    return normalized_gaussian(d, n)

def batch_exponential_distribution(d, n=100):
    return exponential_distribution(d, n)

def batch_dropped_coordinates(d, n=100):
    return dropped_coordinates(d, n, rnd=np.random)

def rnd_dropped_coordinates(d, n=1, rnd=rg, dt=None):
    return dropped_coordinates(d, n, rnd, dt=dt)

def rnd_exponential_distribution(d, n=1, rnd=rg, dt=None):
    return exponential_distribution(d, n, rnd, dt=dt)

def rnd_normalized_gaussian(d, n=1, rnd=rg, dt=None):
    return normalized_gaussian(d, n, rnd, dt=dt)

def rnd_batch_dropped_coordinates(d, n=100, rnd=rg, dt=None):
    return dropped_coordinates(d, n, rnd, dt=dt)

def rnd_batch_exponential_distribution(d, n=100, rnd=rg, dt=None):
    return exponential_distribution(d, n, rnd, dt=dt)

def rnd_batch_normalized_gaussian(d, n=100, rnd=rg, dt=None):
    return normalized_gaussian(d, n, rnd, dt=dt)

def rnd_float_dropped_coordinates(d, n=1, rnd=rg, dt='f'):
    return dropped_coordinates(d, n, rnd=rnd, dt=dt)

def run_methods(dimensions, N, methods):
    data = {}
    for method in methods:
        data[method] = []

    for n in dimensions:
        print(n)
        for method in methods:
            try:
                _, times = aggregate_statistics(method, n, N)
                avg = np.mean(times)
                std = np.var(times)**.5
                
                data[method].append(times)

                print("\t{0}: \t {1:04.4f}, {2:04.4f}".format(method.__name__, avg, std))
            except AssertionError:
                pass

            except MemoryError:
                pass
        print()

    return data

def run_baseline(dimensions, N):
    def rnd(d, n=100):
        return np.random.rand(d, n)

    def rnd_normal(d, n=100):
        return np.random.randn(d, n)

    baseline = {}
    baseline["rnd"] = []
    baseline["normal"] = []
    for n in dimensions:
        _, times = aggregate_statistics(rnd, n, N)
        baseline["rnd"].append(np.mean(times))

        _, times = aggregate_statistics(rnd_normal, n, N)
        baseline["normal"].append(np.mean(times))

    return baseline

def plot_batchsize(dimensions, data, methods, baseline=None):
    fig = plt.figure(figsize=(12, 6))
    for i in range(len(dimensions)):
        ax = fig.add_subplot(2,3,i+1)
        d = [data[method][i] for method in methods if len(data[method]) > i]
        c = cols[0]

        ax.boxplot(d, sym='',
            boxprops=dict(color=c),
            capprops=dict(color=c),
            whiskerprops=dict(color=c),
            flierprops=dict(color=c, markeredgecolor=c),
            medianprops=dict(color=c))
        ax.plot([0, 0], [0, 0], color=c, label=method.__name__)

        if i+1 == 1 or i+1 == 4:
            ax.set_ylabel('time [us]')
        if i+1 > 3:
            ax.set_xlabel("batch size")

        ax.set_xticklabels(["%.E" % 10**i for i in range(0, 6)])
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=8)

        ax.set_yscale('log')

        ax.set_title("n="+str(dimensions[i]))

    if baseline is not None:
        ax.plot(np.arange(1, len(dimensions)+1), baseline["rnd"], '--', color='darkgrey', label='baseline n-cube')
        ax.plot(np.arange(1, len(dimensions)+1), baseline["normal"], ':', color='darkgrey', label='baseline n-normal')

    ax.set_yscale('log')

def plot(dimensions, data, methods, baseline=None):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1,1,1)
    for i, method in enumerate(methods):
        d = data[method]
        c = cols[i]
        ax.boxplot(d, sym='',
            boxprops=dict(color=c),
            capprops=dict(color=c),
            whiskerprops=dict(color=c),
            flierprops=dict(color=c, markeredgecolor=c),
            medianprops=dict(color=c))
        ax.plot([0, 0], [0, 0], color=c, label=method.__name__)

    if baseline is not None:
        ax.plot(np.arange(1, len(dimensions)+1), baseline["rnd"], '--', color='darkgrey', label='baseline n-cube')
        ax.plot(np.arange(1, len(dimensions)+1), baseline["normal"], ':', color='darkgrey', label='baseline n-normal')

    ax.set_xticklabels(dimensions)
    ax.set_xlabel('n')
    ax.set_ylabel('time [us]')

    ax.set_yscale('log')
    ax.legend()

def run():
    normal_methods = [
        rejection_sampling,
        polar_radial_sampling,
        concentric_map,
        normalized_gaussian,
        exponential_distribution,
        dropped_coordinates,
    ]

    rnd_normal_methods = [
        normalized_gaussian,
        rnd_normalized_gaussian,
        exponential_distribution,
        rnd_exponential_distribution,
        dropped_coordinates,
        rnd_dropped_coordinates,
    ]

    batch_methods = [
        dropped_coordinates,
        exponential_distribution,
        normalized_gaussian,
        batch_normalized_gaussian,
        batch_exponential_distribution,
        batch_dropped_coordinates,
    ]

    batch_size_methods = [
        lambda d: rnd_batch_normalized_gaussian(d, 1),
        lambda d: rnd_batch_normalized_gaussian(d, 10),
        lambda d: rnd_batch_normalized_gaussian(d, 100),
        lambda d: rnd_batch_normalized_gaussian(d, 1000),
        lambda d: rnd_batch_normalized_gaussian(d, 10000),
        lambda d: rnd_batch_normalized_gaussian(d, 100000),
        #lambda d: rnd_batch_normalized_gaussian(d, 1000000),
    ]

    rnd_batch_methods = [
        batch_dropped_coordinates,
        rnd_batch_dropped_coordinates,
        batch_normalized_gaussian,
        rnd_batch_normalized_gaussian,
        batch_exponential_distribution,
        rnd_batch_exponential_distribution,
    ]

    dtype_rnd_batch_methods = [
        rnd_float_dropped_coordinates,
        rnd_dropped_coordinates,
    ]

    #methods = batch_methods
    #methods = rnd_batch_methods
    #methods = dtype_rnd_batch_methods
    methods = rnd_normal_methods

    #methods = batch_size_methods

    N = 1000
    #dimensions = np.arange(2, 11)
    dimensions = [2, 10, 50, 100, 1000, 10000]

    #baseline = run_baseline(dimensions, N)
    baseline = None
    data = run_methods(dimensions, N, methods)

    #plot_batchsize(dimensions, data, methods)
    plot(dimensions, data, methods)

def run_profiler(method):
    # wrapper function to call several times - otherwise the resulution is too small
    def test(a, b):
        for _ in range(100000):
            method(a, n=b)
    
    cProfile.runctx('test(100, 1)', globals(), locals(), sort='tottime')#, filename="dropped_coord")

def profile():
    #method = polar_radial_sampling
    method = rnd_float_dropped_coordinates
    #method = dropped_coordinates
    #method = rnd_batch_dropped_coordinates
    #method = batch_exponential_distribution
    #method = dropped_coordinates

    run_profiler(method)

if __name__ == "__main__":
    #profile()
    run()

    plt.savefig('initial.png',
                bbox_inches='tight',
                transparent=True)
    plt.show()
