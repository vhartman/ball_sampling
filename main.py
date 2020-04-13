import numpy as np
import matplotlib.pyplot as plt

from randomgen import Generator, Xoshiro256, Xoroshiro128
rg = Generator(Xoshiro256())
#rg = Generator(Xoroshiro128())

import time

import timeit
import cProfile

cols = plt.rcParams['axes.prop_cycle'].by_key()['color'] 

def rejection_sampling(d, n=1):
    assert d <= 5 # while possible to do it longer, it sucks after

    # batching: create a lot of random variables at onece, check them one after another
    # add the ones that fulfill the condition to the array
    while True:
        r = np.random.rand(d) * 2 - 1

        if np.linalg.norm(r) < 1:
            break

    return r

def polar_radial_sampling(d, n=1):
    assert d == 2 or d == 3

    if d == 2:
        u = np.random.rand()
        v = np.random.rand()

        r = u**0.5
        theta = 2* np.pi *v
        x = r*np.cos(theta)
        y = r*np.sin(theta)

        return np.array([x, y])
    elif d == 3:
        u = np.random.rand()*2 - 1
        phi = 2* np.pi * np.random.rand()
        r = np.random.rand()**(1/3.)

        z = r *u
        x = r * np.cos(phi) * (1-z**2)**0.5
        y = r * np.sin(phi)* (1-z**2)**0.5

        return np.array([x, y, z])

def concentric_map(d, n=1):
    assert d == 2

    u = np.random.rand()
    v = np.random.rand()

    if u==0 and v==0:
        return np.array([0, 0])
    
    theta=0
    r = 1
    a=2*u-1
    b=2*v-1
    if a**2 > b**2:
        r = a
        phi = np.pi/4 * b / a
    else:
        r = b
        phi = np.pi/2 - np.pi/4 * a / b

    x = r*np.cos(theta)
    y = r*np.sin(theta)
    
    return np.array([x, y])

def normalized_gaussian(d, n=1, rnd=np.random, dt=None): # muller
    if n == 1:
        u = rnd.randn(d)
        norm = np.linalg.norm(u)

        r = rnd.rand()**(1/d)
        return u*r / norm
    else:
        u = rnd.randn(d, n)
        norm = np.linalg.norm(u, axis=0)

        r = rnd.rand(n)**(1/d)
        return u*r / norm

def exponential_distribution(d, n=1, rnd=np.random, dt=None):
    if n == 1:
        u = rnd.randn(d) 
        e = rnd.exponential(0.5)

        denom = (np.sum(u**2) + e)**0.5
        return u / denom
    else:
        u = rnd.randn(d, n) 
        e = rnd.exponential(0.5, size=n)

        denom = (np.sum(u**2, axis=0) + e)**0.5
        return u / denom

def dropped_coordinates(d, n=1, rnd=np.random, dt=None):
    if dt is None:
        u = rnd.randn(d+2, n)
    else:
        u = rnd.randn(d+2, n, dtype=dt)

    norm = np.linalg.norm(u, axis=0)
    u = u/norm
    s = u[0:d, :] #take the first d coordinates
    return s

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

    
    def rnd_float_dropped_coordinates(d, rnd=rg, dt='f'):
        return dropped_coordinates(d, rnd=rnd, dt=dt)

    dtype_rnd_batch_methods = [
        rnd_float_dropped_coordinates,
        rnd_dropped_coordinates,
    ]

    methods = batch_size_methods
    #methods = batch_methods
    #methods = rnd_batch_methods
    #methods = dtype_rnd_batch_methods
    #methods = rnd_normal_methods

    N = 100000
    #dimensions = np.arange(2, 11)
    dimensions = [2, 10, 50, 100, 1000, 10000]

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

    def draw_stuff(ax, x, data, label, **args):
        def plt_quantile(q):
            d = []
            for da in data:
                d.append(np.quantile(da, q))

            #ax.plot(x, d,'--', color='tab:blue', alpha=.5)
            return d

        min_ = plt_quantile(.05)
        quart = plt_quantile(.25)
        avg = plt_quantile(.5)
        quart2 = plt_quantile(.75)
        max_ = plt_quantile(.95)

        ax.plot(x, avg, alpha=1, label=label, **args)
        ax.fill_between(x, quart, quart2, alpha=.5, **args)
        ax.fill_between(x, min_, max_, alpha=.3, **args)

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

    fig = plt.figure(figsize=(12, 6))
    #ax = fig.add_subplot(1,1,1)
    #for i, method in enumerate(methods):
    for i in range(len(dimensions)):
        ax = fig.add_subplot(2,3,i+1)
        d = [data[method][i] for method in methods if len(data[method]) > i]
        c = cols[0]

    #    d = data[method]
    #    c = cols[i]
        ax.boxplot(d, sym='',
            boxprops=dict(color=c),
            capprops=dict(color=c),
            whiskerprops=dict(color=c),
            flierprops=dict(color=c, markeredgecolor=c),
            medianprops=dict(color=c))
        ax.plot([0, 0], [0, 0], color=c, label=method.__name__)
        #draw_stuff(ax, np.arange(2, len(data[method])+2), data[method], method.__name__, color=cols[i])

        if i+1 == 1 or i+1 == 4:
            ax.set_ylabel('time [us]')
        if i+1 > 3:
            ax.set_xlabel("batch size")
        ax.set_xticklabels(["%.E" % 10**i for i in range(0, 6)])
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=8)

        ax.set_yscale('log')

        ax.set_title("n="+str(dimensions[i]))

    #ax.plot(np.arange(1, len(dimensions)+1), baseline["rnd"], '--', color='darkgrey', label='baseline n-cube')
    #ax.plot(np.arange(1, len(dimensions)+1), baseline["normal"], ':', color='darkgrey', label='baseline n-normal')

    #ax.set_xticklabels(dimensions)
    #ax.set_xlabel('n')
    #ax.set_ylabel('time [us]')

    ax.set_yscale('log')
    #ax.legend()

def profile():
    #method = polar_radial_sampling
    def rnd_float_dropped_coordinates(d, n=1, rnd=rg, dt='f'):
        return dropped_coordinates(d, n=n, rnd=rnd, dt=dt)
    method = rnd_float_dropped_coordinates
    #method = dropped_coordinates
    #method = rnd_batch_dropped_coordinates
    #method = batch_exponential_distribution
    #method = dropped_coordinates

    def test(a, b):
        for _ in range(100000):
            method(a, n=b)
    
    cProfile.runctx('test(100, 1)', globals(), locals(), sort='tottime')#, filename="dropped_coord")

if __name__ == "__main__":
    #profile()
    #exit()

    run()
    plt.savefig('initial.png',
                bbox_inches='tight',
                transparent=True)
    plt.show()
