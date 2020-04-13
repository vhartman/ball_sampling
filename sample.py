import numpy as np

from randomgen import Generator, Xoshiro256, Xoroshiro128
rg = Generator(Xoshiro256())
#rg = Generator(Xoroshiro128())

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
