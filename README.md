# Sampling an N-Ball
Full post with more information [here](https://vhartmann.com/ball_sampling/).

This repo contains a few implementations to sample an n-ball uniformly, and an analysis/profiling script which I used to analyze a few of the runs and speed things up.

The original motivation was the need for uniform sampling over a ball in [Informed RRT*](https://arxiv.org/abs/1404.2334).

Main takeaways:
- Batching speeds everything up massively.
- Everything else does not matter that much.
- Optimizing things that take little time is usually not really worth it.
