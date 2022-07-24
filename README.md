# Disco Party

A collection of libraries and scripts to take Disco Diffusion (or potentially other notebooks) to the next level.
These libraries are meant to be used in conjunction with a system which allows you to evaluate a function for the weights, camera parameters, etc.
Disco Diffusion currently only allows static weights on prompts, and static series for camera parameters.
I have modified DD to also support evaluation functions for these. I am currently whispering and colluding with Gandamu to have this officially supported in DD >:)


# pnodes.py

This script offers a new system to baking prompts and weights using a network of composable nodes.
This allows prompts to drift in and out on separate tracks, and was originally made to replace the scene system in PyTTI.
Several node types allow for various methods of sequencing.

A demo of this used in PyTTI can be seen in [Inflorescent Perceptions](https://www.youtube.com/watch?v=7FHZFIaeP4s).


# maths.py

Offers various math functions to use for creative coding.
