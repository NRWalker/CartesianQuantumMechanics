Cartesian Quantum System
========================

In essence, this program is a bare-bones single-particle quantum mechanics simulator. The features of the system are built up in a natural progression:

1. Basic cartesian domain - this is the most basic class, which consists of the basic cartesian coordinate system (including boundary conditions)
2. Differentiable domain - this inherits the cartesian coordinates and adds the general finite difference derivative operator
3. Quantum System - this inherits the differentiability and adds in the potential, quantum operators, and the associated features of the mechanics such as time evolutions and mean values of observables

The rest of the files pretty much just implement methods for certain actions that need to be done a lot behind the scenes. For instance, the functions for integration, normalization, generating finite difference schemes, and solving eigenvalue problems are all in extra files. Included in this repository are a bunch of examples, which should hopefully help with understanding how to use the code. A paper and a presentation that go into the details of the mathematics behind this project are also present.


