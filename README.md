# S(P)-NR-O-Quickhull
Sequential and parallel codes for the recursive elimination version of the O-Quickhull algorithm to find the connected orthogonal convex hull of a set of points in the plane.

1. O_Quickhull.jl: It is a sequential algorithm that uses the O-Quickhull algorithm to find the orthogonal convex hull.
2. O_Quickhull_Parallel.jl: It is a parallel version that uses the O-Quickhull algorithm to find the orthogonal convex hull.
3. O_Quickhull_Non-recursive_Parallel_version1.jl: It is an improved version that eliminates recursion and parallelizes the recursive elimination version. This version is used as the result of the paper "Improving and Parallelizing the O-Quickhull Algorithm for Finding the Connected Orthogonal Convex Hull of a Finite Set of Points."
4. O_Quickhull_Non-recursive_Parallel_version2_mask.jl, O_Quickhull_Non-recursive_Parallel_version2.jl, O_Quickhull_Non-recursive_Parallel_version3.jl: These are several other parallel versions of the improved O-Quickhull algorithm.
5. utils.jl: This file generates several types of test data and includes some other preparation functions.
6. main.jl: These files are included in main.jl to run the algorithms.

A "result" directory needs to be created to store the results of the algorithms.
