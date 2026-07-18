# Week 2 report

I started working on the core on Thursday. I chose the Olivetti faces dataset from scikit-learn. I researched the Jacobian eigenvalue decomposition algorithm and implemented the outline of necessary methods. I  found a reference implementation on GitHub ([text](https://github.com/i-djurdjevic/jacobi-eigenvalue-algorithm/blob/master/jacobi_eigenvalue_algorithm.py)) using the largest-element variant. I'm going to rewrite it from scratch in my own structure. I wrote the max_off_diagonal() method which returns the largest off-diagonal element of matrix A. I also created a rough directory hierarchy for my project. (2.5h)

On Friday I digged into testing. I read all the course pages on testing and learned how to implement unit tests. I installed and initialized Poetry. I implemented the first two unit tests for the max_off_diagonal() method I created on thursday. (3h)

On Saturday I worked more on the Jacobian algorithm and tried to udnerstand the source github repo in more detail. (1.5h)

## Problems and challanges

I've had some challanges with installing and initializing Poetry. However, I did manage to get it working. I'm also still trying to fully understand the math behind the Jacobi algorithm.

## Next week

Next week I plan to finish implementing the jacobi algorithm and write more tests. I want to start tracking the test coverage more rigorously as well.
