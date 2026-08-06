# Week 5 report

On Thursday I completed the first peer review. Also, as advised in the testing guidance meeting, I implemented a method for visualizing the computed eigenfaces.
Now, after running the program, it shows a grid of the eigenfaces and the mean face. They all look correct. Picture is in the documentation directory.

## Problems and challanges

When it came to actually testing the full model, I got 85% accuracy. HOWEVER, the Jacobi eigendecomposition did not fully converge within the max iterations.
First I tried with 1000, then 20k, then 40k, and it still would not converge. I researched the topic more and decided to switch from the classic method of finding the largest off-diagonal element
each time (O(n^2) each) to a cyclic sweep method, where every off-diagonal pair (p, q) is rotated once per sweep. I ran the model again and it finally converged fully after about 4 min runtime with accuracy 85% again.

## Next week

Next week I want to polish the code a bit more. I also need to start writing the implementation document and write all the necessary tests for the new scripts I've written and update the documentation.
I also want to implement a small document (for myself, but also for peers) explaining all the math behind the approach in more detail.
