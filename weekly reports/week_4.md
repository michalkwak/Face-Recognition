# Week 4 report

On Tuesday I implemented eigenfaces.py which finally extracts eigenfaces from a set of training images and projects images to eigenspace. (4h)
On Thursday I implemented the classifier, which uses Euclidian distance to return the closest neighbor/matching image. I alos started working on a command line interface. (2h)
On Friday I finished the CLI. (4-5h)
The model now runs and produces results.

## Problems and challanges

When it came to actually testing the full model, I got 85% accuracy. HOWEVER, the Jacobi eigendecomposition did not fully converge within the max iterations.
First I tried with 1000, then 20k, then 40k, and it still would not converge. I researched the topic more and decided to switch from the classic method of finding the largest off-diagonal element
each time (O(n^2) each) to a cyclic sweep method, where every off-diagonal pair (p, q) is rotated once per sweep. I ran the model again and it finally converged fully after about 4 min runtime with accuracy 85% again.

## Next week

Next week I want to polish the code a bit more. I also need to start writing the implementation document and write all the necessary tests for the new scripts I've written and update the documentation.
I also want to implement a small document (for myself, but also for peers) explaining all the math behind the approach in more detail.
