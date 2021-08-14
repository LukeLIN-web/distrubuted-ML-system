8.12-8.18

## Summary:

### Progress:









day by day

8.12

I read paper Pollux.  I find how we could design joint learning.

8.13

I  try get trace, reproduce pollux 



pollux agent:

1. measure:  the time taken for each iteration of training
2.   calculating the gradient noise scale.
3. fits the system throughput model  to the profiled metrics collected 
4.  reports the fitted system throughput parameters, along with the latest gradient statistics, to PolluxSched.
5.  After reporting to PolluxSched, PolluxAgent updates the job’s per-GPU batch size and gradient accumulation steps, by optimizing its now up-to-date goodput function with its currently allocated resources.

pollux scheduler:

input : 

8.14 

I implement the simulator.



