8.5 -8.11

## Summary:

### Progress:

 I summarize some top conference papers in recent years. 

#### What are the problems in past method?

Mirhoseini et al., 2017 proposed to solve the device placement problem using a reinforcement learning approach, based on the policy gradient method.  Unfortunately, the standard policy gradient method is known to be inefficient, as it performs one gradient update for each data sample. With the vanilla policy gradient method, it took 27 hours over a cluster of 160 workers to find a placement that outperforms an existing heuristic. Such training costs are prohibitive and hardly acceptable by machine learning practitioners. 

Spotlight 2018 can improve the reinforcement learning of the policy gradient, the modeling is faster than Google's solution, and the state transition model is more stable than Google's DRL. But this cannot optimize co-located multiple jobs.

#### what is the newest heterogeneity placement?

Narayanan et al., 2020 proposed Gavel, a heterogeneity-aware cluster scheduler that is able to optimize makespan.

How to use it? Perhaps this can be used as a baseline or to allow the device placement network to converge to the specified strategy first.







