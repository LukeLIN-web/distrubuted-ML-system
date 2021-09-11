DRL 的基础知识

问题: 

1. 策略梯度算法是怎么写代码的?
2. DRL是怎么写代码的?

强化学习

<A,S,R,P>经典四元 . A 表示 agent的所有动作, state是状态, reward, P也叫model就是agent交互的env. Try and error学习policy.

reward !=goal, goal是累计reward最大. 也就是value最大. 

DRL 是不使用model的, 

采用了DNN网络进行Q值的函数拟合, 就是DQN. 

有过估计问题, 估计的值函数过大, 因为公式有max. 而且过估计量不均匀, 所以可能选不到最优的策略. double Qlearning 企图改进, 把动作的选择和评估用不同的值函数. 

DQN有个主要改进点

Dueling Network把网络分成一个输出标量V(s)另一个输出动作上Advantage值两部分，最后合成Q值

### DRL是怎么写代码的?

就是写一个网络

### 策略梯度算法是怎么写代码的?

就是回传?  就是梯度回传NN 。



Actor-critic

两个网络, actor网络, 预测下一个action可能的分布,  critic给state 估计reward.

a3c

一个master, 多个actor-critic





怎么写

首先定义critic,  这是一个NN.

定义validate , 是检查是否有效.

pointer 类, 可以计算下一个stat