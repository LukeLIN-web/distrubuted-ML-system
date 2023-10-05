我们介绍了AuTO，一种用于数据中心的端到端DRL系统，可与商品硬件配合使用。

AuTO是一个两级DRL系统，模拟动物的外周和中枢神经系统。外围系统（PS）在所有终端主机上运行，收集ows信息，并在本地对短期ows做出即时决策。PS的决策由中央系统（CS）通知，中央系统提供全球trac信息进行聚合和处理。CS进一步针对长时间ows做出个人决策，可容忍更长的处理延迟。
AuTO可扩展性的关键是将耗时的DRL处理与快速行动分离，以实现短期ows。

为了实现这一点，我们采用多级反馈排队（MLFQ）[8]让PS在一组阈值的引导下安排ows。每个新流从具有最高优先级的rst队列开始，并在其发送的字节超过某些阈值后逐渐降级到较低的队列。使用MLFQ，AuTO的PS可根据本地信息（发送的字节数和阈值）

4立即做出每流决策，而阈值仍由CS中的DRL算法在相对较长的时间内进行优化。通过这种方式，全局到决策以MLFQ阈值的形式（更具延迟容忍度）传递给PS，使AuTO能够仅使用本地信息对大多数ows进行全局到决策。此外，MLFQ自然地将短ows和长ows分开：短ows在前几个队列中完成，长ows下降到最后一个队列。对于长ows，CS使用不同的DRL算法集中处理它们，以确定路由、速率限制和优先级。

## ABSTRACT

数据中心的流量优化（Traffic optimizations  TO，例如流量调度、负载平衡）是困难的在线决策问题。 以前，它们是通过启发式方法完成的，依赖于操作员对工作负载和环境的理解。 因此，设计和实施适当的 TO 算法至少需要数周时间。 受到最近在应用深度强化学习 (DRL) 技术解决复杂在线控制问题方面取得的成功的鼓舞，我们研究了是否可以将 DRL 用于无需人工干预的自动 TO。 然而，我们的实验表明，当前 DRL 系统的延迟无法在当前数据中心的规模上处理流级 TO，因为短流（构成大部分流量）通常在做出决定之前就消失了。

利用数据中心流量的长尾分布，我们开发了一个两级 DRL 系统 AuTO，模仿动物的外周和中枢神经系统，以解决可扩展性问题。 外围系统 (PS) 驻留在终端主机上，收集流信息，并在本地以最小的短流延迟做出 TO 决策。  PS 的决策由中央系统 (CS) 通知，其中汇总和处理全球交通信息。  CS 进一步为长流做出单独的 TO 决策。 使用 CS&PS，AuTO 是一个端到端的自动 TO 系统，可以收集网络信息，从过去的决策中学习，并执行操作以实现运营商定义的目标。我们使用流行的机器学习框架和商品服务器实现 AuTO，并部署它 在 32 台服务器的测试台上。 与现有方法相比，AuTO 将 TO 周转时间从数周减少到 100 毫秒，同时实现了卓越的For example, it demonstrates up to 48.14% reduction in average flow completion time (FCT) over existing solutions

## 1 introduction

数据中心流量优化（TO，例如流/协流调度 [1, 4, 8, 14, 18, 19, 29, 61]，拥塞控制 [3, 10]，负载平衡和路由 [2]）对应用程序性能有显着影响 . 目前，TO 依赖于手工启发式方法来改变流量负载、流量大小分布、流量集中度等。当参数设置与流量不匹配时，TO 启发式方法可能会受到性能损失。 例如，在 PIAS [8] 中，阈值是根据长期流量大小分布计算的，并且在运行时容易与当前/真实大小分布不匹配。 在不匹配的情况下，性能下降可能高达 38.46% [8]。  pFabric [4] 在使用有限的交换机队列实施时存在同样的问题：在某些情况下，即使仔细优化阈值，平均 FCT 也可以减少 30% 以上。 此外，在协流调度中，Aalo [18] 中的固定阈值取决于操作员预先选择好的值的能力，因为没有运行时自适应。

除了参数环境不匹配之外，设计 TO 启发式算法的周转时间很长——至少数周。 因为它们需要运营商洞察力、应用知识和长期收集的流量统计数据。 一个典型的流程包括：首先，部署一个监控系统来收集终端主机和/或交换机的统计信息； 其次，在收集到足够的数据后，运营商对数据进行分析，设计启发式方法，并使用仿真工具和优化工具对其进行测试，以找到合适的参数设置； 最后，测试启发式被强制执行1（应用程序修改 [19, 61]，操作系统内核模块 [8, 14]，开关配置 [10]，或上述的任何组合）。

因此，自动化 TO 流程很有吸引力，我们需要一种自动化的 TO 代理，它可以适应大量、不确定和易变的数据中心流量，同时实现运营商定义的目标。 在本文中，我们研究了强化学习 (RL) 技术 [55]，因为 RL 是机器学习的子领域，涉及决策制定和行动控制。 它研究代理如何学习在复杂、不确定的环境中实现目标。  RL 代理观察先前的环境状态和奖励，然后决定一个动作以最大化奖励。 近年来，随着深度神经网络 (DNN) 的进步，强化学习在许多困难的环境中取得了良好的结果：DeepMind 的 Atari 结果 [40] 和 AlphaGo [52] 使用了深度强化学习 (DRL) 算法，这些算法对其环境几乎没有假设，因此 可以在其他设置中推广。 受这些结果的启发，我们有动力为自动数据中心 TO 启用 DRL。

我们首先验证了 DRL 在 TO 中的有效性。 我们使用基本的 DRL 算法、策略梯度 [55] 实现了一个流级集中式 TO 系统。 然而，在我们的实验中（第 2.2 节），即使是在当前机器学习软件框架 2 和高级硬件 (GPU) 上运行的这种简单算法也无法处理生产数据中心（> 105 个服务器）规模的流量优化任务。 关键是计算时间（~100 毫秒）：短流（构成大部分流）在 DRL 决策回来之前就消失了，使大多数决策变得无用。

DRL 没用了怎么办? 

计算时间不够, 在机器学习中一般不会计算时间不够, 因为都是训练几个小时?

因此，在本文中，我们试图回答关键问题：如何在数据中心规模启用基于 DRL 的自动 TO？ 为了使 DRL 具有可扩展性，我们首先需要了解数据中心流量的长尾分布 [3, 11, 33]：大多数流是短流3，但大部分字节来自长流。 因此，必须快速生成短流量的 TO 决策； 而长流程的决策更具影响力，因为它们需要更长的时间才能完成。

- 我们展示了 AuTO，这是一种适用于数据中心规模 TO 的端到端 DRL 系统，可与商品硬件配合使用。  AuTO 是一个两级 DRL 系统，模仿动物的外周和中枢神经系统。 外围系统 (PS) 在所有终端主机上运行，收集流量信息，并在本地为短流量做出即时 TO 决策。  PS 的决定由中央系统 (CS) 通知，其中全球交通信息被聚合和处理。  CS 进一步为可以容忍更长处理延迟的长流做出单独的 TO 决策。

AuTO 可扩展性的关键是将耗时的 DRL 处理与短流程的快速行动分开。 为了实现这一点，我们采用多级反馈队列（MLFQ）[8]让 PS 来调度由一组阈值引导的流。 每个新流从优先级最高的第一个队列开始，在其发送的字节超过一定阈值后逐渐降级到较低的队列。 使用 MLFQ，AuTO 的 PS 可以根据本地信息（发送的字节数和阈值）4 立即做出每个流的决策，而阈值仍然通过 CS 中的 DRL 算法在相对较长的时间段内进行优化。 通过这种方式，全局 TO 决策以 MLFQ 阈值（更具延迟容忍性）的形式传递给 PS，使 AuTO 能够为仅具有本地信息的大多数流做出全局知情的 TO 决策。 此外，MLFQ 自然地将短流和长流分开：短流在前几个队列中完成，长流下降到最后一个队列。 对于长流，CS 使用不同的 DRL 算法单独集中处理它们，以确定路由、速率限制和优先级。

我们已经使用 Python 实现了一个 Auto 原型。因此，AuTO 与流行的学习框架兼容，例如 Keras/TensorFlow。 这允许网络和机器学习社区轻松开发和测试新算法，因为 AuTO 中的软件组件可在数据中心的其他 RL 项目中重用。 

我们进一步构建了一个由 2 个交换机连接的 32 个服务器的测试平台来评估 AuTO。 我们的实验表明，对于负载和流量大小分布稳定的流量，经过 8 小时的训练，AuTO 与标准启发式算法（shortest-job-first and least-attained-servicefirst）相比，性能提升高达 48.14%。  AuTO 还表现出稳定学习并适应时间和空间异构流量：仅经过 8 小时的训练，与启发式相比，AuTO 的平均（尾）FCT 减少了 8.71%（9.18%）。 在下文中，我们首先概述了 DRL，并在第 2 节中揭示了当前 DRL 系统无法大规模运行的原因。 我们在第 3 节中描述了系统设计，在第 4 节中描述了 DRL 公式和解决方案。 我们在第 5 节中实现了 AuTO，并在第 6 节中使用真实的测试平台通过大量实验对其进行了评估。 最后，我们在 §7 中回顾相关工作，并在 §8 中总结。

## 2 BACKGROUND AND MOTIVATION

In this section, we first overview the RL background. Then, we describe and apply a basic RL algorithm, policy gradient,

2.1 Deep Reinforcement Learning (DRL) As shown in Figure 1, environment is the surroundings of the agent with which the agent can interact through observations, actions, and feedback (rewards) on actions [55]. Specifically, in each time step t, the agent observes state st , and chooses action at . The state of the environment then transits to st+1, and the agent receives reward rt . The state transitions and rewards are stochastic and Markovian [36]. The objective of learning is to maximize the expected cumulative discounted reward E[P∞ t=0γtrt] where γt∈(0,1] is the discounting factor.

1 深度强化学习 (DRL) 如图 1 所示，环境是代理的周围环境，代理可以通过观察、动作和对动作的反馈（奖励）与之交互 [55]。 具体来说，在每个时间步 t 中，智能体观察状态 st ，并在 选择动作。 然后环境状态转移到 st+1，代理收到奖励 rt 。 状态转换和奖励是随机的stochastic和马尔可夫的 [36]。 学习的目标是最大化预期累积折扣奖励 E[ P∞ t=0γtrt] 其中 γt∈(0,1] 是折扣因子。

RL 代理基于策略采取行动，该策略是在状态 s 中采取行动 a 的概率分布：π(s,a)。 对于大多数实际问题，学习状态-动作对的所有可能组合是不可行infeasible的，因此函数逼近 [31] 技术通常用于学习策略。 函数逼近器 πθ (s,a) 由 θ 参数化，其大小比所有可能的状态-动作对的数量小得多（因此在数学上易于处理）。 函数逼近器可以有多种形式，最近，深度神经网络 (DNN) 已被证明可以解决类似于流调度的实际大规模动态控制问题。 因此，我们也使用 DNN 作为 AuTO 中函数逼近器的表示。

通过函数逼近，代理通过在每个时间段/步骤 t 中更新函数参数 θ 与状态 st 、动作 at 和相应的奖励 rt 来学习。 我们专注于一类通过对策略参数执行梯度下降来学习的更新算法。 学习涉及更新 DNN 的参数（链接权重），以便最大化上述目标。

已知的 REINFORCE 算法 [56]。 该变体使用等式 (1) 的修改版本，它减轻alleviate了算法的缺点：收敛速度和方差。 为了减轻这些缺点，蒙特卡罗方法 [28] 用于计算经验奖励 vt ，并使用基线值（每台服务器经验奖励的累积平均值）来减少方差 [51]。 由于其方差管理和保证收敛到至少局部最小值，因此将结果更新规则（等式（2））应用于策略 DNN [56]

我可以和老师说我做过 value network(actor-critic),Experience replay,   和 policy network 一样,  还是好好读读文献. 

The update which follows (2) ensures that poor flow scheduling decisions are discouraged for similar states in the future, and the good ones become more probable for similar states in the future. When the system converges, the policy achieves a sufficient flow scheduling mechanism for a cluster of servers.



问题: 

即使对于1000fps的小流到达率和只有1个隐藏层，所有实现的处理延迟都超过60ms， 大多数 DRL 操作是无用的，因为当操作到达时，相应的流已经消失了。 

总结 当前 DRL 系统的性能不足以为数据中心规模的流量做出在线决策。 即使对于简单的算法和低流量负载，它们也会遭受长时间的处理延迟。

## 3 AUTO DESIGN

###  3.1 Overview

机器学习系统应该是时间比较久, 几个小时, 几天, 不用考虑时间.

Peripheral Systems 外围系统

外围系统 (PS) 在所有终端主机上运行，收集流信息，并在本地做出 TO 决策，对于短流，延迟最小。 中央系统 (CS) 为可以容忍更长处理延迟的长流做出单独的 TO 决策。 此外，PS 的决策由 CS 通知，在那里聚合和处理全局交通信息。

### 3.2 外围系统

 AuTO 可扩展性的关键是使 PS 能够仅使用本地信息就短流做出全局知情的 TO 决策。  PS有两个模块：执行模块和监控模块。 执行模块为了实现上述目标，我们采用多级反馈队列（MLFQ，在 PIAS [8] 中引入）来调度流，而无需集中的每个流控制。 具体来说，PS在每个终端主机的IP报文的DSCP字段中进行报文标记，如图4所示。有K个优先级，Pi,1≤i≤K，和(K-1)个降级阈值，αj,1≤  j≤K-1。 我们将所有交换机配置为根据 DSCP 字段执行严格的优先级排队。 在终端主机，当一个新的流被初始化时，它的数据包被标记为 P1，在网络中给予它们最高的优先级。 随着发送的字节越多，该流的数据包将被标记为优先级递减的 Pj（2≤j≤K），因此它们在网络中以递减的优先级进行调度。 将优先级从 Pj-1 降级到 Pj 的阈值是 αj-1。 使用 MLFQ，PS 具有以下属性： • 它可以仅基于本地信息：发送的字节数和阈值，做出即时的每个流决策。

它自然地将短流和长流分开。 如图 5 所示，短流在前几个队列中结束，长流下降到最后一个队列。 因此，CS 可以单独集中处理长流，以决定路由、速率限制和优先级。 **监控模块**

 对于CS生成阈值，监控模块收集所有完成的流程的流量大小和完成时间，以便CS可以更新流量大小分布。 监控模块还报告在其终端主机上已经降到最低优先级的正在进行的长流，以便 CS 可以做出单独的决定

3.3 中央系统

 CS由两个DRL代理（RLA）组成：短流RLA（sRLA）用于优化MLFQ的阈值，长流RLA（lRLA）用于确定长流的速率、路由和优先级。  sRLA 试图解决 FCT 最小化问题，为此我们开发了一种深度确定性策略梯度算法。 对于 lRLA，我们使用 PG 算法（第 2.2 节）为长流生成动作。 在下一节中，我们将描述两个 DRL 问题和解决方案。

## 4 DRL 公式和解决方案 

在本节中，我们将描述 CS 中的两种 DRL 算法。
4.1 优化 MLFQ 阈值 我们考虑连接多个服务器的数据中心网络。 通过在主机和网络交换机上使用 K 个严格优先级队列（图 4），通过设置

这里用了DDPG, 好像现在都不用DDPG 了用PPO. PPO 收敛不如 DDPG fast, 不过ppo 比ddpg  more likely to use in practice.

MLFQ 的挑战之一是计算主机上 K 个优先级队列的最佳降级阈值。 先前的工作 [8, 9, 14] 提供了优化降级阈值的数学分析和模型：{α1,α2,...,αK−1}。 白等人。  [9] 还建议每周/每月重新计算阈值与收集的流量级跟踪。  AuTO 更进一步，提出了一种 DRL 方法来优化 α 的值。 与之前在数据中心问题中使用机器学习的工作不同 [5, 36, 60]，AuTO 是独一无二的，因为它的目标是优化连续动作空间中的真实值。 我们将阈值优化问题表述为 DRL 问题，并尝试探索 DNN 对复杂数据中心网络进行建模以计算 MLFQ 阈值的能力。

如§2.2 所示，PG 是一种基本的 DRL 算法。 代理遵循由向量 θ 参数化的策略 πθ (a|s) 并根据经验对其进行改进。 然而，REINFORCE 和其他常规 PG 算法只考虑随机stochastic策略，πθ (a|s)=P[a|s;θ]，根据由 θ 参数化的动作集 A 上的概率分布选择状态 s 中的动作 a。  PG 不能用于值优化问题，因为值优化问题计算的是真实值。 因此，我们应用确定性策略梯度 (DPG) [53] 的变体来逼近给定状态 s 的最佳值 {a0,a1,...,an }，使得 ai=µθ (s) for i=0,..  .,n. 图 6 总结了随机策略和确定性策略之间的主要区别。  DPG 是一种用于确定性策略的 actor-critic [12] 算法，它维护一个参数化的 actor 函数 μθ，用于表示当前策略和一个使用 Bellman 方程更新的 critic 神经网络 Q(s,a)（如 Q-learning [  41]）。 我们用方程 (4,5,6) 描述算法如下：actor 对环境进行采样，并根据方程 (4) 更新其参数 θ。 公式（4）的结果源于策略的目标是最大化预期累积折扣奖励公式（5），其梯度可以表示为以下形式的公式（5）。 更多细节请参考[53]。

深度确定性策略梯度（DDPG）[35]是DPG算法的扩展，它利用了深度学习技术[41]。 我们使用 DDPG 作为优化问题的模型，并在下面解释它是如何工作的。 与 DPG 相同，DDPG 也是一个演员-评论家 [12] 算法，它维护四个 DNN。 两个 DNN，critic QθQ (s,a) 和actor µθµ (s)，权重分别为 θQ 和 θµ，在大小为 N 的采样小批量上进行训练，其中一个item代表一个有经验的转换元组 (si ,ai ,ri ,si+  1）当代理与环境交互时。  DNN 是在随机样本上训练的，这些样本存储在缓冲区中，以避免导致 DNN 发散的相关状态 [41]。 另外两个 DNN，目标演员 µ0 θ 和目标评论家 Q 0 0 θQ (s,a)，

分别用于演员和评论家网络的平滑更新（算法（1）[35]）。 更新步骤稳定了演员-评论家网络的训练，并在连续空间动作 [35] 上取得了最先进的结果。  AuTO 应用 DDPG 来优化阈值以实现更好的流量调度决策。

DRL 公式 接下来，我们展示了阈值的优化可以被公式化为可由 DDPG 解决的 actor-critic DRL 问题。我们首先开发了一个优化问题，即选择一组最佳阈值 {αi} 以最小化流的平均 FCT。 然后我们将这个问题转化为 DRL 问题，用 DDPG 算法解决。 将流量大小分布的累积密度函数表示为 F(x)，因此 F(x) 是流量大小不大于 x 的概率。 令 Li 表示当 i=1,...,K 时，给定流带入队列 Qi 的数据包数量。

状态空间：在我们的模型中，状态表示为当前时间步长内整个网络中所有已完成流的集合 Fd。 每个流由其 5 元组 [8, 38] 标识：源/目标 IP、源/目标端口号和传输协议。 由于我们只报告完成的流，我们还将 FCT 和流大小记录为流属性。 每个流总共有 7 个特征。
动作空间：动作空间由中心化代理 sRLA 计算。 在时间步骤 t，代理提供的动作是一组 MLFQ 阈值 {αt i }。

奖励：奖励是对代理的延迟反馈，说明其在前一个时间步的行为有多好。 我们将奖励建模为两个连续时间步长的目标函数之间的比率

DRL 算法我们使用等式（4）（算法1）指定的更新规则。  DNN 为从主机接收到的每个新状态计算 奖励 rt 和下一个状态 st+1 仅在下一次更新来自同一主机时才知道，因此代理缓存 st 和 at 直到收到所有需要的信息。 以随机批次执行参数更新以稳定学习并降低发散概率 [35, 41]。 奖励 rt 在步骤 t 在主机上计算，并与之前的平均 FCT 进行比较。 根据比较，产生适当的奖励（负面或正面），将其作为信号发送给代理，用于评估 处的动作。 通过遵循算法 1，系统可以改进底层的 actor-critic DNN 并收敛到问题 (7

#### 4.2 优化长流

最后一个阈值 αK-1 通过 sRLA 将长流与短流分开，因此 αK-1 会根据当前的流量特征动态更新，这与之前的短流和长流具有固定阈值的工作不同[1，  22]。 对于长流和 lRLA，我们使用类似于 §2.2 中的流调度问题的 PG 算法，唯一的区别在于动作空间。

Action Space：对于每个活动流f，在时间步t，其对应的动作为{Priot(f),Ratet(f),Patht(f)}，其中Priot(f)为流优先级，Ratet(f)为 速率限制，Patht (f) 是流 f 的路径。 我们假设路径的枚举方式与 XPath [32] 中的相同。 

状态空间：与第 2.2 节相同，状态表示为当前时间步长 t 整个网络中所有活动流的集合 Ft a 和所有已完成流的集合 Ft d。 除了它的 5 元组 [8, 38]，每个活动流还有一个附加属性：它的优先级； 每个完成的流都有两个附加属性：FCT 和流大小。

奖励：奖励是为一组完成的流 Ft d 获得的。 奖励函数的选择可以是：发送速率、链路利用率和连续时间步长的吞吐量的差异或比率。 对于链路速度至少为 10Gbps 的现代数据中心，及时获取活动流的流级信息并不容易。 因此，我们选择仅计算已完成流的奖励，并使用两个连续时间步长的平均吞吐量之间的比率作为reward, capped设定了上限.

## 5 实现

 在本节中，我们将描述实现。 我们在 Python 2.7 中开发了 Auto。 语言选择有助于与现代深度学习框架 [17, 45, 57] 的集成，这些框架提供了出色的 Python 接口 [45]。 当前原型使用 Keras [17] 深度学习库（以 TensorFlow 作为后端）。

### 5.1 外围系统

 PS 是运行在每台服务器上的守护进程。 它有一个监控模块 (MM) 和一个执行enforcement模块 (EM)。  MM 线程收集有关流的信息，包括最近完成的流和当前活动的长流（在 MLFQ 的最后一个队列中）。 在每个周期结束时，MM 汇总收集到的信息，并发送给 CS。  PS 的 EM 线程根据当前活动流的 MLFQ 阈值执行标记，以及长流的路由、速率限制和优先级标记。 我们为 PS 和 CS 之间的通信实现了远程过程调用 (RPC) 接口。  CS 使用 RPC 设置 MLFQ 阈值并对活动的长流执行操作。

我们是不是也可以,  一个监控模块监控 这个job的信息, 比如完成的.  但是其实 deviec和 resource 都是全局的, 我还是没想到怎么搞, pollux也是一个局部一个全局,  我们这两个目前还是得全局分配.

5.1.1 监控模块（MM）：。 为了获得最大效率，MM 可以作为 Linux 内核模块来实现，如 PIAS[8]。 然而，对于当前的原型，由于我们使用流生成器（如 [8, 10, 20] 中所见）来生成工作负载，我们选择直接在流生成器内部实现 MM。 这种选择使我们能够获得基本事实并摆脱可能干扰结果的其他网络流。 

对于长流（MLFQ最后一个队列中的流），每T秒，MM将nl个活跃的长流（每个有6个属性）和ml个完成的长流（每个有7个属性）合并到一个列表中。 对于同一时期的短流（在MLFQ的前几个队列中），MM将ms完成的流（每个有7个属性）收集到一个列表中。 最后，MM 连接两个列表并将它们发送到 CS 作为对环境的观察。

5.1.2 执行模块 (EM)：。  EM 定期从 CS 接收动作。 这些操作包括新的 MLFQ 阈值和对本地长流的 TO 决策。 对于 MLFQ 阈值，EM 建立在 PIAS [8] 内核模块之上，并添加了降级阈值的动态配置。

### 5.2 中央系统 

CS 运行 RL 代理（sRLA 和 lRLA）以做出优化的 TO 决策。 我们实现的 CS 在处理传入更新和向流生成服务器发送操作时遵循类似 SEDA 的架构 [58]。 该架构细分为不同的阶段：http 请求处理、深度网络学习/处理和响应发送。 每个阶段都有自己的进程，并通过队列进行通信以将所需的信息传递给下一个阶段。 这种方法确保了 CS 服务器的多个核心参与处理来自主机的请求并分配负载。 由于 Python 编程语言的 CPython 实现中的全局锁问题 [24]，采用了多处理架构。 状态和动作在 CS 中被封装为一个“环境”（类似于 [47]），RL 代理可以直接和编程地与之交互。

他是online 训练的吗?  离线训练 harmony自己写另一个网络, 但是online 需要收集数据也很耗费资源. 好像是online直接训练的, 没有线下pre train

5.2.2 lRLA. For lRLA, we also use Keras to implement
the PG algorithm with a fully connected NN with 10 hidden layer of300 neurons. The RL agent takes a state (136 features per-server (nl=11, ml=10)) and outputs probabilities for the actions for all the active flows.

Summary The hyper-parameters (structure, number of layer, height, and width of DNN) are chosen based on a few empirical training sessions. Our observation is that more complicated DNNs with more hidden layers and more parameters took longer to train and did not perform much better than the chosen topologies. Overall, we nd that such RLA congurations leads to good system performance and is rather reasonable considering the importance of computation delay, as we reveal next in the evaluation

总结 超参数（DNN 的结构、层数、高度和宽度）是根据一些经验训练session选择的。 我们的观察结果是，具有更多隐藏层和更多参数的更复杂的 DNN 需要更长的训练时间，并且性能并不比所选拓扑好多少。 总体而言，我们发现此类 RLA 配置可带来良好的系统性能，并且考虑到计算延迟的重要性，这是相当合理的，正如我们接下来在评估中所揭示的那样。

## 6 评估

在本节中，我们使用真实的测试台实验评估AuTO 的性能。 我们试图理解：1）在流量稳定（流量大小分布和流量负载固定）的情况下，AuTO 与标准启发式相比如何？  2) 针对不同的交通特征，AuTO 能否适应？  3）AuTO对交通动态的响应速度有多快？  4) 性能开销和整体可扩展性是多少？

 结果摘要（按场景分组）： • 同构：对于具有固定流量大小分布和负载的流量，AuTO 生成的阈值收敛，并且与标准启发式算法相比表现出相似或更好的性能，平均 FCT 减少高达 48.14%。

**比较目标** 我们与流调度中的两种流行启发式方法进行了比较：最短作业优先 (SJF) 和最少实现服务优先 (LAS)。 两者之间的主要区别在于 SJF 方案 [1, 4, 29] 需要在流开始时确定流大小，而 LAS 方案 [8, 14, 43] 不需要。 为了让这些算法工作，在计算它们的参数（阈值）之前应该收集足够的数据。 收集足够流量信息以形成准确可靠的流量大小分布的最短时间是一个开放的研究问题 [9, 14, 21, 34]，我们注意到以前报告的分布都是在至少几周的时间内收集的（图 8)，这表明这些算法的周转时间也至少为数周。

​	CS 响应延迟 在实验过程中，CS 服务器（图 17）的响应延迟测量如下：tu 是 CS 从一个服务器接收更新的时刻，ts 是 CS 向该服务器发送动作的时刻， 所以响应时间是 ts-tu。 该指标直接显示调度程序适应 PS 报告的流量动态的速度。 我们观察到，对于我们的 32 台服务器测试平台，CS 平均可以在 10 毫秒内响应更新。 这种延迟主要是由于 DNN 的计算开销，以及服务器更新在 CS 的排队延迟。  AuTO 目前仅使用 CPU。 为了减少这种延迟，一个有前途的方向是 CPU-GPU 混合训练和服务 [46]，其中 CPU 处理与环境的交互，而 GPU 在后台训练模型。

## 7 相关工作

 数据中心中的 TO 一直在不断努力。 一般来说，探索了三类机制：负载平衡、拥塞控制和流量调度。 我们专注于使用机器学习技术的提案。

 自 1990 年代以来，互联网上的路由和负载平衡就采用了基于 RL 的技术 [13]。 然而，它们是基于交换机的机制，难以在具有 >10 GbE 链路的现代数据中心以线速实施。  RL 技术也用于 Pensieve [37] 中的自适应视频流。

 机器学习技术 [59] 已被用于优化拥塞控制的参数设置。 在给定一组流量分布的情况下，参数是固定的，并且在运行时没有调整参数。 

对于流量调度，CODA [61] 使用无监督聚类算法来识别流量信息，而无需修改应用程序。 然而，它的调度决策仍然是由具有固定参数的启发式算法做出的。 

## 结论

受 DRL 技术最近在解决复杂在线控制问题方面取得的成功启发，在本文中，我们尝试为自动 TO 启用 DRL。 然而，我们的实验表明，当前 DRL 系统的延迟是当前数据中心规模的 TO 的主要障碍。 我们通过利用数据中心流量的长尾分布解决了这个问题。 我们开发了一个两级 DRL 系统 AuTO，模仿动物的外周和中枢神经系统，以解决可扩展性问题。 我们在真实的测试平台上部署和评估了 AuTO，并展示了它的性能和对数据中心动态流量的适应性。  AuTO 是实现数据中心 TO 自动化的第一步，我们希望 AuTO 中的许多软件组件可以在数据中心的其他 DRL 项目中重用。 对于未来的工作，虽然本文侧重于使用 RL 执行流调度和负载平衡，但可以开发用于拥塞控制和任务调度的 RL 算法。 除了我们在 §5&6 中提到的潜在改进之外，我们还计划研究 RL 在数据中心之外的应用，例如 WAN 带宽管理。

RL在各个地方的应用. For future work, while this paper focuses on employing RL to perform how scheduling and load balancing, RL algorithms for congestion control and task scheduling can be developed. In addition to the potential improvements we mentioned in §5&6, we also plan to investigate applications of RL beyond datacenters, such as WAN bandwidth management.
