实验过程

没有GPU: 可以申请, 用colab. 让学长帮我申请了GPU.

尝试DQN 问题1 : colab没有显示器.

尝试一个DRL 可以, 它也是用策略梯度

服务器上装个x,   用ssh转发一下

ImportError: libcublas.so.10.0: cannot open shared object file: No such file or directory

方法1:  我把代码抄一遍用pytorch复现一遍.

朱junhao先写model, 然后写input ,然后train, 最后test.

### 尝试DL2

1. 尝试 train.py, 失败了. 无法连接 tensorboard, localhost6006 refuse to connect . 在hku服务器上的问题是, python2 有很多不兼容,  python3 没有numpy
2. 然后卸载了安装 tf1.13.1 , 显示**This page isn’t working right now** **e0844decf715** didn’t send any data.
3. There are not any runs in the log folder.

4. pip21.1.2 找不到tensorflow. pip10.0.1 可以找到但是显示没有.   

完成了优化器, checkpoint. 企图分离agent.

为什么训练过程这么长, over 150 LOC.

8.13

他每个文件都有单元测试的. 我测试了一下, 他没有ali trace , 我改成了随机.

https://adaptdl.readthedocs.io/en/latest/

[petuum/adaptdl: Resource-adaptive cluster scheduler for deep learning training. (github.com)](https://github.com/petuum/adaptdl)



adaptdl 是用两层, 可以试试能不能用DRL来做.

复现一下pollux,

table2 ,  99% tile是什么意思? 就是超过99%的



8.14

**怎么做实验? DRL没有env怎么办?** 

但是不是让你用k8s实际部署一个job, 到machine跑完, 测时间, 而是根据trace 模拟出来大概要多久,这样你就可以在本地训练大量数据

最好能做个模型在这些环境跑一跑,咱们就知道你的方案可不可行

simulation就是说你给一个job的placement和他的资源数量 告诉你这个job跑了多久,不是实际在机器跑 而是直接给你一个时间. 就是RL的env

**trace读了什么?**

我下午做一个45分钟就可以知道. 

可以参考它的simulator , 它fitting了现有的model, 这个已经搞懂了. 就是一个函数, 两个自变量 numps和num worker, 一个因变量speed. 用scipy.interpolate.Rbf 拟合.  

8月14日 , 目标搞懂它输入资源放置和输出速度的关系. 给一个job 的资源, 告诉你reward. reward是运行的epoch数/总epoch. 在哪里告诉你的?

get trace  做了什么?

就是生成一些job,给job赋值, job有很多属性

1. batch szie 
2. 带宽
3. speed func(在job.py中用来得到epoch) 
4. 考虑了误差的 总epoch

trace是有100个job list的字典.  一个job list 10个job

为什么要是字典? TS是什么意思? 就是timeslot

好像看完了。 感觉不对， device placement的 模拟做不来？ 因为没有device 的trace ， co-located ， 这个trace不能用？  device placement的 trace 需要哪些参数？ 



然后明天要干嘛

8.15

 8.19日

debug一晚上, 

1.  一个一模一样的代码, 可以执行, 但是comparsion不能执行.   解决方法: 注释掉一些
2. 发现是trace有问题,   可能是pycharm跑, 输出到stdout之外? 终端试一试 ,还是不行 
3. possess =1 可以, 是电脑卡了吗?  也不至于, possess =2也会卡,  而且一模一样的40也没问题. 
4. imoprt trace 就爆炸了.  把trace 类注释,  发现import 就会爆炸 ,其实可以用相对路径来标记文件是不是自己写的. 
5. 递归寻找,  二分寻找.  二分注释,  找到出问题的那一行.  40个子进程要40个构造函数, 在fit中加入print, 确认是否每个进程执行一次,  是执行了40次, 而且会出错,  看看内存, 没有问题
6. 因为没有封装到函数, 所以执行了很多次, pycharm可以试试 wsl. 封装起来就可以正常运行了.

8.20

1. dict() python2 有问题. python3 需要  list( values()) , 因为 values()返回不是pickleable可序列化对象, 是一个view视图对象, 不能修改.
2. 做 , 验证集是怎么用的? 





