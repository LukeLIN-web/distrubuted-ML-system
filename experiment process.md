实验过程

没有GPU: 可以申请, 用colab. 让学长帮我申请了GPU.

尝试DQN 问题1 : colab没有显示器.

尝试一个DRL 可以, 它也是用策略梯度

ImportError: libcublas.so.10.0: cannot open shared object file: No such file or directory

方法1:  我把代码抄一遍用pytorch复现一遍.

朱junhao先写model, 然后写input ,然后train, 最后test.

### 尝试DL2

1. 尝试 train.py, 失败了. 无法连接 tensorboard, localhost6006 refuse to connect . 在hku服务器上的问题是, python2 有很多不兼容,  python3 没有numpy
2. 然后卸载了安装 tf1.13.1 , 显示**This page isn’t working right now** **e0844decf715** didn’t send any data.
3. There are not any runs in the log folder.



完成了优化器, checkpoint. 企图分离agent.

为什么训练过程这么长, over 150 LOC.

