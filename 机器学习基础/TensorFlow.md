## TensorFlow

机器学习主要分为训练和部署两个步骤。

在训练阶段，TensorFlow不仅支持Python，更提供对Swift语言和JS语言的支持，你可以选择你熟悉的语言来进行开发。

在部署阶段，TensorFlow模型可以跑在不同的平台，支持服务器端部署的TensorFlow Serving， 支持Android，iOS和嵌入式设备等端侧平台部署的TensorFlow Lite，支持浏览器和Node 服务器部署的TensorFlow.js，以及包括C语言，Java 语言，Go语言，C#语言，Rust和R等多种语言。

使用Keras高层API。Keras 是一个用于构建和训练深度学习模型的高阶 API，可用于快速设计原型、研究和生产环境使用。它具有易使用，模块化，可组合以及易于扩展等优点。Keras 是 TensorFlow 2.0 主要推荐的 API。

tf.sigmoid:  sigmoid算法实现

tf.nn.softmax:  softmax算法实现

tf.squared_difference：差平方

tf.train.GradientDescentOptimizer：梯度下降算法优化器

tf.nn.relu：relu算法

tf.tanh：双曲正切函数

tf.nn.conv2d：卷积层

tf.nn.max_pool：池化层

前面提到tensorflow的编程思维是训练算法一步一步逼达到精确，训练的核心就是“损失函数”和“优化器”，其中“损失函数”需要我们自己去实现，“优化器”只需要调用tensorflow提供的即可，如果是大神级别的，估计可以自己实现优化器。
链接：https://zhuanlan.zhihu.com/p/33801947

下面介绍一下tensorflow的关键概念以及之间的联系：

1）输入：指训练数据集中的输入数据，tensorflow的数据对象是张量，因此输入数据也是张量；

2）结果：指训练数据集中和输入数据对应的结果数据，输入数据和结果数据的用途不同，下面会详细介绍，这两个数据都是我们要**提前准备**的。

3）算法（图中的“算法定义”）：算法是需要我们自己根据业务来定义的，算法依赖两个常见的tensorflow概念，一个叫占位符，一个叫变量；算法可以是传统的机器学习算法，也可以是深度学习算法。

4）占位符：为何要多出占位符这个概念，我开始也理解了半天，其实这里的占位符就是和传统编程的函数参数类似，在定义函数的时候我们用参数来代表输入数据，在定义算法或者损失函数的时候我们用占位符来代表训练数据集中的数据，等到真正运行的时候再将训练数据替换占位符。

5）变量：tensorflow的变量和传统编程里面的变量用途差异很大，导致我理解花费了较长时间，后来才明白变量是tensorflow的核心概念，所谓的tensorflow训练，其实就是通过优化器来调整变量的值，使得算法更加精确；

6）损失函数（图中的“损失函数”）：损失函数是tensorflow训练的核心，tensorflow通过优化器调整变量的值，使得算法更加精确，如何判断算法更加精确呢？其实就是通过损失函数来判断，损失函数输出值越小，算法就更加精确。

损失函数其实就是将算法结果（图中“算法结果”）和训练结果（图中的“结果”）进行对比，同算法一样，损失函数在定义的时候也用到了占位符，这个占位符代表的是训练数据集的结果数据。

看了吴恩达的课程，里面介绍算法的时候用了cost function（成本函数），我简单研究了一下，这个函数和损失函数是相关的，简单来说：**损失函数是针对单个训练样本的，成本函数是针对所有训练样本的均值**，我们只需要定义损失函数,更多请参考：[机器学习中的目标函数、损失函数、代价函数有什么区别？](https://www.zhihu.com/question/52398145)

### variable_scope

`with tf.variable_scope(*self*.scope):`

获得参数数量. 

```python
	def get_num_weights(self):
		with tf.variable_scope(self.scope):
			total_parameters = 0
			for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope):
                # 在创建图的过程中，TensorFlow的Python底层会自动用一些collection对op进行归类，方便之后的调用。这部分collection的名字被称为tf.GraphKeys，可以用来获取不同类型的op
				shape = variable.get_shape()
				variable_parameters = 1
				for dim in shape:
					variable_parameters *= dim.value
				total_parameters += variable_parameters
			return total_parameters
```

torch也可以输出模型的参数

```python
	for name,param in policy_net.named_parameters():
		logger.info(f"name: {name}, param: {param}")
```

### placeholder()

 Tensorflow的设计理念称之为计算流图，在编写程序时，首先构筑整个系统的graph，代码并不会直接生效，这一点和python的其他数值计算库（如Numpy等）不同，graph为静态的，类似于docker中的镜像。然后，在实际的运行时，启动一个session，程序才会真正的运行。这样做的好处就是：避免反复地切换底层程序实际运行的上下文，tensorflow帮你优化整个系统的代码。我们知道，很多python程序的底层为C语言或者其他语言，执行一行脚本，就要切换一次，是有成本的，tensorflow通过计算流图的方式，帮你优化整个session需要执行的代码，还是很有优势的。

所以placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存。等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。

#### tf.train.XXXOptimizer

`apply_gradients`和`compute_gradients`是所有的优化器都有的方法。

 \#执行对应变量的更新梯度操作 training_op = optimizer.apply_gradients(capped_gvs)

### sess.run

当我们构建完图后，需要在一个会话中启动图，启动的第一步是创建一个Session对象。

为了取回（Fetch）操作的输出内容，可以在使用Session对象的run()调用执行图时，传入一些tensor，这些tensor会帮助你取回结果。

在python语言中，返回的tensor是numpy ndarray对象。

在执行sess.run()时，tensorflow并不是计算了整个图，只是计算了与想要fetch的值相关的部分。

```python
return self.sess.run([self.output, self.loss], feed_dict={self.input:input, self.label:label})
```



### 创建网络

