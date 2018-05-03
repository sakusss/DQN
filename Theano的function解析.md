unction是theano框架中极其重要的一个函数，另外一个很重要的函数是scan，在学习theano框架中deep learning的教程的时候，几乎所有的实例程序都用到了function和scan，深受这两个函数的折磨，在这里把我理解的记录下来，方便以后使用。
function函数里面最典型的4个参数就是inputs,outputs,updates和givens，下面一次谈一下我对这4个参数的理解。 function是一个由inputs计算outputs的对象，它关于怎么计算的定义一般在outputs里面，这里outputs一般会是一个符号表达式。 
- inputs:输入是一个python的列表list，里面存放的是将要传递给outputs的参数，这里inputs不用是共享变量shared variables.

outputs: 输出是一个存放变量的列表list或者字典dict，如果是字典的话，keys必须是字符串。这里代码段2中的outputs是cost，可以把它看成是一个损失函数值，它由输入inputs，updates以后的shared_variable的值和givens的值共同计算得到。在这里inputs只是在采取minibatch算法时准备抽取的样本集的索引，根据这个索引得到givens数据，是模型的输入变量即输入样本集，而updates中的shared_variable是模型的参数，所以最后由模型的输入和模型参数得到了模型输出就是cost。

updates: 这里的updates存放的是一组可迭代更新的量，是(shared_variable, new_expression)的形式，对其中的shared_variable输入用new_expression表达式更新，而这个形式可以是列表，元组或者有序字典，这几乎是整个算法的关键也就是梯度下降算法的关键实现的地方。 看示例代码段1中updates是怎么来的，cost最后计算出来的可以看作是损失函数，是关于所有模型参数的一个函数，其中的模型参数是self.params，所以gparams是求cost关于所有模型参数的偏导数，其中模型参数params存放在一个列表里面，所有偏导数gparams也存放在一个列表里面，然后用来一个for循环，每次从两个列表里面各取一个量，则是一个模型参数和这个参数之于cost的偏导数，然后把它们存放在updates字典里面，字典的关键字就是一个param，这里一开始声明的所有params都是shared_variable，对应的这个关键字的值就是这个参数的梯度更新，即param-gparam*lr,其实这里的param-gparam*lr就是new_expression，所以这个updates的字典就构成了一对(shared_variable, new_expression)的形式。所以这里updates其实也是每次调用function都会执行一次，则所有的shared_variable都会根据new_expression更新一次值。

givens：这里存放的也是一个可迭代量，可以是列表，元组或者字典，即每次调用function，givens的量都会迭代变化，但是比如上面的示例代码，是一个字典，不论值是否变化，都是x，字典的关键字是不变的，这个x值也是和input一样，传递到outputs的表达式里面的，用于最后计算结果。所以其实givens可以用inputs参数来代替，但是givens的效率更高。
'''
cost = T.mean(self.free_energy(self.input)) - T.mean(
            self.free_energy(chain_end))
gparams = T.grad(cost, self.params, consider_constant=[chain_end])
for gparam, param in zip(gparams, self.params):
     updates[param] = param - gparam * T.cast(
          lr,
          dtype=theano.config.floatX
     )
     '''
'''
train_rbm = theano.function(
        [index], # inputs
        cost,    # outputs
        updates=updates,
        givens={
            x: train_set_x[index×batch_size: (index + 1)×batch_size]
        },
        name='train_rbm'
    )
    '''
    
