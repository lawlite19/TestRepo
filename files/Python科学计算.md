Python科学计算
===========
## 0、python自带
### 1、assert断言
- 可以用于测试，若是错误则会出现异常    
  - `assert 1==2,"1==2错误异常"`


## 一、Numpy
### 1、Numpy特征和导入
- （1）用于多维数组的第三方Python包
- （2）更接近于底层和硬件 (高效)
- （3）专注于科学计算 (方便)
- （4）导入包：`import numpy as np`

### 2、list转为数组
- （1）`a = np.array([0,1,2,3])`
- （2）输出为：`[0 1 2 3]`
- （3）数据类型：`<type 'numpy.ndarray'>`

### 3、一维数组
- （1）`a = np.array([1,2,3,4])`属性
`a.ndim`-->维度为1
`a.shape`-->形状，返回`(4,)`
`len(a)`-->长度，4
- （2）访问数组
`a[1:5:2]`下标1-5，下标关系+2
- （3）逆序

    `a[::-1]`

### 4、多维数组
- （1）二维：`a = np.array([[0,1,2,3],[1,2,3,4]])`
输出为：

    [[0 1 2 3]
     [1 2 3 4]]
a.ndm   -->2
a.shape -->(2,4)-->行数，列数
len(a)  -->2-->第一维大小
- （2）三维：`a = np.array([[[0],[1]],[[2],[4]]])`
`a.shape`-->(2,2,1)

### 5、用函数创建数组
- （1）`np.arange()`

    a = np.arange(0, 10)
b = np.arange(10)
c = np.arange(0,10,2)
输出：

    [0 1 2 3 4 5 6 7 8 9] 
[0 1 2 3 4 5 6 7 8 9] 
[0 2 4 6 8]
- （2）`np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)`
等距离产生num个数
- （3）`np.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None)`
以log函数取

### 6、常用数组
- （1）`a = np.ones((3,3))`
输出：

    [[ 1.  1.  1.]
 [ 1.  1.  1.]
 [ 1.  1.  1.]]

- （2）`np.zeros((3,3))`
- （3）`np.eye(2)`单位矩阵
- （4）`np.diag([1,2,3],k=0)`对角矩阵，k为对角线的偏移

### 7、随机数矩阵
- （1）`a = np.random.rand(4)`
输出：`[ 0.99890402  0.41171695  0.40725671  0.42501804]`范围在[0,1]之间
- （2）`a = np.random.randn(4)` Gaussian函数，
- （3）生成100个0-m的随机数:  `[t for t in [np.random.randint(x-x, m) for x in range(100)]]` 
  - 也可以
 ```
 m_arr = np.arange(0,m)      # 生成0-m-1
 np.random.shuffle(m_arr)    # 打乱m_arr顺序
 ```
 然后取前100个即可



### 8、查看数据类型
- （1）`a.dtype`


### 9、数组复制
- （1）共享内存
```
    a = np.array([1,2,3,4,5])
    b = a
    print np.may_share_memory(a,b)
```
输出：True
说明使用的同一个存储区域，修改一个数组同时另外的也会修改
- （2）不共享内存
 `b = a.copy()`

### 10、布尔型
- （1）
```
    a = np.random.random_integers(0,20,5)
    print a
    print a%3==0
    print a[a % 3 == 0]
```
输出：
    [14  3  6 15  4]
    [False  True  True  True False]
    [ 3  6 15]

### 11、中间数、平均值
- （1）中间数`np.median(a)`
- （2）平均值`np.mean(a)`,
  - 若是矩阵，不指定`axis`默认求所有元素的均值
  - `axis=0`,求列的均值
  - `axis=1`，求行的均值

### 12、矩阵操作
- （1）乘积`np.dot(a,b)`
```
    a = np.array([[1,2,3],[2,3,4]])
    b = np.array([[1,2],[2,3],[2,2]])
    print np.dot(a,b)
```
或者使用`np.matrix()`生成矩阵，相乘需要满足矩阵相乘的条件
- （2）内积`np.inner(a,b)`
行相乘

- （3）逆矩阵`np.linalg.inv(a)`
- （4）列的最大值`np.max(a[:,0])`-->返回第一列的最大值
- （5）每列的和`np.sum(a,0)`
- （6）每行的平均数`np.mean(a,1)`
- （7）求交集`p.intersect1d(a,b)`，返回一维数组
- （8）转置：`np.transpose(a)`
- （9）两个矩阵对应对应元素相乘（点乘）：`a*b`

### 13、文件操作
- （1）保存：`tofile()`
```
    a = np.arange(10)
    a.shape=2,5
    a.tofile("test.bin")
```
读取：（需要注意指定保存的数据类型）
```
    a = np.fromfile("test.bin",dtype=np.int32)
    print a
```
- （2）保存：`np.save("test",a)`-->会保存成test.npy文件
读取：`a = np.load("test")`

### 14、组合两个数组
- （1）垂直组合
```
    a = np.array([1,2,3])
    b = np.array([[1,2,3],[4,5,6]])
    
    c = np.vstack((b,a))
```
- （2）水平组合
```
    a = np.array([[1,2],[3,4]])
    b = np.array([[1,2,3],[4,5,6]])
    
    c = np.hstack((a,b))
```

### 15、读声音Wave文件
- （1）`wave`     
```
    import wave
    from matplotlib import pyplot as plt
    import numpy as np
    
    # 打开WAV文档
    f = wave.open(r"c:\WINDOWS\Media\ding.wav", "rb")
    
    # 读取格式信息
    # (nchannels, sampwidth, framerate, nframes, comptype, compname)
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    
    # 读取波形数据
    str_data = f.readframes(nframes)
    f.close()
    
    #将波形数据转换为数组
    wave_data = np.fromstring(str_data, dtype=np.short)
    wave_data.shape = -1, 2
    wave_data = wave_data.T
    time = np.arange(0, nframes) * (1.0 / framerate)
    
    # 绘制波形
    plt.subplot(211) 
    plt.plot(time, wave_data[0])
    plt.subplot(212) 
    plt.plot(time, wave_data[1], c="g")
    plt.xlabel("time (seconds)")
    plt.show()
```

### 16、`where`
- （1）找到y数组中=1的位置：`np.where(y==1)`

### 17、`np.ravel(y)`
- 将二维的转化为一维的，eg:`(5000,1)-->(5000,)`

### 18、ndarray.flat函数
- 将数据展开对应的数组，可以进行访问
- 应用：0/1映射
```
def dense_to_one_hot(label_dense,num_classes):
    num_labels = label_dense.shape[0]
    index_offset = np.arange(num_labels)*num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot  
```

### 19、数组访问
- X = np.array([[1,2],[3,4]])
  - `X[0:1]和X[0:1,:]`等价，都是系那是第一行数据

### 20、`np.c_()`
- 按照第二维度，即列拼接数据
  -  `np.c_[np.array([[1,2,3]]), 0, 0, np.array([[4,5,6]])]`
输出：`array([[1, 2, 3, 0, 0, 4, 5, 6]])`
- 两个列表list拼接，长度要一致
  - `np.c_[[1,2,3],[2,3,4]]`
  - `np.c_[range(1,5),range(2,6)]`

## 二、Matplotlib
### 1、关于`pyplot `
- （1）matplotlib的pyplot子库提供了和matlab类似的绘图API，方便用户快速绘制2D图表。
- （2）导入包：`from matplotlib import pyplot as plt`

### 2、绘图基础
- （1）sin和cos
```
    x = np.linspace(-np.pi, np.pi,256,endpoint=True)
    C,S = np.cos(x),np.sin(x)
    plt.plot(x,C)
    plt.plot(x,S)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
```
- （2）指定绘图的大小

    plt.figure(figsize=(8,6), dpi=80)
- （3）指定线的颜色、粗细和类型

    plt.plot(x,C,color="blue",linewidth=2.0,linestyle="-",label="cos")
    蓝色、宽度、连续、label（使用legend会显示这个label）
- （4）指定x坐标轴范围

    plt.xlim(-4.0,4.0)
- （5）设置y抽刻度间隔
`plt.yticks(np.linspace(-1, 1, 15, endpoint=True))`
- （6）显示图例

    plt.legend(loc="upper left")
    显示在左上方
- （7）一个figure上画多个图subplot方式

    plt.subplot(1, 2, 1)
    plt.subplot(1, 2, 2)
    例如：plt.subplot(m, n, p)
    代表图共有的m行，n列，第p个图
    p是指第几个图，横向数
    上面代表有一行，两个图
  - [更详细解释]：
 231,232,233表示第一行的1,2,3个位置，接着的223表示把整个矩形分成4分，所以第3个位置就是第二行的第一个位置，但是相比第一行占了1.5列
（每次subplot划分都是整个图重新划分）

- （8）一个figure上画多个图，axes方式

    plt.axes([.1, .1, .8, .8])
    plt.axes([.2, .2, .3, .3])
- （9）填充

    plt.fill_between(x, y1, y2=0, where=None, interpolate=False, step=None, 
                hold=None, data=None)

    eg:
```
    plt.fill_between(X, 1, C+1, C+1>1,color="red")
    plt.fill_between(X, 1, C+1, C+1<1,color="blue")
```

### 3、散点图
- （1）

    `plt.scatter(X,Y)`

### 4、条形图
- （1）

    `plt.bar(X, Y, facecolor="red", edgecolor="blue" )`
    填充颜色为facecolor,边界颜色为edgecolor

### 5、等高线图
- （1）只显示等高线`contour`
- （2）显示表面`contourf`
- （3）注意三维图要用到`meshgrid`转化为网格
```    
    def f(x,y):
        return (1 - x / 2 + x**5 + y**3) * np.exp(-x**2 -y**2)
    
    n = 256
    x = np.linspace(-3,3,n)
    y = np.linspace(-3,3,n)
    X,Y = np.meshgrid(x,y)
    
    plt.contourf(X,Y,f(X,Y),alpha=.5)
    C = plt.contour(X,Y,f(X, Y),colors="black",linewidth=.5)
    plt.clabel(C)
    
    plt.show()
```

### 6、显示图片`imshow`
- （1）
```
    def f(x,y):
    return (1 - x / 2 + x ** 5 + y ** 3 ) * np.exp(-x ** 2 - y ** 2)
    n = 10
    x = np.linspace(-3, 3, 3.5 * n)
    y = np.linspace(-3, 3, 3.0 * n)
    X, Y = np.meshgrid(x, y)
    z = f(X,Y)
    plt.imshow(z)
    plt.show()
```

### 7、饼图`pie`
- （1）传入一个序列
```
    plt.figure(figsize=(8,8))
    n = 20
    Z = np.arange(10)
    plt.pie(Z)
    plt.show()
```


### 8、三维表面图*
- （1）需要导入包：`from mpl_toolkits.mplot3d import Axes3D`
- （2）
```
    fig = plt.figure()
    ax = Axes3D(fig)
    X = np.arange(-4, 4, 0.25)
    Y = np.arange(-4, 4, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = np.sin(R)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.hot)
    ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=plt.cm.hot)
    ax.set_zlim(-2, 2)
    plt.show()
```

### 9、legend显示问题
- （1）
```
    p1, = plt.plot(np.ravel(X[pos,0]),np.ravel(X[pos,1]),'ro',markersize=8)
    p2, = plt.plot(np.ravel(X[neg,0]),np.ravel(X[neg,1]),'g^',markersize=8)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend([p1,p2],["y==1","y==0"])
```
**注意** p1后要加上`,`逗号，里面的数据要是**一维**的，使用`np.ravel()`转化一下


### 10、显示网格
- （1）`plt.grid(True, linestyle = "-.", color = "b", linewidth = "1")`  

### 11、显示正方形的坐标区域
- （1）`plt.axis('square')`


## 三、Scipy
### 1、 Scipy特征
- （1）内置了图像处理， 优化，统计等等相关问题的子模块
- （2）scipy 是Python科学计算环境的核心。 它被设计为利用 numpy 数组进行高效的运行。从这个角度来讲，scipy和numpy是密不可分的。

### 2、文件操作`io`
- （1）导包：`from scipy import io as spio`
- （2）保存`mat`格式文件

    `spio.savemat("test.mat", {'a':a})`


- （3）加载`mat`文件

    `data = spio.loadmat("test.mat")`
    访问值：data['a']-->相当于map
- （4）读取图片文件
导包：`from scipy import misc`
读取：`data = misc.imread("123.png")`
[注1]：与matplotlib中`plt.imread('fname.png')`类似
[注2]：执行`misc.imread`时可能提醒不存在这个模块，那就安装`pillow`的包

### 3、线性代数操作`linalg`
- （1）求行列式`det`

    `res = linalg.det(a)`
- （2）求逆矩阵`inv`

    `res = linalg.inv(a)`
    若是矩阵不可逆，则会抛异常`LinAlgError: singular matrix`
- （3）奇异值分解`svd`
    `u,s,v = linalg.svd(a)`
    [注1]：s为a的特征值（一维），降序排列，
    [注2]：a = u\*s\*v'（需要将s转换一下才能相乘）
```
    t = np.diag(s)
    print u.dot(t).dot(v)
```

### 4、梯度下降优化算法
- （1）`fmin_bfgs`
```
    def f(x):
        return x**2-2*x
    initial_x = 0
    optimize.fmin_bfgs(f,initial_x)
```
    [注]：initial_x为初始点（此方法可能会得到局部最小值）
- （2）`fmin()`、`fmin_cg`等等方法

### 5、拟合（最小二乘法）
- （1）`curve_fit`
```
    #产生数据
    def f(x):
        return x**2 + 10*np.sin(x)
    xdata = np.linspace(-10, 10, num=20)
    ydata = f(xdata)+np.random.randn(xdata.size)
    plt.scatter(xdata, ydata, linewidths=3.0, 
               edgecolors="red")
    #plt.show()
    #拟合
    def f2(x,a,b):
        return a*x**2 + b*np.sin(x)
    guess = [2,2]
    params, params_covariance = optimize.curve_fit(f2, xdata, ydata, guess)
    #画出拟合的曲线
    x1 = np.linspace(-10,10,256)
    y1 = f2(x1,params[0],params[1])
    plt.plot(x1,y1)
    plt.show()
```

### 6、统计检验
- （1）T-检验`stats.ttest_ind`
```
    a = np.random.normal(0, 1, size=10)
    b = np.random.normal(1, 1, size=10)
    print stats.ttest_ind(a, b)  
```
输出：(-2.6694785119868358, 0.015631342180817954)
后面的是概率p: 两个过程相同的概率。如果其值接近1，那么两个过程几乎可以确定是相同的，如果其值接近0，那么它们很可能拥有不同的均值。

### 7、插值
- （1）导入包：`from scipy.interpolate import interp1d`
```
    #产生一些数据
    x = np.linspace(0, 1, 10)
    y = np.sin(2 * np.pi * x)
    computed_time = np.linspace(0, 1, 50)
    #线性插值
    linear_interp = interp1d(x, y)
    linear_results = linear_interp(computed_time)
    #三次方插值
    cubic_interp = interp1d(x, y, kind='cubic')
    cubic_results = cubic_interp(computed_time)
    #作图
    plt.plot(x, y, 'o', ms=6, label='y')
    plt.plot(computed_time, linear_results, label='linear interp')
    plt.plot(computed_time, cubic_results, label='cubic interp')
    plt.legend()
    plt.show()
```

### 8、求解非线性方程组
- （1）`optimize`中的`fsolve`
```
    from scipy.optimize import fsolve
    def func(x):
        x0,x1,x2 = x.tolist()
        return [5*x1-25,5*x0*x0-x1*x2,x2*x0-27]
    initial_x = [1,1,1]
    result = fsolve(func, initial_x)
    print result
```


## 四、pandas
### 1、pandas特征与导入
- （1）包含高级的数据结构和精巧的工具
- （2）pandas建造在NumPy之上
- （3）导入：
```
    from pandas import Series, DataFrame
    import pandas as pd
```

### 2、pandas数据结构
#### （1）Series
- 一维的类似的数组对象
- 包含一个数组的数据（任何NumPy的数据类型）和一个与数组关联的索引
  - 不指定索引：`a = Series([1,2,3])` ，输出为
    ```
    0    1
    1    2
    2    3
    ```
    包含属性`a.index,a.values`，对应索引和值
  - 指定索引：`a = Series([1,2,3],index=['a','b','c'])`
   可以通过索引访问`a['b']`
- 判断某个索引是否存在：`'b' in a`
- 通过字典建立`Series`
```
dict = {'china':10,'america':30,'indian':20}
print Series(dict)
```
输出：
```
america    30
china      10
indian     20
dtype: int64
```
- 判断哪个索引值缺失：
```
dict = {'china':10,'america':30,'indian':20}
state = ['china','america','test']
a = Series(dict,state)
print a.isnull()
```
输出：（test索引没有对应值）
```
china      False
america    False
test        True
dtype: bool
```
- 在算术运算中它会**自动对齐**不同索引的数据
```
a = Series([10,20],['china','test'])
b = Series([10,20],['test','china'])
print a+b
```
输出：
```
china    30
test     30
dtype: int64
```
- 指定`Series`对象的`name`和`index`的`name`属性
```
a = Series([10,20],['china','test'])
a.index.name = 'state'
a.name = 'number'
print a
```
输出：
```
state
china    10
test     20
Name: number, dtype: int64
```

#### （2）DataFrame
- `Datarame`表示一个表格，类似电子表格的数据结构
- 包含一个经过**排序的列表集**（按`列名`排序）
- 每一个都可以有不同的类型值（数字，字符串，布尔等等）
- `DataFrame`在内部把数据存储为一个二维数组的格式，因此你可以采用分层索引以表格格式来表示高维的数据
- 创建：
  - 通过字典
 ```
    data = {'state': ['a', 'b', 'c', 'd', 'd'],
            'year': [2000, 2001, 2002, 2001, 2002],
            'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
    frame = DataFrame(data)
    print frame
 ```
 输出：(按照**列名排好序**的[若是手动分配列名，会按照你设定的]，并且索引会自动分配)
 ```
    pop state  year
0  1.5     a  2000
1  1.7     b  2001
2  3.6     c  2002
3  2.4     d  2001
4  2.9     d  2002
 ```
- 访问
  - **列**：与`Series`一样，通过列名访问：`frame['state']`或者`frame.state`
  - **行**：`ix` 索引成员（field），`frame.ix[2]`，返回每一列的第3行数据
- 赋值：`frame2['debt'] = np.arange(5.)`，若没有`debt`列名，则会新增一列
- 删除某一列：`del frame2['eastern']`
- 像Series一样， `values` 属性返回一个包含在DataFrame中的数据的二维ndarray
- 返回所有的列信息：`frame.columns`
- 转置：`frame2.T`

#### （3）索引对象
- pandas的索引对象用来保存坐标轴标签和其它元数据（如坐标轴名或名称）
- 索引对象是不可变的，因此不能由用户改变
- 创建`index = pd.Index([1,2,3])`
- 常用操作
  - `append`-->链接额外的索引对象，产生一个新的索引
  - `diff`	-->计算索引的差集
  - `intersection`	-->计算交集
  - `union`	-->计算并集
  - `isin`	-->计算出一个布尔数组表示每一个值是否包含在所传递的集合里
  - `delete`	-->计算删除位置i的元素的索引
  - `drop`	-->计算删除所传递的值后的索引
  - `insert`	-->计算在位置i插入元素后的索引
  - `is_monotonic`	-->返回True，如果每一个元素都比它前面的元素大或相等
  - `is_unique`	-->返回True，如果索引没有重复的值
  - `unique`	-->计算索引的唯一值数组

### 3、重新索引`reindex` 
#### （1）Series
- （1）重新排列
```
    a = Series([2,3,1],index=['b','a','c'])
    b = a.reindex(['a','b','c'])
    print b
```
- （2）重新排列，没有的索引补充为0,`b=a.reindex(['a','b','c','d'],fill_value=0)`
- （3）重建索引时对值进行内插或填充
```
    a = Series(['a','b','c'],index=[0,2,4])
    b = a.reindex(range(6),method='ffill')
    print b
```
输出：
```
    0    a
    1    a
    2    b
    3    b
    4    c
    5    c
    dtype: object
```
`method`的参数
ffill或pad---->前向（或进位）填充
bfill或backfill---->后向（或进位）填充

#### （3）DataFrame
- 与Series一样，`reindex` index
- 还可以reindex column列，`frame.reindex(columns=['a','b'])`

### 4、从一个坐标轴删除条目

#### （1）Series
- `a.drop(['a','b'])` 删除a，b索引项
#### （2）DataFrame
- 索引项的删除与`Series`一样
- 删除column--->`a.drop(['one'], axis=1)` 删除column名为one的一列


### 5、索引，挑选和过滤

#### （1）Series
- 可以通过index值或者整数值来访问数据，eg：对于`a = Series(np.arange(4.), index=['a', 'b', 'c', 'd'])`，`a['b']`和`a[1]`是一样的
- 使用标签来切片和正常的Python切片并不一样，它会把结束点也包括在内
```
a = Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
print a['b':'c']
```
输出包含c索引对应的值


#### （2）DataFrame
- 显示前两行：`a[:2]`
- 布尔值访问：`a[a['two']>5]`
- 索引字段 ix 的使用
  - index为2，column为'one'和'two'--->`a.ix[[2],['one','two']]`
  - index为2的一行：`a.ix[2]`

### 6、DataFrame和Series运算

- （1）DataFrame每一行都减去一个Series
```
    a = pd.DataFrame(np.arange(16).reshape(4,4),index=[0,1,2,3],columns=['one',    'two','three','four'])
    print a
    b = Series([0,1,2,3],index=['one','two','three','four'])
    print b
    print a-b
```
输出：
```
       one  two  three  four
    0    0    1      2     3
    1    4    5      6     7
    2    8    9     10    11
    3   12   13     14    15
    one      0
    two      1
    three    2
    four     3
    dtype: int64
       one  two  three  four
    0    0    0      0     0
    1    4    4      4     4
    2    8    8      8     8
    3   12   12     12    12
```

### 7、读取文件
- （1）`csv`文件
`pd.read_csv(r"data/train.csv")`，返回的数据类型是`DataFrame`类型

### 8、查看DataFrame的信息
- （1）`train_data.describe()`
eg:
```
       PassengerId    Survived      Pclass         Age       SibSp  \
count   891.000000  891.000000  891.000000  714.000000  891.000000   
mean    446.000000    0.383838    2.308642   29.699118    0.523008   
std     257.353842    0.486592    0.836071   14.526497    1.102743   
min       1.000000    0.000000    1.000000    0.420000    0.000000   
25%     223.500000    0.000000    2.000000   20.125000    0.000000   
50%     446.000000    0.000000    3.000000   28.000000    0.000000   
75%     668.500000    1.000000    3.000000   38.000000    1.000000   
max     891.000000    1.000000    3.000000   80.000000    8.000000   
```

### 9、定位到一列并替换
- `df.loc[df.Age.isnull(),'Age'] = 23 #'Age'列为空的内容补上数字23`

### 10、将分类变量转化为指示变量`get_dummies()`
- 
```
s = pd.Series(list('abca'))
pd.get_dummies(s)
```
```
   a  b  c
0  1  0  0
1  0  1  0
2  0  0  1
3  1  0  0
```

### 11、list和string互相转化
- string转list
```
>>> str = 'abcde'
>>> list = list(str)
>>> list
['a', 'b', 'c', 'd', 'e']
```
- list转string
```
>>> str_convert = ','.join(list)
>>> str_convert
'a,b,c,d,e'
```

### 12、删除原来的索引，重新从0-n索引
- `x = x.reset_index(drop=True)`

### 13、apply函数
- DataFrame.apply(func, axis=0, broadcast=False, raw=False, reduce=None, .....
- df.apply(numpy.sqrt) # returns DataFrame
  - 等价==》df.apply(lambda x : numpy.sqrt(x))==>使用更灵活
- df.apply(numpy.sum, axis=0) # equiv to df.sum(0)
- df.apply(numpy.sum, axis=1) # equiv to df.sum(1)

### 13、re.search().group()函数
- re.search(pattern, string, flags=0)
- group(num=0)函数返回匹配的字符，默认num=0,可以指定**多个组号**，例如group(0,1)

### 14、pandas.cut()函数
- pandas.cut(x, bins, right=True, labels=None, retbins=False, precision=3, include_lowest=False)
- x为以为数组
- bins可以是**int值**或者**序列**
  - 若是int值就根据x分为bins个数的**区间**
  - 若是序列就是自己指定的区间
- right**包含**最右边的区间，默认为True
- labels **数组**或者**一个布尔值**
  - 若是数组，需要与对应bins的**结果**一致
  - 若是布尔值**False**，返回bin中的一个值

- eg:pd.cut(full["FamilySize"], bins=[0,1,4,20], labels=[0,1,2])

### 15、




## 五、scikit-learn
### 1、手写数字识别（SVM）
```
    from sklearn import datasets
    from sklearn import svm
    import numpy as np
    from matplotlib import pyplot as plt
    
    '''
    使用sciki-learn中的数据集，一般有data,target,DESCR等属性属性
    '''
    
    digits = datasets.load_digits()                 #加载scikit-learn中的数据集
    
    clf = svm.SVC(gamma=0.001,C=100)                    #使用支持向量机进行分类，gamma为核函数的系数
    clf.fit(digits.data[:-4],digits.target[:-4])        #将除最后4组的数据输入进行训练
    
    predict = clf.predict(digits.data[-4:])         #预测最后4组的数据，[-4:]表示最后4行所有数据，而[-4,:]表示倒数第4行数据
    
    print "预测值为：",predict
    print "真实值：",digits.target[-4:]
    
    #显示最后四个图像
    plt.subplot(2,2,1)
    plt.imshow(digits.data[-4,:].reshape(8,8))
    plt.subplot(2,2,2)
    plt.imshow(digits.data[-3,:].reshape(8,8))
    plt.subplot(2,2,3)
    plt.imshow(digits.data[-2,:].reshape(8,8))
    plt.subplot(2,2,4)
    plt.imshow(digits.data[-1,:].reshape(8,8))
    plt.show()

```

svm的参数参数解释：
- （1）**C: 目标函数的惩罚系数C，用来平衡分类间隔margin和错分样本的，default C = 1.0；**
- （2）**kernel：参数选择有RBF, Linear, Poly, Sigmoid, 默认的是"RBF";**
- （3）degree：if you choose 'Poly' in param 2, this is effective, degree决定了多项式的最高次幂；
- （4）**gamma：核函数的系数('Poly', 'RBF' and 'Sigmoid'), 默认是gamma = 1 / n_features;**
- （5）coef0：核函数中的独立项，'RBF' and 'Poly'有效；
- （6）probablity: 可能性估计是否使用(true or false)；
- （7）shrinking：是否进行启发式；
- （8）tol（default = 1e - 3）: svm结束标准的精度;
- （9）cache_size: 制定训练所需要的内存（以MB为单位）；
- （10）class_weight:每个类所占据的权重，不同的类设置不同的惩罚参数C,缺省的话自适应；
- （11）verbose: 跟多线程有关，不大明白啥意思具体；
- （12）**max_iter: 最大迭代次数，default = 1000， if max_iter = -1, no limited;**
- （13）decision_function_shape ： ‘ovo’ 一对一, ‘ovr’ 多对多  or None 无, default=None
- （14）random_state ：用于概率估计的数据重排时的伪随机数生成器的种子。


### 2、保存训练过的模型
`joblib.dump(clf, "digits.pkl")  #将训练的模型保存成digits.pkl文件`
加载模型：
`clf = joblib.load("digits.pkl")`
其余操作数据即可，预测

### 3、鸢尾花分类（svm，分离出测试集）
```
    from sklearn import datasets
    from sklearn.cross_validation import train_test_split
    from sklearn.svm import SVC
    import numpy as np
    '''
    加载scikit-learn中的鸢尾花数据集
    '''
    
    #加载鸢尾花数据集
    iris = datasets.load_iris()
    iris_data = iris.data;          #相当于X
    iris_target = iris.target;      #对应的label种类，相当于y
    
    x_train,x_test,y_train,y_test =     train_test_split(iris_data,iris_target,test_size=0.2)       #将数据分成训练集x_train和测试集x_test，测试集占总数据的0.2
    
    model = SVC().fit(x_train,y_train);     #使用svm在训练集上拟合
    predict = model.predict(x_test)         #在测试集上预测
    right = sum(predict == y_test)          #求预测正确的个数
    
    print ('测试集准确率：%f%%'%(right*100.0/predict.shape[0]))        #求在测试集上预测的正确率，shape[0]返回第一维的长度，即数据个数

```
**[另：留一验证法]：**-->每次取一条数据作为测试集，其余作为训练集
```
    from sklearn import datasets
    from sklearn.svm import SVC
    import numpy as np
    
    def data_svc_test(data,target,index):
        x_train = np.vstack((data[0:index],data[index+1:-1]))#除第index号之外的    数据为训练集
        x_test = data[index].reshape(1,-1)                    #第index号数据为测试集，reshape(1,-1)的作用是只有一条数据时，使用reshap    e(1,-1)，否则有个过时方法的警告
        y_train = np.hstack((target[0:index],target[index+1:-1]))
        y_test = target[index]
        model = SVC().fit(x_train,y_train)    #建立SVC模型
        predict = model.predict(x_test)
        
        return predict == y_test        #返回结果是否预测正确
    
    #读取数据
    iris = datasets.load_iris()
    iris_data = iris.data
    iris_target = iris.target
    m = iris_target.shape[0]
    
    right = 0;
    for i in range(0,m):
        right += data_svc_test(iris_data,iris_target,i)
    print ("%f%%"%(right*100.0/m))
    
```
### 4、房价预测(SVR-->支持向量回归)
```

    from sklearn import datasets
    from sklearn.svm import SVR     #引入支持向量回归所需的SVR模型
    from sklearn.cross_validation import train_test_split
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    #加载数据
    house_dataset = datasets.load_boston()
    house_data = house_dataset.data
    house_price = house_dataset.target
    #数据预处理-->归一化
    x_train,x_test,y_train,y_test =     train_test_split(house_data,house_price,test_size=0.2)  
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train) #训练集
    x_test = scaler.transform(x_test)   #测试集
    
    #回归，预测
    model = SVR().fit(x_train,y_train)  #使用SVR回归拟合
    predict = model.predict(x_test)     #预测
    result = np.hstack((y_test.reshape(-1,1),predict.reshape(-1,1))) #reshape(-1,1)所有行转为1列向量
    print(result)
```
## 六、sk-learn模型总结

### 0、数据处理
#### （1）均值归一化：`from sklearn.preprocessing import StandardScaler`
- scaler = StandardScaler()
- scaler.fit(X_train)
- X_train = scaler.transform(X_train)

#### （2）分割数据：`from sklearn.cross_validation import train_test_split`
- `x_train,x_test,y_train,y_test =     train_test_split(iris_data,iris_target,test_size=0.2)`


### 1、线性模型`from sklearn import linear_model`
#### （1）逻辑回归模型
- linear_model.LogisticRegression()
- 重要参数
  - C：正则化作用，默认值`1.0`，值越小，正则化作用**越强**
  - max_iter：最大梯度下降执行次数，默认值`100`
  - tol：停止执行的容忍度，默认值`1e-4`
- 重要返回值
  - coef_：对应feature的**系数**

### 2、svm模型`from sklearn import svm`
#### （1）分类模型
- svm.SVC()
- 重要参数
  - kernel：使用的核函数，默认是`rbf`径向基函数，还有`linear，poly，sigmoid ，precomputed`核函数
  - C：正则化作用，默认值`1.0`，值越大，`margin`越大
  - tol：停止执行的容忍度，默认值`1e-4`
  - gamma：为核函数的系数，值**越大**拟合的越好，默认是`1/feature的个数`
  - degree：对应`poly`核函数
- 重要返回值


#### （2）回归模型

### 3、ensemble模型`import sklearn.ensemble`
#### （1）随机森林
- ensemble.RandomForestClassifier()
- 重要参数
 - 
- 重要返回值
 - 
















