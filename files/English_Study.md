英语学习笔记
=============================

## 一、常用

## 二、Paper Records
### 1、Notes on Convolutional Neural Networks
#### 1) 词汇
- `derivation`[ˌderɪˈveɪʃn]：起源、衍生
- `straightforward`[ˌstreɪtˈfɔ:wəd]：直截了当的
- `invariance`[ɪn'veərɪəns]：不变性
- `subsampling`[sʌb'zɑ:mplɪŋ]：二次采样
- `snippet`[ˈsnɪpɪt]：片段
- `overstate`：夸大
- `exaggeration`：夸张
- `concatenate`[kɒn'kætɪneɪt]：把 （一系列事件、事情等）联系起来
- `derivatives`[dɪ'rɪvətɪvz]：导数
- `designate`[ˈdezɪgneɪt]：命名为
- `hyperbolic`[ˌhaɪpəˈbɒlɪk]：双曲线的
- `tangent`[ˈtændʒənt]：正切的
- `saturation`[ˌsætʃəˈreɪʃn]：饱和度
- `perturbation`[ˌpɜ:təˈbeɪʃn]：摄动
- `intersperse`[ˌɪntəˈspɜ:s]：点缀; 散布
- `auditory`[ˈɔ:dətri]：听觉的
- `triplets`['trɪpləts]：三个一组
- `vanilla`[vəˈnɪlə]：相对没有新意的，普通的
- `nomenclature`[nəˈmenklətʃə(r)]：术语; 专门名称

#### 2) 短语
- `element-wise multiplication`：元素相乘

### 2、Understanding the difficulty of training deep feedforward neural networks

#### 0)论文目的：
- Our objective here is to understand better why standard gradient descent from `random initialization` is doing so poorly with `deep neural networks`
- 
#### 1)词汇
- `saturation`[ˌsætʃəˈreɪʃn]：饱和
- `albeit`[ˌɔ:lˈbi:ɪt]：即使
- `plateau`[ˈplætəʊ]：高原，平稳时期
- `empirical`[ɪmˈpɪrɪkl]：经验主义的
- `basin`[ˈbeɪsn]：盆地
- `infinite`[ˈɪnfɪnət]：无限的
- `substantial`[səbˈstænʃl]：大量的，重大的
- `synthetic`[sɪnˈθetɪk]：合成的，人造的
- `ellipse`[ɪˈlɪps]：椭圆
- `stochastic`[stə'kæstɪk]：随机的
- `consecutive`[kənˈsekjətɪv]：连续的
- `quadratic`[kwɒˈdrætɪk]：二次的
- `exponential`[ˌekspəˈnenʃl]：指数，指数的
- `asymptote`['æsɪmptəʊt]：渐近线
- `heuristic`：启发式的，探索的
- `reveal`[rɪˈvi:l]：揭露
- `overly`：过度的
- `induce`：引起
- `epochs`[ˈi:pɒk]：训练次数
- `symmetric`[sɪ'metrɪk]：相对称的，均匀的
- `explode`[ɪkˈspləʊd]：激增
- `reminiscent`[ˌremɪˈnɪsnt]：怀旧的，使人联想的
- `magnitude`[ˈmægnɪtju:d]：量级
- `diverge`[daɪˈvɜ:dʒ]：发散
- `presumably`[prɪˈzju:məbli]：大概
- `alleviate`[əˈli:vieɪt]：减轻，缓和
- `discrepancy`[dɪsˈkrepənsi]：矛盾，不符合之处
- `symptomatic`[ˌsɪmptəˈmætɪk]：有症状的

#### 2)短语
-  theoretical appeal：理论的吸引力
-  uniform distribution：均匀分布
-  standard deviation：标准差
-  solid line：实线
-  argument vector：变元向量
-  normalized initialization：归一化的初始化
-  with respect to：关于

### 3、Delving Deep into Rectifiers:Surpassing Human-Level Performance on ImageNet Classification

#### 0)论文目的
- 第一：提出了PRelU
- 第二：对应的健壮的初始化方法
- 成果：
 - we achieve 4.94% top-5 test error on the ImageNet 2012 classification dataset
 - 超越了人类的5.1%

#### 1)单词
- `delve`[delv]：探究
- `rectifier`['rektɪfaɪə]：纠正者
- `Surpassing`[sɜ:'pɑ:sɪŋ]：卓越的
- `demonstrate`[ˈdemənstreɪt]：论证
- `tremendous`[trəˈmendəs]：极大的
- `sophisticated`[səˈfɪstɪkeɪtɪd]：复杂的
- `aggressive`[əˈgresɪv]：侵略的
- `expedite`[ˈekspədaɪt]：加快发展
- `prevalence`['prevələns]：流行
- `negligible`[ˈneglɪdʒəbl]：微不足道的
- `coefficient`：系数
- `slope`[sləʊp]：斜率
- `subscript`：下标
- `Leaky`[ˈli:ki]：有漏洞的
- `variant`[ˈveəriənt]：变式
- `summation`[sʌˈmeɪʃn]：加在一起
- `momentum`[məˈmentəm]：动量
- `decay`[dɪˈkeɪ]：衰退
- `monotonic`[ˌmɒnə'tɒnɪk]：单调的
- `discriminative`：有判别力的
- `hamper`[ˈhæmpə(r)]：妨碍
- `optimum`[ˈɒptɪməm]：最佳效果
- `auxiliary`[ɔ:gˈzɪliəri]：补充的，辅助的
- `intermediate`：充当调解
- `mutually`[ˈmju:tʃuəli]：互相地
- `magnify`[ˈmægnɪfaɪ]：放大
- `exponentially`[ˌekspə'nenʃəlɪ]：以指数方式的
- `diminishing`[dɪ'mɪnɪʃɪŋ]：逐渐缩小的


#### 2)短语
- Rectified activation units：纠正激活单元
- Parametric Rectified Linear Unit(PReLu)：参数校正线性单元
- theoretically sound：理论上合理的
- momentum method：动量法


### 4、Batch Normalization: Accelerating Deep Network Training by Reducing
Internal Covariate Shift

#### 0)论文目的
- Our method draws its strength from making normalization a part of the model architecture and performing the normalization `for each training mini-batch`. 

#### 1)单词
- `notoriously`[nəʊ'tɔ:rɪəslɪ]：著名的，众所周知的
- `ensemble`[ɒnˈsɒmbl]：全体
- `tuning`['tju:nɪŋ]：调整
- `preceding`[prɪˈsi:d]：在....之前发生
- `amplify`[ˈæmplɪfaɪ]：放大
- `stuck`[stʌk]：动不了的
- `subtracte`[səbˈtrækt]：减去
- `subsequent`[ˈsʌbsɪkwənt]：附随的
- `impractical`：不切实际的
- `crucial`：关键性的，决定性的
- `inference`[ˈɪnfərəns]：推断

#### 2)短语
-  blow up：爆炸、爆发
-  unit variance：单位方差
-  identity transform：恒等变换
-  affine transform：仿射变换
-  with respect to：关于
-  differentiable transformation：微分变换



## 三、Competition Records

### 1、Data Science Bowl 2017
#### 1)Description
- `strike`[straɪk]：攻击
- `spearhead`[ˈspɪəhed]：带头
- `milestone`：里程碑
- `convene`[kənˈvi:n]：召开
- `high-resolution`：高分辨率
- `lesion`[ˈli:ʒn]：损害
- `plague`[pleɪg]：灾害
- `intervention`：介入，干涉
- `unprecedent`[ʌnp'resɪdənt]：史无前例的

#### 2)Data
- `sophisticate`[səˈfɪstɪkeɪt]：见多识广的人
- `BitTorrent`：比特流



## 四、Study Records

### 1、CS224d: Deep Learning for Natural Language Processing

#### 0)
- 斯坦福大学课程，网站：http://web.stanford.edu/class/cs224n/





















