import numpy

class BatchKmeansProcess:
    '''创建类BatchKmeansProcess的目的是希望它能模拟多台计算机分布式地计算K-Means算法。
    每个BatchKmeansProcess实例表示分布式系统中的某一台计算机。它的成员变量和成员函数表示每一台计算
    机内部运算时所需的变量和操作。分布式系统中计算机间的数据通信采用实例之间的参数传递模拟'''
    
    def __init__(self,X,k):
        '''输入：
                X:numpy.array m*n矩阵，m个n维度训练样本
                k:int类型，类别数目
                
            初始化参数：
                self.Train     ：numpy.array m*n矩阵
                self.LabelNums ：int类型，类别数
                self.u         ：numpy.array k*n矩阵
                self.x2y       ：numpy.array [j0,j1,j2,...,jl] 表示序号从0到l的样本对应的类别
                self.distlst   ：'''
        
        self.Train = X
        self.LabelNums = k
        self.u = numpy.array([ [ 0 for _ in range(len(X[0])) ] for _ in range(k)])
        self.x2y = numpy.array([0]*len(X))
        self.distlst = numpy.array( [0]*len(X) )
        
        #0号计算机会接收所有其他计算机发来的数据，并会对这些数据进行额外的操作。0号计算机内将会使用以下额外变量。
        self.maxSampleTotal = X[0]
        self.maxDistTotal = 0
        self.changeMarkTotal = True
        self.YxTotal = numpy.array([[0]*len(X[0])]*k)
        self.YnTotal = numpy.array([0]*k)



    def ExtraIniCalulate(self,maxSample,maxDist):
        '''该函数在中心点初始化时使用，它将对每台计算机送来的最大距离值maxDist与
        样本点maxSample进行挑选。
        
            输入：
                   maxSample: numpy.array 1*n矩阵 某一台计算机中距离前i个中心最远的样本点
                   maxDist:   int类型，maxSample距离各个中心点的距离和'''
        if self.maxDistTotal<=maxDist:
            self.maxDistTotal = maxDist
            self.maxSampleTotal = maxSample


            
    def ExtraCenterCalculate(self,Yx,Yn):
        '''该函数在更新K-Means中心时使用。它接收各个计算输出的各个分类样本特征之和与各个分类的样本总数，对自身更新中心点。
            输入：
                    Yx：numpy.array,1*n矩阵 某个计算机输出的各个类别中的样本特征之和。
                    Yn：numpy.array,1*k矩阵 某个计算机输出的各个类别的样本总数。
                    '''
        self.YxTotal += Yx
        self.YnTotal += Yn
        self.u = numpy.array([list(self.YxTotal[ct]/self.YnTotal[ct]) \
                                     for ct in range(self.LabelNums) if self.YnTotal[ct]!=0])
 
    def DistFun(self,x1,x2):
        '''该函数定义K-Means算法下的距离函数。
            输入：
                x1：numpy.array 1*n矩阵
                x2：numpy.array 1*n矩阵'''

        #采用欧式距离
        return (sum((numpy.array(x2)-numpy.array(x1))**2)/len(x1))**0.5
        
    def BatchPartForIni(self,ui):
        '''该函数用于Kmeans聚类的初始化。它将根据输入的训练样本使用如下方法找到k个样本中心：
        KMeans 的初始化方法：
        1、随机选一个样本作为第一个簇中心。
        2、计算每个样本到最近簇中心的距离。
        3、选择
        4、重复步骤2和3，直到找到K个簇中心。
            输入：
                    ui：numpy.array 1*n矩阵 初始化的第i个中心
            输出：
                    maxSample：numpy.array 1*n矩阵 该计算机中距离前i个中心最远的样本点
                    maxDist  ：int，maxSample距离前i个中心的距离和'''
        
        self.distlst = list(map( lambda x,y:y+self.DistFun(x,ui),self.Train,self.distlst  ))
        maxSampleIdx,maxDist = max( enumerate(self.distlst),key=lambda x:x[1])
        return self.Train[maxSampleIdx],maxDist

    def cluster(self):
        '''该函数对应Kmeans算法迭代过程中的step1。函数将会对样本寻找最近的中心点,并通过修改x2y变量将这个样本纳入该点对应的类别，在这个过程中函数会检测x2y在函数调用
        前后是否发生改动，这将用于之后的收敛判断。
        输出：
            ChangeMark：整形变量 0表示x2y无改动，1表示有改动'''
    
        #对每个样本求最近的中心点
        ChangeMark = False
        for Id_x in range(len(self.x2y)):
            Id_x_y = self.x2y[Id_x]
            minDist = 10000     #取足够大的数,要大于最小间隔
            for Id_y in range(len(self.u)):
                dist = self.DistFun( self.Train[Id_x],self.u[Id_y] )
                if minDist>=dist:
                    Id_x_y = Id_y
                    minDist = dist

            
            if Id_x_y != self.x2y[Id_x]:#如果出现改动，将ChangeMark修改为1
                ChangeMark = True
                self.x2y[Id_x] = Id_x_y

        return ChangeMark
    
    def BatchPartForComputeCenter(self):
        '''该函数用于计算当前数据批中各个类别中特征值的总和，以及各个类别中数据个数。
        输出：
                Yx：numpy.array k*n矩阵，存储各个类别中特征值的总和
                Yn：numpy.array k*1矩阵，存储各个类别中数据的个数'''
    
        Yx = numpy.array([ [ 0 for _ in range(len(self.Train[0])) ] for _ in range(self.LabelNums)])
        Yn = numpy.array([0]*self.LabelNums)

        for Id_x in range(len(self.x2y)):
            Yx[self.x2y[Id_x]] += self.Train[Id_x]
            Yn[self.x2y[Id_x]] += 1

        return Yx,Yn

    def BatchPartForError(self):
        '''该函数用于计算该计算机各个样本偏离中心的距离总和，它用于进行误差计算，不属于k-means算法。
        输出：
                类型numpy.array ,1*k维矩阵，各个类别下的样本偏移总和'''
        Error = numpy.array([0]*self.LabelNums)
        for Id_x in range(len(self.x2y)):
            Error[self.x2y[Id_x]]+=self.DistFun(self.Train[Id_x],self.u[self.x2y[Id_x]])
        return Error


        
