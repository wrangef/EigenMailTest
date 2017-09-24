import random
from BatchKmeans import *
import matplotlib as mplt
import matplotlib.pyplot as plt

#####################准备阶段#######################

    #生成训练样本
m = 1000 #样本数目
n = 2    #样本维度
k = 6    #类别数目
X = [[ random.randint(1,1000) for _ in range(n)] for _ in range(m)]




computerNum = 8    #设置计算机数目
trainNumForEachCom = [int(m/computerNum)]*computerNum #每台计算机可计算的样本容量
computerList = []



    #将训练样本按容量trainNumForEachCom 分配给各个计算机
for comID in range(computerNum):
    computer = BatchKmeansProcess(X[0:trainNumForEachCom[comID]],k)
    computerList.append(computer)
    X = X[trainNumForEachCom[comID]::]






###############################K-Means######################################

    #初始化k类中心
    '''KMeans++ 的初始化方法：
        1、随机选一个样本作为第一个簇中心。
        2、计算每个样本到最近簇中心的距离。
        3、选择距离最远的样本作为新的中心点。
        4、重复步骤2和3，直到找到K个簇中心。'''


        #随机选择一台计算机中的一个样本作为第一个中心
pickComputer = random.randint(0,computerNum-1)  
pickSampleIdx = random.randint(0,trainNumForEachCom[pickComputer]-1)
computerList[0].u[0]= computerList[pickComputer].Train[pickSampleIdx]
       

for ct in range(1,k):
    for comID in range(computerNum):
        
        #初始化中的第1-4步：每台计算机输出各自样本集合距离最近中心点距离最远的样本与距离
        maxSample,maxDist = computerList[comID].BatchPartForIni\
                            (numpy.array(computerList[0].u[ct-1]))
        
        '''初始化第5-7步：各台计算机将参数传递给计算机0，计算机0将每台计算机的最远样本进行比较，
        选择距离最远的样本作为新的中心'''                                                                                                   
        computerList[0].ExtraIniCalulate(maxSample,maxDist) 
        
    computerList[0].u[ct] = computerList[0].maxSampleTotal 
    computerList[0].maxDistTotal = 0
    

        #k个中心点生成后，计算机0为每台计算机分配初始的中心点
for comID in range(computerNum):
    computerList[comID].u = computerList[0].u










    #进入K-Means迭代
changeMarkForAll = True         
itNum = 0
while changeMarkForAll == True:
    changeMarkForAll = False
    
    computerList[0].YxTotal = numpy.array([[0]*n]*k)
    computerList[0].YnTotal = numpy.array([0]*k)
    
    ErrTotal = numpy.array([0]*k)#总均方根误差
    for comID in range(computerNum):
        
        '''分布式迭代部分Step1，对每一台计算机，计算它所有样本点到各个中心点的距离，
        并将每一个样本纳入到它距离最近中心点所对应的类别。
        并检测样本点的类别是否有改动，生成changeMark变量。'''
        changeMark = computerList[comID].cluster()

        #收敛判断（该部分应该在计算机0中进行，但为了代码的方便被单独提出）
        changeMarkForAll = changeMarkForAll or  changeMark
        
        #分布式迭代部分Step2，第1-2步
        Yx,Yn = computerList[comID].BatchPartForComputeCenter()
        
        #分布式迭代部分Step2，第3-5步,计算机0利用接收到的数据更新中心点（无收敛判断部分）
        computerList[0].ExtraCenterCalculate(Yx,Yn)


    
    ErrTotal+=computerList[comID].BatchPartForError()
    
    itNum+=1
    print('第{0}次迭代，平均每个样本偏移为{1}'.format(itNum,sum(ErrTotal)/m))
    
    #分布式迭代部分Step2,7-8步，计算机0向其他计算机发送更新的中心点数据。
    for comID in range(computerNum):
        computerList[comID].u = computerList[0].u






        

##################作图部分########################
        
#针对样本维度为2，类别数目<=6的情况的作图程序

if k<=6 and n==2:
    COLOR = {0:'b',1:'g',2:'r',3:'c',4:'m',5:'y'}
    plt.figure('1')
    for comID in range(computerNum):
        for Id_x in range(len(computerList[comID].Train)):
            plt.scatter(computerList[comID].Train[Id_x][0],
                        computerList[comID].Train[Id_x][1],
                        color = COLOR[computerList[comID].x2y[Id_x]])

    plt.show()
else:
    print('放弃作图！该程序仅在样本维度为2，类别数目<=6时进行作图')
    

