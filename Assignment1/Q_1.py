import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Question1
data=pd.read_csv('landslide_data_original.csv')
#1
'''mean=new_data.loc[:,'temperature'].mean()
minimum=new_data.loc[:,'temperature'].min()
maximum=new_data.loc[:,'temperature'].max()
median=new_data.loc[:,'temperature'].median()
std_div=new_data.loc[:,'temperature'].std()'''

def stats(col):
    #mean
    mean=col.sum()/len(col)

    #min & max
    min=max=col[0]
    for i in col:
        if i<min:
            min=i
        elif i>=max:
            max=i
    
    #median
    sort_col=sorted(col)
    median=sort_col[len(col)//2]

    #standard deviation
    sq_num=0
    for i in col:
        sq_num+=(mean-i)**2
    num=sq_num**(1/2)
    std_dev=(sq_num/len(col))**0.5

    return mean,min,max,median,std_dev
temp=data.loc[:,'temperature']
mean_d,min_d,max_d,median_d,std_dev_d=stats(temp)

sol_1=pd.DataFrame({'Attribute':['Mean','Minimum','Maximum','Median','Std'], 'Value':[mean_d,min_d,max_d,median_d,std_dev_d]})
#print(sol_1.round(2))

#2
def correl(c1,c2):
    m1=c1.mean()
    m2=c2.mean()
    num=sum((c1[i]-m1)*(c2[i]-m2) for i in range(len(c1)))
    den=(sum((c1[i]-m1)**2 for i in range(len(c1)))**0.5)*(sum((c2[i]-m2)**2 for i in range(len(c2)))**0.5)
    r=num/den
    return r

features=data.columns.to_list()
features=features[2:]
df_dict={'Feat/Feat':[f for f in features]}
for i in features:
    df_dict[i]=[correl(data[f],data[i]) for f in features]

sol_2=pd.DataFrame(df_dict)
#print(sol_2)

#3
new_data=data.loc[data['stationid']=='t12']['humidity'].round(2)

#no. of bins depend on the starting and end point chosen. let start be (min//5)*5 in our case
#end point be ((max//5)+1)*5
#no. of bins of size 5= (e-s)/5
#next we find frequency of values in each bin(the y axis)
#for x axis we need strings depicting the bin being considered

s=int((new_data.min()//5)*5)
e=int(((new_data.max()//5)+1)*5)
n=int((e-s)/5)

y=[0]*n

for i in new_data:
    index=int((i-s)//5)
    y[index]+=1

x_ends=list(range(s,e,5))
x=[f"{x_ends[i]}-{x_ends[i+1]}" for i in range(n-1)]

plt.bar(x,y[:-1]) #last bin is empty so last index of y is dropped
#plt.show()