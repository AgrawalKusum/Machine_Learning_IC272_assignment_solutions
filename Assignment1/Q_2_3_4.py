import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Q2
df=pd.read_csv('landslide_data_miss.csv')
df2=pd.read_csv('landslide_data_original.csv')
#1
og_index=df.index  #for part b
df_clean=df.dropna(subset=['stationid'])
dropped_index=og_index.difference(df_clean.index)  #for partb

frac=len(df.columns)/3

og_index1=df_clean.index #for part b
df_clean=df_clean[df_clean.apply(lambda row: row.isna().sum()<frac, axis=1)]
dropped_index1=og_index1.difference(df_clean.index)#for part b

#print(df_clean)

#2
#for linear interpolation in a single column
n_r,n_c=df.shape
def interpol(col):
    for i in range(len(col)):
        if pd.isna(col.iloc[i]):
            p = i - 1
            n = i + 1
            while n < len(col) and pd.isna(col.iloc[n]):
                n += 1
            if p >= 0 and n < len(col):
                m = (col.iloc[n] - col.iloc[p]) / (n - p)
                c = col.iloc[p]
                col.iloc[i] = c + (m * (i - p))

    return col

#applying this to all columns
columns=df_clean.columns
s=columns.get_loc('temperature')
interpol_col=columns[s:]

df_clean[interpol_col]=df_clean[interpol_col].apply(interpol)
#print(df_clean)

#a
def stats(col):
    n=len(col)
    #mean
    mean=col.sum()/n
    
    #median
    sort_col=sorted(col)
    median=sort_col[len(col)//2]
    if n % 2 == 0:
        median = (sort_col[n//2 - 1] + sort_col[n//2]) / 2
    else:
        median = sort_col[n//2]

    #standard deviation
    sq_num=0
    for i in col:
        sq_num+=(mean-i)**2
    std_dev=(sq_num/len(col))**0.5

    return mean,median,std_dev

old_mean=[]
old_median=[]
old_std_dev=[]
new_mean=[]
new_median=[]
new_std_dev=[]

for col in interpol_col:
    a1,b1,c1=stats(df_clean[col])
    a2,b2,c2=stats(df2[col])
    new_mean.append(a1)
    new_median.append(b1)
    new_std_dev.append(c1)
    old_mean.append(a2)
    old_median.append(b2)
    old_std_dev.append(c2)    

comparison_table={'Columns':interpol_col, 'New Mean':new_mean, 'Old Mean': old_mean, 'New Median': new_median, 'Old Median': old_median, 'New Std': new_std_dev, 'Old Std': old_std_dev}
comparison_df=pd.DataFrame(comparison_table)
#print(comparison_df)

#b
df2=df2.drop(dropped_index)
df2=df2.drop(dropped_index1)#matching the shape of original and cleaned

#calculating rmse
def rmse(og,cl):
    rmse=np.sqrt((np.mean(og-cl)**2)/len(og))
    return rmse

RMSE=[rmse(df2[i],df_clean[i]) for i in interpol_col]

plt.plot(interpol_col, RMSE)
plt.xlabel('Attributes')
plt.ylabel('RMSE Values')
plt.title('RMSE of Interpolated Dataset')
#plt.show()


#Q3
#1
df_clean.boxplot()
plt.title('Boxplot of Interpolated Data')
#plt.show()

#2
def outlier_replacement(df):
    for col in interpol_col:
        Q1=df[col].quantile(0.25)
        Q3=df[col].quantile(0.75)
        IQR = Q3 - Q1
        l_b = Q1 - 1.5 * IQR
        u_b = Q3 + 1.5 * IQR
        med=new_median[interpol_col.get_loc(col)-2]
        df[col]=df[col].apply(lambda x: med if (x<l_b or x> u_b) else x)

    return df

df_clean2=outlier_replacement(df_clean)
df_clean2.boxplot()
plt.title('Boxplot of Interpolated Data')
#plt.show()

#Q4
#1
old_min_max={'old_min':[], 'old_max':[]}
def normalize(col):
    min_col=col.min()
    max_col=col.max()

    col=[(5+(x-min_col)*7/(max_col-min_col)) for x in col]
    old_min_max['old_min'].append(min_col)
    old_min_max['old_max'].append(max_col)

    return col

df_clean2_norm=df_clean2.copy()
df_clean2_norm[interpol_col]=df_clean2_norm[interpol_col].apply(normalize)

#print(df_clean2_norm)
mean_old=df_clean2[interpol_col].mean()
std_old=df_clean2[interpol_col].std()
mean_new=df_clean2_norm[interpol_col].mean()
std_new=df_clean2_norm[interpol_col].std()

#2
def standardize(col):
    m=col.mean()
    sd=col.std()
    col=[(x-m)/sd for x in col]
        
    return col
df_clean2_std=df_clean2.copy()
df_clean2_std[interpol_col]=df_clean2_std[interpol_col].apply(standardize)
#print(df_clean2_std)
mean_new_s=df_clean2_std[interpol_col].mean()
std_new_s=df_clean2_std[interpol_col].std()