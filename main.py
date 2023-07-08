import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=sns.load_dataset("titanic")
df.head()
df['age'].median()
#Eksik değerlerin yerini medyanı ile doldurma
df['age'].fillna(df['age'].median(),inplace=True)
#------------------------------
pd.set_option("display.max_columns",None)
pd.set_option("display.width",500)
#Kategorik Değişken Analizi
df["sex"].unique() #male,femaile,dtype:object
df["sex"].nunique()#2
cat_cols=[col for col in df.columns if str(df[col].dtypes) in ["category","object","bool"]]
num_but_cat=[col for col in df.columns if df[col].nunique()<10 and df[col].dtypes in ["int64","float64"]]

cat_but_car= [col for col in df.columns if df[col].nunique()>20 and str(df[col].dtypes) in ["int64","float64"]]

cat_cols=cat_cols+num_but_cat
cat_cols=[col for col in cat_cols if col not in cat_but_car]
#-----
def cat_summary(dataframe,col_name,plot=False):
    print(pd.DataFrame({col_name:dataframe[col_name].value_counts(),
                        "Ratio":100*dataframe[col_name].value_counts()/len(dataframe)}))
    print("###################")
    if plot:
        sns.countplot(x=dataframe[col_name],data=dataframe)
        plt.show(block=True)
cat_summary(df,"sex",True)

#Sayısal Değişken Analizi
df[["age","fare"]].describe().T
num_cols=[col for col in df.columns if df[col].dtypes in ["int64","float64"]]
num_cols=[col for col in num_cols if col not in cat_cols]

def num_summary(dataframe,numerical_col,plot=False):
    quantiles=[0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
num_summary(df,"age",plot=True)

#Değişkenlerin Yakalanması ve İşlemlerin Genelleştirilmesi
def grab_col_names(dataframe,cat_th=10,car_th=20):
    """

    :param dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir
    :param cat_th: int,float
        numerik fakat kategorik olan değişkenler için sınır eşik değeri
    :param car_th: int,float
        kategorik fakat kardinal değişkenler için sınır eşik değeri
    :return:
        cat_cols: list
            kategorik değişken listesi
        num_cols: list
            numerik değişken listesi
        cat_but_car: list
            kategorik görünümlü kardinal değişken listesi

    :Notes:
        cat_cols + num_cols + cat_but_car=toplam değişken sayısı
        num_but_car + cat_cols'un içerisinde
        return olan 3 liste toplamı değişken sayısına eşittir.
    """
    cat_cols=[col for col in df.columns if str(df[col].dtypes) in ["category","object","bool"]]

    num_but_cat=[col for col in df.columns if df[col].nunique()<10 and df[col].dtypes in ["int64","float64"]]
    cat_but_car=[col for col in df.columns if df[col].nunique()>20 and str(df[col].dtypes) in ["int64","float64"]]

    cat_cols=cat_cols+num_but_cat
    cat_cols=[col for col in cat_cols if col not in cat_but_car]

    num_cols=[col for col in df.columns if df[col].dtypes in ["int64","float64"]]
    num_cols=[col for col in num_cols if col not in cat_cols]

    print(f"Observation: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_car: {len(num_but_cat)}")
    return cat_cols,num_cols,cat_but_car

cat_cols,num_cols,cat_but_car=grab_col_names(df)
help(grab_col_names)
#bonus
df=sns.load_dataset("titanic")
for col in df.columns:
    if df[col].dtypes=="bool":
        df[col]=df[col].astype(int)

#Korelasyon Analizi(Analysis of Correlation)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns",None)
pd.set_option("display.width",500)
df=sns.load_dataset("titanic")
df=df.iloc[:,1:-1]
df.head()

num_cols=[col for col in df.columns if df[col].dtype in [int,float]]

corr=df[num_cols].corr()

sns.set(rc={"figure.figsize":(12,12)})
sns.heatmap(corr,cmap="RdBu")
plt.show()

##Yüksek Korelasyonlu Değişkenlerin Silinmesi
cor_matrix=df.corr().abs()

upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))


drop_list=[col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col]>0.90)]
cor_matrix[drop_list]

def high_correlated_cols(dataframe,plot=False,corr_th=0.90):
    corr=dataframe.corr()
    cor_matrix=corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list=[col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col]>corr_th)]
    if plot:
        sns.set(rc={"figure.figsize":(15,15)})
        sns.heatmap(corr,cmap="RdBu")
        plt.show()
    return drop_list
high_correlated_cols(df)
drop_list=high_correlated_cols(df,plot=True)
df.drop(drop_list,axis=1)
high_correlated_cols(df.drop(drop_list,axis=1),plot=True)

#Görselleştirme ve analizi
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df2=sns.load_dataset("titanic")
df2.head()
#cinsiyetin hayatta kalma durumu üzerindeki etkisini görselleştirme
sns.countplot(x='survived',hue='sex',data=df2)
#yaş dağılımını görselleştirme
df2['age'].hist(bins=30,color='darkred',alpha=0.7)

#Hayatta kalma ve diğer özellikler arasındaki korelasyonları görselleştirme

plt.figure(figsize=(10,7))
sns.heatmap(df2.corr(),annot=True,cmap='coolwarm')
