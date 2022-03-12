#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.preprocessing import StandardScaler,LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import zscore


from sklearn.linear_model import LinearRegression,Lasso,LassoCV,Ridge,RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR

from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

import warnings
warnings.filterwarnings("ignore")


# In[38]:


data_train=pd.read_csv("D:\\fliprobo\\project\\P4\\Project-Housing--2-\\Project-Housing_splitted\\train.csv")
data_test=pd.read_csv("D:\\fliprobo\\project\\P4\\Project-Housing--2-\\Project-Housing_splitted\\test.csv")


# In[39]:


data_train.head(5)


# In[40]:


data_test.head(5)


# In[41]:


data_train.shape


# In[42]:


data_test.shape


# In[43]:


data_train.info()


# In[44]:


data_train['Alley'].nunique()


# In[45]:


for col in data_train:
    print("-"*30)
    print(f"no. of null values in {col} is {data_train[col].isna().sum()}")
    print(f"no. of unique values in {col} is {data_train[col].nunique()}")
    print("-"*30)


# Id is unique in the dataset no need to this variable as feature.,unique values in Utilities is 1
# 
# 
# null values in LotFrontage is 214, in Alley is 1091, in MasVnrType is 7, in MasVnrArea is 7, in BsmtQual is 30, in BsmtCond is 30, in BsmtExposure is 31, in BsmtFinType1 is 30, BsmtFinType2 is 31, FireplaceQu is 551, in GarageType is 64, in GarageYrBlt is 64, in GarageFinish is 64, in GarageQual is 64, in GarageCond is 64, in PoolQC is 1161, in Fence is 931, in MiscFeature is 1124

# In[46]:


sns.heatmap(data_train.isna())


# In[47]:


sns.heatmap(data_test.isna())


# In[48]:


data_train[data_train["BsmtFinSF1"]+data_train["BsmtFinSF2"]+data_train["BsmtUnfSF"]==data_train["TotalBsmtSF"]]  # Sum of basement type 1 type 2 and unfinished is equal to the total basement of house.


# As we can see total Basement area is equal to the finishe of type 1 and type2 and unfinised too.

# In[49]:


null_feature=[]
for col in data_train:
    if data_train[col].isna().sum() >0:
        null_feature.append(col)
        print(col)
    if (data_train[col].nunique() ==1) or (data_train[col].nunique() ==1168):
        data_train.drop(col,axis=1,inplace=True)


# In[50]:


data_test.drop(columns=["Id","Utilities"],axis=1,inplace=True)


# In[51]:


null_feature_test=[]
for col in data_test:
    if data_test[col].isna().sum() >0:
        null_feature_test.append(col)
        print(col)


# In[52]:


for col in null_feature:
    print(col,"\n","-"*25)
    print(data_train[col].value_counts())
    print("\n\n")


# ###LotFrontage--> median
# ###Alley--> no_alley_access
# ###MasVnrType-->mode
# ###MasVnrArea--> mode
# ####BsmtCond,BsmtQual,BsmtExposure,BsmtFinType1,BsmtFinType2-->noBasement
# $$$$BsmtExposure-->replce(NO Exposur,No basement)
# ####FireplaceQu--> No firplace
# ####GarageYrBlt-->Yearbuilt
# ###GarageFinish,GarageQual,GarageCond--> No garage
# ####PoolQC-->no pool
# ####Fence--> no fence
# ####MiscFeature--> no

# In[53]:


for col in null_feature:
    if col=="LotFrontage":
        data_train[col]=data_train[col].fillna(data_train[col].median)
    elif (col=="MasVnrArea" or col=="MasVnrType"):
        data_train[col]=data_train[col].fillna(data_train[col].mode()[0])
    elif (col=="GarageFinish" or col=="GarageQual" or col=="GarageCond"):
        data_train[col]=data_train[col].fillna("No garage")
    elif col=="Alley":
        data_train[col]=data_train[col].fillna("no_alley_access")
    elif col=="FireplaceQu":
        data_train[col]=data_train[col].fillna("no_Fireplace")
    elif col=="PoolQC":
        data_train[col]=data_train[col].fillna("no_pool")
    elif col=="Fence":
        data_train[col]=data_train[col].fillna("no_Fence")
    elif col=="MiscFeature":
        data_train[col]=data_train[col].fillna("no_other_features")
    elif col=="FireplaceQu":
        data_train[col]=data_train[col].fillna("no_Fireplace")
    elif col=="GarageYrBlt":
        data_train[col]=data_train[col].fillna(1800)
         # this may be beacuse of there is no garage so i replace it by 1800 so that in dateset variance is not more.
    else:
        data_train[col]=data_train[col].fillna("no_basement")


# In[54]:


data_train[["GarageYrBlt","GarageCars","GarageArea"]][data_train['GarageYrBlt']==1800]


# In[55]:


data_train.info()


# In[56]:


##Same we do for test data

for col in null_feature_test:
    if col=="LotFrontage":
        data_test[col]=data_test[col].fillna(data_test[col].median)
    elif (col=="MasVnrArea" or col=="MasVnrType"):
        data_test[col]=data_test[col].fillna(data_test[col].mode()[0])
    elif (col=="GarageFinish" or col=="GarageQual" or col=="GarageCond" or col=="GarageType"):
        data_test[col]=data_test[col].fillna("No garage")
    elif col=="Alley":
        data_test[col]=data_test[col].fillna("no_alley_access")
    elif col=="FireplaceQu":
        data_test[col]=data_test[col].fillna("no_Fireplace")
    elif col=="PoolQC":
        data_test[col]=data_test[col].fillna("no_pool")
    elif col=="Fence":
        data_test[col]=data_test[col].fillna("no_Fence")
    elif col=="MiscFeature":
        data_test[col]=data_test[col].fillna("no_other_features")
    elif col=="FireplaceQu":
        data_test[col]=data_test[col].fillna("no_Fireplace")
    elif col=="Electrical":
        data_test[col]=data_test[col].fillna(data_test[col].mode()[0])
    elif col=="GarageYrBlt":
        continue
    else:
        data_test[col]=data_test[col].fillna("no_basement")


# In[57]:


null_feature_test


# In[58]:


garage_null_list_test=list(np.where(data_test.GarageYrBlt.isna()))
for i in garage_null_list_test[0]:
    data_test["GarageYrBlt"].iloc[i]=1800             # this may be beacuse of there is no garage & replaced with 1800 because no more variance in the column.


# In[59]:


data_test.info()


# In[60]:


data_train.info()


# In[61]:


sns.heatmap(data_test.isna())


# In[62]:


data_train['LotFrontage'].value_counts().to_frame()


# In[63]:


sns.heatmap(data_train.isna())


# In[64]:


type(data_train['LotFrontage'].iloc[0])


# In[65]:


for i in range(1168):
    try:
        data_train["LotFrontage"].iloc[i]=int(data_train["LotFrontage"].iloc[i])
    except:
        data_train["LotFrontage"].iloc[i]=0


# In[66]:


for i in range(292):
    try:
        data_test["LotFrontage"].iloc[i]=int(data_test["LotFrontage"].iloc[i])
    except:
        data_test["LotFrontage"].iloc[i]=0


# In[67]:


data_train.LotFrontage=data_train.LotFrontage.astype(float)
data_test.LotFrontage=data_test.LotFrontage.astype(float)


# In[68]:


data_train.LotFrontage=data_train.LotFrontage.replace({0.0:data_train.LotFrontage.mean()})
data_test.LotFrontage=data_test.LotFrontage.replace({0.0:data_test.LotFrontage.mean()})


# In[69]:


data_train.describe().T


# MasVnrArea, BsmtFinSF1, BsmtFinSF2, 2ndFlrSF, LowQualFinSF, BsmtFullBath, BsmtHalfBath, HalfBath,Fireplaces,WoodDeckSF,EnclosedPorch,3SsnPorch, ScreenPorch, PoolArea, MiscVal  these such features has standard deviations greter then their mean. Lets see why.

# In[70]:


data_train['SalePrice']=data_train['SalePrice']/100000


# In[71]:


sns.scatterplot(data_train['BsmtFinSF1'],data_train['SalePrice'])


# from above graph we know that there is some linear relation between Basement Type 1 finished in sq. ft. and Sale Price of the Houses.

# In[72]:


sns.scatterplot(data_train['YearBuilt'],data_train['SalePrice'])


# from above graph we know that there is some linear relation between House built and Sale Price of the Houses.

# In[73]:


data_train.groupby("YrSold")['SalePrice'].sum().plot(kind='line')


# There is extreme loss is sale price of house in between year 2009 and 2010.

# In[74]:


plt.figure(figsize=(20,5))
data_train.groupby("YearBuilt")['SalePrice'].mean().plot(kind='line')


# Here we analyses that the more the house was built new has their total sales price increasing from 1898.

# In[75]:


plt.figure(figsize=(20,5))
data_train.groupby("YearRemodAdd")['SalePrice'].mean().plot(kind='line')


# Here we analyses that the more the house that were remodellled with increase in the years the average sales price of the houses were increased.

# In[76]:


plt.figure(figsize=(20,5))
data_train.groupby("GarageYrBlt")['SalePrice'].mean().plot(kind='line')


# The old is the garage the lower is sale Price.

# In[77]:


data_train.groupby(["YrSold","MoSold"])['SalePrice'].mean().plot(kind='line')


# In[78]:


plt.figure(figsize=(15,5))
data_train.groupby(["RoofStyle","RoofMatl"])['SalePrice'].mean().plot(kind='bar')
plt.show()


# In[79]:


num=[]
cat=[]
for col in data_train:
    if str(data_train[col].dtype)=='object':
        cat.append(col)
    else:
        num.append(col)


# In[80]:


plt.figure(figsize=(20,20))
g=1
for col in num:
    if g<=36 and col!='SalePrice':
        ax=plt.subplot(6,6,g)
        sns.scatterplot(data_train[col],data_train['SalePrice'])
        plt.xlabel(col,fontsize=15)
        plt.ylabel('SalePrice',fontsize=15)
    g+=1
plt.tight_layout()


# LotFrontage: Linear feet of street connected to property is somehow seems that there is some linear relation between them.
#     
# LotArea: Not so specific but the lot area somehow seems like to be very slight increase in the area leads to extra amount of change in increasing manner of SalesPrice.
# 
# MasVnArea: Masonry veneer area except zero area there is linear reltions that tells that increase in the masonary ares lead to increase the sales Price of the houses.
# 
# BsmtFinSF1: Basement finished Type 1 rather than zero square feet there is some min amount of increase in the area there is good amount of increase in SalePrice of House.
# 
# BsmtFinSF2: Not much impacts it showing to the target variable.
# 
# BsmtUnfSf: Not much impacts it showing to the target variable.
# 
# TotalBsmtSF: Shows that there is increase in the total areas for the basement in the house leads to increase in the SalePrice.
# 
# 1stFlrSF: In this more area in on the 1st Floor of houses will impact to increase in the sale price of the houses.
# 
# 2ndFlrSF: In this more area in on the 2nd Floor of houses will impact to increase in the sale price of the houses.
# 
# LowQualFinSF: Not much impacts it showing to the target variable.
# 
# GrLivArea: In this more increase in above grade (ground) living area will impact to increase in the sale price of the houses.
# 
# GarageArea: Except with the 0 area there is some linear relation that stats that increase in the Garage area increase the Sale price of the House.
# 
# WoodDeckSF: Except with the 0 area sq. ft. there is some linear relation that stats that increase in the Wood Deck area increase the Sale price of the House. And seems to be outliers in it.
# 
# OpenPorchSF:Except with the 0 area sq. ft. there is some linear relation that stats that increase in the open Porch area increase the Sale price of the House. And seems to be outliers in it.
# 
# Enclosed Porch, 3SnPorch,ScreenPorch, PoolArea,MiscVal these area very less founded in the houses due to which not much info we gathered from graph.

# In[81]:


sns.scatterplot(x=data_train['1stFlrSF']+data_train['2ndFlrSF'],y=data_train.SalePrice)


# 1stFlrSF,2ndFlrSF these two features after adding means total area of both 1st and 2nd floor cobinally showing the linear relations w.r.t. SalePrice of the houses.

# In[82]:


sns.scatterplot(x=data_train['OpenPorchSF']+data_train['EnclosedPorch']+data_train['ScreenPorch'],y=data_train.SalePrice)


# In[83]:


int_columns=['MSSubClass','OverallQual','OverallCond','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars']
plt.figure(figsize=(20,30))
g=1
for col in int_columns:
    if g<=24:
        ax=plt.subplot(6,4,g)
        data_train.groupby(col)['SalePrice'].mean().plot(kind='bar')
        plt.xlabel(col)
        plt.ylabel("Averge Sale Price")
        ax=plt.subplot(6,4,g+1)
        data_train[col].value_counts().plot(kind='bar',color='g')
        plt.xlabel(col)
        plt.ylabel("Counts")
    g+=2
plt.show()


# MSSubClass:  20 type(1-STORY 1946 & NEWER ALL STYLES) of dwelling involved in the sale are the most that were sold while 60 type (2-STORY 1946 & NEWER) of dwelling involved in the sale that are highlyy expensive that were sold.   $$
# 
# OverallQual: As in data description there we set a no. from 1 to 10 that specifies the quality so in accordance with that higher the quality higher the price for sale. And there are very less no. for records for quality of houses 1 7 2 that are very poor/ poor.                  $$
# 
# OverallCond: Here I analyse that the average overall condition of houses i.e. 5 were more in sale from both graph as beteer the conditions of houses their prices for sale in increasing.               $$
# 
# BsmFullBath: there are zero no. of full basement bathrrom records are  high from both for these grapd i aalysed that more the no. of basement Full bathrooms more the seles price for the house.                $$
# 
# BsmtHalfBath: from both graphs there is ionly analyse that there are more no. of houss that has no half bathroom in basement.                                                $$
# 
# FullBath: Increased in the no. of full bathrooms will result in the increase in sale price for the houses and there are very less records for the the O & 3 no. of bathrooms.
# 
# HalfBath: there are very no. of houses that has 2 no. of Half bathrooms but the average sale price for having only 1 halfbathroom is maximum.
# 
# BedrooomAbvGr: there no such impacts for the no. of bedroooms to the salePrice but there very less no of records for 6,0,8 no. of bedroooms that lead model to underfiited.                 $$
# 
# KitchenAbvGr: very less records for the houses having 3 or no kitchen above gorund and the sale price are high for having only 1 kitchen in the house.              $$
# 
# TotRmsAbvGrd: here we analyse that the salePrice is increasing w.r.t. increase in the no. of rooms above ground while there are very less records for having 2 & 14 no. of rooms in the house.
# 
# Fireplaces: Increased no. of fireplaces is impacting the increase in the SalePrice of the houses.         $$
# 
# GarageCars: More No. of cars capacity in the house more its sale Price.

# In[85]:


plt.figure(figsize=(20,60))
g=1
for col in cat:
    if g<=84:
        ax=plt.subplot(14,6,g)
        data_train.groupby(col)['SalePrice'].mean().plot(kind='bar')
        plt.xlabel(col)
        plt.ylabel("Averge Sale Price")
        plt.xticks(rotation=50)
        ax=plt.subplot(14,6,g+1)
        data_train[col].value_counts().plot(kind='bar',color='g')
        plt.xlabel(col)
        plt.ylabel("Counts")
        plt.xticks(rotation=50)
    g+=2
plt.tight_layout(pad=0.5)


# MSZoning:Floating Village Residential(FV) has highest average Sale Price of the houses and RL(Residential Low Density) types of Zoning has highest types of houses and very low amount of houses for commercial(C) types of Zoning.
#     
# Street: Paved Type of road access to property has highest average Sale Prices for the houses and Low amount of houses are there that are Gravel Type of road access to property. 
# 
# Alley: No alley access types of such houses are very common and their prices for Sale is highest.
# 
# Lot Shape : Not much info we gathered from Lo Shape but Moderate type of irregularities has highest averge sale Price while Irregular type of records are not much available in our dataset.
# 
# LandContour: HLS(Hillside - Significant slope from side to side) has highest Sales Price while Flat type buildings has higher o. of records in the dataaset.
# 
# LotConfig: Cul-de-sac & Frontage on 3 sides of property(CullDSac & Fr3) having hihest Sale Prices for the houses while such type of houses are raely available.
# 
# LandSlope: No such impact n the SalePrice for the houses while Gentle Slope(Gtl) has highest no. of records among them.
# 
# Neighborhood: North Ames (NAmes) such type houses are mostly available.
# 
# Condition1: Its shows the Proximity to various conditions in which we found that RRNn & PosA (Within 200' of North-South Railroad and Adjacent to postive off-site feature) sch type has highest average SalePrices for the houses.Whilr Norm(NOrmal0 type of houses are easily available for Sale.
# 
# Condition2: It is Proximity to various conditions in which PosA(Adjacent to postive off-site feature) types has highest SalePrice while Norm i.e., Normal proximity to various condition such type are easily available in Sale.
# 
# Bldg Type: it is type of dwelling in the house in which Single-family Detached & Townhouse End Unit	(1Farm & TwnhsE) has highest SalePrice and 2FmCon	type  are rarely found to Sale.
# 
# HouseStyle: Two and one-half story: 2nd level finished(2.5 Fin)  such houses are rarely available to sale and has highest prices for Sale.
# 
# RoofStyle: Shed type of roof in the houses are very rarely available for sale and are expensive.
# 
# RoofMatl: Wood Shingles type of material used for roofs in the houses are rarely available and such houses are expensive.
# 
# Exterior1st: Exterior covering on house in that Imitation Stucco of exterior used in the houses are rarely available and such houses are costly.
# 
# Exterior2nd: Other type Exterior covering on house are rarely availavle in the sale are are costly such houses.
# 
# MasVnrType: Stone type Masonry veneer type of houses are expensive and Brick Common type of masonary venner type of houses are rarely available in the Sales.
# 
# ExterQual: Evaluates the quality of the material on the exterior  in which Excellent quality of houses are raely available and are expensive ones in the Sale.
# 
# ExterCond: Evaluates the present condition of the material on the exterior in which Excellent quality of houses are raely available and are expensive ones in the Sale.
# 
# Foundation: Poured Contrete Type of foundation are costly in Sale of houses while wood type are rarely available in Sale of Houses.
# 
# BsmtQual: Excellent Quality of basement in the houses are expensive while fair type of basement quality are rarely available in the sale of houses.
# 
# BsmtCond: The basement condition is good then such houses are exensive while Poor type of basement Condition are rare.
# 
# Basement Exposure: The exposure of the basement is better their prices for sale is better while No basement in houses are very easy eaily vailable in the sale.
# 
# 
# BsmtFinType1: Better the Living Quarters better the sale Prices for the houses and there bery less houses that doen't have basement in it.
# 
# BsmtFinType2: Better the Living Quarters better the sale Prices for the houses and mostly houses have unfinished basement in it.
# 
# Heating: GasA(Gas forced warm air furnace) type of heating in the houses are mostly in the houses and such type of heating container has high sale Price.
# 
# HeatingQC: Better the heating Quality and its conditions better the Sale Price for the houses.
# 
# 
# CentralAir : Mostly houses has Central Air in it and are Costly.
#     
# Electrical: Mostly houses has Electical system of SBrkr(Standard Circuit Breakers & Romex) in it and are Costly.  
# 
# KitchenQual: Better the quality of kitchen of the houses better their Sale Price.
# 
# Functional: Home functionality Typ(typical) type are in most of houses and such houses have high salePrice.
# 
# FirePlaceQual: Better the quality for fire place having higher Sale Prices for such houses.
# 
# GarageType: Builtin Garage types of houses are expensive in Sale Price  and 2Types are rarely available for sale.
# 
# Garage finish: better the finishing of garagee better the Sale Price and vary rare houses that has not Garage in it.
# 
# Garage Qual: Better the quality of garage in the houses higher the prices of houses. And mostly houses their garage quality is typical/Average.
# 
# GarageCond: Good and Typical/Average type of garage condition in the houses has hhigher SalePrice of the houses.
# 
# Paved drive: Paved(Y) driveway has highest average Sale Price of the houses and mostly founded during Sale of houses.
# 
# PoolQC: Better the qaulity and conditions for the pool of the houses better the Sale Price of the houses. Most of houses hasn't have pool in it.
# 
# Fence: Fence is not impacting much to Sle Price for the houses but mostly houses doen't have fencing in it.
# 
# MiscFeature: Miscellaneous feature not covered in other categories in which houses Tennis Court is available such houss has higher Sale Pice. Mostly houses hases has no ther miscellaneaous features in it.
# 
# SaleType: con(Contract 15% Down payment regular terms) and New(Home just constructed and sold) such types of sales has highest SalePrices. mostly houses are those whose Sale type WD (Warranty Deed - Conventional).
# 
# Sale COndition: Partial	[Home was not completed when last assessed (associated with New Homes)] such condition of Sales are having High Sale prices for the houses. While most of houses has normal Sale Condition.
# 
# 
# 

# In[86]:


for col in cat:
    print(f"Training set:\n----------\n{data_train[col].value_counts()}\n Test set: \n--------\n{data_test[col].value_counts()}\n-----------------------------------\n\n")


# In[87]:


replace_type=[]
for col in cat:
    if len(data_test[col].unique().tolist())!=len(data_train[col].unique().tolist()):
        print(col)
        replace_type.append(col)
    else:
        continue


# In[88]:


label_type=[]
for col in cat:
    if col in replace_type:
        continue
    else:
        label_type.append(col)


# In[89]:


label_type=[]
for col in cat:
    if col in replace_type:
        continue
    else:
        label_type.append(col)


# In[90]:


data_train['MSZoning']=data_train['MSZoning'].replace({"RL":0,"RM":1,"FV":2,"RH":3,"C (all)":4})

data_train['Neighborhood']=data_train['Neighborhood'].replace({
    "NAmes":0,"CollgCr" : 1,"OldTown" :2,"Edwards":3,
    "Somerst" :    4,"Gilbert" :    5,"NridgHt" :    6,"Sawyer" :     7,
    "NWAmes" :     8,"SawyerW":     9,"BrkSide" :    10,"Crawfor" :    11,"NoRidge" :    12,
    "Mitchel" :    13,'IDOTRR'  :    14,'Timber'  :    15,'ClearCr':     16,'SWISU'  :     17,'StoneBr' :    18,
    'Blmngtn' :    19,'BrDale' :     20,'MeadowV' :21,'Veenker':22,'NPkVill' :     23,'Blueste' :     23
})

data_train['Condition1']=data_train['Condition1'].replace({
    "Norm":0,"Feedr":1,"Artery":2,"RRAn":3,'PosN':4,'RRAe':5,'PosA' :6,'RRNn':7,'RRNe':7
})

data_train['Condition2']=data_train['Condition2'].replace({
    "Norm":0,"Feedr":1,"Artery":2,"RRAn":2,'PosN':2,'RRAe':2,'PosA' :2,'RRNn':2             
    #their valu counts are less So I replaced with same no. for such values we says of other categories
})

data_train['RoofStyle']=data_train['RoofStyle'].replace({
    "Gable" :0 ,'Hip': 1,'Flat' :2,'Gambrel' :3,'Mansard' :4,'Shed' :5
})

data_train['RoofMatl']=data_train['RoofMatl'].replace({
    'CompShg' :0,'Tar&Grv':1,'WdShngl':2,'WdShake':3,'Roll':4,'ClyTile':4,'Metal':4,'Membran':4
    #their valu counts are less So I replaced with same no. for such values we says of other categories
})

data_train['Exterior1st']=data_train['Exterior1st'].replace({
    'VinylSd':0,'HdBoard':1,'MetalSd':2,'Wd Sdng':3,'Plywood':4,'CemntBd':5,'BrkFace':6,'Stucco':7,
    'WdShing':8,'AsbShng':9,'Stone':10,'AsphShn':10,'ImStucc':10,'BrkComm':10
    #their valu counts are less So I replaced with same no. for such values we says of other categories
})

data_train['Exterior2nd']=data_train['Exterior2nd'].replace({
    'VinylSd':0,'MetalSd':1,'HdBoard':2,'Wd Sdng':3,'Plywood':4,'CmentBd':5,'Wd Shng':6,
    'Stucco':7,'BrkFace':8,'AsbShng':9,'ImStucc':10,'Brk Cmn':11,'Stone':12,'AsphShn':13,"CBlock":14,'Other':14
})


data_train['Heating']=data_train['Heating'].replace({
    'GasA':0,'GasW':1,'Grav' :2,'Wall':3,'Floor' :4,'OthW':5
})

data_train['Electrical']=data_train['Electrical'].replace({
    'SBrkr':0,'FuseA':1,'FuseF':2,'FuseP':3,'Mix':4
})

data_train['Functional']=data_train['Functional'].replace({
    'Typ':0,'Min2':1,'Min1':2,'Mod':3,'Maj1':4,'Maj2':5,'Sev':6
})

data_train['MiscFeature']=data_train['MiscFeature'].replace({
    'no_other_features':0,'Shed':1,'Gar2':2,'TenC':3,'Othr':4
})

data_train['SaleType']=data_train['SaleType'].replace({
    'WD':0,'New':2,'COD':3,'ConLD':7,'ConLI':6,'ConLw':5,'Oth':8,'CWD':1,'Con':4
})


# In[91]:


data_train['ExterCond']=data_train['ExterCond'].replace({
    'TA':2,'Gd' :3,'Fa':1,'Ex':4,'Po' :0
})

data_train['ExterQual']=data_train['ExterQual'].replace({
    'TA':2,'Gd' :3,'Fa':1,'Ex':4,'Po' :0
})

data_train['BsmtQual']=data_train['BsmtQual'].replace({
    'TA':3,'Gd' :4,'Fa':2,'Ex':5,'Po' :1,"no_basement":0
})

data_train['KitchenQual']=data_train['KitchenQual'].replace({
    'TA':2,'Gd' :3,'Fa':1,'Ex':4,'Po' :0
})

data_train['BsmtCond']=data_train['BsmtCond'].replace({
    'TA':3,'Gd' :4,'Fa':2,'Ex':5,'Po' :1,"no_basement":0
})

data_train['BsmtExposure']=data_train['BsmtExposure'].replace({
    'Mn':2,'Gd' :4,'No':1,'Av':3,'no_basement' :0
})

data_train['HeatingQC']=data_train['HeatingQC'].replace({
    'TA':2,'Gd' :3,'Fa':1,'Ex':4,'Po' :0
})


data_train['PoolQC']=data_train['PoolQC'].replace({
    'no_pool':0,'Gd':2,'Ex':3,'Fa':1
})

data_train['SaleCondition']=data_train['SaleCondition'].replace({
    'Normal':0,'Partial':5,'Abnorml':1,'Family'  :   4,'Alloca':3,'AdjLand':2
})

data_train['BsmtFinType1']=data_train['BsmtFinType1'].replace({
    'GLQ':6,'ALQ' :5,'BLQ':4,'Rec':3,'LwQ':2,'no_basement':0,'Unf' :1
})

data_train['BsmtFinType2']=data_train['BsmtFinType2'].replace({
    'GLQ':6,'ALQ' :5,'BLQ':4,'Rec':3,'LwQ':2,'no_basement':0,'Unf' :1
})

data_train['LandSlope']=data_train['LandSlope'].replace({
    'Gtl':2,'Mod' :1,'Sev':0
})

data_train['FireplaceQu']=data_train['FireplaceQu'].replace({
    'TA':3,'Gd' :4,'Fa':2,'Ex':5,'Po' :1,'no_Fireplace':0
})

data_train['GarageQual']=data_train['GarageQual'].replace({
    'TA':3,'Gd' :4,'Fa':2,'Ex':5,'Po' :1,'No garage':0
})

data_train['GarageCond']=data_train['GarageCond'].replace({
    'TA':3,'Gd' :4,'Fa':2,'Ex':5,'Po' :1,'No garage':0
})

data_train['PavedDrive']=data_train['PavedDrive'].replace({
    'Y':2,'P':1,'N':0
})


# In[92]:


label_type.remove('PavedDrive')
label_type.remove('GarageCond')
label_type.remove('GarageQual')
label_type.remove('FireplaceQu')

label_type.remove('KitchenQual')
label_type.remove('BsmtFinType2')
label_type.remove('BsmtFinType1')
label_type.remove('BsmtExposure')

label_type.remove('BsmtQual')
label_type.remove('LandSlope')


# In[93]:


data_test['MSZoning']=data_test['MSZoning'].replace({"RL":0,"RM":1,"FV":2,"RH":3,"C (all)":4})

data_test['Neighborhood']=data_test['Neighborhood'].replace({
    "NAmes":0,"CollgCr" : 1,"OldTown" :2,"Edwards":3,
    "Somerst" :    4,"Gilbert" :    5,"NridgHt" :    6,"Sawyer" :     7,
    "NWAmes" :     8,"SawyerW":     9,"BrkSide" :    10,"Crawfor" :    11,"NoRidge" :    12,
    "Mitchel" :    13,'IDOTRR'  :    14,'Timber'  :    15,'ClearCr':     16,'SWISU'  :     17,'StoneBr' :    18,
    'Blmngtn' :    19,'BrDale' :     20,'MeadowV' :21,'Veenker':22,'NPkVill' :     23,'Blueste' :     23
})

data_test['Condition1']=data_test['Condition1'].replace({
    "Norm":0,"Feedr":1,"Artery":2,"RRAn":3,'PosN':4,'RRAe':5,'PosA' :6,'RRNn':7,'RRNe':7
})

data_test['Condition2']=data_test['Condition2'].replace({
    "Norm":0,"Feedr":1,"Artery":2,"RRAn":2,'PosN':2,'RRAe':2,'PosA' :2,'RRNn':2             
    #their valu counts are less So I replaced with same no. for such values we says of other categories
})

data_test['RoofStyle']=data_test['RoofStyle'].replace({
    "Gable" :0 ,'Hip': 1,'Flat' :2,'Gambrel' :3,'Mansard' :4,'Shed' :5
})

data_test['RoofMatl']=data_test['RoofMatl'].replace({
    'CompShg' :0,'Tar&Grv':1,'WdShngl':2,'WdShake':3,'Roll':4,'ClyTile':4,'Metal':4,'Membran':4
    #their valu counts are less So I replaced with same no. for such values we says of other categories
})

data_test['Exterior1st']=data_test['Exterior1st'].replace({
    'VinylSd':0,'HdBoard':1,'MetalSd':2,'Wd Sdng':3,'Plywood':4,'CemntBd':5,'BrkFace':6,'Stucco':7,
    'WdShing':8,'AsbShng':9,'Stone':10,'AsphShn':10,'ImStucc':10,'BrkComm':10,'CBlock':10
    #their valu counts are less So I replaced with same no. for such values we says of other categories
})

data_test['Exterior2nd']=data_test['Exterior2nd'].replace({
    'VinylSd':0,'MetalSd':1,'HdBoard':2,'Wd Sdng':3,'Plywood':4,'CmentBd':5,'Wd Shng':6,
    'Stucco':7,'BrkFace':8,'AsbShng':9,'ImStucc':10,'Brk Cmn':11,'Stone':12,'AsphShn':13,"CBlock":14,'Other':14
})


data_test['Heating']=data_test['Heating'].replace({
    'GasA':0,'GasW':1,'Grav' :2,'Wall':3,'Floor' :4,'OthW':5
})

data_test['Electrical']=data_test['Electrical'].replace({
    'SBrkr':0,'FuseA':1,'FuseF':2,'FuseP':3,'Mix':4
})

data_test['Functional']=data_test['Functional'].replace({
    'Typ':0,'Min2':1,'Min1':2,'Mod':3,'Maj1':4,'Maj2':5,'Sev':6
})

data_test['MiscFeature']=data_test['MiscFeature'].replace({
    'no_other_features':0,'Shed':1,'Gar2':2,'TenC':3,'Othr':4
})

data_test['SaleType']=data_test['SaleType'].replace({
    'WD':0,'New':2,'COD':3,'ConLD':7,'ConLI':6,'ConLw':5,'Oth':8,'CWD':1,'Con':4
})


# In[94]:


data_test['ExterCond']=data_test['ExterCond'].replace({
    'TA':2,'Gd' :3,'Fa':1,'Ex':4,'Po' :0
})

data_test['ExterQual']=data_test['ExterQual'].replace({
    'TA':2,'Gd' :3,'Fa':1,'Ex':4,'Po' :0
})

data_test['BsmtQual']=data_test['BsmtQual'].replace({
    'TA':3,'Gd' :4,'Fa':2,'Ex':5,'Po' :1,"no_basement":0
})

data_test['KitchenQual']=data_test['KitchenQual'].replace({
    'TA':2,'Gd' :3,'Fa':1,'Ex':4,'Po' :0
})

data_test['BsmtCond']=data_test['BsmtCond'].replace({
    'TA':3,'Gd' :4,'Fa':2,'Ex':5,'Po' :1,"no_basement":0
})

data_test['BsmtExposure']=data_test['BsmtExposure'].replace({
    'Mn':2,'Gd' :4,'No':1,'Av':3,'no_basement' :0
})

data_test['HeatingQC']=data_test['HeatingQC'].replace({
    'TA':2,'Gd' :3,'Fa':1,'Ex':4,'Po' :0
})


data_test['PoolQC']=data_test['PoolQC'].replace({
    'no_pool':0,'Gd':2,'Ex':3,'Fa':1
})

data_test['SaleCondition']=data_test['SaleCondition'].replace({
    'Normal':0,'Partial':5,'Abnorml':1,'Family'  :   4,'Alloca':3,'AdjLand':2
})

data_test['BsmtFinType1']=data_test['BsmtFinType1'].replace({
    'GLQ':6,'ALQ' :5,'BLQ':4,'Rec':3,'LwQ':2,'no_basement':0,'Unf' :1
})

data_test['BsmtFinType2']=data_test['BsmtFinType2'].replace({
    'GLQ':6,'ALQ' :5,'BLQ':4,'Rec':3,'LwQ':2,'no_basement':0,'Unf' :1
})

data_test['LandSlope']=data_test['LandSlope'].replace({
    'Gtl':2,'Mod' :1,'Sev':0
})

data_test['FireplaceQu']=data_test['FireplaceQu'].replace({
    'TA':3,'Gd' :4,'Fa':2,'Ex':5,'Po' :1,'no_Fireplace':0
})

data_test['GarageQual']=data_test['GarageQual'].replace({
    'TA':3,'Gd' :4,'Fa':2,'Ex':5,'Po' :1,'No garage':0
})

data_test['GarageCond']=data_test['GarageCond'].replace({
    'TA':3,'Gd' :4,'Fa':2,'Ex':5,'Po' :1,'No garage':0
})

data_test['PavedDrive']=data_test['PavedDrive'].replace({
    'Y':2,'P':1,'N':0
})


# In[95]:


encode=LabelEncoder()
for col in label_type:
    data_train[col]=encode.fit_transform(data_train[col])
    data_test[col]=encode.fit_transform(data_test[col])


# In[96]:


data_test.info()


# In[97]:


data_train.info()


# In[98]:


int_columns


# In[99]:


float_type=[]
for i in num:
    if i in int_columns:
        continue
    else:
        float_type.append(i)


# In[100]:


plt.figure(figsize=(20,20))
g=1
for col in float_type:
    if g<=24:
        ax=plt.subplot(4,6,g)
        sns.boxenplot(data_train[col],orient='v')
        plt.xlabel(col)
    g+=1
plt.show()


# In[101]:


outliers_det_train=['LotArea','BsmtFinSF1','BsmtFinSF2','TotalBsmtSF','1stFlrSF','GrLivArea','EnclosedPorch','3SsnPorch','MiscVal']


# In[102]:


plt.figure(figsize=(20,20))
g=1
for col in float_type:
    if g<=24:
        ax=plt.subplot(4,6,g)
        sns.boxenplot(data_test[col],orient='v')
        plt.xlabel(col)
    g+=1
plt.show()


# In[103]:


outliers_det_test=['LotArea','BsmtFinSF2','TotalBsmtSF','1stFlrSF','EnclosedPorch','3SsnPorch','MiscVal']


# In[104]:


data_train[float_type].describe().T


# In[105]:


float_type.remove("SalePrice")


# In[106]:


data_test[float_type].describe().T


# In[107]:


data_train['MasVnrArea']=np.sqrt(data_train['MasVnrArea'])
data_train['BsmtFinSF1']=np.sqrt(data_train['BsmtFinSF1'])
data_train['BsmtFinSF2']=np.sqrt(data_train['BsmtFinSF2'])
data_train['2ndFlrSF']=np.sqrt(data_train['2ndFlrSF'])
data_train['LowQualFinSF']=np.sqrt(data_train['LowQualFinSF'])
data_train['WoodDeckSF']=np.sqrt(data_train['WoodDeckSF'])
data_train['OpenPorchSF']=np.sqrt(data_train['OpenPorchSF'])
data_train['EnclosedPorch']=np.sqrt(data_train['EnclosedPorch'])
data_train['3SsnPorch']=np.sqrt(data_train['3SsnPorch'])
data_train['ScreenPorch']=np.sqrt(data_train['ScreenPorch'])
data_train['PoolArea']=np.sqrt(data_train['PoolArea'])
data_train['MiscVal']=np.sqrt(data_train['MiscVal'])


# In[108]:


data_test['MasVnrArea']=np.sqrt(data_test['MasVnrArea'])
data_test['BsmtFinSF1']=np.sqrt(data_test['BsmtFinSF1'])
data_test['BsmtFinSF2']=np.sqrt(data_test['BsmtFinSF2'])
data_test['2ndFlrSF']=np.sqrt(data_test['2ndFlrSF'])
data_test['LowQualFinSF']=np.sqrt(data_test['LowQualFinSF'])
data_test['WoodDeckSF']=np.sqrt(data_test['WoodDeckSF'])
data_test['OpenPorchSF']=np.sqrt(data_test['OpenPorchSF'])
data_test['EnclosedPorch']=np.sqrt(data_test['EnclosedPorch'])
data_test['3SsnPorch']=np.sqrt(data_test['3SsnPorch'])
data_test['ScreenPorch']=np.sqrt(data_test['ScreenPorch'])
data_test['PoolArea']=np.sqrt(data_test['PoolArea'])
data_test['MiscVal']=np.sqrt(data_test['MiscVal'])


# In[71]:


plt.figure(figsize=(20,40))
g=1
for col in data_train.drop("SalePrice",axis=1):
    if g<=80:
        ax=plt.subplot(8,10,g)
        sns.distplot(data_train[col],color="green")
        plt.xlabel(col,fontsize=15)
    g+=1
plt.show()


# In[72]:


data_train.skew().to_frame()


# In[73]:


plt.figure(figsize=(20,40))
g=1
for col in data_test:
    if g<=80:
        ax=plt.subplot(8,10,g)
        sns.distplot(data_train[col],color="green")
        plt.xlabel(col,fontsize=15)
    g+=1
plt.show()


# In[109]:


data_test.skew().to_frame()


# In[75]:


plt.figure(figsize=(50,50))
sns.heatmap(data_train.corr(),vmax=1,vmin=-1,annot=True,annot_kws={"size":10},cmap="PiYG")
plt.tight_layout()


# Exterior1st & Exterior2nd   ---> 80% collinear to each other.
# 
# GrLivArea & TotRmsAbvGrd are 82% collinear to each other.
# 
# TotalBsmtSF&1stFlrSF--->81% collinear with each other.
# 
# BsmtFinSF2&BsmtFinType2 ----> -81% collinear.
# 
# BsmtFinSF1&BsmtFinType1 ----> -73% collinear.
# 
# GarageCars & GarageArea  ----> 88% collinear to each other.
# 
# FirePlace&FireQual are 72% collinear to each other.
# 
# MiscFeature & MiscVal are 78% collinear to each other.
# 
# PoolArea & PoolQC are collinear about -93%.
# 
# 

# In[76]:


sns.regplot(data_train['Exterior1st'],data_train['Exterior2nd'])


# There seems to be some collinearity but its also deviated too.

# In[77]:


sns.regplot(data_train['GrLivArea'],data_train['TotRmsAbvGrd'])


# Above these two feature are co-linear. Its beter to remove one of these features

# In[78]:


sns.regplot(data_train['TotalBsmtSF'],data_train['1stFlrSF'])


# Above these two feature are co-linear. Its beter to remove one of these features

# In[79]:


sns.regplot(data_train['BsmtFinSF2'],data_train['BsmtFinType2'])


# There seems to be some collinearity but its also deviated too.

# In[80]:


sns.regplot(data_train['BsmtFinSF1'],data_train['BsmtFinType1'])


# There seems to be some collinearity but its also deviated too.

# In[81]:


sns.regplot(data_train['GarageArea'],data_train['GarageCars'])


# Above these two feature are co-linear. Its beter to remove one of these features

# In[82]:


sns.regplot(data_train['FireplaceQu'],data_train['Fireplaces'])


# There seems to be some collinearity but its also deviated too.

# In[83]:


sns.regplot(data_train['MiscFeature'],data_train['MiscVal'])


# There seems to be some collinearity but its also deviated too.

# In[111]:


plt.figure(figsize=(20,5))
data_train.drop("SalePrice",axis=1).corrwith(data_train.SalePrice).plot(kind='bar')
plt.tight_layout()


# #### Data Cleaning

# In[112]:


data_train.drop_duplicates()


# In[113]:


data_test.describe()


# In[114]:


data_train.describe()


# In[115]:


data_train.drop("SalePrice",axis=1).corrwith(data_train.SalePrice)[5:-5]


# ##### these features are giving very less impact to predict the Sales for the houses.
# LandSlope  ->-1.5%
# Condition1->1%
# Condition2->0.8%
# MasVnrType->0.7%
# BsmtFinType2-> 0.7%
# BsmtHalfBath-> 1.1%
# 
# TotRmsAbvGrd is having co-relation with target variable of 52% while GrLivArea  is having co-relation with the label of 70% and these two features are co-related.So, TotRmsAbvGrd this feature we drop.
# 
# 1stFlrSF is 58% related & TotalBsmtSF is 59% related with label so dropping 1stFlrSF features.
# 
# GarageCars is realted  62% while GarageArea is 61.9% related with the label. So its better to drop Garage Area.
# 
# 
# 

# In[116]:


drop_columns=['LandSlope', 'Condition1','Condition2','MasVnrType','BsmtFinType2','BsmtHalfBath','TotRmsAbvGrd','1stFlrSF','GarageArea','PoolArea']


# In[117]:


data_train.drop(columns=drop_columns,axis=1,inplace=True)
data_test.drop(columns=drop_columns,axis=1,inplace=True)


# In[118]:


data_train.drop(columns=["BsmtFinSF1","BsmtFinSF2","BsmtUnfSF"],inplace=True)
data_test.drop(columns=["BsmtFinSF1","BsmtFinSF2","BsmtUnfSF"],inplace=True)

#AS we seen in pre-processing that the sum of Basement finished of type1 and type2 and Unfinised basement areas is eqaul to the sum of toal Basement area of the house.


# In[119]:


outliers_det_train


# In[120]:


outliers_det_train.remove('BsmtFinSF1')
outliers_det_train.remove('BsmtFinSF2')
outliers_det_train.remove('1stFlrSF')

outliers_det_test.remove('BsmtFinSF2')
outliers_det_test.remove('1stFlrSF')


# #### Removing Outliers

# In[121]:


## Removing Outliers using z-score to both data set.

zs=zscore(data_train[outliers_det_train])
filtered=(np.abs(zs)<3).all(axis=1)
data_train=data_train[filtered]

zs=zscore(data_test[outliers_det_test])
filtered=(np.abs(zs)<3).all(axis=1)
data_test=data_test[filtered]


# In[122]:


data_train.shape


# In[123]:


(1168-1066)/1168


# ##### Reducing skewness

# In[124]:


data_train.skew()[5:-5]


# In[125]:


data_train.skew()


# In[126]:


skewed_col_train=[]
for col in data_train:
    if (data_train[col].skew()>0.5) or (data_train[col].skew()<-0.5):
        skewed_col_train.append(col)
    else:
        continue

skewed_col_test=[]
for col in data_train:
    if (data_train[col].skew()>0.5) or (data_train[col].skew()<-0.5):
        skewed_col_test.append(col)
    else:
        continue


# In[127]:


for col in skewed_col_train:
    if col not in data_train.columns.tolist():
        skewed_col_train.remove(col)

for col in skewed_col_test:
    if col not in data_test.columns.tolist():
        skewed_col_test.remove(col)


# In[128]:


skewed_col_train.remove("SalePrice")


# In[129]:


## Reducing skewness to both dataset

from sklearn.preprocessing import PowerTransformer
scalar=PowerTransformer(method='yeo-johnson')


data_train[skewed_col_train]=scalar.fit_transform(data_train[skewed_col_train].values)
data_train[skewed_col_train].head()


data_test[skewed_col_test]=scalar.fit_transform(data_test[skewed_col_test].values)
data_test[skewed_col_test].head()


# In[130]:


data_train[skewed_col_train].head()


# In[131]:


data_train.skew().to_frame()


# In[132]:


data_train.skew()[5:-5]


# Street,CentralAir,3SsnPorch  No changes in the skewness while others features are in range og 0.5 to -0.5 or else their skeness reduced.

# In[133]:


data_train.Street.value_counts()


# In[134]:


data_train.CentralAir.value_counts()


# In[135]:


data_train['3SsnPorch'].value_counts()


# In[136]:


data_train['PoolQC'].value_counts()


# In[137]:


data_train.drop(columns=['Street','3SsnPorch','PoolQC'],inplace=True)
data_test.drop(columns=['Street','3SsnPorch','PoolQC'],inplace=True)


# In[138]:


data_test.shape


# In[139]:


data_train.shape


# In[140]:


## Data loss:
print(f"Data Loss for training dataset after Cleaning:{round(((1168-1066)/1168)*100,2)}%")

print(f"Data Loss for test dataset after Cleaning:{round(((292-267)/292)*100,2)}%")


# In[114]:


print("Features Dropped:",80-62)


# ### Model Deployment

# In[141]:


X=data_train.drop("SalePrice",axis=1)
Y=data_train['SalePrice']


# splitting data into fetures & labels.
# 
# Here X is our features & Y is our label.

# In[143]:


pca=PCA()
pca.fit_transform(X)


# In[144]:


plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Pricipal Components")
plt.ylabel("Variance covered")
plt.title("PCA")
plt.tight_layout()


# In[146]:


new_pca_train=PCA(n_components=20)
new_pcom_train=new_pca_train.fit_transform(X)
pcomp_train=pd.DataFrame(new_pcom_train,columns="PCA1 PCA2 PCA3 PCA4 PCA5 PCA6 PCA7 PCA8 PCA9 PCA10 PCA11 PCA12 PCA13 PCA14 PCA15 PCA16 PCA17 PCA18 PCA19 PCA20".split(" "))
pcomp_train


# In[147]:


new_pca_test=PCA(n_components=50)
new_pcom_test=new_pca_train.fit_transform(data_test)
pcomp_test=pd.DataFrame(new_pcom_test,columns="PCA1 PCA2 PCA3 PCA4 PCA5 PCA6 PCA7 PCA8 PCA9 PCA10 PCA11 PCA12 PCA13 PCA14 PCA15 PCA16 PCA17 PCA18 PCA19 PCA20".split(" "))
pcomp_test


# In[ ]:





# In[148]:


## Normalizing Both train &test dataset.

scalar=StandardScaler()
X_=scalar.fit_transform(pcomp_train)                  #Train Features

X__=scalar.fit_transform(pcomp_test)         #Test Features


# In[149]:


Linear=LinearRegression()
DecisionTree=DecisionTreeRegressor()
RandomForest=RandomForestRegressor()
AdaBoost=AdaBoostRegressor()
Bagging=BaggingRegressor()
knn=KNeighborsRegressor()
GB=GradientBoostingRegressor()
xgb_=xgb.XGBRegressor()
SVM=SVR()
algo=[Linear,DecisionTree,Bagging,RandomForest,AdaBoost,SVM,xgb_,GB]


# In[150]:


model_acc_rs={}
maximum_acc=[]
for model in algo:
    max_accuracy=0
    for i in range(100,300,3):
        X_train,X_test,Y_train,Y_test=train_test_split(X_,Y,test_size=0.2,random_state=i)
        model.fit(X_train,Y_train)
        Y_pred=model.predict(X_test)
        accuracy=r2_score(Y_test,Y_pred)*100
        if accuracy>max_accuracy:
            max_accuracy=accuracy
            rs=i
            mae=mean_absolute_error(Y_test,Y_pred)
            mse=mean_squared_error(Y_test,Y_pred)
            rmse=np.sqrt(mean_squared_error(Y_test,Y_pred))
    maximum_acc.append(max_accuracy)
    model_acc_rs[model]=[max_accuracy,rs]
    print(f"\n\n{model}:\n---------------------------\n---------------------------\n")
    print(f"The highest accuracy is {max_accuracy} of model {model} at random state {rs}")


    print("\nMEAN ABSOLUTE ERROR:",mae)

    print(f"\nMEAN SQUARED ERROR for the model:",mse)
    
    print(f"\nROOT MEAN SQUARED ERROR for the model:",rmse)


# In[151]:


CVmodel={}

for model in algo:
    CVscore_={}
    print(f"\n{model}")
    print("-"*25)
    print("\n")
    for i in range(3,10):
        cvS=cross_val_score(model,X_,Y,cv=i)
        CVscore_[i]=cvS.mean()
        print(f"Mean CV Score of model {model}:: {cvS.mean()} at k-fold::{i}\n")
    CVdata=pd.DataFrame(CVscore_,index=[""])
    CVmodel[str(model)]=CVdata.max(axis=1).tolist()


# In[152]:


CVmodel


# Showing maximun cvScore for each model.

# In[153]:


m=list(CVmodel.keys())

print("The least difference between the accuracy and CV score of each model is::\n")
for i in range(len(algo)):
    print(f"{m[i]}::{round(np.abs(CVmodel[m[i]][0]*100-maximum_acc[i]),2)}")


# here for model RandomForestRegressor we get the least value i.e. the difference between the accuracy and cvScore of this model is 3.64.

# In[154]:


X_train,X_test,Y_train,Y_test=train_test_split(X_,Y,test_size=0.2,random_state=232)


# ### Plotting regplot graph for Random Forest model

# In[155]:


RandomForest.fit(X_train,Y_train)
Y_pred=RandomForest.predict(X_test)
sns.regplot(Y_test,Y_pred)
plt.xlabel("Actual Test Data")
plt.ylabel("Predicted Data")
plt.tight_layout()


# Here we analyse that the data points are close to the best fit line. That means the residual is less.

# #### Regularization

# In[156]:


lasso=Lasso()
parameters={"alpha":np.arange(0.001,0.1,0.002),'normalize':[True,False],'max_iter':[1000,1250,750]}
grid=GridSearchCV(lasso,parameters)
grid.fit(X_train,Y_train)
grid.best_params_


# In[157]:


lasso=Lasso(alpha= 0.003, max_iter= 1000, normalize=False)
lasso.fit(X_train,Y_train)
pred=lasso.predict(X_test)
r2_score(Y_test,pred)*100


# Here we can see that our model is not overfitted or underfitted as the r2 scoe Lasso model is 79.44 while the Random Forest Regressor r2-score is 86.9

# ### Hyperparameter Tuning

# In[162]:


reg=RandomForestRegressor()
param={
    "n_estimators":[95,100,117],
    "min_samples_split":[2,4,6],
    "min_samples_leaf":[1,3,5],
    "max_depth":[25,21,14,7]
    
}
grd=GridSearchCV(reg,param_grid=param)
grd.fit(X_train,Y_train)
print("Best Pramaeters:",grd.best_params_)

reg=grd.best_estimator_   #reinstantiating the best parameter to algo

reg.fit(X_train,Y_train)
ypred=reg.predict(X_test)

print(f"The accuracy is {round(r2_score(ypred,Y_test)*100,2)}% of model Random Forest.")


print("\nMEAN ABSOLUTE ERROR:",round(mean_absolute_error(ypred,Y_test),2))

print(f"\nMEAN SQUARED ERROR for the model:",round(mean_squared_error(ypred,Y_test),2))
    
print(f"\nROOT MEAN SQUARED ERROR for the model:",round(np.sqrt(mean_squared_error(ypred,Y_test)),2))


# AS after doing hypertuned the Random forest regressor model we decresed the accuracy of the model.

# In[163]:


reg.fit(X_train,Y_train)
Y_pred=reg.predict(X_test)
sns.regplot(Y_test,Y_pred)
plt.xlabel("Actual Test Data")
plt.ylabel("Predicted Data")
plt.tight_layout()


# There is very slight decrease in the errors after hypeertuned the model.

# Using RandomForest deafualt parameterized model to predict Sales prices for the houses.

# In[164]:


predictedSalesPreic=RandomForest.predict(X__)


# In[170]:


pd.Series(predictedSalesPreic*100000).to_excel("Predict_Test_SalesPreice.xlsx")


# ### Saving Model

# In[171]:


import pickle
pickle.dump(RandomForest,open("HousePricePrediction.pickle","wb"))   #defaul parameterized model


# In[ ]:




