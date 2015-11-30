from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation as cv,grid_search as gs,preprocessing as pre,svm
from sklearn import feature_extraction as fe,linear_model as lm
import pandas as pd
import xgboost as xg
import numpy as np
import random




###############RANDOM FOREST###########################################

data=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
#random.seed(32712)
ind=random.sample(range(0,data.shape[0]),int(round(data.shape[0]*.75,0)))
data_0=data.ix[ind,]
data_1=data.ix[set(data.index)-set(ind),]

rf=RandomForestClassifier(n_estimators=500,n_jobs=3)

rf.fit(data_0.drop('target',1),data_0['target'])

# for ensemble
meta_randf=rf.predict_proba(data_1.drop('target',1))
meta_randf_df=pd.DataFrame(meta_randf)
meta_randf_df.columns="randf_"+ meta_randf_df.columns.astype('str')

#predicting on test set
p_randf=rf.predict_proba(test)
p_randf_df=pd.DataFrame(p_randf)
p_randf_df.columns="randf_"+ p_randf_df.columns.astype('str')


  
    
#################XG BOOST #####################################################

data_xg=data_0.copy()


data_xg['target']=data_xg['target'].str.replace('Class_','').astype('int')-1

# creating model
xgb=xg.XGBClassifier(n_estimators=500).fit(data_xg.drop('target',1).as_matrix(),data_xg['target'])


#for ensemble
meta_xgb=xgb.predict_proba(data_1.drop('target',1).as_matrix())
#meta_xgb="Class_"+pd.Series(meta_xgb).astype('str')
meta_xgb_df=pd.DataFrame(meta_xgb)
meta_xgb_df.columns="xgb_"+ meta_xgb_df.columns.astype('str')


# predicting test bring into kaggle format
p_xgb=xgb.predict_proba(test.as_matrix())
p_xgb_df=pd.DataFrame(p_xgb)
p_xgb_df.columns="xgb_"+ p_xgb_df.columns.astype('str')



########################## SVM #######################
## should try scaling only variance and also to [0,1]

data_0x=data_0.drop('target',1).copy()
y=data_0['target']

#creating transformation to unit variance and for scaling to [0,1]
scaler=pre.StandardScaler().fit(data_0x)
data_0x=scaler.transform(data_0x)

#unit variance
test_x=scaler.transform(test)
data_1x=data_1.drop('target',1).copy()
data_1x=scaler.transform(data_1x)


'''
#creating transformation to unit variance and for scaling to [0,1]
scaler=pre.StandardScaler(with_mean=False).fit(data_0x)
data_0x=scaler.transform(data_0x)
min_max_scaler=pre.MinMaxScaler()
data_0x=min_max_scaler.fit_transform(data_0x)

#unit variance
test_x=scaler.transform(test)
data_1x=scaler.transform(data_1.drop('target',1))

#scaling to range
test_x=min_max_scaler.transform(test_x)
data_1x=min_max_scaler.transform(data_1x)
'''
'''
# fitting svm linear model
sv=lm.SGDClassifier(n_jobs=3)
sv.fit(data_0x,y)
'''
# fitting svm  rbfmodel
sv=svm.SVC(probability=True)
sv.fit(data_0x,y)

#for ensemble
meta_svm=sv.predict_proba(data_1x)
meta_svm_df=pd.DataFrame(meta_svm)
meta_svm_df.columns="svm_"+ meta_svm_df.columns.astype('str')

# predicting on test data
p_svm=sv.predict_proba(test_x)
p_svm_df=pd.DataFrame(p_svm)
p_svm_df.columns="svm_"+ p_svm_df.columns.astype('str')


################################ ENSEMBLING ##########################
#[get class probabilities and put all together total 27 coluns for logistic]

data_1.reset_index(inplace=True)


ensemble_train=pd.concat([meta_xgb_df,meta_randf_df,meta_svm_df,data_1['target']],1)
#ensemble_train.to_csv('ensemble_train.csv')
#ensemble_train.columns=['xgb','random_forest','svm','target']

# reformatting target variable to integer
target=ensemble_train['target'].str.replace('Class_','').astype('int')
ensemble_x=ensemble_train.drop('target',1).copy()

#pre -processing for logsitic modelling
ensemble_dict_x=[v.to_dict() for k,v in ensemble_x.iterrows()]

vec = fe.DictVectorizer()
ensemble_final_x = vec.fit_transform(ensemble_dict_x).toarray()
ensemble_final_x = pd.DataFrame(ensemble_final_x)

# using logistic regression as the meta learner for blending
meta_log=lm.LogisticRegression()
meta_log.fit(ensemble_final_x,target)

#blending results
ensemble_test=pd.concat([p_xgb_df,p_randf_df,p_svm_df],1)
#ensemble_train.to_csv('/home/drwh00/Documents/DS/Kaggle/Otto Classifier/ensemble_test.csv')

ensemble_dict_testx=[v.to_dict() for k,v in ensemble_test.iterrows()]
vec = fe.DictVectorizer()
ensemble_dict_testx = vec.fit_transform(ensemble_dict_testx).toarray()
ensemble_dict_testx = pd.DataFrame(ensemble_dict_testx)

p=meta_log.predict(ensemble_dict_testx)

p1=pd.DataFrame({'Class':p})
p1['Value']=[1]*p.shape[0]
p1['id']=p1.index
p2=p1.pivot('id','Class','Value')
p2=p2.fillna(0)
p2.index=p2.index+1
p2.columns="Class_"+ p2.columns.astype('str')

p2.to_csv('submission_blended3.csv',)


