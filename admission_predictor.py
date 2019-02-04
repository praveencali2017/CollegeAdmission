

##Data Analysis:
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from keras.layers import Dense
from keras.models import Sequential,load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import pickle
sns.set()
def visualize_data(cols,data,x,y):
    n_cols = cols
    n_rows=int(math.ceil(len(x)/n_cols))
    fig, axes=plt.subplots(n_rows,n_cols, figsize=(n_cols*10,(n_cols*10)/2))
    row_shift=0
    for i,feature in enumerate(x):
        col_shift=i%n_cols
        sns.scatterplot(x=feature,y=y,data=data,ax=axes[row_shift,col_shift])
        if col_shift==n_cols-1:
            row_shift+=1
    plt.tight_layout(h_pad=n_cols, pad=n_cols)
    plt.show()

df=pd.read_csv('Admission_Predict_Ver1.1.csv')
df.columns=df.columns.str.strip()
# print(df.info())
df['Combined Score']=df.apply(lambda x: x['GRE Score']+x['TOEFL Score'],axis=1)
x_features=['GRE Score','Research','University Rating','TOEFL Score','SOP','LOR' ,'CGPA','Combined Score']
y_label='Chance of Admit'
# # 'GRE Score','TOEFL Score','CGPA'
X=df[x_features].values
y=df[[y_label]].values


### EDA
# sns.pairplot(df,x_vars=x_features,y_vars=y_label)
# sns.heatmap(df.corr(),annot=True)
# plt.tight_layout()
# plt.show()
# # sns.distplot(df[['LOR']])
# # sns.boxplot(df[x_features])
# df.boxplot(column=x_features)
#
# plt.show()
# print(X.shape)
# print(y.shape)



from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42,shuffle=True)
# normalizer= Normalizer().fit(X_train)
# X_train=normalizer.transform(X_train)
# X_test=normalizer.transform(X_test)
normalizer= StandardScaler().fit(X_train)
X_train=normalizer.transform(X_train)
X_test=normalizer.transform(X_test)

# # #
# # # ### TRAINING
model=Sequential()
model.add(Dense(100,activation='relu',input_dim=X_train.shape[1]))
model.add(Dense(150,activation='relu'))
model.add(Dense(150,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(1,activation='linear'))
# # model.summary()
# #
epochs=200
model.compile(optimizer='adam',loss='mse', metrics=['mse'])
history= model.fit(X_train,y_train,10,validation_split=0.20,epochs=epochs,verbose=2)

plt.plot(range(1,epochs+1),history.history['loss'],c='blue')
plt.xlabel('Epochs')
plt.ylabel('Train Loss')
plt.plot(range(1,epochs+1),history.history['val_loss'],c='orange')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.show()
# # # model.save('predictor.h5')



from sklearn.metrics import r2_score

# # # ### TESTING

pred=model.predict(X_test)

print("r_square score: ",r2_score(y_test,pred))
