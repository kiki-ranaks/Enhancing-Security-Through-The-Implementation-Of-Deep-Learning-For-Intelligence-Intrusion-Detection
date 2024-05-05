import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU
from keras import metrics
import keras
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import plot_confusion_matrix
import time
import warnings
warnings.filterwarnings("ignore")
intrusion = pd.read_csv('DATA.csv')
intrusion .head(10)
intrusion .shape
intrusion ["label"].value_counts()
intrusion .info()
intrusion .describe()
intrusion .columns
intrusion ['attack_cat'].value_counts()
data_0,data_1 =intrusion ['label'].value_counts()[0] / len(intrusion .index),intrusion ['label'].value_counts()[1] / len(intrusion .index)


print("In data: there are {} % of class 0 and {} % of class 1".format(data_0,data_1))

f, axes = plt.subplots( figsize=(12,6))
sns.countplot(x="attack_cat", data=intrusion , order = intrusion ['attack_cat'].value_counts().index, palette='Reds_r',ax=axes)
axes.tick_params('x',width=2,labelsize=10)
intrusion .isnull().sum()
import matplotlib.pyplot as plt
plt.figure(figsize=(5, 5))
colors = ['grey', 'tomato']
label_counts = intrusion.label.value_counts()
labels = [ 'Attack','Normal']
plt.pie(label_counts, labels=labels, autopct='%0.2f%%', colors=colors)
plt.title("PIE CHART DISTRIBUTION")
plt.legend()
plt.show()
f, axes = plt.subplots( figsize=(10, 5))
sns.histplot(intrusion [intrusion ['label'] == 1]['service'], color='tomato')
sns.histplot(intrusion [intrusion ['label'] ==0]['service'], color='grey')
axes.tick_params('x',width=2,labelsize=10)
intrusion .drop(['id','attack_cat'],axis=1,inplace=True)
# Clamp extreme Values
data = intrusion .select_dtypes(include=[np.number])
data.describe(include='all')
data = intrusion .select_dtypes(include=[np.number])
data_before = data.copy()
DEBUG = 0
for feature in data.columns:
    if DEBUG == 1:
        print(feature)
        print('nunique = '+str(data[feature].nunique()))
        print(data[feature].nunique()>50)
   
    if data[feature].nunique()>50:
        if data[feature].min()==0:
            intrusion [feature] = np.log(intrusion [feature]+1)
        else:
            intrusion [feature] = np.log(intrusion [feature])

data= intrusion .select_dtypes(include=[np.number])
df_cat = intrusion .select_dtypes(exclude=[np.number])
df_cat.describe(include='all')
DEBUG = 0
for feature in df_cat.columns:
    if DEBUG == 1:
        print(feature)
        print('nunique = '+str(df_cat[feature].nunique()))
        print(df_cat[feature].nunique()>6)
        print(sum(intrusion [feature].isin(intrusion [feature].value_counts().head().index)))
       
    if df_cat[feature].nunique()>6:
        intrusion [feature] = np.where(intrusion [feature].isin(intrusion [feature].value_counts().head().index), intrusion [feature], '-')
df_cat = intrusion .select_dtypes(exclude=[np.number])
df_cat.describe(include='all')
intrusion ['proto'].value_counts()
# Feature Selection
from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

X = intrusion.iloc[:, 4:-2]
y = intrusion.iloc[:, -1]

best_features = SelectKBest(score_func=chi2, k='all')
fit = best_features.fit(X, y)

df_scores = pd.DataFrame(fit.scores_)
df_col = pd.DataFrame(X.columns)

feature_score = pd.concat([df_col, df_scores], axis=1)
feature_score.columns = ['feature', 'score']

top_features = feature_score.nlargest(20, 'score')

plt.figure(figsize=(12, 8))
sns.barplot(x='score', y='feature', data=top_features, palette='Reds_r')

plt.title('Top 20 Features')
plt.xlabel('Score')
plt.ylabel('Feature')
plt.show()

intrusion=intrusion.drop(['proto','service','state'], axis=1)
x =intrusion.drop(['label'], axis=1)
y =intrusion['label']
from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler()
x = ms.fit_transform(x)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.3)
print(X_train.shape, X_test.shape)
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
model_performance = pd.DataFrame(columns=['Accuracy','Recall','Precision','F1-Score','time to train','time to predict','total time'])
y_train = np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0],1,X_train.shape[1]))
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
lstm = Sequential()

lstm.add(LSTM(50,input_dim=39))

lstm.add(Dense(1,activation='sigmoid'))
lstm.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
lstm.summary()
start = time.time()
history = lstm.fit(X_train, y_train, epochs=200, batch_size=2000,verbose=2)
end_train = time.time()
X_test = np.reshape(X_test, (X_test.shape[0],1,X_test.shape[1]))
test_results = lstm.evaluate(X_test, y_test, verbose=1)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]*100}%')
y_pred =lstm.predict(X_test)
end_predict = time.time()
from sklearn.metrics import precision_recall_fscore_support as score
acc = accuracy_score(y_test, y_pred.round())
recall = recall_score(y_test, y_pred.round())
precision = precision_score(y_test, y_pred.round())
f1s = f1_score(y_test, y_pred.round())
print("Accuracy: "+ "{:.2%}".format(acc))
print("Recall: "+ "{:.2%}".format(recall))
print("Precision: "+ "{:.2%}".format(precision))
print("F1-Score: "+ "{:.2%}".format(f1s))
print("time to train: "+ "{:.2f}".format(end_train-start)+" s")
print("time to predict: "+"{:.2f}".format(end_predict-end_train)+" s")
print("total: "+"{:.2f}".format(end_predict-start)+" s")
model_performance.loc['LSTM'] = [acc, recall, precision, f1s,end_train-start,end_predict-end_train,end_predict-start]
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, make_scorer, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, auc
plt.figure(figsize=(6,4))
confusion_matrix = confusion_matrix(y_test,y_pred.round())
sns.heatmap(confusion_matrix,0, annot=True, fmt="d",cmap="Reds")
plt.title("LSTM Confusion Matrix")
plt.show()
fpr_model1, tpr_model1, thresholds_model1 = roc_curve(y_test, y_pred, pos_label=1)
auc_model1 = roc_auc_score(y_test, y_pred)
plt.figure(figsize=(6,4))
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr_model1, tpr_model1,color="tomato",label='ROC model (area = {:.3f})'.format(auc_model1))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

gru = Sequential()
gru.add(GRU(64, input_shape=(None, 39), activation='relu', return_sequences=True))
gru.add(GRU(32, activation='relu'))
gru.add(Dense(1, activation='sigmoid'))
gru.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
gru.summary()
start = time.time()
history = gru.fit(X_train, y_train, epochs=200, batch_size=2000,verbose=2)
end_train = time.time()
test_results = gru.evaluate(X_test, y_test, verbose=1)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]*100}%')
y_pred =gru.predict(X_test)
end_predict = time.time()
from sklearn.metrics import precision_recall_fscore_support as score
acc = accuracy_score(y_test, y_pred.round())
recall = recall_score(y_test, y_pred.round())
precision = precision_score(y_test, y_pred.round())
f1s = f1_score(y_test, y_pred.round())
print("Accuracy: "+ "{:.2%}".format(acc))
print("Recall: "+ "{:.2%}".format(recall))
print("Precision: "+ "{:.2%}".format(precision))
print("F1-Score: "+ "{:.2%}".format(f1s))
print("time to train: "+ "{:.2f}".format(end_train-start)+" s")
print("time to predict: "+"{:.2f}".format(end_predict-end_train)+" s")
print("total: "+"{:.2f}".format(end_predict-start)+" s")
model_performance.loc['GRU'] = [acc, recall, precision, f1s,end_train-start,end_predict-end_train,end_predict-start]
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, make_scorer, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, auc
plt.figure(figsize=(6,4))
confusion_matrix = confusion_matrix(y_test,y_pred.round())
sns.heatmap(confusion_matrix,0, annot=True, fmt="d",cmap="Reds")
plt.title("GRU Confusion Matrix")
plt.show()
fpr_model1, tpr_model1, thresholds_model1 = roc_curve(y_test, y_pred, pos_label=1)
auc_model1 = roc_auc_score(y_test, y_pred)
plt.figure(figsize=(6,4))
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr_model1, tpr_model1,color="tomato",label='ROC model (area = {:.3f})'.format(auc_model1))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
