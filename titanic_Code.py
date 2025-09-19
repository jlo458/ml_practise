import pandas as pd 
from sklearn.model_selection import train_test_split

# Model imports
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

# Metrics Imports
from sklearn.metrics import f1_score


data = pd.read_csv("./preprocessed_data3.csv", index_col=None) 


features = data.drop(["Survived"], axis=1)
labels = data["Survived"] 

features_train, features_v_t, labels_train, labels_v_t = train_test_split(features, labels, test_size=0.4)
features_validation, features_test, labels_validation, labels_test = train_test_split(features_v_t, 
                                                                                      labels_v_t, 
                                                                                      test_size=0.5)

'''print(len(features_train))
print(len(labels_train))
print(len(features_validation))
print(len(labels_validation))
print(len(features_test))
print(len(labels_test))'''

# Model Initiation
lr_model = LogisticRegression()
lr_model.fit(features_train, labels_train)

dt_model = DecisionTreeClassifier()
dt_model.fit(features_train, labels_train)

nb_model = GaussianNB()
nb_model.fit(features_train, labels_train)

svm_model = SVC()
svm_model.fit(features_train, labels_train)

rf_model = RandomForestClassifier()
rf_model.fit(features_train, labels_train)

gb_model = GradientBoostingClassifier()
gb_model.fit(features_train, labels_train)

ab_model = AdaBoostClassifier()
ab_model.fit(features_train, labels_train)

# Model evaluation
print("LR: ", lr_model.score(features_train, labels_train))
print("DT: ", dt_model.score(features_train, labels_train))
print("NB: ", nb_model.score(features_train, labels_train))
print("SVM: ", svm_model.score(features_train, labels_train))
print("RF: ", rf_model.score(features_train, labels_train))
print("GB: ", gb_model.score(features_train, labels_train))
print("AB: ", ab_model.score(features_train, labels_train))

# F1-Score 
print("\nF1 Score")

predicted_labels_lr = lr_model.predict(features_validation)
print("lr: ", f1_score(labels_validation, predicted_labels_lr))

predicted_labels_dt = dt_model.predict(features_validation)
print("dt: ", f1_score(labels_validation, predicted_labels_dt))

predicted_labels_nb = nb_model.predict(features_validation)
print("nb: ", f1_score(labels_validation, predicted_labels_nb))

predicted_labels_svm = svm_model.predict(features_validation)
print("svm: ", f1_score(labels_validation, predicted_labels_svm))

predicted_labels_rf = rf_model.predict(features_validation)
print("rf: ", f1_score(labels_validation, predicted_labels_rf))

predicted_labels_gb = gb_model.predict(features_validation)
print("gb: ", f1_score(labels_validation, predicted_labels_gb))

predicted_labels_ab = ab_model.predict(features_validation)
print("ab: ", f1_score(labels_validation, predicted_labels_ab))











