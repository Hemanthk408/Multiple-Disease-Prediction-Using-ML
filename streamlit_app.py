
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

     
diecase

df=pd.read_csv('/content/indian_liver_patient - indian_liver_patient.csv')
df
     
Age	Gender	Total_Bilirubin	Direct_Bilirubin	Alkaline_Phosphotase	Alamine_Aminotransferase	Aspartate_Aminotransferase	Total_Protiens	Albumin	Albumin_and_Globulin_Ratio	Dataset
0	65	Female	0.7	0.1	187	16	18	6.8	3.3	0.90	1
1	62	Male	10.9	5.5	699	64	100	7.5	3.2	0.74	1
2	62	Male	7.3	4.1	490	60	68	7.0	3.3	0.89	1
3	58	Male	1.0	0.4	182	14	20	6.8	3.4	1.00	1
4	72	Male	3.9	2.0	195	27	59	7.3	2.4	0.40	1
...	...	...	...	...	...	...	...	...	...	...	...
578	60	Male	0.5	0.1	500	20	34	5.9	1.6	0.37	2
579	40	Male	0.6	0.1	98	35	31	6.0	3.2	1.10	1
580	52	Male	0.8	0.2	245	48	49	6.4	3.2	1.00	1
581	31	Male	1.3	0.5	184	29	32	6.8	3.4	1.00	1
582	38	Male	1.0	0.3	216	21	24	7.3	4.4	1.50	2
583 rows × 11 columns


#cateceri=("Gender","Dataset")
#contunius=("Age","Total_Bilirubin","Direct_Bilirubin","Alkaline_Phosphotase","Alamine_Aminotransferase","Aspartate_Aminotransferase","Total_Protiens","Albumin","Albumin_and_Globulin_Ratio")
     

df.info()
     
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 583 entries, 0 to 582
Data columns (total 11 columns):
 #   Column                      Non-Null Count  Dtype  
---  ------                      --------------  -----  
 0   Age                         583 non-null    int64  
 1   Gender                      583 non-null    object 
 2   Total_Bilirubin             583 non-null    float64
 3   Direct_Bilirubin            583 non-null    float64
 4   Alkaline_Phosphotase        583 non-null    int64  
 5   Alamine_Aminotransferase    583 non-null    int64  
 6   Aspartate_Aminotransferase  583 non-null    int64  
 7   Total_Protiens              583 non-null    float64
 8   Albumin                     583 non-null    float64
 9   Albumin_and_Globulin_Ratio  583 non-null    float64
 10  Dataset                     583 non-null    int64  
dtypes: float64(5), int64(5), object(1)
memory usage: 50.2+ KB

df.corr()
     
Age	Gender	Total_Bilirubin	Direct_Bilirubin	Alkaline_Phosphotase	Alamine_Aminotransferase	Aspartate_Aminotransferase	Total_Protiens	Albumin	Albumin_and_Globulin_Ratio	Dataset
Age	1.000000	0.056560	0.011763	0.007529	0.080425	-0.086883	-0.019910	-0.187461	-0.265924	-0.217909	-0.137351
Gender	0.056560	1.000000	0.089291	0.100436	-0.027496	0.082332	0.080336	-0.089121	-0.093799	-0.008400	-0.082416
Total_Bilirubin	0.011763	0.089291	1.000000	0.874618	0.206669	0.214065	0.237831	-0.008099	-0.222250	-0.207360	-0.220208
Direct_Bilirubin	0.007529	0.100436	0.874618	1.000000	0.234939	0.233894	0.257544	-0.000139	-0.228531	-0.201315	-0.246046
Alkaline_Phosphotase	0.080425	-0.027496	0.206669	0.234939	1.000000	0.125680	0.167196	-0.028514	-0.165453	-0.236023	-0.184866
Alamine_Aminotransferase	-0.086883	0.082332	0.214065	0.233894	0.125680	1.000000	0.791966	-0.042518	-0.029742	-0.003996	-0.163416
Aspartate_Aminotransferase	-0.019910	0.080336	0.237831	0.257544	0.167196	0.791966	1.000000	-0.025645	-0.085290	-0.071064	-0.151934
Total_Protiens	-0.187461	-0.089121	-0.008099	-0.000139	-0.028514	-0.042518	-0.025645	1.000000	0.784053	0.236079	0.035008
Albumin	-0.265924	-0.093799	-0.222250	-0.228531	-0.165453	-0.029742	-0.085290	0.784053	1.000000	0.690168	0.161388
Albumin_and_Globulin_Ratio	-0.217909	-0.008400	-0.207360	-0.201315	-0.236023	-0.003996	-0.071064	0.236079	0.690168	1.000000	0.164313
Dataset	-0.137351	-0.082416	-0.220208	-0.246046	-0.184866	-0.163416	-0.151934	0.035008	0.161388	0.164313	1.000000

df.columns
     
Index(['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
       'Albumin_and_Globulin_Ratio', 'Dataset'],
      dtype='object')

df.nunique()
     
0
Age	72
Gender	2
Total_Bilirubin	113
Direct_Bilirubin	80
Alkaline_Phosphotase	263
Alamine_Aminotransferase	152
Aspartate_Aminotransferase	177
Total_Protiens	58
Albumin	40
Albumin_and_Globulin_Ratio	69
Dataset	2

dtype: int64

for i in df.select_dtypes(include='object').columns:
  print(i,df[i].unique())
     
Gender ['Female' 'Male']

df=pd.DataFrame(df)
     

# Calculate the A/G ratio using the given formula
df['Albumin_and_Globulin_Ratio'] =df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin'] / (df['Total_Protiens'] - df['Albumin']))
     

df
     
Age	Gender	Total_Bilirubin	Direct_Bilirubin	Alkaline_Phosphotase	Alamine_Aminotransferase	Aspartate_Aminotransferase	Total_Protiens	Albumin	Albumin_and_Globulin_Ratio	Dataset
0	65	Female	0.7	0.1	187	16	18	6.8	3.3	0.90	1
1	62	Male	10.9	5.5	699	64	100	7.5	3.2	0.74	1
2	62	Male	7.3	4.1	490	60	68	7.0	3.3	0.89	1
3	58	Male	1.0	0.4	182	14	20	6.8	3.4	1.00	1
4	72	Male	3.9	2.0	195	27	59	7.3	2.4	0.40	1
...	...	...	...	...	...	...	...	...	...	...	...
578	60	Male	0.5	0.1	500	20	34	5.9	1.6	0.37	2
579	40	Male	0.6	0.1	98	35	31	6.0	3.2	1.10	1
580	52	Male	0.8	0.2	245	48	49	6.4	3.2	1.00	1
581	31	Male	1.3	0.5	184	29	32	6.8	3.4	1.00	1
582	38	Male	1.0	0.3	216	21	24	7.3	4.4	1.50	2
583 rows × 11 columns


df['Dataset']=df['Dataset'].map({1:1,2:0})
     

df.isnull().sum()
     
0
Age	0
Gender	0
Total_Bilirubin	0
Direct_Bilirubin	0
Alkaline_Phosphotase	0
Alamine_Aminotransferase	0
Aspartate_Aminotransferase	0
Total_Protiens	0
Albumin	0
Albumin_and_Globulin_Ratio	0
Dataset	0

dtype: int64

df.set_index('Age',inplace=True)
     

liver_patient=df.copy()
     

df['Gender']=df['Gender'].map({'Female':0,'Male':1})
     

df
     
Gender	Total_Bilirubin	Direct_Bilirubin	Alkaline_Phosphotase	Alamine_Aminotransferase	Aspartate_Aminotransferase	Total_Protiens	Albumin	Albumin_and_Globulin_Ratio	Dataset
Age										
65	0	0.7	0.1	187	16	18	6.8	3.3	0.90	1
62	1	10.9	5.5	699	64	100	7.5	3.2	0.74	1
62	1	7.3	4.1	490	60	68	7.0	3.3	0.89	1
58	1	1.0	0.4	182	14	20	6.8	3.4	1.00	1
72	1	3.9	2.0	195	27	59	7.3	2.4	0.40	1
...	...	...	...	...	...	...	...	...	...	...
60	1	0.5	0.1	500	20	34	5.9	1.6	0.37	0
40	1	0.6	0.1	98	35	31	6.0	3.2	1.10	1
52	1	0.8	0.2	245	48	49	6.4	3.2	1.00	1
31	1	1.3	0.5	184	29	32	6.8	3.4	1.00	1
38	1	1.0	0.3	216	21	24	7.3	4.4	1.50	0
583 rows × 10 columns


x=df.iloc[:,:-1]
y=df.iloc[:,-1]
     

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
x_train.shape,x_test.shape,y_train.shape,y_test.shape
     
((466, 9), (117, 9), (466,), (117,))

y_train.value_counts()

     
count
Dataset	
1	333
2	133

dtype: int64

y_test.value_counts()
     
count
Dataset	
1	83
2	34

dtype: int64

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier().fit(x_train,y_train)
     

y_pred=model.predict(x_test)
     

y_pred
     
array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 2, 1, 1, 1, 1])

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
     

confusion_matrix(y_test,y_pred)
     
array([[ 0, 36,  2],
       [ 0, 76,  3],
       [ 0,  0,  0]])

accuracy_score(y_test,y_pred)
     
0.6495726495726496

import pickle
     

from sklearn.ensemble import RandomForestClassifier
model1=RandomForestClassifier().fit(x_train,y_train)
y_pred=model1.predict(x_test)
print(classification_report(y_test,y_pred))
     
              precision    recall  f1-score   support

           0       0.48      0.39      0.43        38
           1       0.73      0.80      0.76        79

    accuracy                           0.67       117
   macro avg       0.61      0.60      0.60       117
weighted avg       0.65      0.67      0.66       117


from sklearn.linear_model import LogisticRegression
model2=LogisticRegression().fit(x_train,y_train)
y_pred=model2.predict(x_test)
print(classification_report(y_test,y_pred))
     
              precision    recall  f1-score   support

           0       0.50      0.21      0.30        38
           1       0.70      0.90      0.79        79

    accuracy                           0.68       117
   macro avg       0.60      0.55      0.54       117
weighted avg       0.64      0.68      0.63       117

/usr/local/lib/python3.10/dist-packages/sklearn/linear_model/_logistic.py:465: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(

from xgboost import XGBClassifier
model3=XGBClassifier().fit(x_train,y_train)
y_pred=model3.predict(x_test)
print(classification_report(y_test,y_pred))
     
              precision    recall  f1-score   support

         0.0       1.00      1.00      1.00        52
         1.0       1.00      1.00      1.00        28

    accuracy                           1.00        80
   macro avg       1.00      1.00      1.00        80
weighted avg       1.00      1.00      1.00        80



     
kidney_disease

df1=pd.read_csv('/content/kidney_disease - kidney_disease.csv')
df1
     
id	age	bp	sg	al	su	rbc	pc	pcc	ba	...	pcv	wc	rc	htn	dm	cad	appet	pe	ane	classification
0	0	48.0	80.0	1.020	1.0	0.0	NaN	normal	notpresent	notpresent	...	44	7800	5.2	yes	yes	no	good	no	no	ckd
1	1	7.0	50.0	1.020	4.0	0.0	NaN	normal	notpresent	notpresent	...	38	6000	NaN	no	no	no	good	no	no	ckd
2	2	62.0	80.0	1.010	2.0	3.0	normal	normal	notpresent	notpresent	...	31	7500	NaN	no	yes	no	poor	no	yes	ckd
3	3	48.0	70.0	1.005	4.0	0.0	normal	abnormal	present	notpresent	...	32	6700	3.9	yes	no	no	poor	yes	yes	ckd
4	4	51.0	80.0	1.010	2.0	0.0	normal	normal	notpresent	notpresent	...	35	7300	4.6	no	no	no	good	no	no	ckd
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
395	395	55.0	80.0	1.020	0.0	0.0	normal	normal	notpresent	notpresent	...	47	6700	4.9	no	no	no	good	no	no	notckd
396	396	42.0	70.0	1.025	0.0	0.0	normal	normal	notpresent	notpresent	...	54	7800	6.2	no	no	no	good	no	no	notckd
397	397	12.0	80.0	1.020	0.0	0.0	normal	normal	notpresent	notpresent	...	49	6600	5.4	no	no	no	good	no	no	notckd
398	398	17.0	60.0	1.025	0.0	0.0	normal	normal	notpresent	notpresent	...	51	7200	5.9	no	no	no	good	no	no	notckd
399	399	58.0	80.0	1.025	0.0	0.0	normal	normal	notpresent	notpresent	...	53	6800	6.1	no	no	no	good	no	no	notckd
400 rows × 26 columns


pd.set_option('display.max_columns', None)
df1
     
id	age	bp	sg	al	su	rbc	pc	pcc	ba	bgr	bu	sc	sod	pot	hemo	pcv	wc	rc	htn	dm	cad	appet	pe	ane	classification
0	0	48.0	80.0	1.020	1.0	0.0	NaN	normal	notpresent	notpresent	121.0	36.0	1.2	NaN	NaN	15.4	44	7800	5.2	yes	yes	no	good	no	no	ckd
1	1	7.0	50.0	1.020	4.0	0.0	NaN	normal	notpresent	notpresent	NaN	18.0	0.8	NaN	NaN	11.3	38	6000	NaN	no	no	no	good	no	no	ckd
2	2	62.0	80.0	1.010	2.0	3.0	normal	normal	notpresent	notpresent	423.0	53.0	1.8	NaN	NaN	9.6	31	7500	NaN	no	yes	no	poor	no	yes	ckd
3	3	48.0	70.0	1.005	4.0	0.0	normal	abnormal	present	notpresent	117.0	56.0	3.8	111.0	2.5	11.2	32	6700	3.9	yes	no	no	poor	yes	yes	ckd
4	4	51.0	80.0	1.010	2.0	0.0	normal	normal	notpresent	notpresent	106.0	26.0	1.4	NaN	NaN	11.6	35	7300	4.6	no	no	no	good	no	no	ckd
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
395	395	55.0	80.0	1.020	0.0	0.0	normal	normal	notpresent	notpresent	140.0	49.0	0.5	150.0	4.9	15.7	47	6700	4.9	no	no	no	good	no	no	notckd
396	396	42.0	70.0	1.025	0.0	0.0	normal	normal	notpresent	notpresent	75.0	31.0	1.2	141.0	3.5	16.5	54	7800	6.2	no	no	no	good	no	no	notckd
397	397	12.0	80.0	1.020	0.0	0.0	normal	normal	notpresent	notpresent	100.0	26.0	0.6	137.0	4.4	15.8	49	6600	5.4	no	no	no	good	no	no	notckd
398	398	17.0	60.0	1.025	0.0	0.0	normal	normal	notpresent	notpresent	114.0	50.0	1.0	135.0	4.9	14.2	51	7200	5.9	no	no	no	good	no	no	notckd
399	399	58.0	80.0	1.025	0.0	0.0	normal	normal	notpresent	notpresent	131.0	18.0	1.1	141.0	3.5	15.8	53	6800	6.1	no	no	no	good	no	no	notckd
400 rows × 26 columns


df1.info()
     
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 400 entries, 0 to 399
Data columns (total 26 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   id              400 non-null    int64  
 1   age             391 non-null    float64
 2   bp              388 non-null    float64
 3   sg              353 non-null    float64
 4   al              354 non-null    float64
 5   su              351 non-null    float64
 6   rbc             248 non-null    object 
 7   pc              335 non-null    object 
 8   pcc             396 non-null    object 
 9   ba              396 non-null    object 
 10  bgr             356 non-null    float64
 11  bu              381 non-null    float64
 12  sc              383 non-null    float64
 13  sod             313 non-null    float64
 14  pot             312 non-null    float64
 15  hemo            348 non-null    float64
 16  pcv             330 non-null    object 
 17  wc              295 non-null    object 
 18  rc              270 non-null    object 
 19  htn             398 non-null    object 
 20  dm              398 non-null    object 
 21  cad             398 non-null    object 
 22  appet           399 non-null    object 
 23  pe              399 non-null    object 
 24  ane             399 non-null    object 
 25  classification  400 non-null    object 
dtypes: float64(11), int64(1), object(14)
memory usage: 81.4+ KB

column_mapping= {
    'bp': 'Blood Pressure',
    'sg': 'Specific Gravity',
    'al': 'Albumin',
    'su': 'Sugar',
    'rbc': 'Red Blood Cells',
    'pc': 'Pus Cells',
    'pcc': 'Pus Cell Clumps',
    'ba': 'Bacteria',
    'bgr': 'Blood Glucose Random',
    'bu': 'Blood Urea',
    'sc': 'Serum Creatinine',
    'sod': 'Sodium',
    'pot': 'Potassium',
    'hemo': 'Hemoglobin',
    'pcv': 'Packed Cell Volume',
    'wc': 'White Blood Cell Count',
    'rc': 'Red Blood Cell Count',
    'htn': 'Hypertension',
    'dm': 'Diabetes Mellitus',
    'cad': 'Coronary Artery Disease',
    'appet': 'Appetite',
    'pe': 'Pedal Edema',
    'ane': 'Anemia',
}
     

df1.rename(columns=column_mapping, inplace=True)
     

df1
     
id	age	Blood Pressure	Specific Gravity	Albumin	Sugar	Red Blood Cells	Pus Cells	Pus Cell Clumps	Bacteria	...	Packed Cell Volume	White Blood Cell Count	Red Blood Cell Count	Hypertension	Diabetes Mellitus	Coronary Artery Disease	Appetite	Pedal Edema	Anemia	classification
0	0	48.0	80.0	1.020	1.0	0.0	NaN	normal	notpresent	notpresent	...	44	7800	5.2	yes	yes	no	good	no	no	ckd
1	1	7.0	50.0	1.020	4.0	0.0	NaN	normal	notpresent	notpresent	...	38	6000	NaN	no	no	no	good	no	no	ckd
2	2	62.0	80.0	1.010	2.0	3.0	normal	normal	notpresent	notpresent	...	31	7500	NaN	no	yes	no	poor	no	yes	ckd
3	3	48.0	70.0	1.005	4.0	0.0	normal	abnormal	present	notpresent	...	32	6700	3.9	yes	no	no	poor	yes	yes	ckd
4	4	51.0	80.0	1.010	2.0	0.0	normal	normal	notpresent	notpresent	...	35	7300	4.6	no	no	no	good	no	no	ckd
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
395	395	55.0	80.0	1.020	0.0	0.0	normal	normal	notpresent	notpresent	...	47	6700	4.9	no	no	no	good	no	no	notckd
396	396	42.0	70.0	1.025	0.0	0.0	normal	normal	notpresent	notpresent	...	54	7800	6.2	no	no	no	good	no	no	notckd
397	397	12.0	80.0	1.020	0.0	0.0	normal	normal	notpresent	notpresent	...	49	6600	5.4	no	no	no	good	no	no	notckd
398	398	17.0	60.0	1.025	0.0	0.0	normal	normal	notpresent	notpresent	...	51	7200	5.9	no	no	no	good	no	no	notckd
399	399	58.0	80.0	1.025	0.0	0.0	normal	normal	notpresent	notpresent	...	53	6800	6.1	no	no	no	good	no	no	notckd
400 rows × 26 columns


for i in df1.select_dtypes(include='object').columns:
  print(i,df1[i].unique())
     
Red Blood Cells [nan 'normal' 'abnormal']
Pus Cells ['normal' 'abnormal' nan]
Pus Cell Clumps ['notpresent' 'present' nan]
Bacteria ['notpresent' 'present' nan]
Packed Cell Volume ['44' '38' '31' '32' '35' '39' '36' '33' '29' '28' nan '16' '24' '37' '30'
 '34' '40' '45' '27' '48' '?' '52' '14' '22' '18' '42' '17' '46' '23' '19'
 '25' '41' '26' '15' '21' '43' '20' '47' '9' '49' '50' '53' '51' '54']
White Blood Cell Count ['7800' '6000' '7500' '6700' '7300' nan '6900' '9600' '12100' '4500'
 '12200' '11000' '3800' '11400' '5300' '9200' '6200' '8300' '8400' '10300'
 '9800' '9100' '7900' '6400' '8600' '18900' '21600' '4300' '8500' '11300'
 '7200' '7700' '14600' '6300' '7100' '11800' '9400' '5500' '5800' '13200'
 '12500' '5600' '7000' '11900' '10400' '10700' '12700' '6800' '6500'
 '13600' '10200' '9000' '14900' '8200' '15200' '5000' '16300' '12400'
 '10500' '4200' '4700' '10900' '8100' '9500' '2200' '12800' '11200'
 '19100' '?' '12300' '16700' '2600' '26400' '8800' '7400' '4900' '8000'
 '12000' '15700' '4100' '5700' '11500' '5400' '10800' '9900' '5200' '5900'
 '9300' '9700' '5100' '6600']
Red Blood Cell Count ['5.2' nan '3.9' '4.6' '4.4' '5' '4' '3.7' '3.8' '3.4' '2.6' '2.8' '4.3'
 '3.2' '3.6' '4.1' '4.9' '2.5' '4.2' '4.5' '3.1' '4.7' '3.5' '6' '2.1'
 '5.6' '2.3' '2.9' '2.7' '8' '3.3' '3' '2.4' '4.8' '?' '5.4' '6.1' '6.2'
 '6.3' '5.1' '5.8' '5.5' '5.3' '6.4' '5.7' '5.9' '6.5']
Hypertension ['yes' 'no' nan]
Diabetes Mellitus ['yes' 'no' nan]
Coronary Artery Disease ['no' 'yes' nan]
Appetite ['good' 'poor' nan]
Pedal Edema ['no' 'yes' nan]
Anemia ['no' 'yes' nan]
classification ['ckd' 'notckd']

df1.nunique()

     
0
id	400
age	76
Blood Pressure	10
Specific Gravity	5
Albumin	6
Sugar	6
Red Blood Cells	2
Pus Cells	2
Pus Cell Clumps	2
Bacteria	2
Blood Glucose Random	146
Blood Urea	118
Serum Creatinine	84
Sodium	34
Potassium	40
Hemoglobin	115
Packed Cell Volume	43
White Blood Cell Count	90
Red Blood Cell Count	46
Hypertension	2
Diabetes Mellitus	2
Coronary Artery Disease	2
Appetite	2
Pedal Edema	2
Anemia	2
classification	2

dtype: int64

df1.fillna(method='bfill', inplace=True)
     
<ipython-input-15-66f8028945d9>:1: FutureWarning: DataFrame.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.
  df1.fillna(method='bfill', inplace=True)

df1.info()
     
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 400 entries, 0 to 399
Data columns (total 25 columns):
 #   Column                   Non-Null Count  Dtype  
---  ------                   --------------  -----  
 0   age                      400 non-null    float64
 1   Blood Pressure           400 non-null    float64
 2   Specific Gravity         400 non-null    float64
 3   Albumin                  400 non-null    float64
 4   Sugar                    400 non-null    float64
 5   Red Blood Cells          400 non-null    object 
 6   Pus Cells                400 non-null    object 
 7   Pus Cell Clumps          400 non-null    object 
 8   Bacteria                 400 non-null    object 
 9   Blood Glucose Random     400 non-null    float64
 10  Blood Urea               400 non-null    float64
 11  Serum Creatinine         400 non-null    float64
 12  Sodium                   400 non-null    float64
 13  Potassium                400 non-null    float64
 14  Hemoglobin               400 non-null    float64
 15  Packed Cell Volume       400 non-null    object 
 16  White Blood Cell Count   400 non-null    object 
 17  Red Blood Cell Count     400 non-null    object 
 18  Hypertension             400 non-null    object 
 19  Diabetes Mellitus        400 non-null    object 
 20  Coronary Artery Disease  400 non-null    object 
 21  Appetite                 400 non-null    object 
 22  Pedal Edema              400 non-null    object 
 23  Anemia                   400 non-null    object 
 24  classification           400 non-null    object 
dtypes: float64(11), object(14)
memory usage: 78.2+ KB

df1.columns
     
Index(['id', 'age', 'Blood Pressure', 'Specific Gravity', 'Albumin', 'Sugar',
       'Red Blood Cells', 'Pus Cells', 'Pus Cell Clumps', 'Bacteria',
       'Blood Glucose Random', 'Blood Urea', 'Serum Creatinine', 'Sodium',
       'Potassium', 'Hemoglobin', 'Packed Cell Volume',
       'White Blood Cell Count', 'Red Blood Cell Count', 'Hypertension',
       'Diabetes Mellitus', 'Coronary Artery Disease', 'Appetite',
       'Pedal Edema', 'Anemia', 'classification'],
      dtype='object')

df1.drop(['id'], axis=1, inplace=True)
     

df1.info()
     
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 400 entries, 0 to 399
Data columns (total 25 columns):
 #   Column                   Non-Null Count  Dtype  
---  ------                   --------------  -----  
 0   age                      400 non-null    float64
 1   Blood Pressure           400 non-null    float64
 2   Specific Gravity         400 non-null    float64
 3   Albumin                  400 non-null    float64
 4   Sugar                    400 non-null    float64
 5   Red Blood Cells          400 non-null    object 
 6   Pus Cells                400 non-null    object 
 7   Pus Cell Clumps          400 non-null    object 
 8   Bacteria                 400 non-null    object 
 9   Blood Glucose Random     400 non-null    float64
 10  Blood Urea               400 non-null    float64
 11  Serum Creatinine         400 non-null    float64
 12  Sodium                   400 non-null    float64
 13  Potassium                400 non-null    float64
 14  Hemoglobin               400 non-null    float64
 15  Packed Cell Volume       400 non-null    object 
 16  White Blood Cell Count   400 non-null    object 
 17  Red Blood Cell Count     400 non-null    object 
 18  Hypertension             400 non-null    object 
 19  Diabetes Mellitus        400 non-null    object 
 20  Coronary Artery Disease  400 non-null    object 
 21  Appetite                 400 non-null    object 
 22  Pedal Edema              400 non-null    object 
 23  Anemia                   400 non-null    object 
 24  classification           400 non-null    object 
dtypes: float64(11), object(14)
memory usage: 78.2+ KB

df1.nunique()
     
0
age	76
Blood Pressure	10
Specific Gravity	5
Albumin	6
Sugar	6
Pus Cells	2
Pus Cell Clumps	2
Bacteria	2
Blood Glucose Random	146
Blood Urea	118
Serum Creatinine	84
Hemoglobin	115
Packed Cell Volume	43
White Blood Cell Count	90
Red Blood Cell Count	46
Hypertension	2
Diabetes Mellitus	2
Coronary Artery Disease	2
Appetite	2
Pedal Edema	2
Anemia	2
classification	2

dtype: int64

for i in df1.select_dtypes(include='object').columns:
  print(i,df1[i].unique())
     

from sklearn.preprocessing import OrdinalEncoder
encoder=OrdinalEncoder()
     

for i in df1.select_dtypes(include='object').columns:
  df1[i]=encoder.fit_transform(df1[[i]])
     

df1.info()
     
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 400 entries, 0 to 399
Data columns (total 25 columns):
 #   Column                   Non-Null Count  Dtype  
---  ------                   --------------  -----  
 0   age                      400 non-null    float64
 1   Blood Pressure           400 non-null    float64
 2   Specific Gravity         400 non-null    float64
 3   Albumin                  400 non-null    float64
 4   Sugar                    400 non-null    float64
 5   Red Blood Cells          400 non-null    float64
 6   Pus Cells                400 non-null    float64
 7   Pus Cell Clumps          400 non-null    float64
 8   Bacteria                 400 non-null    float64
 9   Blood Glucose Random     400 non-null    float64
 10  Blood Urea               400 non-null    float64
 11  Serum Creatinine         400 non-null    float64
 12  Sodium                   400 non-null    float64
 13  Potassium                400 non-null    float64
 14  Hemoglobin               400 non-null    float64
 15  Packed Cell Volume       400 non-null    float64
 16  White Blood Cell Count   400 non-null    float64
 17  Red Blood Cell Count     400 non-null    float64
 18  Hypertension             400 non-null    float64
 19  Diabetes Mellitus        400 non-null    float64
 20  Coronary Artery Disease  400 non-null    float64
 21  Appetite                 400 non-null    float64
 22  Pedal Edema              400 non-null    float64
 23  Anemia                   400 non-null    float64
 24  classification           400 non-null    float64
dtypes: float64(25)
memory usage: 78.2 KB

df1['classification'].value_counts()
     
count
classification	
0.0	250
1.0	150

dtype: int64

a = df1.drop(['classification'], axis=1)
b = df1['classification']
     

from sklearn.model_selection import train_test_split
a_train,a_test,b_train,b_test=train_test_split(a,b,test_size=0.2)
a_train.shape,a_test.shape,b_train.shape,b_test.shape
     
((320, 24), (80, 24), (320,), (80,))

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier().fit(a_train,b_train)
b_pred=model.predict(a_test)
     

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
     

print(classification_report(b_test,b_pred))
     
              precision    recall  f1-score   support

         0.0       0.94      1.00      0.97        49
         1.0       1.00      0.90      0.95        31

    accuracy                           0.96        80
   macro avg       0.97      0.95      0.96        80
weighted avg       0.96      0.96      0.96        80


from sklearn.ensemble import RandomForestClassifier
model1=RandomForestClassifier().fit(a_train,b_train)
b_pred=model1.predict(a_test)
print(classification_report(b_test,b_pred))
     
              precision    recall  f1-score   support

         0.0       1.00      1.00      1.00        49
         1.0       1.00      1.00      1.00        31

    accuracy                           1.00        80
   macro avg       1.00      1.00      1.00        80
weighted avg       1.00      1.00      1.00        80


from xgboost import XGBClassifier
model3=XGBClassifier().fit(a_train,b_train)
b_pred=model3.predict(a_test)
print(classification_report(b_test,b_pred))
     
              precision    recall  f1-score   support

         0.0       0.94      1.00      0.97        49
         1.0       1.00      0.90      0.95        31

    accuracy                           0.96        80
   macro avg       0.97      0.95      0.96        80
weighted avg       0.96      0.96      0.96        80

parkinsons

df3=pd.read_csv('/content/parkinsons - parkinsons.csv')
df3
     
name	MDVP:Fo(Hz)	MDVP:Fhi(Hz)	MDVP:Flo(Hz)	MDVP:Jitter(%)	MDVP:Jitter(Abs)	MDVP:RAP	MDVP:PPQ	Jitter:DDP	MDVP:Shimmer	...	Shimmer:DDA	NHR	HNR	status	RPDE	DFA	spread1	spread2	D2	PPE
0	phon_R01_S01_1	119.992	157.302	74.997	0.00784	0.00007	0.00370	0.00554	0.01109	0.04374	...	0.06545	0.02211	21.033	1	0.414783	0.815285	-4.813031	0.266482	2.301442	0.284654
1	phon_R01_S01_2	122.400	148.650	113.819	0.00968	0.00008	0.00465	0.00696	0.01394	0.06134	...	0.09403	0.01929	19.085	1	0.458359	0.819521	-4.075192	0.335590	2.486855	0.368674
2	phon_R01_S01_3	116.682	131.111	111.555	0.01050	0.00009	0.00544	0.00781	0.01633	0.05233	...	0.08270	0.01309	20.651	1	0.429895	0.825288	-4.443179	0.311173	2.342259	0.332634
3	phon_R01_S01_4	116.676	137.871	111.366	0.00997	0.00009	0.00502	0.00698	0.01505	0.05492	...	0.08771	0.01353	20.644	1	0.434969	0.819235	-4.117501	0.334147	2.405554	0.368975
4	phon_R01_S01_5	116.014	141.781	110.655	0.01284	0.00011	0.00655	0.00908	0.01966	0.06425	...	0.10470	0.01767	19.649	1	0.417356	0.823484	-3.747787	0.234513	2.332180	0.410335
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
190	phon_R01_S50_2	174.188	230.978	94.261	0.00459	0.00003	0.00263	0.00259	0.00790	0.04087	...	0.07008	0.02764	19.517	0	0.448439	0.657899	-6.538586	0.121952	2.657476	0.133050
191	phon_R01_S50_3	209.516	253.017	89.488	0.00564	0.00003	0.00331	0.00292	0.00994	0.02751	...	0.04812	0.01810	19.147	0	0.431674	0.683244	-6.195325	0.129303	2.784312	0.168895
192	phon_R01_S50_4	174.688	240.005	74.287	0.01360	0.00008	0.00624	0.00564	0.01873	0.02308	...	0.03804	0.10715	17.883	0	0.407567	0.655683	-6.787197	0.158453	2.679772	0.131728
193	phon_R01_S50_5	198.764	396.961	74.904	0.00740	0.00004	0.00370	0.00390	0.01109	0.02296	...	0.03794	0.07223	19.020	0	0.451221	0.643956	-6.744577	0.207454	2.138608	0.123306
194	phon_R01_S50_6	214.289	260.277	77.973	0.00567	0.00003	0.00295	0.00317	0.00885	0.01884	...	0.03078	0.04398	21.209	0	0.462803	0.664357	-5.724056	0.190667	2.555477	0.148569
195 rows × 24 columns


pd.set_option('display.max_columns',None)
df3
     
name	MDVP:Fo(Hz)	MDVP:Fhi(Hz)	MDVP:Flo(Hz)	MDVP:Jitter(%)	MDVP:Jitter(Abs)	MDVP:RAP	MDVP:PPQ	Jitter:DDP	MDVP:Shimmer	MDVP:Shimmer(dB)	Shimmer:APQ3	Shimmer:APQ5	MDVP:APQ	Shimmer:DDA	NHR	HNR	status	RPDE	DFA	spread1	spread2	D2	PPE
0	phon_R01_S01_1	119.992	157.302	74.997	0.00784	0.00007	0.00370	0.00554	0.01109	0.04374	0.426	0.02182	0.03130	0.02971	0.06545	0.02211	21.033	1	0.414783	0.815285	-4.813031	0.266482	2.301442	0.284654
1	phon_R01_S01_2	122.400	148.650	113.819	0.00968	0.00008	0.00465	0.00696	0.01394	0.06134	0.626	0.03134	0.04518	0.04368	0.09403	0.01929	19.085	1	0.458359	0.819521	-4.075192	0.335590	2.486855	0.368674
2	phon_R01_S01_3	116.682	131.111	111.555	0.01050	0.00009	0.00544	0.00781	0.01633	0.05233	0.482	0.02757	0.03858	0.03590	0.08270	0.01309	20.651	1	0.429895	0.825288	-4.443179	0.311173	2.342259	0.332634
3	phon_R01_S01_4	116.676	137.871	111.366	0.00997	0.00009	0.00502	0.00698	0.01505	0.05492	0.517	0.02924	0.04005	0.03772	0.08771	0.01353	20.644	1	0.434969	0.819235	-4.117501	0.334147	2.405554	0.368975
4	phon_R01_S01_5	116.014	141.781	110.655	0.01284	0.00011	0.00655	0.00908	0.01966	0.06425	0.584	0.03490	0.04825	0.04465	0.10470	0.01767	19.649	1	0.417356	0.823484	-3.747787	0.234513	2.332180	0.410335
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
190	phon_R01_S50_2	174.188	230.978	94.261	0.00459	0.00003	0.00263	0.00259	0.00790	0.04087	0.405	0.02336	0.02498	0.02745	0.07008	0.02764	19.517	0	0.448439	0.657899	-6.538586	0.121952	2.657476	0.133050
191	phon_R01_S50_3	209.516	253.017	89.488	0.00564	0.00003	0.00331	0.00292	0.00994	0.02751	0.263	0.01604	0.01657	0.01879	0.04812	0.01810	19.147	0	0.431674	0.683244	-6.195325	0.129303	2.784312	0.168895
192	phon_R01_S50_4	174.688	240.005	74.287	0.01360	0.00008	0.00624	0.00564	0.01873	0.02308	0.256	0.01268	0.01365	0.01667	0.03804	0.10715	17.883	0	0.407567	0.655683	-6.787197	0.158453	2.679772	0.131728
193	phon_R01_S50_5	198.764	396.961	74.904	0.00740	0.00004	0.00370	0.00390	0.01109	0.02296	0.241	0.01265	0.01321	0.01588	0.03794	0.07223	19.020	0	0.451221	0.643956	-6.744577	0.207454	2.138608	0.123306
194	phon_R01_S50_6	214.289	260.277	77.973	0.00567	0.00003	0.00295	0.00317	0.00885	0.01884	0.190	0.01026	0.01161	0.01373	0.03078	0.04398	21.209	0	0.462803	0.664357	-5.724056	0.190667	2.555477	0.148569
195 rows × 24 columns


df3.isnull().sum()
     
0
name	0
MDVP:Fo(Hz)	0
MDVP:Fhi(Hz)	0
MDVP:Flo(Hz)	0
MDVP:Jitter(%)	0
MDVP:Jitter(Abs)	0
MDVP:RAP	0
MDVP:PPQ	0
Jitter:DDP	0
MDVP:Shimmer	0
MDVP:Shimmer(dB)	0
Shimmer:APQ3	0
Shimmer:APQ5	0
MDVP:APQ	0
Shimmer:DDA	0
NHR	0
HNR	0
status	0
RPDE	0
DFA	0
spread1	0
spread2	0
D2	0
PPE	0

dtype: int64

df3.nunique()
     
0
name	195
MDVP:Fo(Hz)	195
MDVP:Fhi(Hz)	195
MDVP:Flo(Hz)	195
MDVP:Jitter(%)	173
MDVP:Jitter(Abs)	19
MDVP:RAP	155
MDVP:PPQ	165
Jitter:DDP	180
MDVP:Shimmer	188
MDVP:Shimmer(dB)	149
Shimmer:APQ3	184
Shimmer:APQ5	189
MDVP:APQ	189
Shimmer:DDA	189
NHR	185
HNR	195
status	2
RPDE	195
DFA	195
spread1	195
spread2	194
D2	195
PPE	195

dtype: int64

df3.columns
     
Index(['name', 'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
       'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
       'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
       'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'status', 'RPDE', 'DFA',
       'spread1', 'spread2', 'D2', 'PPE'],
      dtype='object')

df3.info()
     
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 195 entries, 0 to 194
Data columns (total 24 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   name              195 non-null    object 
 1   MDVP:Fo(Hz)       195 non-null    float64
 2   MDVP:Fhi(Hz)      195 non-null    float64
 3   MDVP:Flo(Hz)      195 non-null    float64
 4   MDVP:Jitter(%)    195 non-null    float64
 5   MDVP:Jitter(Abs)  195 non-null    float64
 6   MDVP:RAP          195 non-null    float64
 7   MDVP:PPQ          195 non-null    float64
 8   Jitter:DDP        195 non-null    float64
 9   MDVP:Shimmer      195 non-null    float64
 10  MDVP:Shimmer(dB)  195 non-null    float64
 11  Shimmer:APQ3      195 non-null    float64
 12  Shimmer:APQ5      195 non-null    float64
 13  MDVP:APQ          195 non-null    float64
 14  Shimmer:DDA       195 non-null    float64
 15  NHR               195 non-null    float64
 16  HNR               195 non-null    float64
 17  status            195 non-null    int64  
 18  RPDE              195 non-null    float64
 19  DFA               195 non-null    float64
 20  spread1           195 non-null    float64
 21  spread2           195 non-null    float64
 22  D2                195 non-null    float64
 23  PPE               195 non-null    float64
dtypes: float64(22), int64(1), object(1)
memory usage: 36.7+ KB

# Column renaming dictionary
new_column_names = {
    'MDVP:Fo(Hz)': 'Median_Frequency_Hz',
    'MDVP:Fhi(Hz)': 'Max_Frequency_Hz',
    'MDVP:Flo(Hz)': 'Min_Frequency_Hz',
    'MDVP:Jitter(%)': 'Jitter_Percentage',
    'MDVP:Jitter(Abs)': 'Jitter_Absolute',
    'MDVP:RAP': 'Relative_Amplitude_Perturbation',
    'MDVP:PPQ': 'Period_Perturbation_Quotient',
    'Jitter:DDP': 'Jitter_DDP',
    'MDVP:Shimmer': 'Shimmer',
    'MDVP:Shimmer(dB)': 'Shimmer_dB',
    'Shimmer:APQ3': 'Amplitude_Perturbation_Quotient_3',
    'Shimmer:APQ5': 'Amplitude_Perturbation_Quotient_5',
    'MDVP:APQ': 'Amplitude_Perturbation_Quotient',
    'Shimmer:DDA': 'Shimmer_DDA',
    'NHR': 'Noise_to_Harmonics_Ratio',
    'HNR': 'Harmonics_to_Noise_Ratio',
    'status': 'Parkinson_Status',
    'RPDE': 'Recurrence_Period_Density_Entropy',
    'DFA': 'Detrended_Fluctuation_Analysis',
    'spread1': 'Frequency_Spread_1',
    'spread2': 'Frequency_Spread_2',
    'D2': 'Correlation_Dimension',
    'PPE': 'Pitch_Period_Entropy'
}

# Rename columns
df3.rename(columns=new_column_names, inplace=True)

     

df3.info()
     
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 195 entries, 0 to 194
Data columns (total 24 columns):
 #   Column                             Non-Null Count  Dtype  
---  ------                             --------------  -----  
 0   name                               195 non-null    object 
 1   Median_Frequency_Hz                195 non-null    float64
 2   Max_Frequency_Hz                   195 non-null    float64
 3   Min_Frequency_Hz                   195 non-null    float64
 4   Jitter_Percentage                  195 non-null    float64
 5   Jitter_Absolute                    195 non-null    float64
 6   Relative_Amplitude_Perturbation    195 non-null    float64
 7   Period_Perturbation_Quotient       195 non-null    float64
 8   Jitter_DDP                         195 non-null    float64
 9   Shimmer                            195 non-null    float64
 10  Shimmer_dB                         195 non-null    float64
 11  Amplitude_Perturbation_Quotient_3  195 non-null    float64
 12  Amplitude_Perturbation_Quotient_5  195 non-null    float64
 13  Amplitude_Perturbation_Quotient    195 non-null    float64
 14  Shimmer_DDA                        195 non-null    float64
 15  Noise_to_Harmonics_Ratio           195 non-null    float64
 16  Harmonics_to_Noise_Ratio           195 non-null    float64
 17  Parkinson_Status                   195 non-null    int64  
 18  Recurrence_Period_Density_Entropy  195 non-null    float64
 19  Detrended_Fluctuation_Analysis     195 non-null    float64
 20  Frequency_Spread_1                 195 non-null    float64
 21  Frequency_Spread_2                 195 non-null    float64
 22  Correlation_Dimension              195 non-null    float64
 23  Pitch_Period_Entropy               195 non-null    float64
dtypes: float64(22), int64(1), object(1)
memory usage: 36.7+ KB

c = df3.drop(['Parkinson_Status'], axis=1)
d = df3['Parkinson_Status']
     

from sklearn.model_selection import train_test_split
c_train,c_test,d_train,d_test=train_test_split(c,d,test_size=0.2)
c_train.shape,c_test.shape,d_train.shape,d_test.shape
     
((156, 23), (39, 23), (156,), (39,))

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier().fit(c_train,d_train)
d_pred=model.predict(c_test)
print(classification_report(c_test,d_pred))
     

from sklearn.ensemble import RandomForestClassifier
model1=RandomForestClassifier().fit(c_train,d_train)
d_pred=model1.predict(c_test)
print(classification_report(c_test,d_pred))
     

from xgboost import XGBClassifier
model3=XGBClassifier().fit(c_train,d_train)
d_pred=model3.predict(c_test)
print(classification_report(c_test,d_pred))
     
