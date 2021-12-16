# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 14:53:00 2019

@author: svsl156
"""
#conn = pyodbc.connect('DSN=SQLSERVER;UID=uno;PWD=uno')

import pandas as pd
import numpy as np
         
#Read LoanWiseConsolidatedTransactions.csv file and store it as Data Frame into df1.
df1 = pd.read_csv("E:\data-analytics\Practice\LoanWiseConsolidatedTransactions.csv")
#Read LoanWiseEmiDefaultCount.csv file and store it as Data Frame into df2.
df2 = pd.read_csv("E:\data-analytics\Practice\LoanWiseEmiDefaultCount.csv")
#Read Customer.csv file and store it as Data Frame into df3.
df3 = pd.read_csv("E:\data-analytics\Practice\Customer.csv")


# Merge df1 and df2 with LoanRefNumber column refrence and assign into merge1
merge1 = df1.merge(df2, on = 'LoanRefNumber')
# It drop column wise, how='all'(If all values are NA, drop that row or column.)
# Here It drop two column(GRADE OF THE PAYEE and INTEREST RECEIVED TILL DATE) which all value is nan
merger1 = merge1.dropna(axis=1, how='all')

#It give the number of merge1.columns and merger1.columns
len(merge1.columns)
len(merger1.columns)

##change datetime format to panda datetime format
merger1['LastDate'] = pd.to_datetime(merger1['LastDate'])
merger1['LAST PAYMENT DATE'] = pd.to_datetime(merger1['LAST PAYMENT DATE'])

#In dt initialize the datetime with format.
dt = pd.to_datetime('2018/04/01', format='%Y/%m/%d')
 
#Find max date from merger1['LAST PAYMENT DATE'] and assign into dt1.
dt1 = pd.to_datetime(max(merger1['LAST PAYMENT DATE']),format = '%Y/%m/%d') 
 
#Adding one column merger1['START DATE OF LOAN'] = merger1['LastDate] - merger1['TENOR'] change this to month wise + add 30 more days.
#This line having error (#pd.offsets.MonthOffset(1))  
merger1['START DATE OF LOAN'] = (merger1['LastDate'] - pd.to_timedelta(merger1['TENOR'], unit='M')) + pd.DateOffset(days=30)#pd.offsets.MonthOffset(1)

##change datetime format to panda datetime format
merger1['START DATE OF LOAN'] = pd.to_datetime(merger1['START DATE OF LOAN'])

##copying merger1['START DATE OF LOAN'] <= '2018/04/01' to merger2 and if > dt assign into mergernot2.
merger2 = merger1[(merger1['START DATE OF LOAN']<= dt)]
mergernot2 = merger1[(merger1['START DATE OF LOAN']> dt)]

#Its checking is there null value in merger2.columns
merger2.columns[merger2.isnull().any()]

##Drop duplicate value of Customer.csv with column 'CUSTCODE' wise and and store it as Data Frame into df3_new.
df3_new =df3.drop_duplicates('CUSTCODE')


##how='inner'(similar to a SQL inner join; preserve the order of the left keys)
##merge with merger1 and df3 on column CUSTCODE wise
merge_cust = merger1.merge(df3_new,on='CUSTCODE',how='inner')
##drop if all value is not avaliable, its checking column wise
merge_cust1 = merge_cust.dropna(axis=1, how='all')

  
################CREATE NEW VARIABLE WITH LATE PAYMENT OF EMI#################
#copying those element from merge_cust1['LastDate'] <= dt1, if in dt1 is not avaliable then check inmerge_cust1['TENOR'],((dt1 - merge_cust1['START DATE OF LOAN'])/np.timedelta64(1, 'M')+1))
merge_cust1['MONTHS COMPLETE IN TENOR'] = np.where((merge_cust1['LastDate'])<= dt1,merge_cust1['TENOR'],((dt1 - merge_cust1['START DATE OF LOAN'])/np.timedelta64(1, 'M')+1))

##defined the type as int.
merge_cust1['MONTHS COMPLETE IN TENOR'] = merge_cust1['MONTHS COMPLETE IN TENOR'].astype(int)

#Assign into merge_cust1 if Data Frame column merge_cust1[merge_cust1['MONTHS COMPLETE IN TENOR'] >=6.
merge_cust1=merge_cust1[merge_cust1['MONTHS COMPLETE IN TENOR']>=6]

##calculating percentage and round off also and assign into column merge_cust1['PERCENTAGE OF DEFAULT'].
merge_cust1['PERCENTAGE OF DEFAULT'] = ((merge_cust1['NUMBER OF TIMES DEFAULT']/merge_cust1['MONTHS COMPLETE IN TENOR']) * 100).round(2)

   
###############################  DEFINING CUSTOMERS  ###############
#if merge_cust1['PERCENTAGE OF DEFAULT']==0 return RELIABLE, if >0 and <=33 return UNRELIABLE, if >33 return RISKY
merge_cust1['CUSTOMER DEFINITION'] = np.where((merge_cust1['PERCENTAGE OF DEFAULT']==0),'Not Likely to Default',np.where((merge_cust1['PERCENTAGE OF DEFAULT']>0) & (merge_cust1['PERCENTAGE OF DEFAULT']<=33),'Likely to Default',np.where((merge_cust1['PERCENTAGE OF DEFAULT']>33),'Most Likely to Default',np.nan)))


######### FIND DISTRICT FROM PINCODE  ############
#Read pincode.csv file and store it as Data Frame into pincode
pincode = pd.read_csv("E:\data-analytics\Practice\pincode.csv") 


# Rename column pincode with PinCode and assign into a variable
pincode=pincode.rename(columns = {'pincode':'PinCode'})

#create a list to  hold 2 elements and  assign into a variable.
cols = ['PinCode','districtname']
#Initialize pincode[cols] into a variable.
pincode = pincode[cols]


#Remove duplicate from pincode DataFrame/dataset
pincode =pincode.drop_duplicates('PinCode',keep='first')

#Dataset drop row which is subset of ['PinCode'] 
merge_cust1 = merge_cust1.dropna(subset=['PinCode'],how='all') 

# Convert dtype string to int assign into merge_cust1 df/ds and updated
merge_cust1['PinCode'] = merge_cust1['PinCode'].astype(int)


# merge pincode takes the left dataframes and matches rows based on the "on" PinCode assing into a variable
merge_cust1 = merge_cust1.merge(pincode,on='PinCode',how='left')


# Convert into uppercase assign into a variable
merge_cust1['districtname'] = merge_cust1['districtname'].apply(lambda x: str(x).upper())


#Preprocessing
# checking multiple condition with merge_cust1 df/ds assign into a varibale
merge_cust1['MODE OF PAYMENT'] = np.where((merge_cust1['MODE OF PAYMENT']=='A')|(merge_cust1['MODE OF PAYMENT']=='E')|(merge_cust1['MODE OF PAYMENT']=='H'), 'E', merge_cust1['MODE OF PAYMENT'])
merge_cust1['MODE OF PAYMENT'] = np.where((merge_cust1['MODE OF PAYMENT']=='N')|(merge_cust1['MODE OF PAYMENT']=='O'), 'O', merge_cust1['MODE OF PAYMENT'])
merge_cust1['MARITAL STATUS'] = np.where((merge_cust1['MARITAL STATUS']=='S')|(merge_cust1['MARITAL STATUS']=='U'), 'S', merge_cust1['MARITAL STATUS'])
merge_cust1['OCCUPATION'] = np.where((merge_cust1['OCCUPATION']=='PRIVATE LIMITED COMPANY')|(merge_cust1['OCCUPATION']=='PRIVATE LIMITED'), 'PRIVATE LIMITED', merge_cust1['OCCUPATION'])
merge_cust1['OCCUPATION'] = np.where((merge_cust1['OCCUPATION']=='PROPRIETERSHIP')|(merge_cust1['OCCUPATION']=='PROPRIETORSHIP'), 'PROPRIETORSHIP', merge_cust1['OCCUPATION'])
merge_cust1['OCCUPATION'] = np.where((merge_cust1['OCCUPATION']=='PROFESSIONAL GROUP')|(merge_cust1['OCCUPATION']=='PROFESSIONAL'), 'PROFESSIONAL', merge_cust1['OCCUPATION'])
merge_cust1['OCCUPATION'] = np.where((merge_cust1['OCCUPATION']=='PUBLIC LIMITED COMPANY')|(merge_cust1['OCCUPATION']=='PUBLIC LIMITED'), 'PUBLIC LIMITED', merge_cust1['OCCUPATION'])
merge_cust1['TWO WHEELER MAKE'] = np.where((merge_cust1['TWO WHEELER MAKE']=='AMPERE')|(merge_cust1['TWO WHEELER MAKE']=='AMPIRE'), 'AMPIRE', merge_cust1['TWO WHEELER MAKE'])
merge_cust1['TWO WHEELER MAKE'] = np.where((merge_cust1['TWO WHEELER MAKE']=='HERO')|(merge_cust1['TWO WHEELER MAKE']=='HERO MOTORS')|(merge_cust1['TWO WHEELER MAKE']=='HERO MOTOR CORP'), 'HERO MOTOR CORP', merge_cust1['TWO WHEELER MAKE'])
merge_cust1['TWO WHEELER MAKE'] = np.where((merge_cust1['TWO WHEELER MAKE']=='MONTO MOTORS')|(merge_cust1['TWO WHEELER MAKE']=='MONTO'),'MONTO MOTOR', merge_cust1['TWO WHEELER MAKE'])
merge_cust1['TWO WHEELER MAKE'] = np.where((merge_cust1['TWO WHEELER MAKE']=='SUSUKI')|(merge_cust1['TWO WHEELER MAKE']=='SUZUKI'),'SUZUKI', merge_cust1['TWO WHEELER MAKE'])

#create list with hold 7 elements assign into a variable.
bins = [0, 6, 12, 18, 24, 30, 37]

#create list  with hold 6 elements assign into  a variable.
names = ['1to6', '7to12', '13to18', '19to24', '25to30', '31to36']

# Convert into dictionary with index wise and assign into a variable
d = dict(enumerate(names, 1))

#Return the indices of the  "bins" to which each value in input array belongs
merge_cust1['TENOR'] = np.vectorize(d.get)(np.digitize(merge_cust1['TENOR'], bins)) 


# Fill in missing ages with median and Replace all the NaN values with 0.
merge_cust1['AGE']=merge_cust1['AGE'].fillna(merge_cust1['AGE'].median())

#create list with hold 6 elements assign into a variable .
bins = [0, 23, 32, 43, 53, np.inf]

#create list with hold 5 elements assign into a variable.
names = ['Young', 'Young Adult', 'Middle aged Adult', 'Senior Adult', 'Old Age']

#Convert into dictionary with index wise and assign into a variable
d = dict(enumerate(names, 1))

# Return the indices of the  "bins" to which each value in input array belongs. Ex- bins[names-1] <= merge_cust1['TENOR'] < bins[names]
merge_cust1['AGE'] = np.vectorize(d.get)(np.digitize(merge_cust1['AGE'], bins))

######### IMPUTE MARITAL STATUS,SEX,OCCUPATION  #############
# Replace the blank('') in the DataFrame with np.NaN assigh into a variable
merge_cust1['MARITAL STATUS'] = merge_cust1['MARITAL STATUS'].replace(' ', np.nan, regex=True)


# Return those value from Data Frame whose MARITAL STATUS is 'M'.
merge_cust1['MARITAL STATUS'] = np.where((pd.isna(merge_cust1['MARITAL STATUS'])),'M',merge_cust1['MARITAL STATUS'])

# Replace the blank('') in the DataFrame with np.NaN assigh into a variable
merge_cust1['SEX'] = merge_cust1['SEX'].replace(' ', np.nan, regex=True)

# isna function check whether a DataFrame has one (or more) NaN values? return Boolean data (True,False)
merge_cust1['SEX'] = np.where((pd.isna(merge_cust1['SEX'])),'M',merge_cust1['SEX'])

# isna function check whether a DataFrame has one (or more) NaN values? return Boolean data (True,False)
merge_cust1['OCCUPATION'] = np.where((pd.isna(merge_cust1['OCCUPATION'])),'SELF-EMPLOYED',merge_cust1['OCCUPATION'])

#####Salary Imputation###
merge_cust1['SALARY'] = np.where(((merge_cust1['SALARY']<6000) | (merge_cust1['SALARY']>300000)), np.nan, merge_cust1['SALARY'])

merge_cust1['INCOME'] = df3_new['AnualIncome']/12

# show data row and col which is use to axis=1
merge_cust1 = merge_cust1.drop(labels='AnualIncome',axis=1)
merge_cust1['INCOME'] = np.where(((merge_cust1['INCOME']<6000) | (merge_cust1['INCOME']>300000)), np.nan, merge_cust1['INCOME'])
merge_cust1 = merge_cust1.dropna(subset=['OCCUPATION'],how='all') 
# replacing na values in merge_cust1['SALARY'] with merge_cust1["INCOME"] 
merge_cust1['FINAL_INCOME'] = merge_cust1["SALARY"].fillna(merge_cust1["INCOME"])
#####Count nas in salary by occupation####

occ = set(list(merge_cust1['OCCUPATION']))

#Assign an index column to dataframe 
df = pd.DataFrame(columns=['occupation','mean','median','na','total'])

# Iterate over the set
for i in occ:
    occupation = i
    #you will get basic statistics of the dataframe and to get mean of specific column you can use
    mean=merge_cust1['FINAL_INCOME'][merge_cust1['OCCUPATION']==i].mean()
    #check occupation and income median function 
    #The sorted() function is very helpful for this median() Median = {(n + 1) / 2}
    median=merge_cust1['FINAL_INCOME'][merge_cust1['OCCUPATION']==i].median()

    # isna function check whether a DataFrame has one (or more) NaN values? return Boolean data
    # merge_cust1 name of column for which you want to calculate the nan values assign into a variable
    na=merge_cust1['FINAL_INCOME'][merge_cust1['OCCUPATION']==i].isna().sum()

    # check a length of merge_cust1 df/ds assign into a variable
    total=len(merge_cust1['FINAL_INCOME'][merge_cust1['OCCUPATION']==i])
    data = pd.DataFrame()

    #SELF-EMPLOYED', u'SALARIED', u'SALARIED,u'FARMER 
    # Append rows of other to the end of this df/ds assign into a variable
    df=df.append({"occupation":occupation,"mean":mean,"median":median,"na":na,"total":total},ignore_index=True)
####IMPUTE SALARY WITH MEDIAN OF THAT OCCUPATION#####
# check final income and occupation
# Iterate over the set
for i in occ:
    #The sorted() function is very helpful for this median() Median = {(n + 1) / 2}
    median=merge_cust1['FINAL_INCOME'][merge_cust1['OCCUPATION']==i].median()

    # replacing na values in merge_cust1 with median 
    merge_cust1['FINAL_INCOME'].fillna(median,inplace=True)

#######   CONVERT SALARY INTO CATEGORICAL   ########
#create list can hold 5 elements assign into a variable .
bins = [6000, 11000, 16000, 21000, np.inf] 

#create list can hold 4 elements assign into a variable .
names = ['6000to11000', '11001to16000', '16001to21000', '>21000']

# Iterate names all key, value pairs assign into a variable
d = dict(enumerate(names, 1))

#Number of elements of merge_cust1 that fall within the grid points bins assign into a variable
merge_cust1['FINAL_INCOME'] = np.vectorize(d.get)(np.digitize(merge_cust1['FINAL_INCOME'], bins))
#create list with hold 12 elements assign into a variable.
drop_cols = ['FIRST TIMER/RETURING','CREDIT RATING OF THE CUSTOMER','VERIFICATION STATUS','VEHICLE OWNERSHIP','INCOME VERIFIED STATUS',
         'TWO WHEELER COST','TWO WHEELER MODEL','LAST PAYMENT AMOUNT','INSTALLMENT TERM','NET_LOAN','SALARY','INCOME']

# Dropping passed columns assign into a variable
merge_cust1 = merge_cust1.drop(labels=drop_cols,axis=1)
dfr = merge_cust1.copy(deep=True)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


#Select the labelEncoder
number = LabelEncoder()
#Encode columns into selecting wise.
dfr['AGE'] = number.fit_transform(dfr['AGE'].astype('str'))
dfr['TENOR'] = number.fit_transform(dfr['TENOR'].astype('str'))
dfr['SEX'] = number.fit_transform(dfr['SEX'].astype('str'))
dfr['HOME OWNERSHIP'] = number.fit_transform(dfr['HOME OWNERSHIP'].astype('str'))
dfr['MODE OF PAYMENT'] = number.fit_transform(dfr['MODE OF PAYMENT'].astype('str'))
dfr['MARITAL STATUS'] = number.fit_transform(dfr['MARITAL STATUS'].astype('str'))
dfr['OCCUPATION'] = number.fit_transform(dfr['OCCUPATION'].astype('str'))
dfr['OTHER LOAN TAKEN'] = number.fit_transform(dfr['OTHER LOAN TAKEN'].astype('str'))
dfr['TWO WHEELER MAKE'] = number.fit_transform(dfr['TWO WHEELER MAKE'].astype('str'))
dfr['FINAL_INCOME'] = number.fit_transform(dfr['FINAL_INCOME'].astype('str'))
dfr['districtname'] = number.fit_transform(dfr['districtname'].astype('str'))

#Creating list of unused columns
features =['CUSTCODE','LAST PAYMENT DATE','CUSTOMER TYPE','LoanRefNumber','LastDate','START DATE OF LOAN','CITY','PinCode','PERCENTAGE OF DEFAULT','NUMBER OF TIMES DEFAULT']
#drop columns
newdfr = dfr.drop(labels=features,axis=1)
newdfr['CUSTOMER DEFINITION'] = number.fit_transform(newdfr['CUSTOMER DEFINITION'].astype('str'))

#Set the target columns
y=newdfr['CUSTOMER DEFINITION']
x=newdfr.drop('CUSTOMER DEFINITION',axis=1)

# make predictions for test data
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

#spliting data into Train_data and Test_data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20,shuffle=True)
#selecting model
model = XGBClassifier()
model.fit(x_train, y_train)
#Predicting data with Customer wise
predict_data=pd.DataFrame(model.predict_proba(x), columns=["Not Likely to Default","Likely to Default","Most Likely to Default"])

#Merge predicted result with Original data.
Final_predicted_result = merge_cust1.merge(predict_data,left_index=True,right_index=True)
#Write into csv
Final_predicted_result.to_csv("Final_predicted_result.csv")

#It gives the result in percentage
predictions = model.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
