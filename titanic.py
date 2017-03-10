def etl(data_set):
    freq_port = data_set.Embarked.dropna().mode()[0]
    data_set['Embarked'] = data_set['Embarked'].fillna(freq_port)
    data_set['Title'] = data_set.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    data_set['Title'] = data_set['Title'].replace(['Dr', 'Rev','Major', 'Col', \
           'Sir', 'Don',  'Jonkheer', 'Capt'], 'Mr')
    data_set['Title'] = data_set['Title'].replace(['Mlle','Ms'], 'Miss')
    data_set['Title'] = data_set['Title'].replace(['Mme','Lady','Countess'], 'Mrs')
    data_set.ix[(data_set.Sex=='female') & (data_set.Title=='Mr'),'Title'] = 'Mrs'

    data_set['Label']=data_set.Cabin.str.get(0)
    data_set.ix[(data_set.Label.isnull()==True) ,'Label'] = 'O'
    data_set['Label'] = data_set['Label'].replace(['A','B','C','D','E','F','G'], 1)
    data_set['Label'] = data_set['Label'].replace(['O','T'], 0)

    data_set['AgeOld'] = 0
    data_set.ix[data_set.Age>=50,'AgeOld'] = 1
    data_set['AgeYoung'] = 0
    data_set.ix[(data_set.Age>8)&(data_set.Age<25),'AgeYoung'] = 1

    data_set['SexB'] = data_set['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

    title_dummies = pd.get_dummies(data_set['Title'])
    data_set = pd.concat([data_set, title_dummies], axis=1)

    embark_dummies = pd.get_dummies(data_set['Embarked'])
    data_set = pd.concat([data_set, embark_dummies], axis=1)

    class_dummies = pd.get_dummies(data_set['Pclass'])
    data_set = pd.concat([data_set, class_dummies], axis=1)

    data_set['FareB'] = 0
    data_set.ix[data_set.Fare<8,'FareB'] = 1
#    data_set.ix[data_set.Fare==0,'FareB'] = 0

    data_set['ParchB'] = 0
    data_set.ix[data_set.Parch>0,'ParchB'] = 1
    data_set['SibSpB'] = 0
    data_set.ix[data_set.SibSp>0,'SibSpB'] = 1
    data_set['Alone'] = 1
    data_set.ix[(data_set.SibSp>0)|(data_set.Parch>0),'Alone'] = 0
    return data_set
