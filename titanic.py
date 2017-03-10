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

train_df = pd.read_csv('Datasets/train.csv')
test_df = pd.read_csv('Datasets/test.csv')
print(train_df.shape)
print(train_df.describe())
print(train_df.describe(include=['O']))
train_df.info()

##train.hist()

train_df = etl(train_df)
test_df = etl(test_df)

#train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
#train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

#train_df['FareBand'] = pd.cut(train_df['Fare'], 3)
#train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)

train_df[['Age', 'Survived']].groupby(['Age'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['Age', 'Survived']].groupby(['Age'], as_index=False).mean().sort_values(by='Age', ascending=False)



for v in ['Title','Embarked','Pclass','Sex',"Parch",'SibSp','Label']:
    print(v)
    print(train_df[[v, 'Survived']].groupby([v], as_index=False).agg([np.size, np.mean]))
    #print(train_df[[v, 'Survived']].groupby([v], as_index=False).apply(lambda x: (x.mean()[-1],x.count()[-1])))
    #print(train_df[[v, 'Survived']].groupby([v], as_index=False).count())
    print('-'*50)

for v in ['Fare','Age']:
    print(v)
    print(train_df[['Survived',v]].groupby(['Survived'], as_index=False).apply(lambda x: (x.mean()[-1],x.count()[-1])))
    #print(train_df[[v, 'Survived']].groupby([v], as_index=False).count())
    print('-'*50)

pd.crosstab(train_df['Title'], train_df['Sex'])
pd.crosstab(train_df['Parch'], train_df['SibSp'])

#g = sns.FacetGrid(train_df, col='Survived')
#g.map(plt.hist, 'Fare', bins=20)
#
#grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
#grid.map(plt.hist, 'Fare', alpha=.5, bins=20)
#grid.add_legend()


target = ['Survived']
features = list(train_df.columns)
features.remove(target[0])
#features.remove('Age')
features.remove('Cabin')
features.remove('PassengerId')
features.remove('Ticket')

X=train_df[features[:]].values
y=train_df[target].values
iv = WOE().woe(X,y)
score = zip(features,list(iv[1]))
#sorted(score, key=lambda x: x[1], reverse=True)
selected_f = [v[0] for v in score if v[1]>0]
selected_f.remove('Title')
selected_f.remove('Sex')
selected_f.remove('Embarked')
selected_f.remove('Pclass')
selected_f.remove('S')
selected_f.remove('Miss')
selected_f.remove('Alone')
selected_f.remove(3)
selected_f.remove('SexB')
#selected_f.remove('AgeYoung')

X_train = train_df[selected_f]
Y_train = train_df[target[0]]
X_test  = test_df[selected_f]

selected_fl=selected_f.copy()
selected_fl.remove('FareB')
selected_fl.remove('AgeYoung')
#selected_fl.remove('SibSpB')
logreg = LogisticRegression()
logreg.fit(X_train[selected_fl], Y_train)
Y_pred = logreg.predict(X_test[selected_fl])
acc_log = round(logreg.score(X_train[selected_fl], Y_train) * 100, 2)
print(acc_log)

coeff_df = pd.DataFrame(selected_fl)
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)
