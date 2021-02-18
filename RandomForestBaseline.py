def RandomForestBaseline(xTrain,yTrain,xTest,yTest,problem_type,range1):

    if problem_type == 'Regression':
        model = RandomForestRegressor(n_estimators = 100)
        base = 1
    if problem_type == 'Classification':
        model = RandomForestClassifier(n_estimators = 100)
        base = 0.5

    rf = model.fit(xTrain,yTrain)
    imp = pd.DataFrame(np.column_stack((xTrain.columns,rf.feature_importances_)),columns = ['features','scores']).sort_values('scores',ascending = False)
    print(imp)
    acc =[]
    n_features = []
    se = []
    for i in range1:

        if i == 0:
            acc.append(base)
            se.append(0)
            n_features.append(i)
            continue

        to_use = imp.head(i)['features'].values
        trial = 0
        acc1 = []
        while trial < 1:
            rf1 = model.fit(xTrain[to_use],yTrain)

            if problem_type == 'Regression':
                pred = rf1.predict(xTest[to_use])
                acc1.append(np.sqrt(np.mean((yTest-pred)**2)))

            if problem_type == 'Classification':
                pred = rf1.predict_proba(xTest[to_use])[:,1]
                acc1.append(sklearn.metrics.roc_auc_score(yTest,pred))

            trial = trial+1

        acc.append(np.mean(acc1))
        se.append(np.std(acc1))
        n_features.append(i)

    return acc,n_features,se
