### Package Implementation of Control Burn Algorithm

import sklearn
import statistics
import scipy as sc
import numpy as np
import pandas as pd
import random
import cvxpy as cp
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import Pool
import time
import re
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import mosek
import statsmodels.api as sm
import gc

### Helper Functions
def converge_test(sequence, threshold,length):
    diff = np.diff(sequence)
    if len(diff) < (length+1):
        return False
    else:
        return ( max(np.abs(diff[-length:])) < threshold)

### Functions for Experiments
# These functions are designed to be used for the experiments in the ControlBurn Paper

### Incremental Depth Bagging Algorithm
# arg is an array [xTrain,yTrain,xTest,yTest,max_depth,problem_type,loss_type,threshold]
def build_trees_bag_experiment(arg):
    xTrain = arg[0]
    yTrain = arg[1]
    xTest= arg[2]
    yTest= arg[3]
    max_depth= arg[4]
    problem_type = arg[5]
    loss_type = arg[6]
    lambd= arg[7]
    threshold= arg[8]

    train = xTrain
    train = train.reset_index().drop('index',axis = 1)
    train['yTrain'] = list(yTrain)
    features = xTrain.columns
    nfeatures = len(features)
    importance_key = pd.DataFrame(features,columns = ['Features'])
    tree_results = []
    i = 0
    depth = 1
    total_trees = 0

    for depth in range (1,max_depth+1):
        i = 0
        ### Early Stopping
        early_stop_pred = []
        early_stop_train_err = []
        converged = False

        while converged == False:
            train1 = train.sample(n = len(train), replace = True)

            yTrain1 = train1['yTrain']
            xTrain1 = train1[features]

            if problem_type == 'Regression':
                clf = DecisionTreeRegressor(max_depth = depth)
            elif problem_type == 'Classification':
                clf = DecisionTreeClassifier(max_depth = depth)

            clf.fit(xTrain1,yTrain1)
            imp = pd.DataFrame(np.column_stack((xTrain1.columns,clf.feature_importances_)), columns = ['Features','Importances'])
            used = imp[imp['Importances']>0]['Features'].values
            feature_indicator = [int(x in used) for x in features]

            if problem_type == 'Regression':
                pred = clf.predict(xTrain[features])
                test_pred = clf.predict(xTest[features])
            elif problem_type == 'Classification':
                if loss_type == 'logistic':
                    pred = clf.predict_proba(xTrain[features])[:,1]
                    test_pred = clf.predict_proba(xTest[features])[:,1]
                elif loss_type == 'hinge':
                    pred = clf.predict(xTrain[features])
                    test_pred = clf.predict(xTest[features])

            feature_importances = pd.merge(importance_key,imp, on = 'Features', how = 'left').fillna(0)['Importances'].values
            tree_results.append([pred,feature_indicator,feature_importances, test_pred  ,clf,xTrain1,yTrain1,features])
            i = i+1
            total_trees = total_trees+1
            early_stop_pred.append(pred)
            early_stop_train_err.append(np.sqrt(np.mean((np.mean(early_stop_pred,axis = 0) - yTrain)**2)))
            converged = converge_test(early_stop_train_err,threshold,5)

    return tree_results

### Optimization Algorithms
def solve_step_experiment(arg, tree_results):
    xTrain = arg[0]
    yTrain = arg[1]
    xTest= arg[2]
    yTest= arg[3]
    max_depth= arg[4]
    problem_type = arg[5]
    loss_type = arg[6]
    lambd= arg[7]
    threshold= arg[8]
    optimization_type = arg[9] #penalized or constrained

    feature_list = xTrain.columns
    tree_pred = np.transpose(np.array([np.array(row[0]) for row in tree_results]))
    test_pred = np.transpose(np.array([np.array(row[3]) for row in tree_results]))
    indicators = np.transpose(np.array([np.array(row[1]) for row in tree_results]))
    w = cp.Variable(len(tree_results),nonneg=True)

    if optimization_type == 'penalized':
        constraints = []
        if problem_type == 'Regression':
            loss = cp.sum_squares(cp.matmul(tree_pred,w)-yTrain)
            objective = (1/len(yTrain))*loss + lambd*cp.norm(cp.matmul(indicators,w),1)
        elif problem_type == 'Classification':
            if loss_type == 'logistic':
                loss = -cp.sum(cp.multiply(yTrain, tree_pred@ w) - cp.logistic(tree_pred @ w))
                objective = (1/len(yTrain))*loss + lambd*cp.norm(cp.matmul(indicators,w),1)
            elif loss_type == 'hinge':
                loss =  cp.sum(cp.pos(1 - cp.multiply(yTrain, tree_pred @ w)))
                objective =  (1/len(yTrain))*loss + lambd*cp.norm(cp.matmul(indicators,w),1)


    if optimization_type == 'constrained':
        if problem_type == 'Regression':
            objective = cp.sum_squares(cp.matmul(tree_pred,w)-yTrain)
        elif problem_type == 'Classification':
            if loss_type == 'logistic':
                objective = -cp.sum(cp.multiply(yTrain, tree_pred@ w) - cp.logistic(tree_pred @ w))
            elif loss_type == 'hinge':
                objective = cp.sum(cp.pos(1 - cp.multiply(yTrain, tree_pred @ w)))
        constraints = [cp.norm(cp.matmul(indicators,w),1)<= lambd]

    prob = cp.Problem(cp.Minimize(objective),constraints)
    prob.solve(solver = cp.MOSEK,mosek_params = {mosek.dparam.optimizer_max_time: 10000.0} )
    weights = np.asarray(w.value)
    low_values_flags = np.abs(weights) < 10**-3
    weights[low_values_flags] = 0
    tree_ind = np.where(weights >0)[0]

    if len(tree_ind)==0:
        if problem_type == 'Regression':
            return([[],np.sqrt(np.mean((yTest )**2)),0,np.sqrt(np.mean((yTrain )**2))])
        else:
            return([[],.5,0,.5])

    importances = np.array([np.array(row[2]) for row in tree_results])
    feature_importances = np.mean(importances[tree_ind],axis = 0)
    nonzero_features = xTrain.columns[np.where(feature_importances >0)[0]]

    if problem_type == 'Regression':
        rf = RandomForestRegressor(n_estimators = 100).fit(xTrain[nonzero_features],yTrain)
        test_pred = rf.predict(xTest[nonzero_features])
        train_pred = rf.predict(xTrain[nonzero_features])
        train_error =  np.sqrt(np.mean((yTrain -train_pred)**2))
        test_error = np.sqrt(np.mean((yTest -test_pred)**2))
        return([feature_importances,test_error,len(nonzero_features),train_error])


    elif problem_type == 'Classification' :
        rf = RandomForestClassifier(n_estimators = 100).fit(xTrain[nonzero_features],yTrain)
        test_pred = rf.predict_proba(xTest[nonzero_features])[:,1]
        train_pred = rf.predict_proba(xTrain[nonzero_features])[:,1]
        train_error = sklearn.metrics.roc_auc_score(yTrain,train_pred)
        test_error = sklearn.metrics.roc_auc_score(yTest,test_pred)
        return([feature_importances,test_error,len(nonzero_features),train_error])


def run_experiment(arg,ntrials,features_to_find,search_limit,l_start):
    test_error_result = []
    nonzero_result = []
    train_error_result = []
    trial = 0
    while trial < ntrials:
        #Build Trees
        tree_results = build_trees_bag_experiment(arg)
        LL = 0
        RL = l_start
        to_find = 0
        counter1 = 0
        count_array = []
        while to_find <= features_to_find:
            arg_list = []
            lambd = (LL + RL)/2
            arg[7] = lambd
            result = solve_step_experiment(arg,tree_results)
            test_acc = result[1]
            nonzero = result[2]
            train_acc = result[3]
            print(nonzero,to_find,lambd)

            #Append Results
            test_error_result.append(test_acc)
            nonzero_result.append(nonzero)
            train_error_result.append(train_acc)
            count_array.append(nonzero)

            freq = pd.DataFrame(np.column_stack(np.unique(count_array, return_counts = True)),columns = ['value','counts'])
            count_to_find = freq.loc[freq['value']==to_find]['counts'].values

            if arg[9] != 'constrained':

                if count_to_find > 0 :
                    counter1 = 0
                    RL = lambd
                    LL = 0
                    to_find = to_find + 1

                elif counter1 >= search_limit:
                    counter1 = 0
                    RL = lambd/2
                    LL = 0
                    to_find = to_find + 1

                elif nonzero < to_find:
                    RL = lambd
                    counter1 = counter1 + 1

                elif nonzero >= to_find:
                    LL = lambd
                    counter1 = counter1 + 1

            elif arg[9] == 'constrained':

                if count_to_find > 0 :
                    counter1 = 0
                    RL = lambd
                    LL = 0
                    to_find = to_find + 1

                elif counter1 >= search_limit:
                    counter1 = 0
                    RL = l_start
                    LL = 0
                    to_find = to_find + 1

                elif nonzero > to_find:
                    RL = lambd
                    counter1 = counter1 + 1

                elif nonzero <= to_find:
                    LL = lambd
                    counter1 = counter1 + 1

        trial = trial + 1

    return test_error_result,nonzero_result,train_error_result

def plot_tradeoff_curve(test_acc,nonzero,color,label):
    results = pd.DataFrame()
    for i in range(0,len(test_acc)):
        results = results.append(pd.DataFrame(np.column_stack((test_acc[i],nonzero[i])),columns = ['test_acc','nonzero']))
    agg = results.groupby(['nonzero'], as_index=False).agg({'test_acc':['mean','std','count']})

    plt.scatter(agg['nonzero'],agg['test_acc']['mean'],color = color,label = label)
    plt.errorbar(agg['nonzero'],agg['test_acc']['mean'], agg['test_acc']['std'],color = color)
    plt.xlabel('Number of Nonzero Features')
    plt.ylabel('Test Error')
    plt.legend()
