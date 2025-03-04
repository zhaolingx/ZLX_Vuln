# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import logging
import sys
import json
import numpy as np
from sklearn.metrics import recall_score,precision_score,f1_score,balanced_accuracy_score,matthews_corrcoef,roc_curve,auc

def read_answers(filename):
    answers={}
    with open(filename) as f:
        for line in f:
            line=line.strip()
            js=json.loads(line)
            answers[js['idx']]=js['target']
    return answers

def read_predictions(filename):
    predictions={}
    with open(filename) as f:
        for line in f:
            line=line.strip()
            idx,label=line.split()
            predictions[int(idx)]=int(label)
    return predictions

def calculate_scores(answers,predictions):
    Acc=[]
    y_trues,y_preds=[],[]
    Result=[]
    Fcount = 0
    Tcount = 0
    TTcount = 0
    TTTcount = 0
    FFcount = 0
    count = 0
    # print(answers)
    for key in answers:
        if key not in predictions:
            count = count + 1
            logging.error("Missing prediction for index {}.".format(key))
            sys.exit()
        y_trues.append(answers[key])
        y_preds.append(predictions[key])
        Acc.append(answers[key]==predictions[key])
        if answers[key] == 1:
            FFcount = FFcount +1
        if answers[key] == 0:
            TTTcount = TTTcount + 1
        if answers[key] != predictions[key]:
            Fcount = Fcount + 1
            Result.append(key)
        if answers[key] == predictions[key]:
            Tcount = Tcount + 1
            if answers[key] == 0:
                TTcount = TTcount + 1
    scores={}
    #Acc = balanced_accuracy_score(y_trues, y_preds)
    scores['Acc']=np.mean(Acc)
    scores['Recall']=recall_score(y_trues, y_preds, average="binary")
    scores['Prediction']=precision_score(y_trues, y_preds)
    scores['F1']=f1_score(y_trues, y_preds)
    scores['MCC']=matthews_corrcoef(y_trues,y_preds)
    fpr, tpr, thresholds = roc_curve(y_true=y_trues, y_score=y_preds)
    scores['AUC'] = auc(fpr, tpr)
    print(2*scores['Recall']*scores['Prediction']/(scores['Recall']+scores['Prediction']))
    return scores, Result, Fcount, Tcount, TTcount, TTTcount, FFcount, count,Acc

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for Defect Detection dataset.')
    parser.add_argument('--answers', '-a',help="filename of the labels, in txt format.")
    parser.add_argument('--predictions', '-p',help="filename of the leaderboard predictions, in txt format.")
    

    args = parser.parse_args()
    answers=read_answers(args.answers)
    predictions=read_predictions(args.predictions)
    scores, Result, Fcount, Tcount, TTcount, TTTcount, FFcount, count, ACC=calculate_scores(answers,predictions)
    print(scores)
    print(Result)
    print(Fcount)
    print("TN:",Tcount-TTcount)
    print("TP:",TTcount)
    print("FN:",TTTcount-TTcount)
    print("FP:",FFcount-Tcount+TTcount)
    print(ACC)

if __name__ == '__main__':
    main()
