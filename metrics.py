import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target, loss):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError
        

class AccumulatedAccuracyMetric(Metric):
    """
    Works with classification model
    """

    def __init__(self,margin=0.):
        self.correct = 0
        self.total = 0
        self.margin = margin

    def __call__(self, q_emb,p_emb, labels):
        
        labels = torch.where (labels == -1,0,labels)
        if self.margin:
            predicted_prob = F.sigmoid(F.cosine_similarity(q_emb,p_emb) - self.margin)
        else:
            predicted_prob = F.sigmoid(F.cosine_similarity(q_emb,p_emb))
        pred = torch.where (predicted_prob >=0.5,1,0)
 
        self.correct += pred.eq(labels).cpu().sum()
        self.total += len(labels)
        return self.value()
    
    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return 100 * float(self.correct) / self.total

    def name(self):
        return 'Accuracy'
    
class AccumulatedF1Metric(Metric):
    """
    Works with classification model
    """

    def __init__(self,margin=0.):
        self.tp,self.fn,self.fp,self.tn = 0,0,0,0
        # self.total = 0
        self.margin = margin

    def __call__(self, q_emb,p_emb, labels):
        
        labels = torch.where (labels == -1,0,labels)
        if self.margin:
            predicted_prob = F.sigmoid(F.cosine_similarity(q_emb,p_emb) - self.margin)
            # pred = torch.where (predicted_prob >=0.5,1,-1)
        else:
            predicted_prob = F.sigmoid(F.cosine_similarity(q_emb,p_emb))
            
        pred = torch.where (predicted_prob >=0.5,1,0)
        confusion_vector = pred / labels
        self.tp +=  torch.sum(confusion_vector == 1).item()
        self.fp +=  torch.sum(confusion_vector == float('inf')).item()
        self.tn +=  torch.sum(torch.isnan(confusion_vector)).item()
        self.fn +=  torch.sum(confusion_vector == 0).item()

        return self.value()
    
    def reset(self):
        self.tp,self.fn,self.fp,self.tn = 0,0,0,0

    def value(self):
        precision = self.tp/(self.tp+self.fp)
        recall = self.tp/(self.tp+self.fn)
        return ( 2*precision*recall ) / ( recall + precision )

    def name(self):
        return 'F1-score'