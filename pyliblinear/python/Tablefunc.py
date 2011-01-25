import numpy, cPickle, sys,os
from linearutil import *
from Classifier import *

#list_tasks = ['toys_games','tools_hardware','software','apparel','video','office_products','automotive','books_small','jewelry_watches','grocery','camera_photo','baby','magazines','cell_phones_service','electronics','dvd_small','outdoor_living','musical_instruments','health_personal_care','music_small','computer_video_games','kitchen_housewares','beauty','sports_outdoors','gourmet_food']

list_tasks = ['software','automotive','jewelry_watches','grocery','baby','magazines','cell_phones_service','outdoor_living','computer_video_games','beauty','gourmet_food']


list_tasks_small =  ['books','kitchen','electronics','dvd']


def modelcreate(task_train,depth,path1 = '/data/lisa/data/multi_domain_sentiment_analysis/all_amazon_acl07_data/pyDLrep/',path2 = '/data/lisa/data/multi_domain_sentiment_analysis/all_amazon_acl07_data/pylibsvm-data/',smallname = False,folds = False,size=5000):
    if not folds:
        if task_train != 'all':
	    TrainVectors = path1 + 'in-domain-train-%s-featsz=%s_DLdepth%s.vec'%(task_train,size,depth)
	    if smallname:
	        TrainLabels = path2 + 'in-domain-train-%s-%s.lab'%(task_train,size)
	    else:
	        TrainLabels = path2 + 'in-domain-train-%s-featsz=%s.lab'%(task_train,size)
	    TrainIndices = path2 + 'in-domain-train-%s.idx'%(task_train)
        else:
	    TrainVectors = path1 + 'all-domain-train-featsz=%s_DLdepth%s.vec'%(depth,size)
	    if smallname:
	        TrainLabels = path2 + 'all-domain-train-%s.lab'%size
	    else:
	        TrainLabels = path2 + 'all-domain-train-featsz=%s.lab'%size
	    TrainIndices = path2 + 'all-domain-train.idx'
        TrainingData, ValidationData = loadTrainDataset(0, TrainLabels, TrainVectors, TrainIndices)
        best_classifier = TrainAndOptimizeClassifer(TrainingData, ValidationData, False, 'small')
        save_model('modelcreatetmp%s%s'%(depth,task_train),best_classifier)
    else:
        for seed in range(folds):
            if task_train != 'all':
                TrainVectors = path1 + 'in-domain-train-%s-featsz=%s_DLdepth%s.vec'%(task_train,size,depth)
                if smallname:
                     TrainLabels = path2 + 'in-domain-train-%s-%s.lab'%(task_train,size)
                else:
                    TrainLabels = path2 + 'in-domain-train-%s-featsz=%s.lab'%(task_train,size)
                TrainIndices = path2 + 'in-domain-train-%s_fold%s.idx'%(task_train,seed)
            else:
                TrainVectors = path1 + 'all-domain-train-featsz=%s_DLdepth%s.vec'%(size,depth)
                if smallname:
                    TrainLabels = path2 + 'all-domain-train-%s.lab'%size
                else:
                    TrainLabels = path2 + 'all-domain-train-featsz=%s.lab'%size
                TrainIndices = path2 + 'all-domain-train_fold%s.idx'%seed
            TrainingData, ValidationData = loadTrainDataset(0, TrainLabels, TrainVectors, TrainIndices)
            best_classifier = TrainAndOptimizeClassifer(TrainingData, ValidationData, False, 'small')
            save_model('modelcreatetmp%s%s%s'%(depth,task_train,seed),best_classifier)


def errorcalculate(task_test,depth,path1 = '/data/lisa/data/multi_domain_sentiment_analysis/all_amazon_acl07_data/pyDLrep/',path2 = '/data/lisa/data/multi_domain_sentiment_analysis/all_amazon_acl07_data/pylibsvm-data/',smallname = False,folds = False,size=5000,ind=False):
    if task_test != 'all':
        if task_test =='books_small':
            testt = 'books'
        elif task_test =='dvd_small':
            testt = 'dvd'
        elif task_test == 'music_small':
            testt = 'music'
        else:
            testt = task_test
        TestVectors = path1 + 'in-domain-test-%s-featsz=%s_DLdepth%s.vec'%(testt,size,depth)
        if smallname:
            TestLabels = path2 + 'in-domain-test-%s-%s.lab'%(testt,size)
        else:
            TestLabels = path2 + 'in-domain-test-%s-featsz=%s.lab'%(testt,size)
    else:
        TestVectors = path1 + 'all-domain-test-featsz=%s_DLdepth%s.vec'%(size,depth)
        if smallname:
            TestLabels = path2 + 'all-domain-test-%s.lab'%size
        else:
            TestLabels = path2 + 'all-domain-test-featsz=%s.lab'%size
    TestData = loadTestDataset(0, TestLabels, TestVectors)
    res = {}
    if smallname:
        list_tasks = list_tasks_small
    if ind:
        list_tasks = [task_test]
    if not folds:
        for i in list_tasks:
            best_classifier = load_model('modelcreatetmp%s%s'%(depth,i))
            result = Classifier(best_classifier, TestData, TestVectors.rpartition('/')[2].rpartition('.')[0]+'_task'+str(0))
            res.update({i:result})
    else:
        for i in list_tasks:
            for seed in range(folds):
                best_classifier = load_model('modelcreatetmp%s%s%s'%(depth,i,seed))
                result = Classifier(best_classifier, TestData, TestVectors.rpartition('/')[2].rpartition('.')[0]+'_task'+str(0)+'_fold%s'%seed)
                if i in res.keys():
                    res[i] += [result]
                else:
                    res.update({i:[result]})
    f = open('ressavetmp%s%s.pkl'%(depth,task_test),'w')
    cPickle.dump(res,f,-1)



def errorvalid(task_train,depth,path1 = '/data/lisa/data/multi_domain_sentiment_analysis/all_amazon_acl07_data/pyDLrep/',path2 = '/data/lisa/data/multi_domain_sentiment_analysis/all_amazon_acl07_data/pylibsvm-data/',smallname = False,folds = False,size=5000,ind=False):
    if not folds:
        if task_train != 'all':
            TrainVectors = path1 + 'in-domain-train-%s-featsz=%s_DLdepth%s.vec'%(task_train,size,depth)
            if smallname:
                TrainLabels = path2 + 'in-domain-train-%s-%s.lab'%(task_train,size)
            else:
                TrainLabels = path2 + 'in-domain-train-%s-featsz=%s.lab'%(task_train,size)
            TrainIndices = path2 + 'in-domain-train-%s.idx'%(task_train)
        else:
            TrainVectors = path1 + 'all-domain-train-featsz=%s_DLdepth%s.vec'%(size,depth)
            if smallname:
                TrainLabels = path2 + 'all-domain-train-%s.lab'%size
            else:
                TrainLabels = path2 + 'all-domain-train-featsz=%s.lab'%size
            TrainIndices = path2 + 'all-domain-train.idx'
        TrainingData, ValidationData = loadTrainDataset(0, TrainLabels, TrainVectors, TrainIndices)
        if smallname:
            list_tasks = list_tasks_small
        if ind:
            list_tasks = [task_train]
        res = {}
        for i in list_tasks:
            best_classifier = load_model('modelcreatetmp%s%s'%(depth,i))
            result = Classifier(best_classifier, ValidationData, TrainVectors.rpartition('/')[2].rpartition('.')[0]+'_task'+str(0))
            res.update({i:result})
    else:
        for seed in range(folds):
            if task_train != 'all':
                TrainVectors = path1 + 'in-domain-train-%s-featsz=%s_DLdepth%s.vec'%(task_train,size,depth)
                if smallname:
                    TrainLabels = path2 + 'in-domain-train-%s-%s.lab'%(task_train,size)
                else:
                    TrainLabels = path2 + 'in-domain-train-%s-featsz=%s.lab'%(task_train,size)
                TrainIndices = path2 + 'in-domain-train-%s_fold%s.idx'%(task_train,seed)
            else:
                TrainVectors = path1 + 'all-domain-train-featsz=%s_DLdepth%s.vec'%(size,depth)
                if smallname:
                    TrainLabels = path2 + 'all-domain-train-%s.lab'%size
                else:
                    TrainLabels = path2 + 'all-domain-train-featsz=%s.lab'%size
                TrainIndices = path2 + 'all-domain-train_fold%s.idx'%seed
            TrainingData, ValidationData = loadTrainDataset(0, TrainLabels, TrainVectors, TrainIndices)
            if smallname:
                list_tasks = list_tasks_small
            if ind:
                list_tasks = [task_train]
            res = {}
            for i in list_tasks:
                best_classifier = load_model('modelcreatetmp%s%s%s'%(depth,i,seed))
                result = Classifier(best_classifier, ValidationData, TrainVectors.rpartition('/')[2].rpartition('.')[0]+'_task'+str(0)+'_fold%s'%seed)
                if i in res.keys():
                    res[i] += [result]
                else:
                    res.update({i:[result]})
    f = open('resvalidsavetmp%s%s.pkl'%(depth,task_train),'w')
    cPickle.dump(res,f,-1)
    f.close()

if __name__ == '__main__':
    typ = sys.argv[1]
    task = sys.argv[2]
    depth = int( sys.argv[3])
    if len(sys.argv)>4:
        path1 = sys.argv[4]
        path2 = sys.argv[5]
        smallname = eval(sys.argv[6])
    if len(sys.argv)>7:
        folds = eval(sys.argv[7])
    if len(sys.argv)>8:
        size = int(sys.argv[8])
    else:
        size = 5000
    if len(sys.argv)>9:
        ind = eval(sys.argv[9])
    else:
        ind = False
    if typ == 'model':
        print 'modelcreate',task,depth
        if len(sys.argv)>7:
            modelcreate(task,depth,path1,path2,smallname,folds,size=size)
        elif len(sys.argv)>4:
            modelcreate(task,depth,path1,path2,smallname,size=size)
        else:
            modelcreate(task,depth,size=size)
    elif typ == 'res':
        print 'res',task,depth
        if len(sys.argv)>7:
            errorcalculate(task,depth,path1,path2,smallname,folds,size=size,ind=ind)
        elif len(sys.argv)>4:
            errorcalculate(task,depth,path1,path2,smallname,size=size,ind=ind)
        else:
            errorcalculate(task,depth,size=size,ind=ind)
    elif typ == 'valid':
        print 'valid',task,depth
        if len(sys.argv)>7:
            errorvalid(task,depth,path1,path2,smallname,folds,size=size,ind=ind)
        elif len(sys.argv)>4:
            errorvalid(task,depth,path1,path2,smallname,size=size,ind=ind)
        else:
            errorvalid(task,depth,size=size,ind=ind)
