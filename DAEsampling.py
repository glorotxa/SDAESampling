import cPickle,numpy, os, os.path, sys, time
import scipy

from theano.tensor.shared_randomstreams import RandomStreams as RandomStreamsCPU
from theano.sandbox.rng_mrg  import MRG_RandomStreams as RandomStreamsGPU
import theano
import theano.tensor as T
import theano.sparse

from linearutil import *
from Classifier import *

data_path = '/data/lisa/data/multi_domain_sentiment_analysis/'
c_path = 'hard_code_your_/pyliblinear/python path'


def binomial_NLP_noise(inp,zer_mask,one_mask):
    return zer_mask * inp + (inp==0) * one_mask
    #return theano_rng.binomial( size = inp.shape, n = 1, p =  1 - noise_lvl[0], dtype=theano.config.floatX) * inp \
    #                    + (inp==0) * theano_rng.binomial( size = inp.shape, n = 1, p =  noise_lvl[1], dtype=theano.config.floatX)

def cross_entropy_sampled_cost(target, output_act,pattern,scaling = False):
    XE = target * (- T.log(1 + T.exp(-output_act))) + (1 - target) * (- T.log(1 + T.exp(output_act)))
    if pattern != None:    
        XE = XE * pattern
    if scaling != False:
        XE = XE * T.cast(scaling[0],dtype=theano.config.floatX) * target + XE * T.cast(scaling[1],dtype=theano.config.floatX) * (1-target)
    return -T.mean(T.sum(XE, axis=1),axis=0), -T.mean(XE,axis=0)

def MSE_sampled_cost(target, output_act,pattern,scaling = False):
    MSE = (T.nnet.sigmoid(output_act) - target) * (T.nnet.sigmoid(output_act) - target)
    if pattern != None:
        MSE = MSE * pattern
    if scaling != False:
        MSE = MSE * T.cast(scaling[0],dtype=theano.config.floatX) * target + MSE * T.cast(scaling[1],dtype=theano.config.floatX) * (1-target)
    return T.mean(T.sum(MSE, axis=1),axis=0), T.mean(MSE,axis=0)


def vectosparsemat(path,NBDIMS):
    """
    This function converts the unlabeled training data into a scipy
    sparse matrix and return it.
    """
    print >> sys.stderr , "Read and converting data file: %s to a sparse matrix"%path 
    # We first count the number of line in the file
    f = open(path, 'r')
    i = f.readline()
    ct = 0
    while i!='':
        ct+=1
        i = f.readline()
    f.close()
    # Then we allocate and fill the sparse matrix as a lil_matrix 
    # for efficiency.
    NBEX = ct
    train = scipy.sparse.lil_matrix((NBEX,NBDIMS),dtype=theano.config.floatX)
    f = open(path, 'r')
    i = f.readline()
    ct = 0
    next_print_percent = 0.1
    while i !='':
        if ct / float(NBEX) > next_print_percent:
            print >> sys.stderr , "\tRead %s %s of file"%(next_print_percent*100,'%')
            next_print_percent += 0.1
        i = i[:-1]
        i = list(i.split(' '))
        for j in i:
            if j!='':
                idx,dum,val = j.partition(':')
                train[ct,int(idx)-1] = 1
        i = f.readline()
        ct += 1
    print >> sys.stderr , "Data converted" 
    # We return a csr matrix because for efficiency 
    # because we will later shuffle the rows.
    return train.tocsr()


def createdensebatch(spmat,size,batchnumber):
    """ 
    This function creates and return dense matrix corresponding to the 
    'batchnumber'_th slice of length 'size' of the sparse data matrix.
    """
    NB_DENSE = int(numpy.ceil(spmat.shape[0] / float(size)))
    assert batchnumber>=0 and batchnumber<NB_DENSE 
    realsize = size
    if batchnumber< NB_DENSE-1:
        batch = numpy.asarray(spmat[size*batchnumber:size*(batchnumber+1),:].toarray(),dtype=theano.config.floatX)
    else:
        batch = numpy.asarray(spmat[size*batchnumber:,:].toarray(),dtype=theano.config.floatX)
        realsize = batch.shape[0]
        if batch.shape[0] < size:
            batch = numpy.concatenate([batch,numpy.zeros((size-batch.shape[0],batch.shape[1]),dtype=theano.config.floatX)])
    return batch,realsize
 

def createvecfile(Wenc,benc,PathData,depth,OutFile,act = 'rect'):
    """
    This function builds a 'OutFile' .vec file corresponding to 'PathData' taken at 
    the layer 'depth' of the 'PathLoad' model.
    """
    print >> sys.stderr, "Creating vec file %s ( depth=%d, datafiles=%s)..." % (repr(OutFile), depth,PathData)
    inp = T.matrix()
    hid_lin = T.dot(inp,Wenc)+benc
    if act == 'rect':
        hid_out = hid_lin * (hid_lin>=0)
    if act == 'sigmoid':
        hid_out = T.nnet.sigmoid(hid_lin)
    outputs = [hid_out]
    func = theano.function([inp],outputs)

    full_train = vectosparsemat(PathData,Wenc.value.shape[0])
    NB_BATCHS = int(numpy.ceil(full_train.shape[0] / float(500)))

    f = open(OutFile,'w')

    for i in range(NB_BATCHS):
        if i < NB_BATCHS-1:
            rep = func(numpy.asarray(full_train[500*i:500*(i+1),:].toarray(),dtype=theano.config.floatX))[0]
        else:
            rep = func(numpy.asarray(full_train[500*i:,:].toarray(),dtype=theano.config.floatX))[0]
        textr = ''
        for l in range(rep.shape[0]):
            idx = rep[l,:].nonzero()[0]
            for j,v in zip(idx,rep[l,idx]):
                textr += '%s:%s '%(j,v)
            textr += '\n'
        f.write(textr)
    f.close()
    print >> sys.stderr, "...done creating vec files"


def createvecfilesparse(Wenc,benc,PathData,depth,OutFile,act = 'rect'):
    """
    This function builds a 'OutFile' .vec file corresponding to 'PathData' taken at 
    the layer 'depth' of the 'PathLoad' model.
    """
    print >> sys.stderr, "Creating vec file %s ( depth=%d, datafiles=%s)..." % (repr(OutFile), depth,PathData)
    inp = theano.sparse.csr_matrix()
    hid_lin = theano.sparse.dot(inp,Wenc)+benc
    if act == 'rect':
        hid_out = hid_lin * (hid_lin>=0)
    if act == 'sigmoid':
        hid_out = T.nnet.sigmoid(hid_lin)
    outputs = [hid_out]
    func = theano.function([inp],outputs)

    full_train = vectosparsemat(PathData,Wenc.value.shape[0])
    NB_BATCHS = int(numpy.ceil(full_train.shape[0] / float(500)))

    f = open(OutFile,'w')

    for i in range(NB_BATCHS):
        if i < NB_BATCHS-1:
            rep = func(full_train[500*i:500*(i+1),:])[0]
        else:
            rep = func(full_train[500*i:,:])[0]
        textr = ''
        for l in range(rep.shape[0]):
            idx = rep[l,:].nonzero()[0]
            for j,v in zip(idx,rep[l,idx]):
                textr += '%s:%s '%(j,v)
            textr += '\n'
        f.write(textr)
    f.close()
    print >> sys.stderr, "...done creating vec files"


def createWbshared(rng,n_inp,n_hid,tag,trans=False):
    wbound = numpy.sqrt(6./(n_inp+n_hid))
    if not(trans):
        W_values = numpy.asarray( numpy.random.uniform( low = -wbound, high = wbound, \
                                    size = (n_inp, n_hid)), dtype = theano.config.floatX)
    else:
        W_values = numpy.asarray( numpy.random.uniform( low = -wbound, high = wbound, \
                                    size = (n_hid, n_inp)), dtype = theano.config.floatX)
    W = theano.shared(value = W_values, name = 'W'+tag)
    b_values = numpy.zeros((n_hid,), dtype= theano.config.floatX)
    b = theano.shared(value= b_values, name = 'b'+tag)
    return W,b


def SamplingdenseSDAEexp(state,channel):
    """
    This script launch a SDAE experiment, training in a greedy layer wise fashion.
    The hidden layer activation function is the rectifier activation (i.e. max(0,y)). The reconstruction activation function
    is the sigmoid. The reconstruction cost is the cross-entropy. From one layer to the next we need to scale the
    parameters in order to ensure that the representation is in the interval [0,1].
    The noise of the input layer is a salt and pepper noise ('binomial_NLP'), for deeper layers it is a zero masking
    noise (binomial).
    """
    SavePath = channel.remote_path+'/' if hasattr(channel,'remote_path') else channel.path+'/'
    
    numpy.random.seed(state.seed)
    
    Wenc,benc = createWbshared(numpy.random,state.n_inp,state.n_hid,'enc')
    Wdec,bdec = createWbshared(numpy.random,state.n_hid,state.n_inp,'dec')

    # Load the entire training data
    full_train = vectosparsemat(data_path + state['path_data'],state['ninputs'])
    full_test = vectosparsemat(data_path + state['path_data_test'],state['ninputs'])
    full_train = full_train[numpy.random.permutation(full_train.shape[0]),:]
    NB_DENSE_train = int(numpy.ceil(full_train.shape[0] / float(state['dense_size'])))
    NB_DENSE_test = int(numpy.ceil(full_test.shape[0] / float(state['dense_size'])))
    
    # Create the dense batch shared variable
    train = theano.shared(createdensebatch(full_train,state['dense_size'],0)[0])
    zer_mask_shared = theano.shared(numpy.asarray(numpy.random.binomial(n = 1, p = 1-state.zeros, size = train.value.shape),dtype=theano.config.floatX))
    one_mask_shared = theano.shared(numpy.asarray(numpy.random.binomial(n = 1, p = state.ones, size = train.value.shape),dtype=theano.config.floatX))
    #------------------------------
    state.bestindomain = -1
    state.bestindomainstd = -1
    state.bestindomainval = -1
    state.bestindomainvalstd = -1
    state.bestindomainvalde = -1
    state.bestrec = -1
    state.bestrecde = -1
    state.bestonlinerec = -1
    state.bestonlinerecde = -1
    epochsl = []
    indomain = []
    indomainstd = []
    indomainval = []
    indomainvalstd = []
    rec = []
    recdense = []
    reconline = []
    #-------------------------------

    # Model initialization:
    inp = T.matrix()
    RandomStreams = RandomStreamsGPU(state.seed)
    zer_mask = T.matrix()
    one_mask = T.matrix()
    
    if state.pattern == 'inp':
        pattern =  T.cast((inp + RandomStreams.binomial(size = inp.shape,n=1,p=state.ratio,dtype=theano.config.floatX)) > 0,dtype=theano.config.floatX)
    elif state.pattern == 'noise':
        pattern =  T.cast(((1-zer_mask)*inp + one_mask + RandomStreams.binomial(size = inp.shape,n=1,p=state.ratio,dtype=theano.config.floatX)) > 0,dtype=theano.config.floatX)
    elif state.pattern == 'inpnoise':
        pattern = T.cast((inp + one_mask + RandomStreams.binomial(size = inp.shape,n=1,p=state.ratio,dtype=theano.config.floatX)) > 0,dtype=theano.config.floatX)
    elif state.pattern == 'random':
        pattern = RandomStreams.binomial(size = inp.shape,n=1,p=state.ratio,dtype=theano.config.floatX)
    elif state.pattern == None:
        pattern = None
    else:
        assert False
    
    inp_noise = binomial_NLP_noise(inp,zer_mask,one_mask)
    hid_lin = T.dot(inp_noise,Wenc)+benc
    if state.act == 'rect':
        hid_out = hid_lin * (hid_lin > 0)
    if state.act == 'sigmoid':
        hid_out = T.nnet.sigmoid(hid_lin)
    L1_reg = T.mean(T.sum(hid_out * hid_out,axis=1),axis=0)
    rec_lin = T.dot(hid_out,Wdec)+bdec
    # the sigmoid is inside the cross_entropy function.
    if not hasattr(state,'scaling'):
        state.scaling = False
    if state.cost == 'CE':
        cost, dum = cross_entropy_sampled_cost(inp, rec_lin,pattern,state.scaling)
        cost_dense, cost_decoupled_dense = cross_entropy_sampled_cost(inp, rec_lin,None)
    if state.cost == 'MSE':
        cost, dum = MSE_sampled_cost(inp, rec_lin,pattern,state.scaling)
        cost_dense, cost_decoupled_dense = MSE_sampled_cost(inp, rec_lin,None)
    if state.regcoef != 0.:
        cost = cost + state.regcoef * L1_reg
    grad = T.grad(cost,[Wenc,Wdec,benc,bdec])
    updates = dict( (p,p-state.lr*g) for p,g in zip([Wenc,Wdec,benc,bdec],grad) )
    givens = {}
    index = T.lscalar()
    givens.update({inp:train[index*state.batchsize:(index+1)*state.batchsize]})
    givens.update({zer_mask:zer_mask_shared[index*state.batchsize:(index+1)*state.batchsize]})
    givens.update({one_mask:one_mask_shared[index*state.batchsize:(index+1)*state.batchsize]})
    TRAINFUNC = theano.function([index],cost, updates = updates, givens = givens)
    #givens = {}
    #givens.update({inp:train[index*state.batchsizeerr:(index+1)*state.batchsizeerr]})
    #givens.update({zer_mask:zer_mask_shared[index*state.batchsizeerr:(index+1)*state.batchsizeerr]})
    #givens.update({one_mask:one_mask_shared[index*state.batchsizeerr:(index+1)*state.batchsizeerr]})
    #ERRNOISE = theano.function([index],[cost_dense,cost_decoupled_dense], givens = givens)
    givens = {}
    givens.update({inp:train[index*state.batchsizeerr:(index+1)*state.batchsizeerr]})
    givens.update({zer_mask:zer_mask_shared[index*state.batchsizeerr:(index+1)*state.batchsizeerr]})
    givens.update({one_mask:one_mask_shared[index*state.batchsizeerr:(index+1)*state.batchsizeerr]})
    ERR = theano.function([index],[cost_dense,cost_decoupled_dense], givens = givens) 
    
    # Train the current DAE
    for epoch in range(state['nepochs']):
    # Load sequentially dense batches of the training data
        reconstruction_error_batch = 0
        update_count1 = 0
        for batchnb in range(NB_DENSE_train):
	    train.container.value[:], realsize = createdensebatch(full_train,state.dense_size,batchnb) 
	    zer_mask_shared.container.value[:] = numpy.asarray(numpy.random.binomial(n = 1, p = 1-state.zeros, size = train.value.shape),dtype=theano.config.floatX)
            one_mask_shared.container.value[:] = numpy.asarray(numpy.random.binomial(n = 1, p = state.ones, size = train.value.shape),dtype=theano.config.floatX)
            for j in range(realsize/state.batchsize):
	        tmp = TRAINFUNC(j) 
	        reconstruction_error_batch += tmp
                update_count1 += 1
	    print >> sys.stderr, "\t\tAt depth %d, epoch %d, finished training over batch %s" % (1, epoch+1, batchnb+1)
	    print >> sys.stderr, "\t\tMean reconstruction error %s" % (reconstruction_error_batch/float(update_count1))
        print >> sys.stderr, '...finished training epoch #%s' % (epoch+1)
        full_train = full_train[numpy.random.permutation(full_train.shape[0]),:]
        if epoch+1 in state.epochs:
            #rec test err
            update_count2 = 0
            test_recerr = 0
            test_recerrd = numpy.zeros((state.ninputs,))
            for batchnb in range(NB_DENSE_test):
                train.container.value[:], realsize = createdensebatch(full_test,state.dense_size,batchnb)
                zer_mask_shared.container.value[:] = numpy.ones(train.value.shape, dtype=theano.config.floatX)
                one_mask_shared.container.value[:] = numpy.zeros(train.value.shape, dtype=theano.config.floatX)
                for j in range(realsize/state.batchsizeerr):
                    # Update function
                    recerr,recerrd = ERR(j)
                    test_recerr += recerr
                    test_recerrd += recerrd
                    update_count2 += 1
	    if not os.path.isdir(SavePath):
	        os.mkdir(SavePath)
	    modeldir = os.path.join(SavePath, 'currentmodel' )
	    if not os.path.isdir(modeldir):
	        os.mkdir(modeldir)
            f = open(modeldir+'/params.pkl','w')
            cPickle.dump(Wenc.value,f,-1)
            cPickle.dump(Wdec.value,f,-1)
            cPickle.dump(benc.value,f,-1)
            cPickle.dump(bdec.value,f,-1)
            f.close()
	    createdatafiles(1,Wenc,benc,SavePath,state.small,state.ninputs,state.act)
	    currentresults =  validtest(1,SavePath,state.small,state.folds,state.ninputs)
            epochsl += [epoch+1] 
            indomainval += [currentresults[1][0]]
            indomainvalstd += [currentresults[1][1]]
            indomain += [currentresults[0][0]]
            indomainstd += [currentresults[0][1]]
            rec += [test_recerr/float(update_count2)]
            recdense += [test_recerrd/float(update_count2)]
            reconline += [reconstruction_error_batch/float(update_count1)]
	    print '###### RESULTS :'
	    print 'Depth:',1
	    print 'Epoch:',epoch+1
	    print 'Online Reconstruction:',reconstruction_error_batch/float(update_count1)
            print 'Reconstruction:',test_recerr/float(update_count2)
	    print 'in-domain val:', currentresults[1][0], '+/-', currentresults[1][1]
	    print 'in-domain test:', currentresults[0][0], '+/-', currentresults[0][1]
	    print ' '
	    f = open('results.pkl','w')
	    cPickle.dump(epochsl,f,-1)
            cPickle.dump(rec,f,-1)
            cPickle.dump(recdense,f,-1)
            cPickle.dump(reconline,f,-1)
            cPickle.dump((indomainval,indomainvalstd),f,-1)
            cPickle.dump((indomain,indomainstd),f,-1)
            f.close()
            if test_recerr/float(update_count2) < state.bestrec  or state.bestrec==-1:
	        state.bestrec = test_recerr/float(update_count2)
	        state.bestrecde =(1,epoch+1)
	    if reconstruction_error_batch/float(update_count1) < state.bestonlinerec  or state.bestonlinerec==-1:
	        state.bestonlinerec = reconstruction_error_batch/float(update_count1)
	        state.bestonlinerecde =(1,epoch+1)
	    if currentresults[1][0] < state.bestindomainval or state.bestindomainval==-1:
                modeldir = os.path.join(SavePath, 'bestmodel' )
	        if not os.path.isdir(modeldir):
	            os.mkdir(modeldir)
                f = open(modeldir+'/params.pkl','w')
                cPickle.dump(Wenc.value,f,-1)
                cPickle.dump(Wdec.value,f,-1)
                cPickle.dump(benc.value,f,-1)
                cPickle.dump(bdec.value,f,-1)
                f.close()
	        state.bestindomain = currentresults[0][0]
	        state.bestindomainstd = currentresults[0][1]
	        state.bestindomainval = currentresults[1][0]
	        state.bestindomainvalstd = currentresults[1][1]
	        state.bestindomainvalde =(1,epoch+1)
	state.currentepoch = epoch+1
	channel.save()
    return channel.COMPLETE
    

task_train = ['books','kitchen','electronics','dvd']
task_train_full = ['software','jewelry_watches','grocery','magazines','cell_phones_service','computer_video_games','beauty']

def statscalc(res,small=True):
    tt = task_train if small else task_train_full
    diag = {}
    meandiag = []
    for i in tt:
        name = i
        namet = name
        if name not in res[name].keys():
            namet = name[:-6]
        else:
            namet = name
        diag.update({name:res[name][namet]})
        meandiag += [res[name][namet]]

    return numpy.mean(meandiag),numpy.std(meandiag)


def validtest(depth,path = '',small = True,folds = False,size = 5000,ind=True):
    tt = task_train if small else task_train_full
    depth = depth 
    if path == '':
        path=os.getcwd()+'/'
    pathref = data_path +'amazon/' if small else data_path + 'fullamazon/'
    smallb = small
    res={}
    for i in tt:
        res.update({i:{}})
        os.system('python %sacl_07/pyliblinear/python/Tablefunc.py model %s %s %s %s %s %s %s %s'%(c_path,i,depth,path,pathref,smallb,folds,size,ind))
        os.system('python %sacl_07/pyliblinear/python/Tablefunc.py res %s %s %s %s %s %s %s %s'%(c_path,i,depth,path,pathref,smallb,folds,size,ind))
        f = open('ressavetmp%s%s.pkl'%(depth,i),'r')
        result = cPickle.load(f)
        for j in result.keys():
            res[i].update({j:result[j]})
    testres = statscalc(res,small)
    for i in tt:
        os.system('python %sacl_07/pyliblinear/python/Tablefunc.py valid %s %s %s %s %s %s %s %s'%(c_path,i,depth,path,pathref,smallb,folds,size,ind))
        f = open('resvalidsavetmp%s%s.pkl'%(depth,i),'r')
        result = cPickle.load(f)
        for j in result.keys():
            res[i].update({j:result[j]})
    validres = statscalc(res,small)
    return testres,validres

def createdatafiles(depth,Wenc,benc,SavePath='.',small=True,ninputs = 5000,act = 'rect'):
    tt = task_train if small else task_train_full
    if not small:
        for i in tt:
            createvecfile(Wenc,benc,data_path + 'fullamazon/in-domain-train-%s-featsz=%s.vec'%(i,ninputs),depth,SavePath + '/in-domain-train-%s-featsz=%s_DLdepth%s.vec'%(i,ninputs,depth),act)
            createvecfile(Wenc,benc,data_path + 'fullamazon/in-domain-test-%s-featsz=%s.vec'%(i,ninputs),depth,SavePath + '/in-domain-test-%s-featsz=%s_DLdepth%s.vec'%(i,ninputs,depth),act)
    else:
        for i in tt:
            createvecfile(Wenc,benc,data_path+ 'amazon/in-domain-train-%s-%s.vec'%(i,ninputs),depth,SavePath + '/in-domain-train-%s-featsz=%s_DLdepth%s.vec'%(i,ninputs,depth),act)
            createvecfile(Wenc,benc,data_path + 'amazon/in-domain-test-%s-%s.vec'%(i,ninputs),depth,SavePath + '/in-domain-test-%s-featsz=%s_DLdepth%s.vec'%(i,ninputs,depth),act)


def createdatafilessparse(depth,Wenc,benc,SavePath='.',small=True,ninputs = 5000,act='rect'):
    tt = task_train if small else task_train_full
    if not small:
        for i in tt:
            createvecfilesparse(Wenc,benc,data_path +'fullamazon/in-domain-train-%s-featsz=%s.vec'%(i,ninputs),depth,SavePath + '/in-domain-train-%s-featsz=%s_DLdepth%s.vec'%(i,ninputs,depth),act)
            createvecfilesparse(Wenc,benc,data_path + 'fullamazon/in-domain-test-%s-featsz=%s.vec'%(i,ninputs),depth,SavePath + '/in-domain-test-%s-featsz=%s_DLdepth%s.vec'%(i,ninputs,depth),act)
    else:
        for i in tt:
            createvecfilesparse(Wenc,benc,data_path + 'amazon/in-domain-train-%s-%s.vec'%(i,ninputs),depth,SavePath + '/in-domain-train-%s-featsz=%s_DLdepth%s.vec'%(i,ninputs,depth),act)
            createvecfilesparse(Wenc,benc,data_path + 'amazon/in-domain-test-%s-%s.vec'%(i,ninputs),depth,SavePath + '/in-domain-test-%s-featsz=%s_DLdepth%s.vec'%(i,ninputs,depth),act)


def SamplingsparseSDAEexp(state,channel):
    """
    This script launch a SDAE experiment, training in a greedy layer wise fashion.
    The hidden layer activation function is the rectifier activation (i.e. max(0,y)). The reconstruction activation function
    is the sigmoid. The reconstruction cost is the cross-entropy. From one layer to the next we need to scale the
    parameters in order to ensure that the representation is in the interval [0,1].
    The noise of the input layer is a salt and pepper noise ('binomial_NLP'), for deeper layers it is a zero masking
    noise (binomial).
    """
    SavePath = channel.remote_path+'/' if hasattr(channel,'remote_path') else channel.path+'/'
    numpy.random.seed(state.seed)
    
    Wenc,benc = createWbshared(numpy.random,state.n_inp,state.n_hid,'enc')
    Wdec,bdec = createWbshared(numpy.random,state.n_hid,state.n_inp,'dec',trans = True)
    
    # Load the entire training data
    full_train = vectosparsemat(data_path+state['path_data'],state['ninputs'])
    full_test = vectosparsemat(data_path+state['path_data_test'],state['ninputs'])
    full_train = full_train[numpy.random.permutation(full_train.shape[0]),:]
    #------------------------------
    state.bestindomain = -1
    state.bestindomainstd = -1
    state.bestindomainval = -1
    state.bestindomainvalstd = -1
    state.bestindomainvalde = -1
    state.bestrec = -1
    state.bestrecde = -1
    state.bestonlinerec = -1
    state.bestonlinerecde = -1
    epochsl = []
    indomain = []
    indomainstd = []
    indomainval = []
    indomainvalstd = []
    rec = []
    recdense = []
    reconline = []
    #-------------------------------

    # Model initialization:

    inp = theano.sparse.csr_matrix()
    pattern = T.matrix()
    target = T.matrix()
    
    hid_lin = theano.sparse.dot(inp,Wenc)+benc
    if state.act == 'rect':
        hid_out = hid_lin * (hid_lin > 0)
    if state.act == 'sigmoid':
        hid_out = T.nnet.sigmoid(hid_lin)
    L1_reg = T.mean(T.sum(hid_out * hid_out,axis=1),axis=0)
    rec_lin = theano.sparse.sampling_dot( hid_out, Wdec , pattern)+bdec
    # the sigmoid is inside the cross_entropy function.
    if not hasattr(state,'scaling'):
        state.scaling = False
    if state.cost == 'CE':
        cost, dum = cross_entropy_sampled_cost(target, rec_lin,pattern,state.scaling)
        cost_dense, cost_decoupled_dense = cross_entropy_sampled_cost(target, T.dot(hid_out,Wdec.T) +bdec,None)
    if state.cost == 'MSE':
        cost, dum = MSE_sampled_cost(target, rec_lin,pattern,state.scaling)
        cost_dense, cost_decoupled_dense = MSE_sampled_cost(target, T.dot(hid_out,Wdec.T)+bdec ,None)
    if state.regcoef != 0.:
        cost = cost + state.regcoef * L1_reg

    grad = T.grad(cost,[Wenc,Wdec,benc,bdec])
    updates = dict( (p,p-state.lr*g) for p,g in zip([Wenc,Wdec,benc,bdec],grad) )
    TRAINFUNC = theano.function([inp,target,pattern],cost, updates = updates)
    ERR = theano.function([inp,target],[cost_dense,cost_decoupled_dense]) 
    
    # Train the current DAE
    for epoch in range(state['nepochs']):
    # Load sequentially dense batches of the training data
        reconstruction_error_batch = 0
        update_count1 = 0
        for batchnb in range(full_train.shape[0]/state.batchsize):
            tmpinp = numpy.asarray(full_train[batchnb*state.batchsize:(batchnb+1)*state.batchsize].toarray(),dtype=theano.config.floatX)
            zer = numpy.asarray(numpy.random.binomial(n = 1, p = 1-state.zeros, size = tmpinp.shape),dtype=theano.config.floatX)
            one = numpy.asarray(numpy.random.binomial(n = 1, p = state.ones, size = tmpinp.shape),dtype=theano.config.floatX)
            tmpinpnoise = scipy.sparse.csr_matrix(zer * tmpinp + (tmpinp==0) * one,dtype=theano.config.floatX)
            if state.pattern == 'inp':
                tmppattern = numpy.asarray((tmpinp + numpy.random.binomial(size = tmpinp.shape,n=1,p=state.ratio))>0,dtype=theano.config.floatX )
            elif state.pattern == 'noise':
                tmppattern = numpy.asarray(((1-zer)*tmpinp + one + numpy.random.binomial(size = tmpinp.shape,n=1,p=state.ratio))>0,dtype=theano.config.floatX)
            elif state.pattern == 'inpnoise':
                tmppattern = numpy.asarray((tmpinp + one + numpy.random.binomial(size = tmpinp.shape,n=1,p=state.ratio))>0,dtype=theano.config.floatX)
            elif state.pattern == 'random':
                tmppattern = numpy.asarray(numpy.random.binomial(size = tmpinp.shape,n=1,p=state.ratio),dtype=theano.config.floatX)
            tmp = TRAINFUNC(tmpinpnoise,tmpinp,tmppattern) 
	    reconstruction_error_batch += tmp
            update_count1 += 1
        print >> sys.stderr, '...finished training epoch #%s' % (epoch+1)
	print >> sys.stderr, "\t\tMean reconstruction error %s" % (reconstruction_error_batch/float(update_count1))
        full_train = full_train[numpy.random.permutation(full_train.shape[0]),:]
        if epoch+1 in state.epochs:
            #rec test err
            update_count2 = 0
            test_recerr = 0
            test_recerrd = numpy.zeros((state.ninputs,))
            for batchnb in range(full_test.shape[0]/state.batchsizeerr):
                tmpinp = scipy.sparse.csr_matrix(numpy.asarray(full_test[batchnb*state.batchsizeerr:(batchnb+1)*state.batchsizeerr].toarray(),dtype=theano.config.floatX))
                recerr,recerrd = ERR(tmpinp,numpy.asarray(tmpinp.toarray(),dtype=theano.config.floatX))
                test_recerr += recerr
                test_recerrd += recerrd
                update_count2 += 1
	    if not os.path.isdir(SavePath):
	        os.mkdir(SavePath)
	    modeldir = os.path.join(SavePath, 'currentmodel' )
	    if not os.path.isdir(modeldir):
	        os.mkdir(modeldir)
            f = open(modeldir+'/params.pkl','w')
            cPickle.dump(Wenc.value,f,-1)
            cPickle.dump(Wdec.value,f,-1)
            cPickle.dump(benc.value,f,-1)
            cPickle.dump(bdec.value,f,-1)
            f.close()
	    createdatafilessparse(1,Wenc,benc,SavePath,state.small,state.ninputs,state.act)
	    currentresults =  validtest(1,SavePath,state.small,state.folds,state.ninputs)
            epochsl += [epoch+1] 
            indomainval += [currentresults[1][0]]
            indomainvalstd += [currentresults[1][1]]
            indomain += [currentresults[0][0]]
            indomainstd += [currentresults[0][1]]
            rec += [test_recerr/float(update_count2)]
            recdense += [test_recerrd/float(update_count2)]
            reconline += [reconstruction_error_batch/float(update_count1)]
	    print '###### RESULTS :'
	    print 'Depth:',1
	    print 'Epoch:',epoch+1
	    print 'Online Reconstruction:',reconstruction_error_batch/float(update_count1)
            print 'Reconstruction:',test_recerr/float(update_count2)
	    print 'in-domain val:', currentresults[1][0], '+/-', currentresults[1][1]
	    print 'in-domain test:', currentresults[0][0], '+/-', currentresults[0][1]
	    print ' '
	    f = open('results.pkl','w')
	    cPickle.dump(epochsl,f,-1)
            cPickle.dump(rec,f,-1)
            cPickle.dump(recdense,f,-1)
            cPickle.dump(reconline,f,-1)
            cPickle.dump((indomainval,indomainvalstd),f,-1)
            cPickle.dump((indomain,indomainstd),f,-1)
            f.close()
            if test_recerr/float(update_count2) < state.bestrec  or state.bestrec==-1:
	        state.bestrec = test_recerr/float(update_count2)
	        state.bestrecde =(1,epoch+1)
	    if reconstruction_error_batch/float(update_count1) < state.bestonlinerec  or state.bestonlinerec==-1:
	        state.bestonlinerec = reconstruction_error_batch/float(update_count1)
	        state.bestonlinerecde =(1,epoch+1)
	    if currentresults[1][0] < state.bestindomainval or state.bestindomainval==-1:
                modeldir = os.path.join(SavePath, 'bestmodel' )
	        if not os.path.isdir(modeldir):
	            os.mkdir(modeldir)
                f = open(modeldir+'/params.pkl','w')
                cPickle.dump(Wenc.value,f,-1)
                cPickle.dump(Wdec.value,f,-1)
                cPickle.dump(benc.value,f,-1)
                cPickle.dump(bdec.value,f,-1)
                f.close()
	        state.bestindomain = currentresults[0][0]
	        state.bestindomainstd = currentresults[0][1]
	        state.bestindomainval = currentresults[1][0]
	        state.bestindomainvalstd = currentresults[1][1]
	        state.bestindomainvalde =(1,epoch+1)
	state.currentepoch = epoch+1
	channel.save()
    return channel.COMPLETE
