#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Runs an entire experiment for benchmarking PURE_RANDOM_SEARCH on a testbed.

CAPITALIZATION indicates code adaptations to be made.
This script as well as files bbobbenchmarks.py and fgeneric.py need to be
in the current working directory.

Under unix-like systems:
    nohup nice python exampleexperiment.py [data_path [dimensions [functions [instances]]]] > output.txt &

"""
import sys # in case we want to control what to run via command line args
import time
import numpy as np
import fgeneric as fg
import bbobbenchmarks
from neuralnetwork import NeuralNetwork
e = fg.LoggingFunction(datapath='ellipsoid', algid='BFGS', comments='x0 uniformly sampled in [0, 1]^5, default settings')
argv = sys.argv[1:] # shortcut for input arguments
copy = []
datapath = '/home/cs4/ma1017/Desktop/bbob.v15.03/python/bbob_pproc/results'  if len(argv) < 1 else argv[0] #'PUT_MY_BBOB_DATA_PATH'

dimensions = [2] if len(argv) < 2 else eval(argv[1])
function_ids = [1]#bbobbenchmarks.nfreeIDs if len(argv) < 3 else eval(argv[2])
# function_ids = bbobbenchmarks.noisyIDs if len(argv) < 3 else eval(argv[2])
instances = range(1, 6) + range(41, 51) if len(argv) < 4 else eval(argv[3])

opts = dict(algid='tester', #'PUT ALGORITHM NAME',
            comments='PUT MORE DETAILED INFORMATION, PARAMETER SETTINGS ETC')
maxfunevals = '10 * dim' # 10*dim is a short test-experiment taking a few minutes
# INCREMENT maxfunevals SUCCESSIVELY to larger value(s)
minfunevals = 'dim + 2'  # PUT MINIMAL sensible number of EVALUATIONS before to restart
maxrestarts = 10000      # SET to zero if algorithm is entirely deterministic


def run_optimizer(fun, dim, maxfunevals, ftarget=-np.Inf):
    """start the optimizer, allowing for some preparation.
    This implementation is an empty template to be filled

    """
    # prepare
    x_start = 8. * np.random.rand(dim) - 4

    # call, REPLACE with optimizer to be tested
    PURE_RANDOM_SEARCH(fun, x_start, maxfunevals, ftarget)
    #NN(fun, x_start, maxfunevals, ftarget)

def sphere(inputTuple):
        answer=0
        for  index in inputTuple:
            answer += index**2
        #answer = inputTuple[0] * inputTuple[0]
        return answer

def NN(fun, x, maxfunevals, ftarget):
    dim = len(x)
    maxfunevals = min(1e8 * dim, maxfunevals)
    popsize = min(maxfunevals, 200)
    fbest = np.inf

    for _ in range(0, int(np.ceil(maxfunevals / popsize))):
    	xpop = 10. * np.random.rand(popsize, dim) - 5.
        nn = NeuralNetwork()
        b = nn.start(xpop)

        #print("TYPE OF B ==", b)

        fvalues = fun(np.asarray(nn.start(xpop)))#np.asarray(nn.start(xpop))
        print("fVALUE TYPE==",type(fvalues))
        print("XPOPPPPPPP", xpop)
        #for values in fvalues:
        #copy.append(value)
        print("COOOOOOOPPPPPPY = ", copy)
        print("TYPE FVALUES", fun(nn.start(xpop)))
        #print("FTARGET ==", ftarget)
        print("12345", type(nn.start(xpop)))
        print("Fvalues in NN", fvalues)
        idx = np.argsort(fvalues)
        print("IDX==", idx)
        if fbest > fvalues[idx[0]]:
            fbest = fvalues[idx[0]]
            xbest = xpop[idx[0]]
            print("XPOP", xpop[idx[0]])
        if fbest < ftarget:
            break
    print("XBEST VALUE", xbest)
    return xbest

def PURE_RANDOM_SEARCH(fun, x, maxfunevals, ftarget):
    """samples new points uniformly randomly in [-5,5]^dim and evaluates
    them on fun until maxfunevals or ftarget is reached, or until
    1e8 * dim function evaluations are conducted.

    """

    dim = len(x)
    maxfunevals = min(1e8 * dim, maxfunevals)
    popsize = min(maxfunevals, 200)
    fbest = np.inf
    numPoints = 0
    accumulation = 0


    for _ in range(0, int(np.ceil(maxfunevals / popsize))):
        xpop = 10. * np.random.rand(popsize, dim) - 5.
        print("contents OF XPOP", xpop)
        nn = NeuralNetwork()
        fvalues = fun(xpop)
        b= fun_id
        mvalues =  nn.start(xpop,fun_id)
        for index in xpop:
            numPoints+=1
        for i in range(0,len(mvalues)):
            accumulation += (fvalues[i]-mvalues[i])/numPoints
        test = []
        for index in xpop:
            test.append(sphere(index))
        print ("FVALUES ==", fvalues)
        print ("MVALUES ==", mvalues)
        print("SOLUTIONS", test)
        print("ACCUMULATION ==", accumulation)
        print ("NUM POINTS==", numPoints)


        idx = np.argsort(fvalues)
        if fbest > fvalues[idx[0]]:
            fbest = fvalues[idx[0]]
            xbest = xpop[idx[0]]
        if fbest < ftarget:  # task achieved
            break
    return xbest

t0 = time.time()
np.random.seed(int(t0))

f = fg.LoggingFunction(datapath, **opts)
for dim in dimensions:  # small dimensions first, for CPU reasons
    for fun_id in function_ids:
        for iinstance in instances:
            f.setfun(*bbobbenchmarks.instantiate(fun_id, iinstance=iinstance))

            # independent restarts until maxfunevals or ftarget is reached
            for restarts in xrange(maxrestarts + 1):
                if restarts > 0:
                    f.restart('independent restart')  # additional info
                run_optimizer(f.evalfun, dim,  eval(maxfunevals) - f.evaluations,
                              f.ftarget)
                #print"Min", minfunevals, "Dim is", dim
                #print  "Max", maxfunevals, "Dim is", dim

                if True:
                    break

            f.finalizerun()

            print('  f%d in %d-D, instance %d: FEs=%d with %d restarts, '
                  'fbest-ftarget=%.4e, elapsed time [h]: %.2f'
                  % (fun_id, dim, iinstance, f.evaluations, restarts,
                     3 - 2, (time.time()-t0)/60./60.))

        print '      date and time: %s' % (time.asctime())
    print '---- dimension %d-D done ----' % dim
