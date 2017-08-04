# -*- coding: utf-8 -*-

# Imports.
import cPickle                              as pkl
import cv2
import h5py                                 as H
import logging                              as L
import numpy                                as np
import os, pdb, random, re, sys
import pysnips.experiment
import time
import yaml                                 as Y
import theano                               as T
import theano.gradient                      as TG
import theano.tensor                        as TT
import theano.scan_module                   as TS
import theano.tensor.nnet                   as TTN
import theano.tensor.nnet.conv              as TTNC
import theano.tensor.nnet.bn                as TTNB
import theano.tensor.signal.pool            as TTSP
from   theano import config                 as TC
import theano.printing                      as TP
TC.dnn.conv.algo_bwd_filter = "deterministic"
TC.dnn.conv.algo_bwd_data   = "deterministic"
TC.floatX                   = "float32"
TC.mode                     = "FAST_RUN"
TC.optimizer                = "fast_run"


# Utilities
def getExperiment(d):
	return Eqprop(d=d)
def getMNIST(dataDir):
	import gzip
	with gzip.open(os.path.join(dataDir, "mnist.pkl.gz"), "rb") as f:
		trainSet, validSet, testSet = pkl.load(f)
	return trainSet[0], trainSet[1],\
	       validSet[0], validSet[1],\
	       testSet [0], testSet [1]


# Parameters Descriptors
class Param(object):
	def __init__(self, *args, **kwargs):
		self.__dict__.update(kwargs)
	@property
	def name(self): raise UnimplementedException("Need a name!")
	@property
	def shape(self): return ()
	@property
	def ndim(self): return len(self.shape)
	@property
	def dtype(self): return TC.floatX
	@property
	def broadcastable(self): return (False,)*len(self.shape)
	@property
	def fanin(self):
		shape = self.shape
		return 1 if len(shape) <= 0 else int(np.prod(shape[1:]))
	@property
	def numParams(self): return int(np.prod(self.shape))
	@property
	def npinit(self):
		scale = float(np.sqrt(1.0/self.fanin))
		return np.random.normal(0, scale, self.shape).astype(self.dtype)
	@property
	def newTheanoSV(self):
		return T.shared(self.npinit,
		                self.name,
		                broadcastable=self.broadcastable)
	@property
	def newTheanoVar(self):
		return TT.TensorType(dtype         = self.dtype,
		                     broadcastable = self.broadcastable)(name=self.name)
class ParamBias(Param):
	@property
	def name(self): return "b_"+str(self.layerNum)
	@property
	def shape(self): return (1, self.fmaps)
	@property
	def fanin(self): raise Exception("Fan-in meaningless for biases.")
	@property
	def broadcastable(self): return (True, False)
	@property
	def npinit(self):
		return np.zeros(self.shape, self.dtype)
class ParamWeightW(Param):
	@property
	def name(self): return "W_"+str(self.layerFrom)+"_"+str(self.layerTo)
	@property
	def shape(self): return (self.fmapsFrom, self.fmapsTo)
class ParamWeightV(Param):
	@property
	def name(self): return "V_"+str(self.layerFrom)+"_"+str(self.layerTo)
	@property
	def shape(self): return (self.fmapsFrom, self.fmapsTo)


# State Descriptors
class State(object):
	def __init__(self, *args, **kwargs):
		self.__dict__.update(kwargs)
	@property
	def name(self): raise UnimplementedException("Need a name!")
	@property
	def shape(self): return ()
	@property
	def ndim(self): return len(self.shape)
	@property
	def dtype(self): return TC.floatX
	@property
	def broadcastable(self): return (False,)*len(self.shape)
	@property
	def numStates(self): return int(np.prod(self.shape))
	@property
	def npinit(self):
		return np.random.uniform(0, 1, self.shape).astype(self.dtype)
	@property
	def newTheanoVar(self):
		return TT.TensorType(dtype         = self.dtype,
		                     broadcastable = self.broadcastable)(name=self.name)
	@property
	def newTheanoZero(self):
		return TT.TensorType(dtype         = self.dtype,
		                     broadcastable = self.broadcastable).make_constant([[0]], name=self.name)
class HiddenState(State):
	def __init__(self, **kwargs):
		kwargs["_shape"] = kwargs["shape"]
		super(HiddenState, self).__init__(**kwargs)
	@property
	def name(self):
		return "h_"+str(self.layerNum)
	@property
	def shape(self): return self._shape


class Eqprop(pysnips.experiment.Experiment):
	def __init__(self, *args, **kwargs):
		super(Eqprop, self).__init__(*args, **kwargs)
		
		datasets = {"cifar10":  (784,10),
		            "cifar100": (784,100),
		            "svhn":     (1024,10)}
		self.arch = [datasets[self.d.dataset][0]]                             + \
		            list(np.fromstring(self.d.arch, dtype="uint64", sep=",")) + \
		            [datasets[self.d.dataset][1]]
		
		self.trainX, self.trainY, \
		self.validX, self.validY, \
		self.testX,  self.testY   = getMNIST(self.dataDir)
		
		self.NepochStart = 0
		self.Nepoch      = self.d.num_epochs
		self.NbatchTrain = len(self.trainX)/self.d.batch_size
		self.NbatchValid = 0
		
		self.buildTSVs()
		self.buildDotOp()
		self.buildGOp()
		self.buildFns()
	def getThetaIter(self):
		for i in xrange(len(self.arch)):
			yield ParamBias   (layerNum=i, fmaps=self.arch[i])
		for i in xrange(1, len(self.arch)):
			yield ParamWeightW(layerFrom = i-1, layerTo = i,
			                   fmapsFrom = self.arch[i-1],
			                   fmapsTo   = self.arch[i])
			yield ParamWeightV(layerFrom = i,   layerTo = i-1,
			                   fmapsFrom = self.arch[i],
			                   fmapsTo   = self.arch[i-1])
	def getStateIter(self):
		for i in xrange(0, len(self.arch)):
			yield HiddenState(layerNum=i, shape=(None, self.arch[i]))
	def getGOpArgNames(self):
		for t in self.getThetaIter():
			yield t.name
		for s in self.getStateIter():
			yield s.name
		yield "beta"
		yield "y"
	def getGOpFreeArgNames(self):
		for t in self.getThetaIter():
			yield t.name
		for s in self.getStateIter():
			yield "free_"+s.name
		yield "beta"
		yield "y"
	def getGOpClampedArgNames(self):
		for t in self.getThetaIter():
			yield t.name
		for s in self.getStateIter():
			yield "clamped_"+s.name
		yield "beta"
		yield "y"
	def getGOpRetNames(self):
		for s in self.getStateIter():
			yield s.name
	def getFPIArgNames(self):
		for n in self.getFPISequenceArgNames():    yield n
		for n in self.getFPIInOutArgNames():       yield n
		for n in self.getFPINonSequenceArgNames(): yield n
	def getFPISequenceArgNames(self):
		return
		yield
	def getFPIInOutArgNames(self):
		for t in self.getThetaIter():
			yield t.name
		for s in self.getStateIter():
			yield "free_"+s.name
		for s in self.getStateIter():
			yield "clamped_"+s.name
		yield "i"
	def getFPINonSequenceArgNames(self):
		yield "beta"
		yield "y"
		yield "lr"
	def getFPIRetNames(self):
		# Initial Values/Outputs
		for t in self.getThetaIter():
			yield t.name
		for s in self.getStateIter():
			yield "free_"+s.name
		for s in self.getStateIter():
			yield "clamped_"+s.name
		yield "i"
	def buildTSVs(self):
		self.thetaTSVs = {t.name:t.newTheanoSV for t in self.getThetaIter()}
	def buildDotOp(self):
		a = TT.fmatrix("a")
		W = TT.fmatrix("W")
		V = TT.fmatrix("V")
		b = TT.dot(a, W)
		self.dot = T.OpFromGraph([a, W, V], [b], inline=True,
		                         grad_overrides=[
		    lambda inps, grads: TT.dot(grads[0], inps[2]),
		    "default",
		    "default"
		])
	def buildGOp(self):
		#
		#          Dynamical System function ds/dt = f(\theta, v, s)
		#
		# The extended version is ds/dt = g(\theta, v, s, \beta) with
		#
		#     g(\theta, v, s, \beta) = f(\theta, v, s) - \beta dC(v,s)/ds
		#
		
		ins   = {}
		ins.update({t.name:t.newTheanoVar for t in self.getThetaIter()})
		ins.update({s.name:s.newTheanoVar for s in self.getStateIter()})
		ins.update({"beta": TT.fscalar("beta")})
		ins.update({"y":    TT.fmatrix("y")})
		outs  = {}
		
		
		beta  = ins["beta"]
		y     = ins["y"]
		for i in xrange(len(self.arch)):
			if   i == 0:
				dh = TT.zeros_like(ins["h_"+str(i)])
			elif i == len(self.arch)-1:
				bbelow = ins["b_"+str(i-1)]
				Wbelow = ins["W_"+str(i-1)+"_"+str(i)]
				Vbelow = ins["V_"+str(i)  +"_"+str(i-1)]
				
				hbelow = ins["h_"+str(i-1)]
				hhere  = ins["h_"+str(i)]
				
				dh = self.dot(self.rho(hbelow+bbelow), Wbelow, Vbelow) - \
				     beta*(hhere-y)
			else:
				bbelow = ins["b_"+str(i-1)]
				Wbelow = ins["W_"+str(i-1)+"_"+str(i)]
				Vbelow = ins["V_"+str(i)  +"_"+str(i-1)]
				babove = ins["b_"+str(i+1)]
				Wabove = ins["W_"+str(i)  +"_"+str(i+1)]
				Vabove = ins["V_"+str(i+1)+"_"+str(i)]
				
				hbelow = ins["h_"+str(i-1)]
				habove = ins["h_"+str(i+1)]
				
				dh = self.dot(self.rho(habove+babove), Wabove, Vabove) + \
				     self.dot(self.rho(hbelow+bbelow), Wbelow, Vbelow)
			outs["h_"+str(i)] = dh
		
		inputs  = [ins [k] for k in self.getGOpArgNames()]
		outputs = [outs[k] for k in self.getGOpRetNames()]
		self.g  = T.OpFromGraph(inputs, outputs, name="gOp")
	def buildFns(self):
		#
		# Fixed-Point Loop Iteration
		#
		
		def fixedPointIteration(*args):
			fpiIns         = dict(zip(self.getFPIArgNames(), args))
			gOpFreeIns     = [fpiIns[k] for k in self.getGOpFreeArgNames()]
			gOpClampedIns  = [fpiIns[k] for k in self.getGOpClampedArgNames()]
			gOpFreeRets    = self.g(*gOpFreeIns)
			gOpClampedRets = self.g(*gOpClampedIns)
			gOpDiffs       = [c-f for c,f in zip(gOpClampedRets, gOpFreeRets)]
			for i, (s, fi, fr, ci, cr) in enumerate(zip(self.getStateIter(),
			    gOpFreeIns,    gOpFreeRets, gOpClampedIns, gOpClampedRets)):
				if i>0:
					fpiIns["free_"   +s.name] = self.rho(fi+fr)
					fpiIns["clamped_"+s.name] = self.rho(ci+cr)
			
			allStates      = TT.concatenate([s.flatten() for s in gOpFreeRets])
			allDiffs       = TT.concatenate([d.flatten() for d in gOpDiffs])
			
			for t, tName in [(fpiIns[t.name], t.name) for t in self.getThetaIter()]:
				J  = TT.jacobian(allStates, t.flatten(), disconnected_inputs="ignore")
				dt = J.T.dot(allDiffs).reshape(t.shape)
				fpiIns[tName] += fpiIns["lr"]*dt
			
			fpiIns["i"]   += 1
			
			fpiRets        = [fpiIns[k] for k in self.getFPIRetNames()]
			return fpiRets
		
		#
		# Fixed-Point Loop
		#
		
		v    = TT.fmatrix("v")
		y    = TT.fmatrix("y")
		beta = TT.fscalar("beta")
		lr   = TT.fscalar("lr")
		trainFnVars = {}
		trainFnVars.update({"free_"   +s.name:s.newTheanoZero for s in self.getStateIter()})
		trainFnVars.update({"clamped_"+s.name:s.newTheanoZero for s in self.getStateIter()})
		trainFnVars.update({t.name:self.thetaTSVs[t.name] for t in self.getThetaIter()})
		trainFnVars.update({"beta": beta})
		trainFnVars.update({"y":    y})
		trainFnVars.update({"lr":   lr})
		trainFnVars.update({"i":    TT.as_tensor_variable(np.array(0, dtype="int32"))})
		trainFnVars["free_h_0"] = trainFnVars["clamped_h_0"] = v
		nSteps = 20
		
		
		fpiSeq    = [trainFnVars[k] for k in self.getFPISequenceArgNames()]
		fpiInOut  = [trainFnVars[k] for k in self.getFPIInOutArgNames()]
		fpiNonSeq = [trainFnVars[k] for k in self.getFPINonSequenceArgNames()]
		
		outputs, updates = T.scan(fixedPointIteration, # Loop Body
		                          fpiSeq,              # Sequences
		                          fpiInOut,            # Initial Values/Output Info
		                          fpiNonSeq,           # Non-Sequences
		                          nSteps,              # Number of Steps
		                          strict = True)       # Be strict
		
		for i, t in enumerate(self.getThetaIter()):
			updates += [(self.thetaTSVs[t.name], outputs[i][-1])]
		
		self.trainFn = T.function([v, y, beta, lr], [], updates=updates,
		                          on_unused_input='warn')
	def rho(self, x):
		return TT.clip(x, 0, 1)
	
	# Checkpoint & Reload Machinery
	def load(self, path):         return self
	def dump(self, path):         return self
	def fromScratch(self):        return self
	def fromSnapshot(self, path): return self
	def run(self):
		for e in xrange(self.NepochStart, self.Nepoch):
			for i in xrange(self.NbatchTrain):
				self.trainFn(self.trainX[(i+0)*self.d.batch_size:(i+1)*self.d.batch_size],
				             np.eye(self.arch[0], dtype=TC.floatX)[self.trainY[(i+0)*self.d.batch_size:(i+1)*self.d.batch_size]],
				             5e-2, self.d.lr)
			#for i in xrange(self.NbatchValid):
			#	self.validFn(X, Y)
			
			
			
			#self.snapshot()

