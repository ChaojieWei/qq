#!/usr/bin/env/python
# System libraries
import sys,os
import multiprocessing
# Numpy, Theano, Lasagne, Nolearn, Sklearn, Scipy
import numpy as np
import theano
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from lasagne.objectives import categorical_crossentropy
from nolearn.lasagne import BatchIterator, TrainSplit
from nolearn.lasagne import NeuralNet
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from lib.sklearn import grid_search
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

# Import argument parsing library
import argparse, argcomplete

# My own utils library
from utils import load_data, saveVar, loadVar, AdjustVariable, EarlyStopping, float32, my_objective, printNet,  testNetworkInit

parser = argparse.ArgumentParser(description='CNN using Theano, Lasagne and NoLearn.')
parser.add_argument('--load_network',help='Load a previously saved network',action='store_true')
parser.add_argument('--load_best_network',help='Load the best network for the selected dataset',action='store_true')
parser.add_argument('--print_network',help='Print the loaded network',action='store_true')
parser.add_argument('--short_print_network',help='Print the main parameters of the network',action='store_true')
parser.add_argument('--test_seed_initialization',help='Test for the correct seed initialization',action='store_true')
parser.add_argument('--network_type',help='The type of network to use if not loaded [CONVNET,LOGREG] (default CONVNET)')
parser.add_argument('--num_epochs',help='Max number of epochs for training (default 5000)')
parser.add_argument('--learning_rate',help='The learning rate (default 0.001)')
parser.add_argument('--kernel_size',help='The size of the kernel (default 87)')
parser.add_argument('--num_kernel',help='The number of kernels to use (default 2)')
parser.add_argument('--momentum',help='The momentum of the Nesterov Gradient Descend (default 0.7)')
parser.add_argument('--num_classes',help='Number of classes (deprecated)')
parser.add_argument('--batch_size',help='Batch size (default 1)')
parser.add_argument('--stride',help='The convolutional stride (default 33)')
parser.add_argument('--lamda1',help='The constant for the L2 norm regularization (default 0)')
parser.add_argument('--lamda2',help='The constant for the proximity L2 norm regularization (default 0)')
parser.add_argument('--earlystop',help='Use early stop (default false)',action='store_true')
parser.add_argument('--earlystop_iteration',help='The number of iteration to wait for Early-stop (default 100)')
parser.add_argument('--normalize_y',help='Normalize between 0 and 1 the observed variable y (default: no)',action='store_true')
parser.add_argument('--only_validation',help='Don\'t use the test set (default false)',action='store_true')
parser.add_argument('--dataset_seed',help='Seed to use to shuffle the datasets (default -1 means no shuffle)')
parser.add_argument('--lasagne_seed',help='Seed to use to initialize the weights of the network (-1 means no shuffle, default ranodom)')
parser.add_argument('--hyperparameter_optimization',help='Use hyperparameter optimiztion [NONE|RANDOM|EXHAUSTIVE] (default NONE)')
parser.add_argument('--dataset',help='The dataset to use (default extFTIR beer dataset)')
parser.add_argument('--max_cpus',help='Maximum number of cpus available (default is the half of the system cpus for security reasons)')

argcomplete.autocomplete(parser)
args = parser.parse_args()
# Network parameters
NETWORK_TYPE = args.network_type if args.network_type else 'CONVNET'
LOAD_NETWORK = bool(args.load_network) if args.load_network else False
PRINT_NETWORK = bool(args.print_network) if args.print_network else False
LOAD_BEST_NETWORK = bool(args.load_best_network) if args.load_best_network else False
SHORT_PRINT_NETWORK = bool(args.short_print_network) if args.short_print_network else False
TEST_SEED_INITIALIZATION = bool(args.test_seed_initialization) if args.test_seed_initialization else False
NUM_EPOCHS = int(args.num_epochs) if args.num_epochs else 5000
LEARNING_RATE = float(args.learning_rate) if args.learning_rate else 0.001
KERNEL_SIZE = int(args.kernel_size) if args.kernel_size else 87
STRIDE = int(args.stride) if args.stride else 33
NUM_KERNEL = int(args.num_kernel) if args.num_kernel else 1
MOMENTUM = float(args.momentum) if args.momentum else 0.7
NUM_CLASSES = int(args.num_classes) if args.num_classes else 2
BATCH_SIZE = int(args.batch_size) if args.batch_size else 1
BATCH_SIZE_TEST = BATCH_SIZE

LAMDA1 = float(args.lamda1) if args.lamda1 else 1.0
LAMDA2 = float(args.lamda2) if args.lamda2 else 0

EARLYSTOP_ITER = int(args.earlystop_iteration) if args.earlystop_iteration else 100
EARLYSTOP=bool(args.earlystop) if args.earlystop else False
if args.earlystop_iteration:
    EARLYSTOP=True
    
# Parameters for samples' analysis
ONLY_VALIDATION = bool(args.only_validation) if args.only_validation else False
SEED = int(args.dataset_seed) if args.dataset_seed else -1
LASAGNE_SEED = int(args.lasagne_seed) if args.lasagne_seed else np.random.randint(100)


# Set it to:
#		'RANDOM' for random hyperparameter optimization with grid search
#		'EXHAUSTIVE' for exhaustive hyperparameter optimization with grid search
#		'NONE' to use the network parameters defined above
HYPERPARAM_OPT = args.hyperparameter_optimization if args.hyperparameter_optimization else 'NONE'
# Environment variable
DATASET = args.dataset if args.dataset else 'extFTIR'
NETWORK_NAME = ('best_' if LOAD_BEST_NETWORK else 'last_') + NETWORK_TYPE
NAME_FILTERS = 'components_' + NETWORK_NAME
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/'

OUTPUT_DIR = '.'

MAX_CPUS = int(args.max_cpus) if args.max_cpus else multiprocessing.cpu_count()

if LOAD_BEST_NETWORK:
    LOAD_NETWORK=True
# Loading dataset
X, y, X_test,y_test = load_data(ROOT_DIR,DATASET,only_validation=ONLY_VALIDATION,scale_dataset=(not HYPERPARAM_OPT=='RANDOM' and not HYPERPARAM_OPT=='EXHAUSTIVE') ,shuffle=SEED)
NO_TEST_SET=(X_test.shape[0]==0)
# Num of classes
NUM_CLASSES=len(np.unique(y)) 
     
# Cross-validation folds
N_FOLDS = 10 

if BATCH_SIZE<0:
  BATCH_SIZE=y.shape[0]
  BATCH_SIZE_TEST=BATCH_SIZE

SAVED_VAR_DIR = ROOT_DIR + 'saved_networks/' + DATASET  + '/'
if not os.path.exists(SAVED_VAR_DIR):
    os.makedirs(SAVED_VAR_DIR)

 


if LASAGNE_SEED>0 and not LOAD_NETWORK and HYPERPARAM_OPT=='NONE':
  if not args.lasagne_seed:
    print('Using seed %d for initializing weights') % (LASAGNE_SEED)	
    raw_input('Press ENTER to continue...')
  lasagne.random.set_rng(np.random.RandomState(LASAGNE_SEED))

print('Train-valid set size:',X.shape[0],'Test set size:',X_test.shape[0],'#Classes:',NUM_CLASSES,'Batch size:',BATCH_SIZE,'#Features:',X.shape[1])

# Load a saved network
if LOAD_NETWORK:
  best_net = loadVar(NETWORK_NAME ,SAVED_VAR_DIR)
  if PRINT_NETWORK:
    printNet(best_net)
  elif SHORT_PRINT_NETWORK:
    params_=getCNNParams(NETWORK_NAME,SAVED_VAR_DIR)
    for key,value in params_.iteritems():
      try:
	print('%s\t=>\t%f')%(key.upper(),value)
      except:
	pass
  HYPERPARAM_OPT='NONE'
else:
  # Set up the network

  if 'CONVNET' in NETWORK_TYPE:
      adjust_and_stop=[
        AdjustVariable('update_learning_rate', start=LEARNING_RATE, stop=LEARNING_RATE*0.01)
      ]
      if EARLYSTOP and EARLYSTOP_ITER>0:
        adjust_and_stop.append(EarlyStopping(patience=EARLYSTOP_ITER))
      layers=[('input', layers.InputLayer),
		  ('conv1d', layers.Conv1DLayer),

		  ('output', layers.DenseLayer),
		  ]
      net1 = NeuralNet(
	  layers=layers,
	  # input layer
	  input_shape=(None, 1, X.shape[1]),
	  # layer conv1d
	  conv1d_num_filters=NUM_KERNEL,
	  conv1d_filter_size= KERNEL_SIZE,
	  conv1d_stride=STRIDE,
	  conv1d_pad='valid',
	  conv1d_nonlinearity=lasagne.nonlinearities.rectify,
	  conv1d_W=lasagne.init.GlorotUniform(gain=np.sqrt(2)),
	  conv1d_b=lasagne.init.Constant(0.),

	  # output layer
	  output_nonlinearity=lasagne.nonlinearities.softmax ,
	  output_W=lasagne.init.GlorotUniform() ,
	  output_b=lasagne.init.Constant(0.) ,
	  output_num_units=NUM_CLASSES,
	  # optimization method params
	  objective=my_objective,
	  objective_lamda1=LAMDA1,
	  objective_lamda2=LAMDA2,
	  
	  update=nesterov_momentum,
	  update_momentum=theano.shared(float32(MOMENTUM)),

	  train_split=TrainSplit(eval_size=0.2 if EARLYSTOP and EARLYSTOP_ITER>0 else 0.0),
	  update_learning_rate=theano.shared(float32(LEARNING_RATE)),
	  max_epochs=NUM_EPOCHS,
	  batch_iterator_train=BatchIterator(batch_size=BATCH_SIZE),
	  batch_iterator_test=BatchIterator(batch_size=BATCH_SIZE_TEST),
	  on_epoch_finished=adjust_and_stop,
	  regression=False,
	  seed=LASAGNE_SEED,
	  verbose=(not HYPERPARAM_OPT=='RANDOM' and not HYPERPARAM_OPT=='EXHAUSTIVE'),
	  )

  elif NETWORK_TYPE == 'LOGREG':
    layers=[('input', layers.InputLayer),
      ('output', layers.DenseLayer),
    ]
    net1 = NeuralNet(
	  layers=layers,
	  # input layer
	  input_shape=(None, 1, X.shape[1]),
	  # output layer
	  output_nonlinearity=lasagne.nonlinearities.softmax,
	  output_W=lasagne.init.GlorotUniform() ,
	  output_b=lasagne.init.Constant(0.) ,
	  output_num_units=NUM_CLASSES,
	  # optimization method params
	  objective=my_objective,
	  objective_lamda1=LAMDA1,
	  
	  update=nesterov_momentum,
	  update_momentum=theano.shared(float32(MOMENTUM)),

	  train_split=TrainSplit(eval_size=0.0),
	  update_learning_rate=theano.shared(float32(LEARNING_RATE)),
	  max_epochs=NUM_EPOCHS,
	  batch_iterator_train=BatchIterator(batch_size=BATCH_SIZE),
	  batch_iterator_test=BatchIterator(batch_size=BATCH_SIZE_TEST),
	  on_epoch_finished=[
	      AdjustVariable('update_learning_rate', start=LEARNING_RATE, stop=LEARNING_RATE*0.01),
	      ],
	  regression=('_reg' in DATASET),
	  seed=LASAGNE_SEED,
	  verbose=(not HYPERPARAM_OPT=='RANDOM' and not HYPERPARAM_OPT=='EXHAUSTIVE' ),
	  )
  else:
    print('Impossible to setup network"',NETWORK_TYPE,'"')
    sys.exit(1)
      
# Test if the random initialization of weights is correctly working
if TEST_SEED_INITIALIZATION:
  print('Initialization of parameters is equal' if testNetworkInit(net1,LASAGNE_SEED) else 'Different initialization for parameters','with seed:',LASAGNE_SEED)
# Train the network and Hyperparameter optimization with GridSearchCV or RandomizedSearchCV
if HYPERPARAM_OPT=='RANDOM':
  if NETWORK_TYPE == 'CONVNET':
    param_dist={
	      'conv1d_filter_size': sp_randint(5,40),
	      'on_epoch_finished_start' : [1e-8,1e-7,1e-6,1e-5,1e-4,1e-3, 1e-2, 1e-1],
	      'update_momentum' : [0.1,0.3,0.5,0.7,0.9],
	      'conv1d_num_filters' : [2,4],
	      'conv1d_stride' : sp_randint(1,20),
	      'objective_lamda1' : [1e-3,1e-2,1e-1,0,1,1e1,1e2,1e3],
	      'objective_lamda2' : [1e-3,1e-2,1e-1,0,1,1e1,1e2,1e3],
	      'seed': [LASAGNE_SEED] if args.lasagne_seed and LASAGNE_SEED>=0 else sp_randint(1,10000000)

    }
  elif 'LOGREG' in NETWORK_TYPE:
    param_dist={
	      'on_epoch_finished_start' : [1e-4,1e-3, 1e-2, 1e-1],
	      'objective_lamda1' : [1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3],
	      'seed': [LASAGNE_SEED] if args.lasagne_seed and LASAGNE_SEED>=0 else sp_randint(1,10000000)

    }


  clf = grid_search.RandomizedSearchCV(net1, 
				      param_distributions=param_dist,
				      cv=N_FOLDS ,
				      scoring=None ,
				      n_jobs=int(MAX_CPUS),
				      refit=True,
				      n_iter=5*len(param_dist),verbose=10)
elif HYPERPARAM_OPT=='EXHAUSTIVE':
  if NETWORK_TYPE == 'CONVNET':
    parameters={
	    'conv1d_filter_size': [3,5,11,31,41,53,61,71,83,91], 
	    'update_momentum' : [0.1,0.3,0.5,0.7,0.9],
	    'conv1d_num_filters' : [2,4],
	    'conv1d_stride' : range(1,40),
	    'objective_lamda1' : [1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3],
	    'objective_lamda2' : [1e-3,1e-2,1e-1,0,1,1e1,1e2,1e3],
	    'on_epoch_finished_start' : [1e-8,1e-7,1e-6,1e-5,1e-4,1e-3, 1e-2, 1e-1],
	    'seed': [LASAGNE_SEED] if args.lasagne_seed and LASAGNE_SEED>=0 else np.randint(10000000,size=10)
	   }
  elif 'LOGREG' in NETWORK_TYPE:
    parameters={
	    'on_epoch_finished_start' : [1e-5,1e-4,1e-3, 1e-2, 1e-1],
	    'objective_lamda1' : [1e-3,1e-2,1e-1,0,1e0,1e1,1e2,1e3],
	    'seed': [LASAGNE_SEED] if arg.lasagne_seed and LASAGNE_SEED>=0 else np.randint(10000000,size=10)
    }
  
  clf = grid_search.GridSearchCV(net1, 
				parameters,
				cv=N_FOLDS,
				n_jobs=int(MAX_CPUS),verbose=10)
log_file=open(SAVED_VAR_DIR + NETWORK_NAME + '.log','w')
if HYPERPARAM_OPT=='RANDOM' or HYPERPARAM_OPT=='EXHAUSTIVE':
    clf.fit(X.reshape((-1, 1, X.shape[1])), y.astype(np.uint8))
    print('Train Accuracy:\t %.4f with parameters: \n')%(clf.best_score_)
    print(clf.grid_scores_)
    print(clf.best_params_)
    best_net=clf.best_estimator_
    saveVar(best_net,NETWORK_NAME ,SAVED_VAR_DIR)
    saveVar(clf.best_params_,NETWORK_NAME + '_params' ,SAVED_VAR_DIR)
    saveVar(clf.grid_scores_,NETWORK_NAME + '_cvscores' ,SAVED_VAR_DIR)
    log_file.write('Accuracy train set: ' + str(clf.best_score_) + '\n')

else:
  if not LOAD_NETWORK:
    best_net = net1.fit(X.reshape((-1, 1, X.shape[1])), y.astype(np.uint8))
    saveVar(best_net,NETWORK_TYPE,SAVED_VAR_DIR)
  preds = best_net.predict(X.reshape((-1, 1, X.shape[1])))
  print(classification_report(y.astype(np.uint8), preds))
  cm = confusion_matrix(y.astype(np.uint8), preds)
  print(cm)
  print("Accuracy:\t%.4f") % ( accuracy_score(y.astype(np.uint8), preds))
  log_file.write('Accuracy train set: ' + str(accuracy_score(y.astype(np.uint8), preds)) + '\n')
  if not NO_TEST_SET:
    preds = best_net.predict(X_test.reshape((-1, 1, X_test.shape[1])))
    print(classification_report(y_test.astype(np.uint8), preds))
    print("Accuracy test:\t%.4f") % (accuracy_score(y_test.astype(np.uint8), preds))
    
    log_file.write('Accuracy test set: ' + str(accuracy_score(y_test.astype(np.uint8), preds)) + '\n')
    
  # Visualize the confusion matrix in case of a test set
    preds = best_net.predict(X_test.reshape((-1, 1, X_test.shape[1])))
    cm = confusion_matrix(y_test.astype(np.uint8), preds)
    print(cm)
log_file.close()