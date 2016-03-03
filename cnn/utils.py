#!/usr/bin/python

import sys
import os


import theano
import theano.tensor as T

from math import sqrt,ceil,floor
from collections import Counter
from random import shuffle
from lasagne.regularization import regularize_layer_params, l2, l1
import lasagne
from lasagne.objectives import aggregate
from lasagne.layers import get_output, get_all_layers, get_all_param_values
import numpy

from scipy.io import savemat, loadmat

import pickle

def pickle2mat(filename='',input_path='.',output_path='.',varname=''):
    
  """Convert .pickle file to .mat file and save it.
  Parameters
  ----------
  filename : str
    the name of the input/output file (default '')
  input_path : str
    the parent directory path of .pickle file
  output_path : str
    the parent directory path  path of .mat file
  varname : str
    the name of the variable inside the .pickle to save into the .mat file (default '')
  Returns
  -------
  nothing
  
  """

  var=loadVar(filename,input_path)
  if not output_path.endswith('/'):
    output_path+='/'
  if not varname:
      if filename.endswith('.pickle'):
	varname=filename[filename.rfind('_')+1:filename.rfind('.')]
      else:
	varname=filename[filename.rfind('_')+1:]
  if not varname:
    varname='_X'
  savemat(output_path + filename + '.mat', mdict={varname: var})

def softmax(w):
    
    """Applies the softmax function to an input array
    Parameters
    ----------
    w : array_like
        the input array
    Returns
    -------
    numpy ndarray
        the output array
    """
    
    w = numpy.array(w)

    maxes = numpy.amax(w, axis=1)
    maxes = maxes.reshape(maxes.shape[0], 1)
    e = numpy.exp(w - maxes)
    dist = e / numpy.sum(e, axis=1, keepdims=True)
    return dist

def floatX(arr):
    """Converts data to a numpy array of dtype ``theano.config.floatX``.
    Parameters
    ----------
    arr : array_like
        The data to be converted.
    Returns
    -------
    numpy ndarray
        The input array in the ``floatX`` dtype configured for Theano.
        If `arr` is an ndarray of correct dtype, it is returned as is.
    """
    return numpy.asarray(arr, dtype=theano.config.floatX)

def float32(k):
    """Converts a number or an array of numbers into the numpy.float32 format
    Parameters
    ----------
    k : array_like or number
        The data to be converted.
    Returns
    -------
    numpy ndarray or number
        The converted array/number
    """
    return numpy.cast['float32'](k)
def l2_paired(x):
    """Applies a modified L2 norm to a 1D vector that takes 
    into account the locality of the information
    Parameters
    ----------
    x : theano tensor 
        The input tensor.
    Returns
    -------
    theano tensor
        The output tensor
    """
  shapes=x.shape.eval()
  mask=numpy.eye(shapes[-1])
  mask[-1,-1]=0
  rolled=T.roll(x,-1,axis=len(shapes)-1)
  return T.sum((x - T.dot(rolled,mask))**2)
def my_objective(layers,
                 loss_function,
                 target,
                 lambda1,
                 lambda2,
                 aggregate=aggregate,
                 deterministic=False,
                 get_output_kw=None):
    """Custom objective function for Nolearn that include 2 different
    type of regularization terms
    Parameters
    ----------
    layers : array of Lasagne layers
        All the layers of the neural network
    loss_function : function
        The loss function to use
    lambda1 : float
        Constant for the L2 regularizaion term
    lambda2 : float
        Constant for the paired L2 regularizaion term
    aggregate : function
        Lasagne function to aggregate an element
        or item-wise loss to a scalar loss.
    deterministic : boolean
    
    Returns
    -------
    float
        The aggregated loss value
    """
    if get_output_kw is None:
        get_output_kw = {}
    net_out = get_output(
        layers[-1],
        deterministic=deterministic,
        **get_output_kw
        )
    hidden_layers = layers[1:-1]
    losses = loss_function(net_out, target) 
    if not deterministic:
      for i,h_layer in enumerate(hidden_layers):
	zeros = numpy.zeros(i).astype(int).astype(str)                
        denom = '10' +  ''.join(zeros) 
        losses = losses + i/float(denom) * (lambda1*regularize_layer_params(h_layer, l2) + lambda2*regularize_layer_params(h_layer, l2_paired))
    return aggregate(losses)
class EarlyStopping(object):
     """Class to apply early stopping during the learning phase"""
    def __init__(self, patience=100):
        """Init function for class EarlyStopping
        Parameters
        ----------
        patience : int
            How many iteration to wait before stopping if the accuracy is not higher
        """
        self.patience = patience
        self.best_valid = numpy.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        """Call function for class EarlyStopping
        Parameters
        ----------
        nn : Nolearn NeuralNetwork object
        train_history : array_like
            Contains information about accuracy for different epochs
        """
	#print(numpy.mean(nn.layers_['conv1d'].W.get_value()))
        current_valid = train_history[-1]['valid_loss']
        #print(train_history[-1])
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()

class AdjustVariable(object):
    """Class to update vairable values during the learning phase"""
    def __init__(self, name, start=0.03, stop=0.001):
        """Init function for class AdjustVariable
        Parameters
        ----------
        name : str
            Name of the variable to update
        start : float
            Start value
        stop : float
            Last value
        """
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        """Call function for class EarlyStopping
        Parameters
        ----------
        nn : Nolearn NeuralNetwork object
        train_history : array_like
            Contains information about accuracy for different epochs
        """
        if self.ls is None:
            self.ls = numpy.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)
def saveVar(var,name,path='./saved_vars/'):
  """Save a python variable into a .pickle file
  Parameters
  ----------
  var : python variable
    the python variable to save
  name : str
    the name of the .pickle output file
  path : str
    the directory where to save the .pickle file
  Returns
  -------
  nothing
  
  """
  try:
    if not path.endswith('/'):
      path+='/'
    f=open(path + name + '.pickle', 'w')
    pickle.dump(var, f)
    print('Writing variable:',name,'in path:',path)
  except IOError:
    print('Error writing variable:',name,'in path:',path)
def loadVar(filename,path='./saved_vars/'):
  """Load a python variable from a .pickle file
  Parameters
  ----------
  name : str
    the name of the .pickle output file
  path : str
    the directory where the .pickle file is located
  Returns
  -------
  var : python variable
    the loaded python variable
  
  """
  try:
    if not filename.endswith('.pickle'):
      filename+='.pickle'
    if not path.endswith('/'):
      path+='/'
    f=open(path + filename , 'r')
    var=pickle.load(f)
    print('Reading filename:',filename,'in path:',path)
    return var
  except IOError:
    print('Error reading filename:',filename,'in path:',path)
    return numpy.array([])
def fromCSV(path="dataset.csv",validation_path="",test_path="", perc_split=[],seed=-1,label_pos='last',num_labels=1,scale_dataset=False):
  """Load data from a .csv file
  Parameters
  ----------
  path : str
    the full path of the .csv input file
  validation_path : str
    the full path of the .csv input file to use for validation
  test_path : str
    the full path of the .csv input file to use for test
  perc_split : array_like
    the percentage of samples to include for train,validatio,test
  seed : int
    seed for shuffling the data (-1 means no shuffle)
  label_pos : str
    position of the label in the .csv file (first,last)
  num_labels : int
    the number of labels for each row of the .csv file
  sclae_dataset : boolean
    whether to normalize or not the dataset
  Returns
  -------
  (data_train,label_train) : pair of array_like elements
    dataset and labels for the train set
  (data_valid,label_valid) : pair of array_like elements
    dataset and labels for the validation set
  (data_test,label_test) : pair of array_like elements
    dataset and labels for the test set
  """
  data=None
  validation_data=None
  test_data=None
  try:
    data = numpy.genfromtxt(path, delimiter=',')
    if seed >0:
      numpy.random.seed(seed)
      data=numpy.random.permutation(data)
      numpy.savetxt('/tmp/permutated_' + str(seed) +'.csv',data,delimiter=',')
  except IOError:
    print("The file '",path,"' doesn't exist")
    sys.exit(1)


    #TODO add validation load
  if len(perc_split)<3 or validation_data:
    raise('Error deprecated way to use this function')
    try:
      validation_data = genfromtxt(validation_path, delimiter=',')
    except IOError:
      print("The file '",path,"' doesn't exist")
      sys.exit(1)
    try:
      test_data = genfromtxt(test_path, delimiter=',')
    except IOError:
      test_data=validation_data
    if label_pos=='first':
      return (data[:,1:],data[:,0]),(validation_data[:,1:],validation_data[:,0]),(test_data[:,1:],test_data[:,0])
    else:
      return (data[:,0:-1],data[:,-1]),(validation_data[:,0:-1],validation_data[:,-1]),(test_data[:,0:-1],test_data[:,-1])
  else:
    train_min=0
    train_max=train_min + int(ceil(float(data.shape[0]*perc_split[0])/100.0))
    if train_max>=data.shape[0]:
      train_max=train_max-((train_max+1)%data.shape[0])	  
    validation_min=train_max
    validation_max=validation_min+int(ceil(float(data.shape[0]*perc_split[1])/100.0))
    if validation_max>=data.shape[0]:
      validation_max=validation_max-((validation_max+1)%data.shape[0])
    test_min=validation_max
    test_max=test_min+int(ceil(float(data.shape[0]*perc_split[2])/100.0))
    if test_max>=data.shape[0]:
      test_max=test_max-(test_max%data.shape[0])
    #print(train_min,train_max,validation_min,validation_max,test_min,test_max)
    #sys.exit(0)
    if label_pos=='first':
      return (data[train_min:train_max,num_labels:],data[train_min:train_max,:num_labels]), (data[validation_min:validation_max,num_labels:],data[validation_min:validation_max,:num_labels]), (data[test_min:test_max,num_labels:],data[test_min:test_max,:num_labels])
    else:
      if perc_split[1]==0:
        data_train=data[train_min:train_max+1,0:-num_labels]
        label_train=data[train_min:train_max+1,-num_labels:]
        data_valid=data[validation_min:validation_max,0:-num_labels]
        label_valid=data[validation_min:validation_max,-num_labels:]
        
        data_test=data[test_min:test_max,0:-num_labels]
        label_test=data[test_min:test_max,-num_labels:]
      elif perc_split[2]==0:
        data_train=data[train_min:train_max,0:-num_labels]
        label_train=data[train_min:train_max,-num_labels:]
        data_valid=data[validation_min:validation_max+1,0:-num_labels]
        label_valid=data[validation_min:validation_max+1,-num_labels:]
        data_test=data[test_min:test_max,0:-num_labels]
        label_test=data[test_min:test_max,-num_labels:]
        #print(data_train.shape,label_train.shape,data_valid.shape,label_valid.shape,data_test.shape,label_test.shape)
      else:
        data_train=data[train_min:train_max,0:-num_labels]
        label_train=data[train_min:train_max,-num_labels:]
        data_valid=data[validation_min:validation_max,0:-num_labels]
        label_valid=data[validation_min:validation_max,-num_labels:]
        
        data_test=data[test_min:test_max,0:-1]
        label_test=data[test_min:test_max,-1]

    if scale_dataset:
      scaler = preprocessing.MinMaxScaler()
      if data_train.shape[0]>0:
	data_train=scaler.fit_transform(data_train)
      if data_valid.shape[0]>0:
	data_valid=scaler.transform(data_valid)
      if data_test.shape[0]>0:
	data_test=scaler.transform(data_test)

    return (data_train,label_train),(data_valid,label_valid),(data_test,label_test)
  
def load_data(root_dir='./',dataset_name='extFTIR',only_validation=False,scale_dataset=False,shuffle=-1,conv_version=False):
    """Function that handles the loading of the considered datasets
    Parameters
    ----------
    root_dir : str
        the parent directory of the dataset directory
    dataset_name : str
        the name of the dataset
    only_validation : boolean
        whether to return only a train set instead of both train and test set
    scale_dataset : boolean
        whether to scale or not the train set and the test set according to it
    shuffle : int
        shuffle the dataset according to the input value or not if is equal to -1
    conv_version : boolean
        whether to return the transformed version of the dataset achieved using the trained convolutional layer
    Returns
    -------
    X_train : numpy array
    y_train : numpy array
    X_test : numpy array
    y_test : numpy array
    """
    first_perc=68
    second_perc=32
    num_labels=1
    dataset_conv=''
    dataset_conv_test=''
    if dataset_name=='extWINE':
      first_perc=68
      second_perc=32
      dataset=root_dir + 'datasets/wine/Wine_ext.csv'
      dataset_conv=root_dir + 'datasets/wine/conv_data_best_CONVNET.mat'
      dataset_conv_test=root_dir + 'datasets/wine/conv_data_test_best_CONVNET.mat'
    elif dataset_name=='extSTRAWBERRY':
      first_perc=67.7
      second_perc=32.3
      dataset=root_dir + 'datasets/strawberry/Strawberry_ext.csv'
      dataset_conv=root_dir + 'datasets/strawberry/conv_data_best_CONVNET.mat'
      dataset_conv_test=root_dir + 'datasets/strawberry/conv_data_test_best_CONVNET.mat'
    elif dataset_name=='COFFEE':
      first_perc=67.8
      second_perc=32.2
      dataset=root_dir + 'datasets/coffee/Coffee_ext.csv'
      dataset_conv=root_dir + 'datasets/coffee/conv_data_best_CONVNET.mat'
      dataset_conv_test=root_dir + 'datasets/coffee/conv_data_test_best_CONVNET.mat'
    elif dataset_name=='OIL':
      first_perc=67.8
      second_perc=32.2
      dataset=root_dir + 'datasets/oil/Oil_ext.csv'
      dataset_conv=root_dir + 'datasets/oil/conv_data_best_CONVNET.mat'
      dataset_conv_test=root_dir + 'datasets/oil/conv_data_test_best_CONVNET.mat'
    elif dataset_name=='TABLET_NIR':
      first_perc=68
      second_perc=32
      dataset=root_dir + 'datasets/tablets/NIR/Tablet_ext.csv'
      dataset_conv=root_dir + 'datasets/tablets/NIR/conv_data_best_CONVNET.mat'
      dataset_conv_test=root_dir + 'datasets/tablets/NIR/conv_data_test_best_CONVNET.mat'
    elif dataset_name=='TABLET_Raman':
      first_perc=68
      second_perc=32
      dataset=root_dir + 'datasets/tablets/Raman/Tablet_ext.csv'
      dataset_conv=root_dir + 'datasets/tablets/Raman/conv_data_best_CONVNET.mat'
      dataset_conv_test=root_dir + 'datasets/tablets/Raman/conv_data_test_best_CONVNET.mat'
    elif dataset_name=='extFTIR':
      dataset=root_dir + 'datasets/beers/FTIR/RvsotherR_ext.csv'
      dataset_conv=root_dir + 'datasets/beers/FTIR/conv_data_best_CONVNET.mat'
      dataset_conv_test=root_dir + 'datasets/beers/FTIR/conv_data_test_best_CONVNET.mat'
      first_perc=59
      second_perc=41 
    elif dataset_name=='extNIR':
      dataset=root_dir + 'datasets/beers/NIR/RvsotherR_ext.csv'
      dataset_conv=root_dir + 'datasets/beers/NIR/conv_data_best_CONVNET.mat'
      dataset_conv_test=root_dir + 'datasets/beers/NIR/conv_data_test_best_CONVNET.mat'
      first_perc=56
      second_perc=44
    elif dataset_name=='extRaman':
      dataset=root_dir + 'datasets/beers/Raman/RvsotherR_ext.csv'
      dataset_conv=root_dir + 'datasets/beers/Raman/conv_data_best_CONVNET.mat'
      dataset_conv_test=root_dir + 'datasets/beers/Raman/conv_data_test_best_CONVNET.mat'
      first_perc=56
      second_perc=44
    else:
      dataset=root_dir + '/' + dataset_name + '.csv'
      dataset_conv=root_dir + '/conv_data_best_CONVNET.mat'
      dataset_conv_test=root_dir  + '/conv_data_test_best_CONVNET.mat'
    if only_validation:
      second_perc=0
      if not 'ext' in dataset_name:
	first_perc=100
    train_set, test_set, _ =fromCSV(path=dataset,validation_path="",perc_split=[first_perc,second_perc,0],num_labels=num_labels,seed=shuffle,scale_dataset=scale_dataset)
    X_train, y_train = train_set
    if y_train.size:
      y_train=y_train-numpy.amin(y_train)
      if num_labels==1 and not '_reg' in dataset_name:
	y_train=y_train.flatten()
    X_test, y_test = test_set
    if y_test.size:
      y_test=y_test-numpy.amin(y_test)
      if num_labels==1 and not '_reg' in dataset_name:
	y_test=y_test.flatten()
    if conv_version and dataset_conv:      
      X_train=loadmat(dataset_conv)['conv_data']
      X_train=X_train.reshape((-1,X_train.shape[1]*X_train.shape[2]))
      X_test=loadmat(dataset_conv_test)['conv_data']
      X_test=X_test.reshape((-1,X_test.shape[1]*X_test.shape[2]))
    return X_train, y_train, X_test, y_test

def testNetworkInit(net,seed):
  """Function to test the correctness of the random initialization of the weights of the network"""
  are_equals=True
  all_prev_p=[]
  for i in range(10):
    lasagne.random.set_rng(numpy.random.RandomState(seed))
    net.initialize()
    ls=get_all_layers(net.layers_['output'])
    prev_p=all_prev_p
    all_prev_p=[]
    for l in range(len(ls)):
      l1=ls[l]
      all_param_values = get_all_param_values(l1)
      if i==0:
	all_prev_p.append(all_param_values)
	continue
      for j in range(len(all_param_values)):
	p=all_param_values[j]
	are_equals=numpy.array_equal(numpy.asarray(prev_p[l][j]),numpy.asarray(p))
      all_prev_p.append(all_param_values)
    if not are_equals:
      break
  return are_equals

def getCNNParams(filename,path='./saved_vars/'):
  """ Function to get parameters of a saved neural network"""
  obj={}
  white_list=[
    'update_momentum',
    'conv1d_filter_size',
    'conv1d_num_filters',
    'conv1d_stride',
    'gaussian_sigma',
    'objective_lamda1',
    'objective_lamda2',
    'seed',
    
  ]
  # The '-' character is to indicate the name of the property in the second object
  getvalue_list=[
    'on_epoch_finished-start'
  ]
  rename_getvalue_list=[
    'learning_rate'
  ]
  net=loadVar(filename,path)
  
  for attr in dir(net):
    if not attr in white_list and not any(attr in s for s in getvalue_list):
      continue
    if any(attr in s for s in getvalue_list):
      tmp_attrs=[s for s in getvalue_list if attr in s]
      if not len(tmp_attrs)==1:
	print('Sub attributes found > 1!! Not implemented',tmp_attrs,'\n SKIPPING')
	break  
      white_list2=[tmp_attrs[0].split('-')[0]]
      sub_attrs=tmp_attrs[0].split('-')[1:]
      #print(white_list2,sub_attrs)
      tmp_val=getattr(net,white_list2[0])
      for i in tmp_val:
	for attr2 in dir(i):
	  if not attr2 in sub_attrs:
	    continue
	  if len(rename_getvalue_list)>getvalue_list.index(attr+'-'+attr2):
	    obj[rename_getvalue_list[getvalue_list.index(attr+'-'+attr2)]]=getattr(i,attr2)
	  else:
	    obj[attr+'-'+attr2]=getattr(i,attr2)
    else:
      #print(attr,getattr(net,attr))
      obj[attr]=getattr(net,attr)
  return obj




def printNet(net):
  """Simply dump to stdout a netowrk object"""
  print(net)

