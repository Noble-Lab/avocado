################################################################
# Implemented by Naozumi Hiranuma (hiranumn@uw.edu)            #
#                                                              #
# Keras-compatible implmentation of Integrated Gradients       # 
# proposed in "Axiomatic attribution for deep neuron networks" #
# (https://arxiv.org/abs/1703.01365).                          #
#                                                              #
# Keywords: Shapley values, interpretable machine learning     #
################################################################
from __future__ import division
import numpy as np
import numpy
from time import sleep
import sys
import keras.backend as K
import time

from keras.models import Model, Sequential

'''
Integrated gradients approximates Shapley values by integrating partial
gradients with respect to input features from reference input to the
actual input. The following class implements this concept.
'''
class integrated_gradients:
    # model: Keras model that you wish to explain.
    # outchannels: In case the model are multi tasking, you can specify which channels you want.
    def __init__(self, model, outchannels=[], verbose=1):
    
        # Bacnend: either tensorflow or theano)
        self.backend = K.backend()
        
        #load model supports keras.Model and keras.Sequential
        if isinstance(model, Sequential):
            self.model = model.model
        elif isinstance(model, Model):
            self.model = model
        else:
            print("Invalid input model")
            return -1
        
        #load input tensors
        self.input_tensors = []
        for i in self.model.inputs:
            self.input_tensors.append(i)
        # The learning phase flag is a bool tensor (0 = test, 1 = train)
        # to be passed as input to any Keras function that uses 
        # a different behavior at train time and test time.
        self.input_tensors.append(K.learning_phase())
        
        #If outputchanel is specified, use it.
        #Otherwise evalueate all outputs.
        self.outchannels = outchannels
        if len(self.outchannels) == 0: 
            if verbose: print("Evaluated output channel (0-based index): All")
            if K.backend() == "tensorflow":
                self.outchannels = range(self.model.output.shape[1]._value)
            elif K.backend() == "theano":
                self.outchannels = range(model.output._keras_shape[1])
        else:
            if verbose: 
                print("Evaluated output channels (0-based index):")
                print(','.join([str(i) for i in self.outchannels]))
                
        #Build gradient functions for desired output channels.
        self.get_gradients = {}
        if verbose: print("Building gradient functions")
        
            # Evaluate over all channels.
        for c in self.outchannels:
            # Get tensor that calcuates gradient
            if K.backend() == "tensorflow":
                gradients = self.model.optimizer.get_gradients(self.model.output[:, c], self.model.input)
            if K.backend() == "theano":
                gradients = self.model.optimizer.get_gradients(self.model.output[:, c].sum(), self.model.input)
                
            # Build computational graph that calculates the tensfor given inputs
            self.get_gradients[c] = K.function(inputs=self.input_tensors, outputs=gradients)
            
            # This takes a lot of time for a big model with many tasks.
            # So lets pring the progress.
            if verbose:
                sys.stdout.write('\r')
                sys.stdout.write("Progress: "+str(int((c+1)*1.0/len(self.outchannels)*1000)*1.0/10)+"%")
                sys.stdout.flush()
        # Done
        if verbose: print("\nDone.")
            
                
    '''
    Input: sample to explain, channel to explain
    Optional inputs:
        - reference: reference values (defaulted to 0s).
        - steps: # steps from reference values to the actual sample.
    Output: list of numpy arrays to integrated over.
    '''
    def explain(self, X, reference=None, num_steps=50):
        
        # Each element for each input stream.
        #num_steps = float(num_steps)
        reference = reference or numpy.zeros(X.shape[1])

        samples = numpy.zeros((X.shape[0] * num_steps, X.shape[1]))
        step_sizes = numpy.empty((X.shape[0], X.shape[1]))

        for i, x in enumerate(X):
            step_sizes[i] = reference + (x - reference) * (1. / num_steps)
            for j in range(num_steps):
                samples[i*num_steps + j] = reference + (x - reference) * (1. * j / num_steps)

        
        gradients = self.get_gradients[0]([samples, 0])
        
        explanations = []
        for i in range(X.shape[0]):
            start, end = i * num_steps, (i+1) * num_steps

            explanation = np.sum(gradients[start:end], axis=0) * step_sizes[i]
            explanations.append(explanation)

        explanations = numpy.array(explanations)
        return explanations
