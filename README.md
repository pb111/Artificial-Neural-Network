# Artificial Neural Network Project


===============================================================================

## Table of Contents


===============================================================================

## 1. Introduction to Deep Learning

-	**Deep Learning** is a class of machine learning algorithms which depends on the structure and function of the brain called **Artificial Neural Network (ANN)**.
-	The term **Deep Learning** was introduced to the machine learning community by Rina Dechter in 1986 and to artificial neural networks by Igor Aizenberg and colleagues in 2000.
-	It uses multiple layers to progressively extract higher level features from raw input.
-	The term `deep` in deep learning refers to the number of layers through which the data is transformed.
-	In deep learning, each level learns to transform its input data into a slightly more abstract and composite representation.
-	Deep learning systems have a substantial `credit assignment path` (CAP) depth.
-	The CAP is the chain of transformations from input to output. CAPs describe potentially casual connections between input and output. 
-	Deep learning architectures are often constructed with a greedy layer-by-layer method. 
-	Deep learning helps to disentangle these abstractions and pick out which features improve performance.
-	Deep Learning systems are based on Neural Networks.
-	So, I will first discuss Neural Networks.

===============================================================================

## 2. Introduction to Neural Network 

-	Neural networks are a set of algorithms which are based on human brain.
-	These are artificial systems that were inspired by biological neural networks.
-	These are designed to recognize patterns in the data.
-	They interpret sensory data through a kind of machine perception, labelling or clustering raw input.
-	The patterns they recognize are numerical, contained in vectors, into which all real-world data (images, sound, text or time-series) must be translated.
-	Neural networks are based on computational models for threshold logic.
-	Threshold logic is a combination of algorithms and mathematics.
-	Components of a typical neural network involve neurons, connections, weights, biases, propagation function, and a learning rule. 
-	Neurons will receive an input pj(t) from predecessor neurons that have an activation aj(t), threshold , an activation function f, and an output function fout . 
-	Connections consist of connections, weights and biases which dictates how neuron i transfers output to neuron j. 
-	Propagation computes the input and outputs the output and sums the predecessor neurons function with the weight. The learning rule modifies the weights and thresholds of the variables in the network.

===============================================================================

## 3. Types of Neural Network

-	There are seven types of Neural Networks used in practice. These are discussed below-

1.	The first type is a multilayer perceptron. It is called Artificial Neural Network. It has three or more layers and uses a non-linear activation function.
2.	The second is the Convolutional Neural Network that uses a variation of the multilayer perceptrons.
3.	The third is the Recursive Neural Network that uses weights to make structured predictions.
4.	The fourth is the Recurrent Neural Network that makes connections between the neurons in a directed cycle. The long short-term memory neural network uses the recurrent neural network architecture and does not use activation function.
5.	The final two are sequence to sequence modules which uses two recurrent networks and shallow neural networks which produces a vector space from an amount of text. 


===============================================================================

## 4. Introduction to Artificial Neural Network (ANN)

-	An Artificial Neural Network (ANN) is an imitation of the human brain. It functions just like human brain.
-	ANNs are software implementations of the neuronal structure of our brains.
-	So, first I will discuss functionality of the human brain.

### 4.1 The Human Brain

-	The human brain is very capable. It has the ability to learn new things, adapt and work accordingly to changing environment. 
-	Our brain can analyze incomplete and unclear information and take decisions.
-	It has the ability to perform tasks such as pattern recognition, perception and control much faster than any computer.
-	The basic building block of the brain is **biological neuron**.


### 4.2 Biological Neurons

-	The brain contains about 1010 (100 billion) basic units called **neurons** or biological neurons.
-	Each neuron can connect to about 104 other neurons.
-	A biological neuron is made up of cell body (soma), axon and dendrite.
-	The nucleus of the neuron (the cell body) is called **soma** which processes the input.
-	It contains long irregularly shaped filaments attached to the soma which act as input channels. These are called **dendrites**.
-	The neuron receives signals from other neurons through dendrites.
-	Another type of link attached to the soma act as output channels. They are called **axon**.
-	 Output of the axon is **voltage pulse (spike)** that lasts for a millisecond.
-	Axon carries the signal from neuron to other neurons.
-	Axon terminates in a specialized contact called the **synaptic junction**- the electrochemical contact between neurons.
-	Connection between dendrites of two neurons or neuron to muscle cells is called **synapse**.
-	The size of synapses are believed to be linked with learning.
-	When the strength of the signal exceeds a certain threshold, the neuron triggers its own signal to be passed on to the next neuron via the axon using synapses.
-	The signal sent to other neurons through synapses trigger them, and this process continues.
-	A huge number of such neurons work simultaneously.
-	The brain has the capacity to store large amounts of data.

### 4.3 Artificial Neurons

-	An artificial neuron tries to replicate the structure and behaviour of biological neurons.
-	It consists of inputs (dendrites) and one output (synapse via axon).
-	The neuron has a function that determines the activation of the neuron.
-	The diagram below demonstrates the artificial neuron model and activation function.

### Artiicial Neuron Model
![Artificial Neuron Model](https://github.com/pb111/Artificial-Neural-Network/blob/master/Images/artificial%20neurons%20model.png)


### 4.4 Working principle of Artificial Neurons

-	The diagram below demonstrates working of artificial neural network model.

![Working Principle of Artificial Neurons](https://github.com/pb111/Artificial-Neural-Network/blob/master/Images/Working%20principle%20of%20artificial%20neural%20network%20model.jpg)

-	At the entrance of artificial neuron the inputs are weighted. It means that every input value is multiplied with individual weight.
-	So, the information comes into the body of an artificial neuron via inputs that are weighted (each input can be individually multiplied with a weight)
-	In the middle section of artificial neuron is sum function that sums all weighted inputs and bias.
-	So, the body of an artificial neuron then sums the weighted inputs, bias and then processes the sum with a transfer function.
-	At the exit of artificial neuron, the sum of previously weighted inputs and bias passed through activation function which is also called transfer function.

### 4.5 Transfer function

-	We can see from the artificial neuron model that the major unknown variable in the model is the **transfer function**.
-	Transfer function defines the properties of artificial neuron and can take any mathematical form.
-	 We choose it on the basis of problem that our ANN model needs to solve.
-	In most cases, we choose it from the following set of functions – 
`Step function`, `Linear function` and `Non-linear (Sigmoid) function`.

**Step function**

-	Step function is a binary function that has only two possible output values (e.g. 0 and 1).
-	That means if input value meets specific threshold, the output results in one value and if specific threshold does not meet, the output results in another value.
-	When this type of transfer function is used in artificial neuron, it is called **perceptron**.

**Perceptron**

-	Perceptron is used for solving classification problems.
-	It can be most commonly found in the last layer of artificial neural network.

**Linear transfer function**

-	In case of linear transfer function, artificial neuron is doing simple linear transformation over the sum of weighted inputs and bias.
-	Such an artificial neuron is in contrast to perceptron, which is most commonly used in the input layer of artificial neural networks.

**Non-linear function**

-	In case of non-linear functions, sigmoid function is the most commonly used.
-	Sigmoid function has easily calculated derivative.
-	It is important when calculating weight updates in ANN.


### 4.6 Artificial Neural Network (ANN)

-	When we combine two or more artificial neurons, we get an Artificial Neural Network (ANN).
-	An Artificial Neural Network (ANN) is a mathematical model that tries to simulate the structure and functionalities of biological neural networks.
- The diagram below represent the similarities between biological neuron and artificial neural network.

![Biological Neuron Vs Artificial Neural Network](https://github.com/pb111/Artificial-Neural-Network/blob/master/Images/biological%20neuron%20vs%20ANN.png)


-	Artificial Neural Networks are capable of solving complex real-life problems by processing information in their basic building blocks.
-	Basic building block of every artificial neural network is artificial neuron.
-	The nodes created in the ANN are supposedly programmed to behave like actual neurons, and hence they are artificial neurons.
-	Such a model has three simple sets of rules – multiplication, summation and activation.

===============================================================================

===============================================================================

===============================================================================

===============================================================================

===============================================================================

===============================================================================
