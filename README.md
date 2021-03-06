# Artificial Neural Network Project

**Artificial Neural Network (ANN)** imitates the functioning of human brain. I discuss ANN fundamentals, its structure and functioning, 
ANN architectures, learning process, back-propagation algorithm, its advantages and disadvantages and ANN applications.

===============================================================================

## Table of Contents

1.	Introduction to Deep Learning

2.	Introduction to Neural Network

3.	Types of Neural Network

4.	Introduction to Artificial Neural Network

      4.1	The human brain 
      
      4.2	Biological neurons
      
      4.3	Artificial neurons
      
      4.4	Working principle of artificial neurons
      
      4.5   Perceptron
      
      4.6	Transfer function
      
      4.7	Artificial Neural Network
      
5.	ANN architectures

      5.1 Feed-forward ANNs
      
      5.2 Recurrent ANNs
      
      5.3 Hopfield ANNs
      
      5.4 Elman and Jordan ANNs
      
      5.5 Long Short Term Memory
      
      5.6 Bi-directional ANNs
      
      5.7 Self-Organizing Maps
      
      5.8 Stochastic ANNs
      
      5.9 Physical ANNS
      
6.	Learning process in neural networks

      6.1 Supervised learning
      
      6.2 Unsupervised learning
      
      6.3 Reinforced learning
      
7.  The back-propagation algorithm

8. Advantages and disadvantages of ANN

9. ANN Applications

10. References

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

### Artificial Neuron Model and Activation Function
![Artificial Neuron Model](https://github.com/pb111/Artificial-Neural-Network/blob/master/Images/artificial%20neurons%20model.png)


### 4.4 Working principle of Artificial Neurons

-	The diagram below demonstrates working of artificial neural network model.

### Working Principle of Artificial Neurons
![Working Principle of Artificial Neurons](https://github.com/pb111/Artificial-Neural-Network/blob/master/Images/Working%20principle%20of%20artificial%20neural%20network%20model.jpg)

-	At the entrance of artificial neuron the inputs are weighted. It means that every input value is multiplied with individual weight.
-	So, the information comes into the body of an artificial neuron via inputs that are weighted (each input can be individually multiplied with a weight)
-	In the middle section of artificial neuron is sum function that sums all weighted inputs and bias.
-	So, the body of an artificial neuron then sums the weighted inputs, bias and then processes the sum with a transfer function.
-	At the exit of artificial neuron, the sum of previously weighted inputs and bias passed through activation function which is also called transfer function.


### 4.5 Perceptron

-	A single layer of artificial neurons is called **Perceptron**.
-	A multi-layer of artificial neurons or multi-layer perceptron is called **Neural Networks**.
-	Perceptron is a linear binary classifier.
-	It is used in supervised learning.
-	It helps to classify the given input data. 
-	The perceptron consists of 4 parts.
      - Input values or One input layer
      - Weights and Bias
      - Net sum
      - Activation Function

-	The perceptron can be represented diagrammatically as follows:-

### Perceptron
![Perceptron](https://github.com/pb111/Artificial-Neural-Network/blob/master/Images/Perceptron.jpg)


-	The perceptron works as follows:-

1.	All the inputs x are multiplied with their weights w.

![inputs are multiplied by weights](https://github.com/pb111/Artificial-Neural-Network/blob/master/Images/inputs%20are%20multiplied%20by%20weights.png)

2.	Add all the multiplied values and call them Weighted Sum.

![weighted sum](https://github.com/pb111/Artificial-Neural-Network/blob/master/Images/Weighted%20sum.gif)

3.	Apply that weighted sum to the correct Activation Function.

![apply the weights to activation function](https://github.com/pb111/Artificial-Neural-Network/blob/master/Images/Activation%20function.png)


### 4.6 Transfer function

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


### 4.7 Artificial Neural Network (ANN)

-	When we combine two or more artificial neurons, we get an Artificial Neural Network (ANN).
-	An Artificial Neural Network (ANN) is a mathematical model that tries to simulate the structure and functionalities of biological neural networks.
- The diagram below represent the similarities between biological neuron and artificial neural network.

### Biological Neurons vs Artificial Neural Network]
![Biological Neuron Vs Artificial Neural Network](https://github.com/pb111/Artificial-Neural-Network/blob/master/Images/biological%20neuron%20vs%20ANN.png)


-	Artificial Neural Networks are capable of solving complex real-life problems by processing information in their basic building blocks.
-	Basic building block of every artificial neural network is artificial neuron.
-	The nodes created in the ANN are supposedly programmed to behave like actual neurons, and hence they are artificial neurons.
-	Such a model has three simple sets of rules – multiplication, summation and activation.

===============================================================================

## 5. ANN architectures
-	The way that individual artificial neurons are interconnected is called **topology**, **architecture** or **graph of an ANN**.
-	The interconnection can be done in numerous ways results in numerous possible topologies (architectures) that are divided into two basic classes.
-	The figure below shows these two topologies.
-	The left side of the figure represent **simple feed-forward topology** (acyclic graph) where information flows from inputs to outputs in only one direction.
-	The right side of the figure represent **simple recurrent topology** (semi cyclic graph) where information flows not only in one direction from input to output but also in opposite direction.

### Feed-forward (FNN) and recurrent (RNN) topology of an Artificial Neural Network
![Feed-forward (FNN) and recurrent (RNN) topology of an Artificial Neural Network](https://github.com/pb111/Artificial-Neural-Network/blob/master/Images/FNN%20and%20RNN%20topology.jpg)
-	
-	The ANN architecture comprises of :
a.	Input layer: Receives the input values.
b.	Hidden layer(s): A set of neurons between input and output layers. There can be single or multiple layers.
c.	Output layer: Usually, it has one neuron and its output ranges between 0 and 1. But, multiple outputs can also be present.
-	The ANN architecture can be represented diagrammatically as follows:-

### ANN Architecture
![ANN Architecture](https://github.com/pb111/Artificial-Neural-Network/blob/master/Images/ANN%20architecture.jpg)


-	Neural Networks are known to be universal function approximators.
-	Various ANN architectures are available to approximate any non-linear function.
-	Different architectures allow to generate functions with different complexity and power.
-	These different types of architectures are discussed below.

### 5.1 Feed-forward ANNs

-	ANN with feed-forward topology is called **Feed-Forward ANN**.
-	It has only one condition – information must flow from input to output in forward direction.
-	The simplest feed-forward ANN is a single perceptron that is only capable of learning linear separable problems.
-	There are no limitations on number of layers, type of transfer functions or number of connections between individual artificial neurons. 
-	**Input layer** - Number of neurons in this layer corresponds to the number of inputs to the neural network. This layer consists of passive nodes. These nodes do not take part in the actual signal modification, but only transmits the signal to the following layer.
-	**Hidden layer** - This layer has arbitrary number of layers with arbitrary number of neurons. The nodes in this layer take part in the signal modification. Hence, they are active. 
-	**Output layer** - The number of neurons in the output layer corresponds to the number of output values of the neural network. 
The nodes in this layer are active ones.
-	**Feed-forward Neural Network** (FFNN) can have more than one hidden layer.
-	However, it has been proved that FFNNs with one hidden layer has enough to approximate any continuous function. 
- Feed-forward Neural Network can be represented with the following diagram.

### Feed-forward Neural Network
![Feed-forward Neural Network](https://github.com/pb111/Artificial-Neural-Network/blob/master/Images/Feed-forward%20neural%20network.jpg) 


### 5.2 Recurrent ANNs

-	ANN with recurrent topology is called **Recurrent Artificial Neural Network**.
-	It is similar to feed-forward neural network with no limitations on back loops.
-	In these cases information is no longer transmitted only in forward direction but it is also transmitted backwards.
-	This creates an internal state of the network which allows it to exhibit dynamic temporal behaviour.
-	Recurrent ANNs can use their internal memory to process any sequence of inputs. 
-	The most basic topology of recurrent artificial neural network is fully recurrent artificial network where every basic building block (artificial neuron) is directly connected to every other basic building block in all direction. 
-	Other recurrent artificial neural networks such as Hopfield, Elman, Jordan, bi-directional and other networks are just special cases of recurrent artificial neural networks.
- Recurrent neural network can be represented diagrammatically as follows:-

### Recurrent Neural Network
![Recurrent Neural Network](https://github.com/pb111/Artificial-Neural-Network/blob/master/Images/Recurrent%20Neural%20Network.jpg)


### 5.3 Hopfield ANNs

-	A Hopfield artificial neural network is a type of recurrent artificial neural network that is
      used to store one or more stable target vectors. 
-	These stable vectors can be viewed as memories that the network recalls when provided with similar vectors that act as a cue to the network memory. 
-	These binary units only take two different values for their states that are determined by whether or not the units' input exceeds their threshold. 
-	Binary units can take either values of 1 or -1, or values of 1 or 0. Consequently there are two possible definitions for binary unit activation.

### 5.4 Elman and Jordan ANNs

-	Elman network also referred as **Simple Recurrent Network** is special case of recurrent artificial neural networks. 
-	It differs from conventional two-layer networks in that the first layer has a recurrent connection. 
-	It is a simple three-layer artificial neural network that has back-loop from hidden layer to input layer trough so called context unit. 
-	This type of artificial neural network has memory that allow it to both detect and generate time-varying patterns. 
-	The Elman artificial neural network has typically sigmoid artificial neurons in its hidden
      layer, and linear artificial neurons in its output layer.  
-	This combination of artificial neurons transfer functions can approximate any function with arbitrary accuracy if only there is enough artificial neurons in hidden layer. 
-	Being able to store information Elman artificial neural network is capable of generating temporal patterns as well as spatial patterns and responding on them. 
-	The only difference is that context units are fed from the output layer instead of the hidden layer.

### 5.5 Long Short Term Memory

-	**Long Short Term Memory** is one of the recurrent artificial neural networks topologies. 
-	In contrast with basic recurrent artificial neural networks it can learn from its experience to process, classify and predict time series with very long time lags of unknown size between important events. 
-	This makes Long Short Term Memory to outperform other recurrent artificial neural networks, Hidden Markov Models and other sequence learning methods. 
-	Long Short Term Memory artificial neural network is build from Long Short Term Memory blocks that are capable of remembering value for any length of time. 
-	This is achieved with gates that determine when the input is significant enough remembering it, when continue to remembering or forgetting it, and when to output the value.

### 5.6 Bi-directional Artificial Neural Networks (Bi-ANN)

-	**Bi-directional ANNs** are designed to predict complex time series. 
-	They consist of two individual interconnected artificial neural (sub) networks that performs direct and inverse (bidirectional) transformation. 
-	Interconnection of artificial neural sub networks is done through two dynamic artificial neurons that are capable of remembering their internal states. 
-	This type of interconnection between future and past values of the processed signals increase time series prediction capabilities. 
-	As such these artificial neural networks not only predict future values of input data but also past values. 
-	That brings need for two phase learning; in first phase we teach one artificial neural sub network for predicting future and in the second phase we teach a second artificial neural sub network for predicting past.


### 5.7 Self-Organizing Map (SOM)

-	**Self-organizing map (SOM)** is an artificial neural network that is related to feed-forward networks but it needs to be told that this type of architecture is fundamentally different in arrangement of neurons and motivation. 
-	Common arrangement of neurons is in a hexagonal or rectangular grid. 
-	Self-organizing map is different in comparison to other artificial neural networks in the sense that they use a neighbourhood function to preserve the topological properties of the input space. 
-	They uses unsupervised learning paradigm to produce a low-dimensional, discrete representation of the input space of the training samples, called a map what makes them especially useful for visualizing low-dimensional views of high-dimensional data. 
-	Such networks can learn to detect regularities and correlations in their input and adapt their future responses to that input accordingly.

### 5.8 Stochastic Artificial Neural Network

-	Stochastic artificial neural networks are a type of an artificial intelligence tool. 
-	They are built by introducing random variations into the network, either by giving the network's neurons stochastic transfer functions or by giving them stochastic weights. 
-	This makes them useful tools for optimization problems, since the random fluctuations help it escape from local minima. 
-	Stochastic neural networks that are built by using stochastic transfer functions are often called **Boltzmann machine**.

### 5.9 Physical Artificial Neural Network

-	Most of the artificial neural networks today are software-based but that does not exclude the possibility to create them with physical elements which base on adjustable electrical current resistance materials. 
-	Although these artificial neural networks were commercialized they did not last for long due to their incapability for scalability. 
-	After this attempt several others followed such as attempt to create physical artificial neural network based on nanotechnology or phase change material.

===============================================================================

## 6. Learning Process in Neural Networks
-	The **learning process** in ANN can be viewed as the problem of updating network architecture and connection weights so that a network can perform a specific task.
-	**Learning** implies that a processing unit is capable of changing its input output behaviour as a result of changes in the environment.
-	The network learn the connection weights from available training patterns.
-	Performance is improved over time by iteratively updating the weights in the network.
-	The learning methods in neural networks are classified into three basic types :–
      - Supervised learning
      - Unsupervised learning
      - Reinforced learning
      
-	These three types are classified based on :-
      - presence or absence of teacher
      - the information provided for the system to learn.
      
-	These learning methods are further categorized, based on the rules used as follows:-
      - Hebbian
      - Gradient descent
      - Competitive
      - Stochastic learning

### 6.1 Supervised Learning
-	A teacher is present during learning process and presents expected output.
-	Every input pattern is used to train the network.
-	Learning process is based on comparison, between network’s computed output and the correct expected output, generating `error`.
-	The `error` generated is used to change network parameters that result in improved performance.
-	The **back-propagation algorithm** belongs to this category.

### 6.2 Unsupervised Learning
-	No teacher is present.
-	So, expected or desired output is not presented to the network.
-	The system learns of its own by discovering and adapting to the structural features in the input pattern.
-	The **Kohonen algorithm** belongs to this category.

### 6.3 Reinforced Learning
-	A teacher is present but does not present the expected or desired output but only indicated if the computed output is correct or incorrect.
-	The information provided helps the network in the learning process.
-	A reward is given for correct answer computed and a penalty for a wrong answer.

===============================================================================

## 7. The back-propagation algorithm

-	The back-propagation algorithm is used in layered feed-forward ANNs. 
-	This means that the artificial neurons are organized in layers, and send their signals “forward”, and then the errors are propagated backwards. 
-	The network receives inputs by neurons in the input layer, and the output of the network is given by the neurons on an output layer. 
-	There may be one or more intermediate hidden layers. 
-	The back-propagation algorithm uses supervised learning, which means that we provide the algorithm with examples of the inputs and outputs we want the network to compute, and then the error (difference between actual and expected results) is calculated. 
-	The idea of the back-propagation algorithm is to reduce this error, until the ANN learns the training data. 
-	The training begins with random weights, and the goal is to adjust them so that the       error will be minimal.

===============================================================================

## 8. Advantages and disadvantages of ANN

-	**Advantages of ANN are as follows:-**

1.	A neural network can perform tasks that a linear program cannot. They can work in case of incomplete information.
2.	A neural network learns and does not need to be reprogrammed.
3.	They do not require knowledge of algorithm to solve the problem (automatic learning).
4.	They can process information in a highly parallel way. When an element of a neural network fails, it can continue without any problem by their parallel nature.
5.	They can generalize to unknown cases. It can be implemented in any application.
6.	They are resistant to partial damage. It can be implemented without any problem.


-	**Disadvantages of ANN are given below:-**

1.	The neural network needs training to operate,
2.	It requires high processing time for large neural networks.
3.	The architecture of a neural network is different from the architecture of microprocessors therefore needs to be emulated.

===============================================================================

## 9. ANN Applications

-	Applications of ANN include:-

1.	**Handwriting recognition** - we can use neural networks to recognize handwritten characters.
2.	**Speech recognition** - speech occupies a prominent role in human-human interaction. Neural networks can be used for speech recognition.
3.	**Character recognition** - neural networks have been used for character recognition.
4.	**Classification** – the aim is to predict the class of an input vector.
5.	**Pattern matching** – the task is to produce the best matching pattern associated with a given input vector.
6.	**Image compression** - vast amounts of information is received and processed at once by neural networks.
7.	**Stock exchange prediction** - neural networks can be used to predict stock prices.
8.	**Optimization** – the task is to find the optimal values of parameters in an optimization problem.
9.	**Function approximation** - the aim is to learn the functional relationships between input and desired output vectors.
10.	**Data mining** - the aim is to discover hidden patterns in the data.

===============================================================================

## 10. References

The ideas and concepts in this projects have been taken from the following books and websites:-

1.	Python Deep Learning Cookbook by Indra den Bakker

2.	Hands on Machine Learning with Scikit-Learn and Tensorflow by Aurélién Géron

3.	https://en.wikipedia.org/wiki/Deep_learning

4.	https://en.wikipedia.org/wiki/Artificial_neural_network

5.	https://www.geeksforgeeks.org/neural-networks-a-beginners-guide/

6.	https://www.geeksforgeeks.org/introduction-to-artificial-neutral-networks/

7.	https://www.geeksforgeeks.org/introduction-artificial-neural-network-set-2/

8.	https://www.geeksforgeeks.org/introduction-ann-artificial-neural-networks-set-3-hybrid-systems/

9.	https://www.geeksforgeeks.org/introduction-to-ann-set-4-network-architectures/

10.	https://skymind.ai/wiki/neural-network

