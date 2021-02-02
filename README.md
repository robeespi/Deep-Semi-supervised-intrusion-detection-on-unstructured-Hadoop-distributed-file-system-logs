# 1. Deep-Semi-supervised-intrusion-detection-on-unstructured-system-logs

Deep learning has been applied in cybersecurity domain, however limited work has been done to detect intrusion on unstructured system logs. Due to the prohibitive cost of large-scale labeled anomaly data, the solution is a semi-supervised approach by labelling a few suspicious logs. 

This piece of work introduces a semi-supervised Long Short Term memory approach to overcome these challenges. LSTM neural network can learn an intermediate representation of sequential data from unstructured system logs and leverage a few labeled examples to detect anomalies. Extensive experiments and results on HDFS logs dataset shows that the proposed model reach state of art by leveraging only 12% of the anomalies from the anomalies training set as labeled example. 

# 2. Dataset

HDFS log data set.
It is generated through running Hadoop-based map-reduce jobs on more than 200 Amazon’s EC2 nodes, and labeled by Hadoop domain experts. Among 11, 197, 954 log entries being collected, about 2.9% are abnormal, including events such as “write exceptionç2.

# 3. Solution Components

This repo contains the code for the last components of this solution

# 3.1 Log Parser
![alt text](https://github.com/robeespi/Deep-intrusion-detection-on-unstructured-system-logs/blob/main/Log%20Parser%20Example.jpeg)

Log parser is done by parsing unstructured logs which are free-text log entries into a structured representation and then extract a “log key” (also known as “message type”) from each log entry. A log key of a log entry “e” refers to the string constant k from the print statement in the source code which print “e” during the execution of that code. For example, the log key k for log entry e =“Took 10 seconds to build instance.” is k =Took * seconds to build instance., which is the string constant from the print statement printf(”Took %f seconds to build instance.”, t).

Note that the parameter(s) are abstracted as asterisk(s) in a log key. These metric values reﬂect the underlying system state and performance status. Values of certain parameters may serve as identiﬁers for a particular execution sequence, such as block_id in a HDFS log. These identiﬁers can group log entries together as singlethread sequential sequences. The thread sequences identified are grouped by session and in this way, we can obtain a list of array. Each array corresponds to a session, which is a sequence of log keys. Each session group is can be seen as a life cycle of one block or a VM instance respectively. In the figure 3 it is show a diagram to explain the previous workflow

This log parser method is called "Spell", you can fin more details on this paper below and repo. Additionally, you can find other methods to parse the logs there.
https://www.cs.utah.edu/~lifeifei/papers/spell.pdf
https://github.com/logpai/logparser


# 3.2 Feature Representation

The figures describes how unstructured system logs are transformed into a set of log key messages as a session. 

Sessions from the HDFS dataset are found on /hdfs data folder. Before to feed the neural network is required to process this representation according to the windows size of the LSTM network. This pre-processing step is not included in this repo.


![alt text](https://github.com/robeespi/Deep-intrusion-detection-on-unstructured-system-logs/blob/main/scheme.png)

# 3.3 Long Short Term Memory NN

Long Short-Term Memory (LSTM) networks are a type of recurrent neural network capable of learning over sequences, LSTM are particularly effective for a sequence classification problems. Each LSTM block remembers a state for its input as a vector of a ﬁxed dimension. The state of an LSTM block from the previous time step is also fed into its next input, together with its (external) data input, to compute a new state and output. this is how historical information is passed to and maintained in a single LSTM block. A series of LSTM blocks form an unrolled version of the recurrent model in one layer as shown in the following figure.

![alt text](https://github.com/robeespi/Deep-intrusion-detection-on-unstructured-system-logs/blob/main/lstm.jpeg)

Each cell maintains a hidden vector (previous output) and a cell state vector (previous cell state) . Both are passed to the next block to initialize its state. In our case, we use one LSTM block for each log key from an input sequence. Hence, a single layer consists of unrolled LSTM blocks. In our case we use 2 stacked layers, within a single LSTM block, the input and the previous output are used to decide how much of the previous cell state to retain in state, how to use the current input and the previous output to inﬂuence the state, and how to construct the output, it is accomplished using a set of gating functions to determine state dynamics by controlling the amount of information to keep from input and previous output, and the information ﬂow going to the next step. Each gating function is parameterized by a set of weights to be learned. The expressive capacity of an LSTM block is determined by the number of memory units, in our case 64 memory units for both layers. Thus, the training step entails ﬁnding proper assignments to the weights so that the ﬁnal output of the sequence of LSTMs produces the desired label (output) that reflect an anomaly score. During the training process, each input/output pair incrementally updates these weights, through binary cross entropy loss minimization via gradient descent.

# 4. Results

Model | #1 | #2 | #3 | #4 | #5 | #6 | #7 | #8 | #9 | #10 | #11
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |---
Seconds | 301 | 283 | 290 | 286 | 289 | 285 | 287 | 287 | 272 | 276 | 26
Seconds | 301 | 283 | 290 | 286 | 289 | 285 | 287 | 287 | 272 | 276 | 26


