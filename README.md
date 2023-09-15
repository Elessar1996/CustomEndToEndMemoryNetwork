# End-To-End Memory Network
An Implementation of End-To-End memory network

This repo is an implemetation of **End-To-End memory network** which is a Deep Learning model for doing Natural Language Processing tasks. The paper proposing this model is:

> Sukhbaatar, S., Weston, J. and Fergus, R., 2015. End-to-end memory networks. Advances in neural information processing systems, 28.

## Data 

The data which is used for training this model is bAbI dataset which is introduced in the paper below: 

> Weston, J., Bordes, A., Chopra, S., Rush, A.M., Van Merriënboer, B., Joulin, A. and Mikolov, T., 2015. Towards ai-complete question answering: A set of prerequisite toy tasks. arXiv preprint arXiv:1502.05698.

The data is then preprocessed and encoded using the method `Bag Of The Words`.

Then, it flows through the model. 

## Model 

This model is basically a `simple Neural Network` which has an explicit memory module. I have implemented this for the sake of my M.S. thesis.    
