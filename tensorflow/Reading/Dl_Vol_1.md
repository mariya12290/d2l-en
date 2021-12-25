- ML  models can be trained to form decisions based on past experience



Applications of ML:

* spam detection systme in Gamil based on keywords such as invoice, payment, free, 





ML :

Field in CS, that gives computers ability to learn without being explicitly programmed. 



### Chapter 1 :

Read Unsupervised learning, Generators, RL, DL 

### Chapter 2 :

Read full

### Chapter 3 :

Read full

### Chapter 4 :

Read full

### Chapter 5 : 

Read full

### Chapter 6 :

Read full

### Chapter 7 : classifier 



### Chapter 8 :

Read full



### Chapter 9:

DIfferent methods to avoid overfitting?

* Regularization
* Batch normalization
* dropout

Why we use regularization?

* To avoid overfitting, or to reduce generalization error
* 

When under fitting can occur?

* when there are less data for training, so model did not learn enough
* Reduce under fitting using more data

Under fitting can be okay, not overfitting



High variance/ low bias - overfitting

Low variance / High bias - underfitting

#### read bayes rule page number - 363



### Chapter 10 :

done



### Chapter 11:

Read full 

### Chapter 12 :

Read full 

### Chapter 13 :

Read full

### Chapter 14 :

Read full

### Chapter 15 :

Do not read it 

### Chapter 16 :

Read synchronous and asynchronous

### Chapter 17 :

##### Activation functions:

Sigmoid/logistic regression  - between 0 and 1 (squash

Problem : Vanish gradients , which means minimal weights updates in the last layer due to small gradients during back propagation

This makes network to take more time to learn the features

Tanh  - between 1 and -1 , vanish gradients still exists

ReLU : simple and fast to introduce the non linearity  into artificial neurons

 output 0 for  negative values, effective against vanish gradients

However, ReLU units are vulnerable to die during training. For instance, a large gradient
passing through a ReLU neuron might cause the weights to be updated where the neuron
will never activate again on any data point. If this occurs, the gradient flowing through
that neuron will be zero from that point on. This is less problematic when the learning
rate is set properly

Dying neuron - Probelm in ReLU

Leaky ReLU - scaled negative value for input for below 0, 

Parametric Leaky ReLU - we can choose how much we have to scale down the negative value 

Softmax :  not a activation function, it turns the output comes from the neurons of last layers into class probabilities

### Chapter 18 :

Purpose of regularization - to maintain the values of weights in certain range for an example between [-1,1], avoid overfitting

Read from 18.2.1

### Chapter 19 :

Read Now

### Chapter 20 :

Read Now

### Chapter 21 :

Read Now

### Chapter 22 :

Read full 

### Chapter 23 :

Read full

### Chapter 24 :

Read full 

### Chapter 25 :

Read Now

### Chapter 26 :

Read full 

### Chapter 27 :

Read full

### Chapter 28 :

Read full













| # Probability score map, scale = [None, 200/100, 176/120, 2] |
| ------------------------------------------------------------ |
|                                                              |

| p_map_p = ConvMD(2, 768, 2, 1, (1, 1), (0, 0), temp_conv, |
| --------------------------------------------------------- |
|                                                           |

| training=self.training, activation=False, bn=False, name='convp20')  #make output channel 4 |
| ------------------------------------------------------------ |
|                                                              |

| # Regression(residual) map, scale = [None, 200/100, 176/120, 14] |
| ------------------------------------------------------------ |
|                                                              |

| r_map_p = ConvMD(2, 768, 14, 1, (1, 1), (0, 0), |
| ----------------------------------------------- |
|                                                 |

| temp_conv, training=self.training, activation=False, bn=False, name='convp21')  #make output channel 28 |
| ------------------------------------------------------------ |
|                                                              |

| # softmax output for positive anchor and negative anchor, scale = [None, 200/100, 176/120, 1] |
| ------------------------------------------------------------ |
|                                                              |

| self.p_pos_p = tf.compat.v1.sigmoid(p_map_p,name='probp') # |
| ----------------------------------------------------------- |
|                                                             |

| #self.p_pos = tf.compat.v1.nn.softmax(p_map, dim=3) |
| --------------------------------------------------- |
|                                                     |

|      |
| ---- |
|      |

| self.cls_pos_loss_p = (-self.pos_equal_one_p * tf.compat.v1.log(self.p_pos_p + small_addon_for_BCE)) / self.pos_equal_one_sum_p |
| ------------------------------------------------------------ |
|                                                              |

|      |
| ---- |
|      |

| self.cls_neg_loss_p = (-self.neg_equal_one_p * tf.compat.v1.log(1 - self.p_pos_p + small_addon_for_BCE)) / self.neg_equal_one_sum_p |
| ------------------------------------------------------------ |
|                                                              |

|      |
| ---- |
|      |

| self.cls_pos_loss_rec_p = tf.compat.v1.reduce_sum( self.cls_pos_loss_p ) |
| ------------------------------------------------------------ |
|                                                              |

| self.cls_neg_loss_rec_p = tf.compat.v1.reduce_sum( self.cls_neg_loss_p ) |
| ------------------------------------------------------------ |
|                                                              |

|      |
| ---- |
|      |

| self.reg_loss_p = smooth_l1(r_map_p * self.pos_equal_one_for_reg_p, self.targets_p * |
| ------------------------------------------------------------ |
|                                                              |

| self.pos_equal_one_for_reg_p, sigma) / self.pos_equal_one_sum_p |
| ------------------------------------------------------------ |
|                                                              |

| self.reg_loss_p = tf.compat.v1.reduce_sum(self.reg_loss_p) |
| ---------------------------------------------------------- |
|                                                            |

|      |
| ---- |
|      |

| #cyclist |
| -------- |
|          |

|      |
| ---- |
|      |

| #Probability score map, scale = [None, 200/100, 176/120, 2] |
| ----------------------------------------------------------- |
|                                                             |

| p_map_c = ConvMD(2, 768, 2, 1, (1, 1), (0, 0), temp_conv, |
| --------------------------------------------------------- |
|                                                           |

| training=self.training, activation=False, bn=False, name='convc20')  #make output channel 4 |
| ------------------------------------------------------------ |
|                                                              |

| # Regression(residual) map, scale = [None, 200/100, 176/120, 14] |
| ------------------------------------------------------------ |
|                                                              |

| r_map_c = ConvMD(2, 768, 14, 1, (1, 1), (0, 0), |
| ----------------------------------------------- |
|                                                 |

| temp_conv, training=self.training, activation=False, bn=False, name='convc21')  #make output channel 28 |
| ------------------------------------------------------------ |
|                                                              |

| #softmax output for positive anchor and negative anchor, scale = [None, 200/100, 176/120, 1] |
| ------------------------------------------------------------ |
|                                                              |

| self.p_pos_c = tf.compat.v1.sigmoid(p_map_c,name='probc') # |
| ----------------------------------------------------------- |
|                                                             |

| #self.p_pos = tf.compat.v1.nn.softmax(p_map, dim=3) |
| --------------------------------------------------- |
|                                                     |

|      |
| ---- |
|      |

| #stacking the pedestrain and cyclist regression and prob map together in axis 0 |
| ------------------------------------------------------------ |
|                                                              |

| r_map = tf.compat.v1.stack([r_map_p,r_map_c],axis=0,name='conv22') |
| ------------------------------------------------------------ |
|                                                              |

| p_pos = tf.compat.v1.stack([self.p_pos_p,self.p_pos_c],axis=0,name='prob') |
| ------------------------------------------------------------ |
|                                                              |

|      |
| ---- |
|      |

| self.output_shape = [cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH] |
| ------------------------------------------------------- |
|                                                         |

| #cross entropy |
| -------------- |
|                |

| self.cls_pos_loss_c = (-self.pos_equal_one_c * tf.compat.v1.log(self.p_pos_c + small_addon_for_BCE)) / self.pos_equal_one_sum_c |
| ------------------------------------------------------------ |
|                                                              |

|      |
| ---- |
|      |

| self.cls_neg_loss_c = (-self.neg_equal_one_c * tf.compat.v1.log(1 - self.p_pos_c + small_addon_for_BCE)) / self.neg_equal_one_sum_c |
| ------------------------------------------------------------ |
|                                                              |

|      |
| ---- |
|      |

|      |
| ---- |
|      |

| self.cls_pos_loss_rec_c = tf.compat.v1.reduce_sum( self.cls_pos_loss_c) |
| ------------------------------------------------------------ |
|                                                              |

| self.cls_neg_loss_rec_c = tf.compat.v1.reduce_sum( self.cls_neg_loss_c) |
| ------------------------------------------------------------ |
|                                                              |

|      |
| ---- |
|      |

|      |
| ---- |
|      |

| self.reg_loss_c = smooth_l1(r_map_c * self.pos_equal_one_for_reg_c, self.targets_c * |
| ------------------------------------------------------------ |
|                                                              |

| self.pos_equal_one_for_reg_c, sigma) / self.pos_equal_one_sum_c |
| ------------------------------------------------------------ |
|                                                              |

| self.reg_loss_c = tf.compat.v1.reduce_sum(self.reg_loss_c) |
| ---------------------------------------------------------- |
|                                                            |

|      |
| ---- |
|      |

| self.cls_loss_p = tf.compat.v1.reduce_sum( 1.5 * self.cls_pos_loss_p + 1.0 * self.cls_neg_loss_p )  #hyperparameters  alpha and beta |
| ------------------------------------------------------------ |
|                                                              |

| self.cls_loss_c = tf.compat.v1.reduce_sum( 1.5 * self.cls_pos_loss_c + 1.0 * self.cls_neg_loss_c ) #hyperparameters   alpha1 and beta1 |
| ------------------------------------------------------------ |
|                                                              |

|      |
| ---- |
|      |

| self.cls_loss = tf.compat.v1.reduce_sum(1* self.cls_loss_p + 1.3 * self.cls_loss_c)  #hyperparameters A and B |
| ------------------------------------------------------------ |
|                                                              |

| self.reg_loss = tf.compat.v1.reduce_sum(1 * self.reg_loss_p + 1.3 * self.reg_loss_c) #hyperparameters  A1 and B1 |
| ------------------------------------------------------------ |
|                                                              |

|      |
| ---- |
|      |

self.loss=tf.compat.v1.reduce_sum(self.cls_loss+self.reg_loss)



