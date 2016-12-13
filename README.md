## Direction over speed..
#### Effectivess & Efficiency

Hola a todos, I'm Eloisa and I'm a data scientist that loves math, python & statistics. Thank you for viewing my portfolio. 

I've been working on these programs in a daily basis for my personal reference and for my readers. I hope you will enjoy them as much I did when working on them. My favorites ones so far are 11 and 12.

..for my amusement, practice and better comprehension



                                                       Gracias totales
                                                            Elo



Note: [Firefox - Visualizing the mathematical formulas(LaTex) in ipython notebook](http://docs.mathjax.org/en/latest/installation.html#firefox-and-local-fonts) 


---
### Content
---
###26_AdaBoosting_Classsifier
  	Summary: Gradient boosting is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees. 
  	Goal: Usage of Sklearn Boost algorithms and SearchGrid.

  	
  	Tradeoffs: 
  		- Advantages:   		
  		- Downside: 
  		- Solution: 
  	
  	Dataset: 
  	
  	The fun part: Grid search
###25_Boosting
  	Summary: Gradient boosting is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees. 
  	Goal: Usage of Sklearn Boost algorithms and SearchGrid.

  	
  	Tradeoffs: 
  		- Advantages:   		
  		- Downside: 
  		- Solution: 
  	
  	Dataset: 
  	
  	The fun part: Grid search
###24_SVM_hyperparameter_C
  	Summary: Support Vector Machine 
  	Goal: Use of hyperparameter C and grid search optimization 

  	
  	Tradeoffs: SVMs have a tradeoff between maximizing the margin and minimizing the classification error.
  		- Advantages: Best parameters set found by using tuning hyper-parameters GridSearchCV() 
  		- Downside: Takes a long time for multiple folds
  		- Solution: Training with a sample data, or reduce the folds qty.
  	
  	Dataset: sklearn load_digits() & other cvs files.
  	
  	The fun part: Grid search
###23_SVM
  	Summary: Support Vector Machine - Linear SVM
  	Goal: Logistic Regression boundary and SVM boundary comparison

  	
  	Tradeoffs: 
  		- Advantages: a)Linear and non-linear classification, b)Usage of soft margins, c)Kernel transformation, d)regularisation parameter, to avoiding over-fitting, e)SVM is defined by a convex optimisation problem
  		- Downside: See, a)Journal of Machine Learning Research 11 (2010) 2079-2107 On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation
  		- Solution: Over-fitting in model selection can be over come using methods that have already been effective in preventing over-fitting during trainin g, such as regularisation as  Kernel Ridge Regression (KRR) classifier
  	
  	Dataset: cvs files.
  	
  	The fun part: SVM boundaries

  	
###22_Random_forest_ROC
  	
  	Summary: Receiver Operating Characteristics (ROC) graphs are a useful technique for organizing classifiers and visualizing their performance. 
 
  	Goal:
  	  	Tradeoffs:
  		- Advantages: Cross validation is not strictly necessary. 
  		- Downside:Confidence scores (threshold) used to build ROC curves may be difficult to assign.
  		- Solution: Alternatives to ROC graphs: DET curves, Cost curves.
  	  		
  	Dataset: Churn.csv
  	Related Programs: 
  	
  	The fun part: The confusion matrix and the receiver operating characteristic, amd feature importance.
  	
###21_Random_forest
  	Summary: (RF) is a non-parametric, non linear supervised learning method used for classification (Nominal/Discrete data) and regression (Continuous data). 
  	Goal: Step by step manual RF
  	
  	Tradeoffs:
  		- Advantages: Cross validation is not strictly necessary
  		- Downside:RF  is a predictive modelling tool, slow to create predictions once trained, More accurate ensembles require more trees.
  		- Solution:RF is a highly parallel algorithm , so if you have multiple cores, you can get a significant speedup.  

  	Dataset: https://archive.ics.uci.edu/ml/datasets
  	Related Programs: RF
  	 
  	The fun part: The RF class and nodes.
  	
###20_Decision_tree

  	Summary: Decision Trees (DTs) are a non-parametric, non linear supervised learning method used for classification (Nominal/Discrete data) and regression (Continuous data). 
  	Goal: Step by step manual DT
  	
  	Tradeoffs: 	
  		- Advantages: No complex data preparation, discrete and continuous data usage, good performance in large datasets 
  		- Downside: Overfitting, computationally expensive to train.
  		- Solution: Prepruning, Pruning, Random Forests
	
	Dataset: playgolf.csv
  	Related Programs: DecisionTree_elo.py, DecisionTree_run.py, TreeNode_elo.py
	
	The fun part: The concept of Entropy in terms of information theory.

###19_KNearest_Neightbor
  	Summary: Nearest neighbor search (NNS), also known as proximity search, similarity search or closest point search, is an optimization problem for finding closest (or most similar) points
  	
  	Goal: Step by step manual KNN
  	
  	Tradeoffs:
  		- Advantages: a)Simple implementation, 
  		- Downside: a)Determine the value of parameter K, b)Computationally intensive, c)It doesn't handle categorical variables very well, d)Highly susceptible to the curse of dimensionality
  		- Solution: Two classical algorithms can be used to speed up the NN search 1)Bucketing(a.k.a Elias’s algorithm) [Welch 1971], 2)k-d trees [Bentley, 1975; Friedman et al, 1977]
  		
  	Dataset: from sklearn.datasets import make_classification
  	Related Programs: Knn.py
  	
  	The fun part:Data needs no preparation for the the algorithm

###18_Gradient_descend
  	Summary: Machine learning - Optimization algorithm
  	Goal: The goal of gradient descent is to minimize a function (the cost function of the hypothesis or the square errors of the hypothesis). For this case is Logistic regression function. Obtain the parameters that minimize my function. h(θ) --> j(θ) --> min_θ j(θ).
  	
  	Tradeoffs:
  		- Advantages: The use of vectorization.
  		- Downside: Overfitting
  		- Solution: Feature scaling, manual selection of features, Ridge-Lasso regularization.
  	
  	Related optimization algorithms: Conjugated gradient, BFGS, L-BFGS.
  	Dataset: from sklearn.datasets import make_classification
  	Related Programs: Gradient.py
  	
  	The fun part: The math and the gradient class function
  	
###17_Logistic_regression
    Summary: Classifier algorithm, ROC, Kfold and AUC
    Goal: Obtain ROC curve
  	
  	Tradeoffs:
  		- Advantages: a)Logistic regression will work better if there's a single decision boundar, b)Logistic regression is intrinsically simple. c)Important to consider regularization
  		- Downside: a)The explanatory variables should not be highly correlated with one another because this could cause problems with estimation.response variable.
  		- Solution: Correct for multicolinearity among features.
  		
  	Dataset: from sklearn.datasets import make_classification
  	
  	The fun part: the ROC curve
  	
###16_Regularization_Regression
  	Summary: Ridge and Lasso   	
  	Goal: Addressing overfitting
  	
  	Tradeoffs:
  		- Advantages: Works well when we have a lot of features each of which contributes a bit to predicting y. Keep all features, but reduce magnitude/values of parameters θj. 
  		- Downside: LASSO - a) For n<<p case (high dimensional case), LASSO can at most select n features. b) For usual case where we have correlated features which is usually the case for real word datasets, LASSO will select only one feature from a group of correlated features. c) For n>>p case, it is seen that for correlated features , Ridge (Tikhonov Regularization) regression has better prediction power than LASSO. RIDGE a') Compared to ordinary least squares, ridge regression is not unbiased. It accepts little bias to reduce variance and the mean square error, and helps to improve the prediction accuracy. Thus, ridge estimator yields more stable solutions by shrinking coefficients but suffers from the lack of sensitivity to the data. LASSO  & RIDGE a)  LASSO regularization can occasionally produce non-unique solutions. A simple example is provided in the figure when the space of possible solutions lies on a 45 degree line. This can be problematic for certain applications, and is overcome by combining LASSO and RIDGE regularization in elastic net regularization
  		- Solution: Model selection algorithm
  	
  	Dataset: sklearn.datasets - load_diabetes()
  	
  	
  	The fun part: Visualizing the best alpha for the model.
  	
###15_Cross_Validation
  	Summary: Comparing models - sklear dataset
  	Goal: The goal is to evaluate the model given  metric I'm interested in.
  	
  	Tradeoffs:
  		- Advantages: The error value will plateau out after a certain m, or training set size.
  		- Downside: If a learning algorithm is suffering from high bias, getting more training data will not (by itself) help much.
  		- Solution:a)For high variance, we have the following relationships in terms of the training set size: With high variance a1)Low training set size:  Jtrain(Θ) will be low and JCV(Θ) will be high., a2)Large training set size: Jtrain(Θ) increases with training set size and JCV(Θ) continues to decrease without leveling off. Also, Jtrain(Θ)<JCV(Θ) but the difference between them remains significant.b)If a learning algorithm is suffering from high variance, getting more training data is likely to help.
  		
  	Dataset: from sklearn.datasets import load_boston

  	
  	The fun part: The training test size estimator
  	
###14_LinearRegression_Credit_card_balances
  	Summary: Credit card analysis - multivariate regression
  	Goal:The task is to predict an individual's balance based on various variables and feature engineering - Using adjusted R^2 or F-tests and VIF to detect multicollinearity.
  	

  	Dataset: csv
  	
  	The fun part: Feature engineering
  	
###13_Multivariate_linear_regression
  	Summary: Linear regression   	
  	Goal: Develop multivariate linear regression
  	
  	Tradeoffs: 
  		- Advantages: Linear regression implements a statistical model that, when relationships between the independent variables and the dependent variable are almost linear, shows optimal results. 
  		- Downside: If you are using regression analysis to generate predictions. Prediction intervals are calculated based on the assumption that the residuals are normally distributed. If the residuals are non-normal, the prediction intervals may be inaccurate.
  		- Solution: Normalizing the dataset, Independence of the observations, Avoiding multicollinearity among features by Compare the heteroscedasticity of residuals before and after taking log.
  	
  	Dataset: csv
  	Related Programs:
  	  		
  	The fun part: Using plotly for graphics
  	  	
###12_Exploratory_Data_Analysis_EDA
  	Summary: Business analysis - Bike rental 
  	Goal: Develop exploratory data analysis and apply linear regression algoritm in order to recomend the specific date with maximum spread of promotional campaign for a business (rental). The goal is to find the coefficients β which fit the equations "best," in the sense of solving the quadratic minimization problem
  	
  	Tradeoffs:
  		- Advantages: The numerical methods for linear least squares are important because linear regression models are among the most important types of model, both as formal statistical models and for exploration of data-sets. The majority of statistical computer packages contain facilities for regression analysis that make use of linear least squares computations. Hence it is appropriate that considerable effort has been devoted to the task of ensuring that these computations are undertaken efficiently and with due regard to round-off error.
  		- Downside:In these cases, the least squares estimate amplifies the measurement noise and may be grossly inaccurate
  		- Solution:Various regularization techniques can be applied e.g. LASSO|RIDGE
  	
  	Dataset: cvs
  		
  	The fun part: Obtaining the normal behavior of the rental business and the use of the basemap
  	
###11_Linear_algebra_PageRank_algorithm
  	Summary: Markov chain and linear algebra
  	Goal: Implementing the PageRank algorithm
  	
  	Tradeoffs
  		- Advantages: No complex ranking algorithm
  	
  	Dataset: Sklearn dataset - Iris.csv
  	
  	The fun part: page ranking algorithm using basic linear algebra, one of the first google pagerank algorithm
  	
###10_Covariance_Joint_distribution
  	Summary: Analizing a university dataset, obtaining potential threshold for admission
  	Goal:
  	
  	Tradeoffs:
  		- Advantages:
  		- Downside:
  		- Solution:
  	
  	Dataset:
  	Related Programs:
  		
  	The fun part: visualizing the pdf for admission vs income
  	
###09_Probability
  	Summary: Probability excersises
  	Goal:
  	
  	Tradeoffs:
  		- Advantages:
  		- Downside:
  		- Solution:
  	
  	Dataset:
  	Related Programs:
  		
  	The fun part: from basic probabilty obtaining interesting inferences
  	
###08_Bayesian_analysis.ipynb
  	Summary: Visualizing Bayes step by step
  	Goal:
  	
  	Tradeoffs:
  		- Advantages:
  		- Downside:
  		- Solution:
  	
  	Dataset:
  	Related Programs:
  		
  	The fun part: The transformation of the probabilities into a distribution
  	
###07_Power_A/B_test
    Summary: Hypothesis test and power calculation
    Goal:
  	
  	Tradeoffs:
    	- Advantages:
  		- Downside:
  		- Solution:
  	
  	Dataset:
  	Related Programs:
  		
  	The fun part: Power is everything, know your sample size 
  
###06_Multi_armed_bandit
  	Summary: Selection of a slot machine strategy 
  	Goal:
  	
  	Tradeoffs:
  		- Advantages:
  		- Downside:
  		- Solution:
  	
  	Dataset:
  	Related Programs:
  		
  	The fun part: Visualize the the house always win
  
###05_Bayesian_bandit
  	Summary: Comparing websites
  	Goal:
  	
  	Tradeoffs:
  		- Advantages:
  		- Downside:
  		- Solution:
  	
  	Dataset:
  	Related Programs:
  		
  	The fun part: Visualizing Bayes
  
###04_Estimation_sampling
   	Summary:Data distribution
   	Goal:
  	
  	Tradeoffs:
   		- Advantages:
  		- Downside:
  		- Solution:
  	
  	Dataset:
  	Related Programs:
  		
  	The fun part: the Gamma vs Normal distributions
  
###03_AB_test
    Summary: Z-test function
    Goal:
  	
  	Tradeoffs:
    	- Advantages:
  		- Downside:
  		- Solution:
  	
  	Dataset:
  	Related Programs:
  		
  	The fun part: the z-test function
  
###02_CTR
    Summary: Click through rate methodology and t-test function
    Goal:
  	
  	Tradeoffs:
    	- Advantages:
  		- Downside:
  		- Solution:
  	
  	Dataset:
  	Related Programs:
  		
  	The fun part: Using a NYT dataset 
  
###01_Classes
  	Summary: Encapsulation programs
  	Goal: Understanding classes
  	
  	Tradeoffs:
  		- Advantages: Object Oriented programming
  		
  	The fun part: Learning another way to program
  
###00_UCI_Iris_&_wine_df
  	Summary: Exploratory data analysis (EDA) & logistic regression model
  	Goal: Visualizing the data
  	
  	Tradeoffs:
  		- Advantages: Visualize the shape of our data
  		- Downside: Takes some valuable time
  		- Solution: It's worth to visualize our datase before start doing stats
  	
  	Dataset: https://archive.ics.uci.edu/ml/machine-learning-databases & Iris.csv
  	
  	The fun part: Graphs
  


  
  
  
  
  
  
  
  
  
  
  
  
  
 


