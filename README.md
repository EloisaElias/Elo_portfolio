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
###23_
  	Summary: Machine learning algorithm
  	Goal:
  	
  	Tradeoffs:
  		- Advantages:
  		- Downside:
  		- Solution:
  	
  	Dataset:
  	Related Programs: 
  	
  	The fun part: In process
  	
###22_Random_forest_ROC
  	
  	Summary: Receiver Operating Characteristics (ROC) graphs are a useful technique for organizing classifiers and visualizing their performance. 
 
  	Goal:
  	  	Tradeoffs:
  		- Advantages: Cross validation is not strictly necessary. 
  		- Downside:Confidence scores (threshold) used to build ROC curves may be difficult to assign.
  		- Solution: Alternatives to ROC graphs: DET curves, Cost curves.
  	  		
  	Dataset: Churn.csv
  	Related Programs: 
  	
  	The fun part: The confusion matrix and the receiver operating characteristic
  	
###21_Random_forest
  	Summary: (RF) is a non-parametric, non linear supervised learning method used for classification (Nominal/Discrete data) and regression (Continuous data). 
  	Goal:
  	
  	Tradeoffs:
  		- Advantages: Cross validation is not strictly necessary
  		- Downside:RF  is a predictive modelling tool, slow to create predictions once trained, More accurate ensembles require more trees.
  		- Solution:RF is a highly parallel algorithm , so if you have multiple cores, you can get a significant speedup.  


  	
  	Dataset: https://archive.ics.uci.edu/ml/datasets
  	Related Programs: 
  	 
  	The fun part: The RF class and nodes.
  	
###20_Decision_tree

  	Summary: Decision Trees (DTs) are a non-parametric, non linear supervised learning method used for classification (Nominal/Discrete data) and regression (Continuous data). 
  	Goal: 
  	
  	Tradeoffs: 	
  		- Advantages: No complex data preparation, discrete and continuous data usage, good performance in large datasets 
  		- Downside: Overfitting, computationally expensive to train.
  		- Solution: Prepruning, Pruning, Random Forests
	
	Dataset: playgolf.csv
  	Related Programs: DecisionTree_elo.py, DecisionTree_run.py, TreeNode_elo.py
	
	The fun part: The concept of Entropy in terms of information theory.

###19_KNearest_Neightbor
  	Summary: Machine learning algorithm
  	
  	Goal:
  	
  	Tradeoffs:
  		- Advantages:
  		- Downside:
  		- Solution:
  		
  	Dataset:
  	Related Programs:
  	
  	The fun part:Data needs no preparation for the the algorithm

###18_Gradient_descend
  	Summary: Machine learning - Optimization algorithm
  	Goal: The goal of gradient descent is to minimize a function (the cost function of the hypothesis or the square errors of the hypothesis). For this case is Logistic regression function. Obtain the parameters that minimize my function. h(θ) --> j(θ) --> min_θ j(θ).
  	
  	Tradeoffs:
  		- Advantages: The use of vectorization.
  		- Downside: Overfitting
  		- Solution: Feature scaling, manual selection of features, Ridge-Lasso regularization.
  	
  	Related optimization algorithms: Conjugated gradient, BFGS, L-BFGS.
  	Dataset:
  	Related Programs:
  	
  	The fun part: The math and the gradient class function
  	
###17_Logistic_regression
    Summary: Classifier algorithm, ROC, Kfold and AUC
    Goal:
  	
  	Tradeoffs:
  		- Advantages:
  		- Downside:
  		- Solution:
  		
  	Dataset:
  	Related Programs:
  	
  	The fun part: the ROC curve
  	
###16_Regularization_Regression
  	Summary: Ridge and Lasso   	
  	Goal: Addressing overfitting
  	
  	Tradeoffs:
  		- Advantages: Works well when we have a lot of features each of which contributes a bit to predicting y. Keep all features, but reduce magnitude/values of parameters θj. 
  		- Downside: a) For n<<p case (high dimensional case), LASSO can at most select n features. b) For usual case where we have correlated features which is usually the case for real word datasets, LASSO will select only one feature from a group of correlated features. c) For n>>p case, it is seen that for correlated features , Ridge (Tikhonov Regularization) regression has better prediction power than LASSO. 
  		- Solution: Model selection algorithm
  	
  	Dataset: sklearn.datasets - load_diabetes()
  	
  	
  	The fun part: Visualizing the best alpha for the model.
  	
###15_Cross_Validation
  	Summary: Comparing models - sklear dataset
  	Goal:
  	
  	Tradeoffs:
  		- Advantages:
  		- Downside:
  		- Solution:
  		
  	Dataset:
  	Related Programs:
  	
  	The fun part: The training test size estimator
  	
###14_LinearRegression_Credit_card_balances
  	Summary: Credit card analysis - multivariate regression
  	Goal:
  	
  	Tradeoffs:
  		- Advantages:
  		- Downside:
  		- Solution:
  		
  	Dataset:
  	Related Programs:
  		
  	The fun part: Feature engineering
  	
###13_Multivariate_linear_regression
  	Summary: Linear regression   	
  	Goal:
  	
  	Tradeoffs: 
  		- Advantages:
  		- Downside:
  		- Solution:
  	
  	Dataset:
  	Related Programs:
  	  		
  	The fun part: Using plotly for graphics
  	  	
###12_Exploratory_Data_Analysis_EDA
  	Summary: Business analysis - Bike rental 
  	Goal:
  	
  	Tradeoffs:
  		- Advantages:
  		- Downside:
  		- Solution:
  	
  	Dataset:
  	Related Programs:
  		
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
  	Goal:
  	
  	Tradeoffs:
  		- Advantages:
  		- Downside:
  		- Solution:
  	
  	Dataset:
  	Related Programs:
  		
  	The fun part: Learning another way to program
  
###00_UCI_Iris_&_wine_df
  	Summary: Exploratory data analysis (EDA) & logistic regression model
  	Goal:
  	
  	Tradeoffs:
  		- Advantages:
  		- Downside:
  		- Solution:
  	
  	Dataset: Iris.csv
  	Related Programs:
  	
  	The fun part: Using different kind of graphs
  


  
  
  
  
  
  
  
  
  
  
  
  
  
 


