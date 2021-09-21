#!/usr/bin/env python
# coding: utf-8

# Hanna Born,
# ANLY 590 HW2.1

# References:
# code files provided on class github at https://github.com/jh2343/590-CODES
# https://towardsdatascience.com/linear-regression-derivation-d362ea3884c2
# http://pillowlab.princeton.edu/teaching/mathtools16/slides/lec10_LeastSquaresRegression.pdf

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
from   scipy.optimize import minimize

#USER PARAMETERS
IPLOT=True
INPUT_FILE='weight.json'
FILE_TYPE="json"
DATA_KEYS=['x','is_adult','y']
X_KEYS = ['x']
Y_KEYS = ['y'] #'is_adult'
OPT_ALGO='BFGS'

#SAVE HISTORY FOR PLOTTING AT THE END
epoch=1; epochs=[]; loss_train=[];  loss_val=[]

# ---- PARADIGM SELECTION ------------------------------------------------------
PARADIGM='batch'
#PARADIGM='mini-batch'
#PARADIGM='stochastic'
# ------------------------------------------------------------------------------

# ---- MODEL SELECTION ---------------------------------------------------------
# choosing second logistic model for columns corresponding to age vs. weight relationship
# choose model type by uncommenting
# model_type="linear";   NFIT=2; xcol=1; ycol=2;       # age vs. weight (linear)
model_type="logistic"; NFIT=4; xcol=1; ycol=2;       # age vs weight (logistic)
# model_type="logistic";   NFIT=4; xcol=2; ycol=0;     # weight vs. adult/child
# ------------------------------------------------------------------------------

#READ FILE
with open(INPUT_FILE) as f:
	input1 = json.load(f)  #read into dictionary

#CONVERT DICTIONARY INPUT AND OUTPUT MATRICES #SIMILAR TO PANDAS DF   
X=[]; Y=[]
for key in input1.keys():
	if(key in X_KEYS): X.append(input1[key])
	if(key in Y_KEYS): Y.append(input1[key])

#MAKE ROWS=SAMPLE DIMENSION (TRANSPOSE)
X=np.transpose(np.array(X))
Y=np.transpose(np.array(Y))
print('--------INPUT INFO-----------')
print("X shape:",X.shape); print("Y shape:",Y.shape,'\n')

#TAKE MEAN AND STD DOWN COLUMNS (I.E DOWN SAMPLE DIMENSION)
XMEAN=np.mean(X,axis=0); XSTD=np.std(X,axis=0) 
YMEAN=np.mean(Y,axis=0); YSTD=np.std(Y,axis=0) 

#NORMALIZE 
X=(X-XMEAN)/XSTD;  Y=(Y-YMEAN)/YSTD  

#------------------------
#PARTITION DATA
#------------------------
#TRAINING: 	 DATA THE OPTIMIZER "SEES"
#VALIDATION: NOT TRAINED ON BUT MONITORED DURING TRAINING
#TEST:		 NOT MONITORED DURING TRAINING (ONLY USED AT VERY END)

f_train=0.8; f_val=0.15; f_test=0.05;

if(f_train+f_val+f_test != 1.0):
	raise ValueError("f_train+f_val+f_test MUST EQUAL 1");

#PARTITION DATA
rand_indices = np.random.permutation(X.shape[0])
CUT1=int(f_train*X.shape[0]); 
CUT2=int((f_train+f_val)*X.shape[0]); 
train_idx, val_idx, test_idx = rand_indices[:CUT1], rand_indices[CUT1:CUT2], rand_indices[CUT2:]
print('------PARTITION INFO---------')
print("train_idx shape:",train_idx.shape)
print("val_idx shape:"  ,val_idx.shape)
print("test_idx shape:" ,test_idx.shape)

#------------------------
#MODEL
#------------------------
def model(x,p):
	if(model_type=="linear"):   return  p[0]*x+p[1]  
	if(model_type=="logistic"): return  p[0]+p[1]*(1.0/(1.0+np.exp(-(x-p[2])/(p[3]+0.0001))))

#FUNCTION TO MAKE VARIOUS PREDICTIONS FOR GIVEN PARAMETERIZATION
def predict(p):
	global YPRED_T,YPRED_V,YPRED_TEST,MSE_T,MSE_V
	YPRED_T=model(X[train_idx],p)
	YPRED_V=model(X[val_idx],p)
	YPRED_TEST=model(X[test_idx],p)
	MSE_T=np.mean((YPRED_T-Y[train_idx])**2.0)
	MSE_V=np.mean((YPRED_V-Y[val_idx])**2.0)

#------------------------
#LOSS FUNCTION
#------------------------
def loss(p,index_2_use):
	errors=model(X[index_2_use],p)-Y[index_2_use]  #VECTOR OF ERRORS
	training_loss=np.mean(errors**2.0)				#MSE
	return training_loss


#SAVE HISTORY FOR PLOTTING AT THE END
iteration=0; iterations=[]; loss_train=[];  loss_val=[]
#------------------------
#OPTIMIZER FUNCTION
#------------------------
def optimizer(f,xi, algo='GD', LR=0.01):
	global epoch,epochs, loss_train,loss_val 
	# x0=initial guess, (required to set NDIM)
	# algo=GD or MOM
	# LR=learning rate for gradient decent

	#PARAM
	iteration=1			#ITERATION COUNTER
	dx=0.0001			#STEP SIZE FOR FINITE DIFFERENCE
	max_iter=5000		#MAX NUMBER OF ITERATION
	tol=10**-10			#EXIT AFTER CHANGE IN F IS LESS THAN THIS 
	NDIM=len(xi)		#DIMENSION OF OPTIIZATION PROBLEM
	#ICLIP=False

	# hyperparameter tuning for stochastic
	if(PARADIGM=='stochastic'):
		LR=0.002
		max_iter=25000
		#ICLIP=True

        
	#OPTIMIZATION LOOP
	while(iteration<=max_iter):

		#-------------------------
		#DATASET PARTITION BASED ON TRAINING PARADIGM
		#-------------------------
		if(PARADIGM=='batch'):
			if(iteration==1): index_2_use=train_idx
			if(iteration>1):  epoch+=1
                
		if(PARADIGM=='mini-batch'): # for mini-batch use a 0.5 batch size
			if(iteration==1):
				# break data-set into two chunks
				batch1 = train_idx[:len(train_idx)//2]
				batch2 = train_idx[len(train_idx)//2:]        
				if(iteration%2==0): # use batch 1
					index_2_use=batch1
				else: # use batch 2
					index_2_use=batch2
			if(iteration>1):  epoch+=1
            
		if(PARADIGM=='stochastic'):   
			if(iteration==1): # if on first iteration use the first data point
				index_2_use=0
			else:
				if(index_2_use==train_idx.shape[0]):
				# or if(index_2_use==len(train_idx)):
					index_2_use=0 # reset back to the beginning
				else:
					index_2_use+=1 # otherwise increment it         


# 		else:
# 			print("REQUESTED PARADIGM NOT CODED"); exit()

		#-------------------------
		#NUMERICALLY COMPUTE GRADIENT 
		#-------------------------
		df_dx=np.zeros(NDIM);	#INITIALIZE GRADIENT VECTOR
		for i in range(0,NDIM):	#LOOP OVER DIMENSIONS

			dX=np.zeros(NDIM);  #INITIALIZE STEP ARRAY
			dX[i]=dx; 			#TAKE SET ALONG ith DIMENSION
			xm1=xi-dX; 			#STEP BACK
			xp1=xi+dX; 			#STEP FORWARD 

			#CENTRAL FINITE DIFF
			grad_i=(f(xp1,index_2_use)-f(xm1,index_2_use))/dx/2

			# UPDATE GRADIENT VECTOR 
			df_dx[i]=grad_i 
			
		#TAKE A OPTIMIZER STEP
		if(algo=="GD"):  xip1=xi-LR*df_dx
		# fill in GD+momentum ... same formula as above + one more term for momentum

		# momentun includes one more term momentum: alpha (exponential decay factor)*change
		dx_m1=0
		alpha = 0.1        
		if(algo=="MOM"):  xip1=xi-LR*df_dx+alpha*dx_m1
            
		# EXTRA CREDIT ... incomplete, will revisit later time-permitting
		if(algo=="RMSprop"): print("REQUESTED ALGORITHM (RMSprop) NOT CODED"); exit()
		if(algo=="ADAM"): print("REQUESTED ALGORITHM (ADAM) NOT CODED"); exit()
		if(algo=="Nelder-Mead"): print("REQUESTED (Nelder-Mead) ALGORITHM NOT CODED"); exit()

		if(iteration==0):
			print("iteration \t epoch \t MSE_T \t MSE_V")
		if(iteration%250==0): # print info for every 250th iteration
			print(iteration,"	",epoch,"	",MSE_T,"	",MSE_V)

		#REPORT AND SAVE DATA FOR PLOTTING
		if(iteration%1==0):
			predict(xi)	#MAKE PREDICTION FOR CURRENT PARAMETERIZATION

			#UPDATE
			epochs.append(epoch); 
			loss_train.append(MSE_T);  loss_val.append(MSE_V);

			#STOPPING CRITERION (df=change in objective function)
			df=np.absolute(f(xip1,index_2_use)-f(xi,index_2_use))
			if(df<tol):
				print("STOPPING CRITERION MET (STOPPING TRAINING)")
				break

		xi=xip1 #UPDATE FOR NEXT PASS
		iterations.append(iteration)
		iteration=iteration+1

	return xi


#------------------------
#FIT MODEL
#------------------------

#RANDOM INITIAL GUESS FOR FITTING PARAMETERS
po=np.random.uniform(2,1.,size=NFIT)

#TRAIN MODEL USING OPTIMIZER(MINIMIZER)
# have the optimizer take the loss function as an argument as well as various default options
#popt=optimizer(loss,po)
popt = optimizer(loss, po, algo='MOM')
print("OPTIMAL PARAM:",popt)
predict(popt)
    


#UN-NORMALIZE
def unnorm_x(x): 
	return XSTD*x+XMEAN  
def unnorm_y(y): 
	return YSTD*y+YMEAN 

# generate plots
#FUNCTION PLOTS
if(IPLOT):
	fig, ax = plt.subplots()
	ax.plot(unnorm_x(X[train_idx]), unnorm_y(Y[train_idx]), 'o', label='Training set')
	ax.plot(unnorm_x(X[test_idx]), unnorm_y(Y[test_idx]), 'x', label='Test set')
	ax.plot(unnorm_x(X[val_idx]), unnorm_y(Y[val_idx]), '*', label='Validation set')
	ax.plot(unnorm_x(X[train_idx]),unnorm_y(YPRED_T), '.', c='red', label='Model')
	plt.xlabel('x', fontsize=18)
	plt.ylabel('y', fontsize=18)
	plt.legend()
	plt.show()

#PARITY PLOTS
if(IPLOT):
	fig, ax = plt.subplots()
	ax.plot(model(X[train_idx],popt), Y[train_idx], 'o', label='Training set')
	ax.plot(model(X[val_idx],popt), Y[val_idx], 'o', label='Validation set')
	plt.xlabel('y predicted', fontsize=18)
	plt.ylabel('y data', fontsize=18)
	plt.legend()
	plt.show()

#MONITOR TRAINING AND VALIDATION LOSS  
if(IPLOT):
	fig, ax = plt.subplots()
	ax.plot(iterations, loss_train, 'o', label='Training loss')
	ax.plot(iterations, loss_val, 'o', label='Validation loss')
	plt.xlabel('optimizer iterations', fontsize=18)
	plt.ylabel('loss', fontsize=18)
	plt.legend()
	plt.show()