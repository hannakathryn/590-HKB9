#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Hanna Born,
# ANLY 590 HW2.2.1

# References:
# code files provided on class github at https://github.com/jh2343/590-CODES


# In[2]:


# SEABORN VISUALIZER SCRIPT

import  pandas  as  pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 

#CONVERT LIST OF INDICES TO DF KEYS
    #NOTE pd_indices SHOULD BE MONOTONICALLY INCREASING
    # i.e. [2,4,5] NOT [5,2,4] 
    #THIS FUNCTION WILL RETURN THEM SORTED
def index_to_keys(df,pd_indices):
    pd_indices=sorted(pd_indices)
    #FIND KEYS FOR PLOTTING BASED ON PROVIDED INDICES
    pd_keys=[]; indx=0;  
    for col in df:
        if(indx in pd_indices ):
            pd_keys.append(col); 
        indx+=1 # for col in df:
    return pd_keys


#SUPPLY PANDAS DATAFRAM AND GET A BASIC SUMMARY OF CONTENTS
def get_pd_info(df):

    #ERROR CHECK 
    if(str(type(df)) != "<class 'pandas.core.frame.DataFrame'>"): 
        raise ValueError("input variable is not panda DataFrame")

    #GENERAL STUFF THAT IS 
    print("----------------------")
    print("GENERAL:")
    print("----------------------")
    print("number of rows:", len(df.index))
    print("number of col:",  len(df.columns))
    print("keys:",  df.keys(),type(df.keys()))
    print("info",df.info() )
    print("head", df.head())
    print("TYPES", df.dtypes)

    print("----------------------")
    print("BASIC STATISTICS:")
    print("----------------------")
    print(df.describe())

    print("----------------------")
    print("CORRELATION MATRIX:")
    print("----------------------")
    print(df.corr())


#-------------------------
#BASIC DATAFRAME PLOTTING 
#-------------------------
#(GENERAL) SHOULD WORK WITH ARBITRARY DATAFRAME
#ACTS ON ENTIRE DATAFRAME

def pd_general_plots(df,HUE=None):
    #NOTE: CERTAIN PLOTS ONLY WORK IF HUE=CATERGORICAL

    #-------------------------
    #FULL PAIR PLOT
    #-------------------------
    sns.pairplot(df, diag_kind='kde', kind="hist", hue=HUE)  # SLOWER
    plt.show()


    #-------------------------
    #FULL CORRELOGRAM
    #-------------------------  
    sns.set_theme(style="white")
    corr = df.corr()  #Compute the correlation matrix

    # # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool)) 
    f, ax = plt.subplots(figsize=(11, 9)) #initialize figure

    cmap = sns.diverging_palette(230, 20, as_cmap=True) #custom diverging colormap

    # # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()


    #-------------------------
    #DENSITY PLOT PROJECTED ONTO FIRST COL
    #-------------------------

    if(HUE!=None):
        #GET NAME OF FIRST COLUMN
        colname = df.columns[0]

        # Plot the distribution of clarity ratings, conditional on carat
        sns.displot(
            data=df,
            x=colname, hue=HUE,
            kind="kde", height=6,
            multiple="fill", alpha=.5,    palette="ch:rot=-.25,hue=1,light=.5",

        )
        plt.show()

    #-------------------------
    #CORRELOGRAM+DENDROGRAMS
    #-------------------------  
    sns.clustermap(corr, center=0, cmap=cmap, vmax=.3, 
                 linewidths=4, cbar_kws={"shrink": .5})
    plt.show()


    #-------------------------
    #VIOLIN PLOTS (LOGSCALE)
    #-------------------------
    f, ax = plt.subplots(figsize=(7, 6))
    ax.set_xscale("log")
    sns.set_theme()
    sns.violinplot(data=df, palette="Set3", inner="points", orient="h")
    plt.show()


#------------------------------------------
#DATAFRAME PLOTTING USING 3 COLUMNS (X,Y,Z)
#-------------------------------------------
#DEFAULT IS FIRST 3 COLUMNS
def pandas_2D_plots(df,col_to_plot=[0,1,2], HUE=None):

    #NOTES:
    # X=NUMERIC CONTINUOUS col_to_plot[0]
    # Y=NUMERIC CONTINUOUS col_to_plot[1]
    # Z=NUMERIC CONTINUOUS col_to_plot[2] (USED FOR SIZE AND COLORING)

    # HUE=DISCRETE OR CATEGORICAL

    #GET COLUMN NAMES FOR PLOTTING
    xname=df.columns[col_to_plot[0]]
    yname=df.columns[col_to_plot[1]]
    zname=df.columns[col_to_plot[2]]
    # print(xname,yname,zname); exit()

    #ERROR CHECK 
    if(str(type(df)) != "<class 'pandas.core.frame.DataFrame'>"): 
        raise ValueError("input variable is not panda DataFrame")

    if(len(df.columns)<3): raise ValueError("not enough columns")
    #print("number of col:",  len(df.columns))


    # #-------------------------
    # #SCATTER PLOT
    # #-------------------------
    plt.figure(figsize=(12,8))
    sns.scatterplot(x=xname,y=yname,data=df,hue=HUE)
    plt.show()

    sns.scatterplot(x=xname,y=yname,data=df,hue=zname) #HUE CAN BE
    sns.kdeplot(x=xname,y=yname,data=df, levels=5, color="b", linewidths=1)
    plt.show()

    sns.scatterplot(x=xname,y=yname,data=df,hue=HUE,size=zname) #HUE CAN BE
    plt.show()


    #-------------------------
    #VARIOUS PAIR PLOT STYLES
    #-------------------------
    keys_to_plot=index_to_keys(df,col_to_plot)

    if(HUE!=None and HUE not in keys_to_plot): keys_to_plot.append(HUE) 
    print("keys_to_plot",keys_to_plot)  

    sns.pairplot(df[keys_to_plot], kind='kde',hue=HUE)       #VERY SLOW BUT LOOKS GOOD
    plt.show()

    sns.pairplot(df[keys_to_plot], diag_kind='kde',hue=HUE)  #FAST
    plt.show()

    plt1=sns.pairplot(df[keys_to_plot], diag_kind='kde',hue=HUE)  #FAST
    plt1.map_lower(sns.kdeplot, levels=4, color=".2")  #SLOWER BUT BETTER 
    plt.show()


    #-------------------------
    #JOINTPLOT
    #-------------------------
    # Show the joint distribution using kernel density estimation
    sns.jointplot(
        data=df,
        x=xname, y=yname,
        kind="kde", hue=HUE
    )   
    plt.show()


    # #-------------------------
    # #BOX AND WHISKER
    # #-------------------------
    # #ONLY WORKS FOR CATEGORICAL HUE
    if(HUE!=None):
        for name in [xname,yname]:
            # Plot the orbital period with horizontal boxes
            sns.boxplot(x=name, y=HUE, data=df,
                        whis=[0, 100], width=.6, palette="vlag")

            # # Add in points to show each observation
            sns.stripplot(x=name, y=HUE, data=df,
                          size=4, color=".3", linewidth=0)
            plt.show()


            sns.boxenplot(x=HUE, y=name,
                      color="b", 
                      scale="linear", data=df)
            plt.show()

            #SWARM PLOT
            ax = sns.swarmplot(data=df, x=xname, y=HUE, hue=HUE)
            ax.set(ylabel="")
            plt.show()


    #-------------------------
    #RELPLOT
    #-------------------------

    sns.relplot(x=xname, y=yname, hue=HUE, size=zname,
            sizes=(40, 400), alpha=.5, palette="muted",
            height=6, data=df)
    plt.show()






#-------------------------------------------
#RUN SCRIPT 
#-------------------------------------------
#NOTE: CERTAIN PLOTS ONLY WORK IF HUE=CATERGORICAL

# #----------------------
# #DATASET-1: 
# #CALIFORNIA HOUSING CENSUS
# #----------------------

# ##source=https://medium.com/analytics-vidhya/house-price-prediction-regression-with-tensorflow-keras-4fc49fae7123

# #READ CSV DATA INTO PANDAS DATAFRAME
# df=pd.read_csv('./DATA/housing.csv')

# # #EVALUATE FUNCTIONS FOR GIVEN DATAFRAM
# get_pd_info(df);  

# # pd_general_plots(df)  
# pd_general_plots(df,HUE='ocean_proximity')  #SLOWER BUT NICER

# # pandas_2D_plots(df)
# # pandas_2D_plots(df,HUE='ocean_proximity')
# pandas_2D_plots(df,col_to_plot=[0,1,8],HUE='ocean_proximity')


# #----------------------
# #DATASET-2: CAR MPG DATA 
# #----------------------

# #GET DATA FROM SEABORN
# df = sns.load_dataset("mpg")

# get_pd_info(df); 

# # pd_general_plots(df)  
# pd_general_plots(df,HUE='origin') 

# # pandas_2D_plots(df)
# # pandas_2D_plots(df,HUE='origin')
# pandas_2D_plots(df,col_to_plot=[4,5,0],HUE='origin')


# In[13]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
from   scipy.optimize import minimize
import csv
import pandas as pd

# #CASE 1: Comment/Uncomment accordingly -------------------------------------
# INPUT_FILE='planar_x1_x2_y.json'
# DATA_KEYS=['x1','x2','y']
# X_KEYS = ['x1', 'x2']
# Y_KEYS = ['y'] 
# # choose model type by uncommenting
# model_type="linear";   NFIT=3
# # model_type="logistic"; NFIT=3
# # --------------------------------------------------------------------------

# #CASE 2: Comment/Uncomment accordingly -------------------------------------
# INPUT_FILE='planar_x1_x2_x3_y.json'
# DATA_KEYS=['x1','x2','x3','y']
# X_KEYS = ['x1', 'x2', 'x3']
# Y_KEYS = ['y'] 
# # choose model type by uncommenting
# model_type="linear";   NFIT=4
# # model_type="logistic"; NFIT=4
# # --------------------------------------------------------------------------


#MODIFYING HW2.2.1 TO GET MPG DATA -----------------------------------------

INPUT_FILE='mpg.csv'
#INPUT_FILE = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'


DATA_KEYS=['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']
X_KEYS = ['Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration']#, 'Model Year', 'Origin']
Y_KEYS = ['MPG'] 

# choose model type by uncommenting
model_type="linear";   NFIT=6
# model_type="logistic"; NFIT=6
# --------------------------------------------------------------------------


#USER PARAMETERS ---------------------
IPLOT=True
FILE_TYPE="csv"
OPT_ALGO='BFGS'
# ------------------------------------


#SAVE HISTORY FOR PLOTTING AT THE END
epoch=1; epochs=[]; loss_train=[];  loss_val=[]

# ---- PARADIGM SELECTION ------------------------------------------------------
PARADIGM='batch'
#PARADIGM='mini-batch'
#PARADIGM='stochastic'
# ------------------------------------------------------------------------------


#READ FILE
column_names = DATA_KEYS
input1 = pd.read_csv('mpg.csv', header = 0, names=column_names,
                          na_values='?', comment='\t',
                          sep=',', skipinitialspace=True)#.dropna()

# # EXPLORE DATA
# get_pd_info(input1)
# pd_general_plots(input1)

# #SELECT COLUMNS TO UDE AS VARIABLES
# x_col = [1,2,3,4,5];
# y_col=[0]; xy_col=x_col+y_col
# X_KEYS = index_to_keys(input1, x_col) #dependent var
# Y_KEYS = index_to_keys(input1, y_col) #independent var

#CONVERT DICTIONARY INPUT AND OUTPUT MATRICES #SIMILAR TO PANDAS DF   
X=[]; Y=[]
for key in input1.keys():
	if(key in X_KEYS): X.append(input1[key])
	if(key in Y_KEYS): Y.append(input1[key])

# CONVERT SELECT DF TO NP
x = input1[X_KEYS].to_numpy()
y = input1[Y_KEYS].to_numpy()

# REMOVE NAN IF PRESENT
xtmp=[]; ytmp=[]
for i in range(0, len(x)):
	if(not 'nan' in str(x[i])):
		xtmp.append(x[i])
		ytmp.append(y[i])

#MAKE ROWS=SAMPLE DIMENSION (TRANSPOSE)
X=np.array(xtmp) #X))
Y=np.array(ytmp) #Y))
NFIT=X.shape[1]+1 # plus one for the bias term
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
#SIGMOID
#MODEL
#------------------------
def S(x): 
	return 1.0/(1.0+np.exp(-x))
if(model_type=="logistic"): 
	Y=S(Y)
def model(x,p):
	linear=p[0]+np.matmul(x,p[1:].reshape(NFIT-1,1))   
	if(model_type=="linear"):   
		return linear 
	if(model_type=="logistic"): 
		return S(linear)

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
def optimizer(f,xi, algo='GD', LR=0.001):
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
				if(iteration%250==0): # use batch 1
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
            
# 		# EXTRA CREDIT
# 		if(algo=="RMSprop"): print("REQUESTED ALGORITHM (RMSprop) NOT CODED"); exit()
# 		if(algo=="ADAM"): print("REQUESTED ALGORITHM (ADAM) NOT CODED"); exit()
# 		if(algo=="Nelder-Mead"): print("REQUESTED (Nelder-Mead) ALGORITHM NOT CODED"); exit()

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

# # generate plots
# #FUNCTION PLOTS
# if(IPLOT):
# 	fig, ax = plt.subplots()
# 	ax.plot(X[train_idx], Y[train_idx], 'o', label='Training set')
# 	ax.plot(X[test_idx], Y[test_idx], 'x', label='Test set')
# 	ax.plot(X[val_idx], Y[val_idx], '*', label='Validation set')
# 	ax.plot(X[train_idx], YPRED_T, '.', c='red', label='Model')
# 	plt.xlabel('x', fontsize=18)
# 	plt.ylabel('y', fontsize=18)
# 	plt.legend()
# 	plt.show()

# #PARITY PLOTS
# if(IPLOT):
# 	fig, ax = plt.subplots()
# 	ax.plot(model(X[train_idx],popt), Y[train_idx], 'o', label='Training set')
# 	ax.plot(model(X[val_idx],popt), Y[val_idx], 'o', label='Validation set')
# 	plt.xlabel('y predicted', fontsize=18)
# 	plt.ylabel('y data', fontsize=18)
# 	plt.legend()
# 	plt.show()

#MONITOR TRAINING AND VALIDATION LOSS  
if(IPLOT):
	fig, ax = plt.subplots()
	ax.plot(epochs, loss_train, 'o', label='Training loss')
	ax.plot(epochs, loss_val, 'o', label='Validation loss')
	plt.xlabel('epochs', fontsize=18)
	plt.ylabel('loss', fontsize=18)
	plt.legend()
	plt.show()
    
    
#FUNCTION PLOTS
def plot_1(xcol=1, xla='x', yla='y'):
    fig, ax = plt.subplots()
    ax.plot(X[train_idx][:,xcol],Y[train_idx], 'o', label='Training')
    ax.plot(X[val_idx][:,xcol],Y[val_idx], 'x', label='Validation')
    ax.plot(X[test_idx][:,xcol],Y[test_idx], '*', label='Test')
    ax.plot(X[train_idx][:,xcol],YPRED_T, '.', label='Model')
    plt.xlabel(xla, fontsize=18)
    plt.ylabel(yla, fontsize=18)
    plt.legend()
    plt.show()

    

#UNNORMALIZE RELEVANT ARRAYS
X=XSTD*X+XMEAN;
Y=YSTD*Y+YMEAN;
YPRED_T=YSTD*YPRED_T+YMEAN;
YPRED_V=YSTD*YPRED_V+YMEAN;
YPRED_TEST=YSTD*YPRED_TEST+YMEAN;

print("here")
i=0
for key in X_KEYS:
    plot_1(i, xla=key, yla=Y_KEYS[0])
    i=i+1
#plot_2()

# if(IPLOT):   
#     i=0
#     for key in X_KEYS:
#         plot_1(i, xla=key, yla=Y_KEYS[0])
#         i=i+1

fig, ax = plt.subplots()
ax.plot(Y[train_idx],YPRED_T, 'o', label='Training')
ax.plot(Y[val_idx],YPRED_V, 'x', label='Validation')
ax.plot(Y[test_idx],YPRED_TEST, '*', label='Test')
plt.xlabel("y_data", fontsize=18)
plt.ylabel("y_predict", fontsize=18)
plt.legend()
plt.show()


# In[ ]:




