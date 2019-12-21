
# coding: utf-8

# # 1 Matrix operations
# 
# ## 1.1 Create a 4*4 identity matrix

# In[5]:


#This project is designed to get familiar with python list and linear algebra
#You cannot use import any library yourself, especially numpy

A = [[1,2,3], 
     [2,3,3], 
     [1,2,5]]

B = [[1,2,3,5], 
     [2,3,3,5], 
     [1,2,5,1]]

#TODO create a 4*4 identity matrix 
def eye(size):
    # Create size x size identity matrix
    result = []
    for i in range(size):
        result.append([0]*size)
    for i in range(size):
        result[i][i] = 1
    return result


I = eye(4)


# ## 1.2 get the width and height of a matrix. 

# In[6]:


#TODO Get the height and weight of a matrix.
def shape(M):
    if type(M) is None:
        return 0, 0
    else:
        row = len(M)
        if type(M[0]) is (int or float):
            return row, 1
        else:
            col = len(M[0])
            return row, col


# In[7]:


# run following code to test your shape function
get_ipython().magic('run -i -e test.py LinearRegressionTestCase.test_shape')


# ## 1.3 round all elements in M to certain decimal points

# In[8]:


# TODO in-place operation, no return value
# TODO round all elements in M to decPts
def matxRound(M, decPts=4):
    row, col = shape(M)
    for i in range(row):
        for j in range(col):
            M[i][j] = round(float(M[i][j]), decPts)
    pass


# In[9]:


# run following code to test your matxRound function
get_ipython().magic('run -i -e test.py LinearRegressionTestCase.test_matxRound')


# ## 1.4 compute transpose of M

# In[10]:


#TODO compute transpose of M
def transpose(M):
    return [list(i) for i in map(list, zip(*M))]


# In[11]:


# run following code to test your transpose function
get_ipython().magic('run -i -e test.py LinearRegressionTestCase.test_transpose')


# ## 1.5 compute AB. return None if the dimensions don't match

# In[12]:


#TODO compute matrix multiplication AB, return None if the dimensions don't match
def matxMultiply(A, B):
    rowA, colA = shape(A)
    rowB, colB = shape(B)
    if (colA != rowB):
        raise ValueError()
        return None
    else:
        result = [[sum((a*b) for a,b in zip(row,col)) for col in zip(*B)] for row in A]
        return result


# In[13]:


# run following code to test your matxMultiply function
get_ipython().magic('run -i -e test.py LinearRegressionTestCase.test_matxMultiply')


# ---
# 
# # 2 Gaussian Jordan Elimination
# 
# ## 2.1 Compute augmented Matrix 
# 
# $ A = \begin{bmatrix}
#     a_{11}    & a_{12} & ... & a_{1n}\\
#     a_{21}    & a_{22} & ... & a_{2n}\\
#     a_{31}    & a_{22} & ... & a_{3n}\\
#     ...    & ... & ... & ...\\
#     a_{n1}    & a_{n2} & ... & a_{nn}\\
# \end{bmatrix} , b = \begin{bmatrix}
#     b_{1}  \\
#     b_{2}  \\
#     b_{3}  \\
#     ...    \\
#     b_{n}  \\
# \end{bmatrix}$
# 
# Return $ Ab = \begin{bmatrix}
#     a_{11}    & a_{12} & ... & a_{1n} & b_{1}\\
#     a_{21}    & a_{22} & ... & a_{2n} & b_{2}\\
#     a_{31}    & a_{22} & ... & a_{3n} & b_{3}\\
#     ...    & ... & ... & ...& ...\\
#     a_{n1}    & a_{n2} & ... & a_{nn} & b_{n} \end{bmatrix}$

# In[14]:


#TODO construct the augment matrix of matrix A and column vector b, assuming A and b have same number of rows
import copy
def augmentMatrix(A, b):
    result = copy.deepcopy(A)
    for i in range(shape(A)[0]):
        result[i].append(b[i][0])
    return result


# In[15]:


# run following code to test your augmentMatrix function
get_ipython().magic('run -i -e test.py LinearRegressionTestCase.test_augmentMatrix')


# ## 2.2 Basic row operations
# - exchange two rows
# - scale a row
# - add a scaled row to another

# In[16]:


# TODO r1 <---> r2
# TODO in-place operation, no return value
def swapRows(M, r1, r2):
    M[r1], M[r2] = M[r2], M[r1]
    pass


# In[17]:


# run following code to test your swapRows function
get_ipython().magic('run -i -e test.py LinearRegressionTestCase.test_swapRows')


# In[24]:


# TODO r1 <--- r1 * scale
# TODO in-place operation, no return value
def scaleRow(M, r, scale):
    if scale == 0:
        raise ValueError()
    M[r] = [i * scale for i in M[r]]
    pass


# In[25]:


# run following code to test your scaleRow function
get_ipython().magic('run -i -e test.py LinearRegressionTestCase.test_scaleRow')


# In[26]:


# TODO r1 <--- r1 + r2*scale
# TODO in-place operation, no return value
def addScaledRow(M, r1, r2, scale):
    M[r1] = [i+j*scale for i,j in zip(M[r1],M[r2])]
    pass


# In[27]:


# run following code to test your addScaledRow function
get_ipython().magic('run -i -e test.py LinearRegressionTestCase.test_addScaledRow')


# ## 2.3  Gauss-jordan method to solve Ax = b
# 
# ### Hint：
# 
# Step 1: Check if A and b have same number of rows
# Step 2: Construct augmented matrix Ab
# 
# Step 3: Column by column, transform Ab to reduced row echelon form [wiki link](https://en.wikipedia.org/wiki/Row_echelon_form#Reduced_row_echelon_form)
#     
#     for every column of Ab (except the last one)
#         column c is the current column
#         Find in column c, at diagonal and under diagonal (row c ~ N) the maximum absolute value
#         If the maximum absolute value is 0
#             then A is singular, return None （Prove this proposition in Question 2.4）
#         else
#             Apply row operation 1, swap the row of maximum with the row of diagonal element (row c)
#             Apply row operation 2, scale the diagonal element of column c to 1
#             Apply row operation 3 mutiple time, eliminate every other element in column c
#             
# Step 4: return the last column of Ab
# 
# ### Remark：
# We don't use the standard algorithm first transfering Ab to row echelon form and then to reduced row echelon form.  Instead, we arrives directly at reduced row echelon form. If you are familiar with the stardard way, try prove to yourself that they are equivalent. 

# In[30]:


#TODO implement gaussian jordan method to solve Ax = b

""" Gauss-jordan method to solve x such that Ax = b.
        A: square matrix, list of lists
        b: column vector, list of lists
        decPts: degree of rounding, default value 4
        epsilon: threshold for zero, default value 1.0e-16
        
    return x such that Ax = b, list of lists 
    return None if A and b have same height
    return None if A is (almost) singular
"""
def gj_Solve(A, b, decPts=4, epsilon=1.0e-16):
    rowA, colA = shape(A)
    rowB, colB = shape(b)
    if colA != rowB:
        return None
    
    Ab = augmentMatrix(A,b)
    matxRound(Ab)
    
    for c in range(colA):
        absoluteMaxValue, maxRowIndex = abs(Ab[c][c]), c
        for i in range(c, rowA):
            if abs(Ab[i][c]) > absoluteMaxValue:
                absoluteMaxValue = abs(Ab[i][c])
                maxRowIndex = i

        if abs(absoluteMaxValue - 0) < epsilon :
            return None

        swapRows(Ab, c, maxRowIndex)
        scaleRow(Ab, c, 1/Ab[c][c])
        for j in range(rowA):
            if j != c:
                addScaledRow(Ab, j, c, -Ab[j][c])

    return [[Ab[r][rowB]] for r in range(rowA)]


# In[31]:


# run following code to test your addScaledRow function
get_ipython().magic('run -i -e test.py LinearRegressionTestCase.test_gj_Solve')


# ## 2.4 Prove the following proposition:
# 
# **If square matrix A can be divided into four parts: ** 
# 
# $ A = \begin{bmatrix}
#     I    & X \\
#     Z    & Y \\
# \end{bmatrix} $, where I is the identity matrix, Z is all zero and the first column of Y is all zero, 
# 
# **then A is singular.**
# 
# Hint: There are mutiple ways to prove this problem.  
# - consider the rank of Y and A
# - consider the determinate of Y and A 
# - consider certain column is the linear combination of other columns

# TODO Please use latex （refering to the latex in problem may help）
# 
# TODO Proof：
# 
# **Consider matrix Y:** 
# 
# If the first column of Y is zero, then Y is not full rank, because any row in Y can be expressed as linear combination of other rows.
# 
# **Define the lower part of matrix A to be matrix B:** 
# 
# $ B = \begin{bmatrix}
#     Z    & Y \\
# \end{bmatrix} $
# 
# Because Z is zero, the rank of B is same as Y. Therefore, B is not full rank.
# 
# It means that, in the lower part of A, any row can be expressed as linear combination of other rows, so matrix A is not full rank.
# 
# **Then A is singular.**
# 

# ---
# 
# # 3 Linear Regression: 
# 
# ## 3.1 Compute the gradient of loss function with respect to parameters 
# ## (Choose one between two 3.1 questions)
# 
# We define loss funtion $E$ as 
# $$
# E(m, b) = \sum_{i=1}^{n}{(y_i - mx_i - b)^2}
# $$
# and we define vertex $Y$, matrix $X$ and vertex $h$ :
# $$
# Y =  \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     ... \\
#     y_n
# \end{bmatrix}
# ,
# X =  \begin{bmatrix}
#     x_1 & 1 \\
#     x_2 & 1\\
#     ... & ...\\
#     x_n & 1 \\
# \end{bmatrix},
# h =  \begin{bmatrix}
#     m \\
#     b \\
# \end{bmatrix}
# $$
# 
# 
# Proves that 
# $$
# \frac{\partial E}{\partial m} = \sum_{i=1}^{n}{-2x_i(y_i - mx_i - b)}
# $$
# 
# $$
# \frac{\partial E}{\partial b} = \sum_{i=1}^{n}{-2(y_i - mx_i - b)}
# $$
# 
# $$
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix} = 2X^TXh - 2X^TY
# $$

# TODO Please use latex （refering to the latex in problem may help）
# 
# TODO Proof：
# 
# **Take partial derivative of $E$:**
# 
# With respect to $m$, treat $b$ as constant and apply chain rule on the squared middle term:
# 
# $$
# \frac{\partial E}{\partial m} = \sum_{i=1}^{n}{-2x_i(y_i - mx_i - b)}
# $$
# 
# With respect to $b$, treat $m$ as constant and apply chain rule to the squared b term:
# 
# 
# $$
# \frac{\partial E}{\partial b} = \sum_{i=1}^{n}{-2(y_i - mx_i - b)}
# $$
# 
# ** Evaluate the right hand side of the equation:**
# 
# $$X^TX = \begin{bmatrix}
#     \sum_{i=1}^{n}{(x_i)^2} & \sum_{i=1}^{n}{x_i} \\ 
#     \sum_{i=1}^{n}{x_i}     &          n          \\
#     \end{bmatrix}$$
#     
# $$2X^TXh = \begin{bmatrix}
#     \sum_{i=1}^{n}2x_i({mx_i} + b) \\ 
#     \sum_{i=1}^{n}2(mx_i + b) \\
#     \end{bmatrix}$$
#     
# $$2X^TY = \begin{bmatrix}
#     \sum_{i=1}^{n}2x_iy_i \\ 
#     \sum_{i=1}^{n}2y_i \\
#     \end{bmatrix}$$
# 
# $$2X^TXh - 2X^TY = \begin{bmatrix}
#     \sum_{i=1}^{n}2x_i(-y_i + mx_i + b) \\ 
#     \sum_{i=1}^{n}2(-y_i + mx_i + b) \\
#     \end{bmatrix} = \begin{bmatrix}
#     \sum_{i=1}^{n}-2x_i(y_i - mx_i - b) \\ 
#     \sum_{i=1}^{n}-2(y_i - mx_i - b) \\
#     \end{bmatrix}$$
#     
# ** Left hand side equals to right hand side, therefore proved: **
# 
# $$
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix} = 2X^TXh - 2X^TY
# $$

# ## 3.1 Compute the gradient of loss function with respect to parameters 
# ## (Choose one between two 3.1 questions)
# We define loss funtion $E$ as 
# $$
# E(m, b) = \sum_{i=1}^{n}{(y_i - mx_i - b)^2}
# $$
# and we define vertex $Y$, matrix $X$ and vertex $h$ :
# $$
# Y =  \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     ... \\
#     y_n
# \end{bmatrix}
# ,
# X =  \begin{bmatrix}
#     x_1 & 1 \\
#     x_2 & 1\\
#     ... & ...\\
#     x_n & 1 \\
# \end{bmatrix},
# h =  \begin{bmatrix}
#     m \\
#     b \\
# \end{bmatrix}
# $$
# 
# Proves that 
# $$
# E = Y^TY -2(Xh)^TY + (Xh)^TXh
# $$
# 
# $$
# \frac{\partial E}{\partial h} = 2X^TXh - 2X^TY
# $$

# TODO Please use latex （refering to the latex in problem may help）
# 
# TODO Proof：

# ## 3.2  Linear Regression
# ### Solve equation $X^TXh = X^TY $ to compute the best parameter for linear regression.

# In[34]:


#TODO implement linear regression 
'''
points: list of (x,y) tuple
return m and b
'''
def linearRegression(points):
    X = [[] for i in range(len(points))]
    Y = [[] for i in range(len(points))]
    b = []
    for i in range(len(points)):
        X[i].append(points[i][0])
        X[i].append(1.0)
        Y[i].append(points[i][1])
    A = matxMultiply(transpose(X), X)
    tmp = matxMultiply(transpose(X), Y)
    row, col = shape(tmp)
    for i in range(row):
        for j in range(col):
            b.append([tmp[i][j]])
    return gj_Solve(A, b)


# ## 3.3 Test your linear regression implementation

# In[35]:


#TODO Construct the linear function
def linearParam():
    # Create the linear function parameters
    m = round(random() * 10 - 5, 4)
    b = round(random() * 10 + 5, 4)
    return m, b

#TODO Construct points with gaussian noise
from random import *

def randomPoints(m, b, num=100):
    # Construct points with gaussian noise based on input parameters
    mu = 0
    sigma = 0.1
    points = [[None, None] for i in range(num)]
    x = sample(range(0, 100), num)
    for i in range(num):
        points[i][0] = x[i]
        points[i][1] = m * x[i] + b + gauss(mu, sigma)
    return points

#TODO Compute m and b and compare with ground truth
m_true, b_true = linearParam()
data = randomPoints(m_true, b_true, 100)
m_lr, b_lr = linearRegression(data)
print("The true linear parameters: m = [{}], b = [{}]".format(m_true, b_true))
print("The linear regression result: m = {}, b = {}".format(m_lr, b_lr))

