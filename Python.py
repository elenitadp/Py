#!/usr/bin/env python
# coding: utf-8

# <h1 style="font-size:3rem;color:blue;">Linear algebra with complex scalar</h1>

# <h2 style="font-size:2rem;color:orange;">un qubit in superpozitie - doar pentru exercitiu</h2>

# In[14]:


import numpy as np
import math

#sqare root of 2
DivSqrt2 = 1.0/math.sqrt(2)
print(DivSqrt2)

#identity 2x2 matrix
I = np.array([[1, 0], [0, 1]]) 
#Hadamard 2x2 matrix
H = np.array([[DivSqrt2, DivSqrt2], [DivSqrt2, -DivSqrt2]]) 

#qbit |0>
a1 = 1.0
b1 = 0.0
a2 = 0.0
b2 = 0.0
c1=complex(a1,b1)
c2=complex(a2,b2)

q = np.array([c1,c2])

print('superposition of |0>')
print(H.dot(q))

#qbit |1>
a1 = 0.0
b1 = 0.0
a2 = 1.0
b2 = 0.0
c1=complex(a1,b1)
c2=complex(a2,b2)

q = np.array([c1,c2])

print('superposition of |1>')
print(H.dot(q))



# <h2 style="font-size:2rem;color:orange;">un qubit in superpozitie pe repeat - doar pentru exercitiu</h2>

# In[24]:


import numpy as np
import math

#sqare root of 2
DivSqrt2 = 1.0/math.sqrt(2)
print(DivSqrt2)

#Hadamard 2x2 matrix
H = np.array([[DivSqrt2, DivSqrt2], [DivSqrt2, -DivSqrt2]]) 

#qbit |0>
a1 = 1.0
b1 = 0.0
a2 = 0.0
b2 = 0.0
c1=complex(a1,b1)
c2=complex(a2,b2)
q = np.array([c1,c2])

counter= int(input("Enter the number of steps: "))
print('superposition of |0> after' , counter , 'Hadamard apply')
for i in range(counter):
    q = H.dot(q)
    print(q)


# <h2 style="font-size:2rem;color:orange;">Generate Hadamard matrix of given order M - doar pentru exercitiu</h2>

# In[ ]:


# Python3 code to implement the approach
def generate(M):
 
    # Computing n = 2^M
    n = 2 ** M
 
    # Initializing a matrix of order n
    hadamard = [ [0] * n for _ in range(n)]
     
    # Initializing the 0th column and
    # 0th row element as 1
    hadamard[0][0] = 1
     
    k = 1
    while (k  < n):
 
        # Loop to copy elements to
        # other quarters of the matrix
        for i in range(k):
            for j in range(k):
                hadamard[i + k][j] = hadamard[i][j];
                hadamard[i][j + k] = hadamard[i][j];
                hadamard[i + k][j + k] = -hadamard[i][j];
        k *= 2
 
    # Displaying the final hadamard matrix
    for i in range(n):
        for j in range(n):
            print(hadamard[i][j], end = " ")
        print()
 
# Driver code
M = 2;
 
# Function call
generate(M);
 
# This code is contributed by phasing17 from GeeksforGeeks
#https://www.geeksforgeeks.org/generate-hadamard-matrix-of-given-order/


# <h2 style="font-size:2rem;color:orange;">GC(generalized coin) si matricea A - doar pentru exercitiu</h2>

# In[2]:


import numpy as np
import math
import random

#square root of 2
DivSqrt2 = 1.0/math.sqrt(2)

#Hadamard 2x2 matrix
H = np.array([[DivSqrt2, DivSqrt2], [DivSqrt2, -DivSqrt2]]) 

#Generalized Coin 4x4 matrix (according to J.Kempe[19])
gammaForCoinFromPaper = np.array([[-0.5,0.5,0.5,0.5],[0.5,-0.5,0.5,0.5],[0.5,0.5,-0.5,0.5],[0.5,0.5,0.5,-0.5]])

#A matrix 4x4 matrix theta0 theta1
alpha=5
beta=5
theta0 = random.betavariate(alpha, beta)*2 * math.pi
theta1 = random.betavariate(alpha, beta)*2 * math.pi
A = np.array([[math.cos(theta0),-math.sin(theta0),0,0],
              [math.sin(theta0),math.cos(theta0),0,0],
              [0,0,math.cos(theta1),-math.sin(theta1)],
              [0,0,math.sin(theta1),math.cos(theta1)]])

#gamma 4x4 matrix

gamma = A.dot(gammaForCoinFromPaper)




print(gammaForCoinFromPaper)
print(A)
print(gamma)
    
  


# <h2 style="font-size:2rem;color:orange;">Quantum Discrete Oriented Random Walker on a Circle - functional</h2>

# In[12]:


import numpy as np
import math
import random

#square root of 2
DivSqrt2 = 1.0/math.sqrt(2)

#Hadamard 2x2 matrix
H = np.array([[DivSqrt2, DivSqrt2], [DivSqrt2, -DivSqrt2]]) 

#M+ matrix 8x8 matrix move forward
Mpls = np.array([[0,0,0,0,0,0,0,1],
               [1,0,0,0,0,0,0,0],
               [0,1,0,0,0,0,0,0],
               [0,0,1,0,0,0,0,0],
               [0,0,0,1,0,0,0,0],
               [0,0,0,0,1,0,0,0],
               [0,0,0,0,0,1,0,0],
               [0,0,0,0,0,0,1,0]])

#M- matrix 8x8 matrix move backward
Mmin = np.array([[0,1,0,0,0,0,0,0],
               [0,0,1,0,0,0,0,0],
               [0,0,0,1,0,0,0,0],
               [0,0,0,0,1,0,0,0],
               [0,0,0,0,0,1,0,0],
               [0,0,0,0,0,0,1,0],
               [0,0,0,0,0,0,0,1],
               [1,0,0,0,0,0,0,0]])

#position vector
#spatiul Hilbert de dimensiune 2x8 (2 exp 1 Hcoin  x 2 exp 3 H pozitie pe axa liniara cu 8 puncte)
#aruncarea de moneda Hadamard influenteaza diferit |0> si respectiv |1> si deci coeficientii de pozitie rezultati de aici

dim=8
#pentru orientare |0>
re0 = np.zeros((dim))
im0 = np.zeros((dim))
poz0 = np.empty((dim),complex)

#pentru orientare |1>
re1 = np.zeros((dim))
im1 = np.zeros((dim))
poz1 = np.empty((dim),complex)

#initial position
# moneda (1,0)=alpha0|0>+alpha1|1>, pozitia (1,0,0,0,0,0,0)
re0[0] = 1.

for i in range(dim):
    poz0[i]=complex(re0[i],im0[i])
    poz1[i]=complex(re1[i],im1[i])

print(poz0)
print(poz1)

counter= int(input("Enter the number of steps: "))
#after first toss
alpha0 = DivSqrt2
alpha1 = DivSqrt2
print("steps ",counter)
for i in range(int(counter)):
    auxpoz0 = poz0
    auxpoz1 = poz1
    poz0 = alpha0*Mpls.dot(auxpoz0)+alpha1*Mpls.dot(auxpoz1)
    poz1 = alpha0*Mmin.dot(auxpoz0)-alpha1*Mmin.dot(auxpoz1)
print(poz0)
print(poz1)
        
for i in range(dim):
    print("Probability to be in pozition ",i, " ", poz0[i].real*poz0[i].real+poz0[i].imag*poz0[i].imag+poz1[i].real*poz1[i].real+poz1[i].imag*poz1[i].imag)
    

probab=0
for i in range(dim):
    probab=probab+poz0[i].real*poz0[i].real+poz0[i].imag*poz0[i].imag+poz1[i].real*poz1[i].real+poz1[i].imag*poz1[i].imag
print(probab)


# <h2 style="font-size:2rem;color:orange;">Inmultire matrice, cu repozitionare</h2>
# 

# In[57]:


#M+ matrix 8x8 matrix move forward (positive)
Mmin = np.array([[0,1,0,0,0,0,0,0],
                  [0,0,1,0,0,0,0,0],
                  [0,0,0,1,0,0,0,0],
                  [0,0,0,0,1,0,0,0],
                  [0,0,0,0,0,1,0,0],
                  [0,0,0,0,0,0,1,0],
                  [0,0,0,0,0,0,0,1],
                  [1,0,0,0,0,0,0,0]])

#M- matrix 8x8 matrix move backward (negative)
Mpls = np.array([[0,0,0,0,0,0,0,1],
                  [1,0,0,0,0,0,0,0],
                  [0,1,0,0,0,0,0,0],
                  [0,0,1,0,0,0,0,0],
                  [0,0,0,1,0,0,0,0],
                  [0,0,0,0,1,0,0,0],
                  [0,0,0,0,0,1,0,0],
                  [0,0,0,0,0,0,1,0]])


#position vector
#spatiul Hilbert de dimensiune 2x2x8x8 (2 exp 2 GC  x 2 exp 3 H pozitie pe axa lini1ara X cu 8 puncte x 2 exp 3 H pozitie pe axa liniara Y cu 8 puncte)
#aruncarea de moneda Hadamard influenteaza diferit |00>, |01>, |10> si respectiv |1> si deci si pe coeficientii de pozitie rezultati de aici

dim=8
#cu efect pe X
#pentru orientare |00>
re00 = np.zeros((dim,dim))
im00 = np.zeros((dim,dim))
poz00 = np.empty((dim,dim),complex)
auxpoz00 = np.empty((dim,dim),complex)

#pentru orientare |01>
re01 = np.zeros((dim,dim))
im01 = np.zeros((dim,dim))
poz01 = np.empty((dim,dim),complex)
auxpoz01 = np.empty((dim,dim),complex)

#cu efect pe Y
#pentru orientare |10>
re10 = np.zeros((dim,dim))
im10 = np.zeros((dim,dim))
poz10 = np.empty((dim,dim),complex)
auxpoz10 = np.empty((dim,dim),complex)

#pentru orientare |11>
re11 = np.zeros((dim,dim))
im11 = np.zeros((dim,dim))
poz11 = np.empty((dim,dim),complex)
auxpoz11 = np.empty((dim,dim),complex)

#initial pozition
# moneda (1,0,0,0)
# pozitia X (DivSqrt2+jDivSqrt2,0,0,0,0,0,0)
# pozitia Y (DivSqrt2+jDivSqrt2,0,0,0,0,0,0)

re00[0][0] = 0.5
im00[0][0] = 0. 
re01[0][0] = 0.5
im01[0][0] = 0. 
re10[0][0] = 0.5
im10[0][0] = 0.  
re11[0][0] = 0.5
im11[0][0] = 0. 

for x in range(dim):
        for y in range(dim):
            poz00[x][y]=complex(re00[x][y],im00[x][y])
            poz01[x][y]=complex(re01[x][y],im01[x][y])
            poz10[x][y]=complex(re10[x][y],im10[x][y])
            poz11[x][y]=complex(re11[x][y],im11[x][y])
            
# move
def move():
    print("poz00")
    print(poz00)
    print("poz01")
    print(poz01)
    print("poz10")
    print(poz10)
    print("poz11")
    print(poz11)

    auxpoz00 = Mpls.dot(poz00)
    auxpoz00=auxpoz00.transpose()
    auxpoz01 = Mmin.dot(poz01.transpose())  
    auxpoz01=auxpoz01.transpose()
    auxpoz10 = Mpls.dot(poz10)
    auxpoz11 = Mmin.dot(poz11) 
    print("auxpoz00")
    print(auxpoz00)
    print("auxpoz01")
    print(auxpoz01)
    print("auxpoz10")
    print(auxpoz10)
    print("auxpoz11")
    print(auxpoz11)
   
move()


# <h2 style="font-size:2rem;color:orange;">Quantum Discrete Oriented Random Walker on a Sphere - in lucru</h2>
# 

# In[1]:


import numpy as np
import math
import random

#from qiskit import *

f = open("out.txt", "w")

#square root of 2
DivSqrt2 = 1.0/math.sqrt(2)

#Generalized Coin 4x4 matrix

#H = np.array([[0.5, 0.5, 0.5, 0.5], 
#              [0.5, -0.5, 0.5, -0.5], 
#              [0.5, 0.5, -0.5, -0.5], 
#              [0.5, -0.5, -0.5, 0.5]]) 
H = np.array([[-0.5,  0.5,  0.5,  0.5], 
              [ 0.5, -0.5,  0.5,  0.5], 
              [ 0.5,  0.5, -0.5,  0.5], 
              [ 0.5,  0.5,  0.5, -0.5]]) 

#M+ matrix 8x8 matrix move forward (positive)
Mpls = np.array([[0,0,0,0,0,0,0,1],
               [1,0,0,0,0,0,0,0],
               [0,1,0,0,0,0,0,0],
               [0,0,1,0,0,0,0,0],
               [0,0,0,1,0,0,0,0],
               [0,0,0,0,1,0,0,0],
               [0,0,0,0,0,1,0,0],
               [0,0,0,0,0,0,1,0]])

#M- matrix 8x8 matrix move backward (negative)
Mmin = np.array([[0,1,0,0,0,0,0,0],
               [0,0,1,0,0,0,0,0],
               [0,0,0,1,0,0,0,0],
               [0,0,0,0,1,0,0,0],
               [0,0,0,0,0,1,0,0],
               [0,0,0,0,0,0,1,0],
               [0,0,0,0,0,0,0,1],
               [1,0,0,0,0,0,0,0]])

#position vector
#spatiul Hilbert de dimensiune 2x2x8x8 (2 exp 2 GC  x 2 exp 3 H pozitie pe axa lini1ara X cu 8 puncte x 2 exp 3 H pozitie pe axa liniara Y cu 8 puncte)
#aruncarea de moneda Hadamard influenteaza diferit |00>, |01>, |10> si respectiv |1> si deci si pe coeficientii de pozitie rezultati de aici

dim=8
#cu efect pe X
#pentru orientare |00>
re00 = np.zeros((dim,dim))
im00 = np.zeros((dim,dim))
poz00 = np.empty((dim,dim),complex)
auxpoz00 = np.empty((dim,dim),complex)

#pentru orientare |01>
re01 = np.zeros((dim,dim))
im01 = np.zeros((dim,dim))
poz01 = np.empty((dim,dim),complex)
auxpoz01 = np.empty((dim,dim),complex)

#cu efect pe Y
#pentru orientare |10>
re10 = np.zeros((dim,dim))
im10 = np.zeros((dim,dim))
poz10 = np.empty((dim,dim),complex)
auxpoz10 = np.empty((dim,dim),complex)

#pentru orientare |11>
re11 = np.zeros((dim,dim))
im11 = np.zeros((dim,dim))
poz11 = np.empty((dim,dim),complex)
auxpoz11 = np.empty((dim,dim),complex)

#initial pozition
# moneda (1,0,0,0)
# pozitia X (DivSqrt2+jDivSqrt2,0,0,0,0,0,0)
# pozitia Y (DivSqrt2+jDivSqrt2,0,0,0,0,0,0)

re00[0][0] = 0.5
im00[0][0] = 0. 
re01[0][0] = 0.5
im01[0][0] = 0. 
re10[0][0] = 0.5
im10[0][0] = 0.  
re11[0][0] = 0.5
im11[0][0] = 0. 

for x in range(dim):
        for y in range(dim):
            poz00[x][y]=complex(re00[x][y],im00[x][y])
            poz01[x][y]=complex(re01[x][y],im01[x][y])
            poz10[x][y]=complex(re10[x][y],im10[x][y])
            poz11[x][y]=complex(re11[x][y],im11[x][y])


#functii
            
def poz_copy(poz00,poz01,poz10,poz11,auxpoz00,auxpoz01,auxpoz10,auxpoz11):
        
#    print("auxpoz00")
#    print(auxpoz00)
#    print("auxpoz01")
#    print(auxpoz01)
#    print("auxpoz10")
#    print(auxpoz10)
#    print("auxpoz11")
#    print(auxpoz11)
    
    for x in range(dim):
        for y in range(dim):
            poz00[x][y]=auxpoz00[x][y]
            poz01[x][y]=auxpoz01[x][y]
            poz10[x][y]=auxpoz10[x][y]
            poz11[x][y]=auxpoz11[x][y]       
    
#    print("poz00")
#    print(poz00)
#    print("poz01")
#    print(poz01)
#    print("poz10")
#    print(poz10)
#    print("poz11")
#    print(poz11)
    

            
#toss
def toss(poz00,poz01,poz10,poz11,auxpoz00,auxpoz01,auxpoz10,auxpoz11):
    for x in range(dim):
        for y in range(dim):
                auxpoz00[x][y] = H[0][0] * poz00[x][y] + H[1][0] * poz01[x][y] + H[2][0] * poz10[x][y] + H[3][0] * poz11[x][y]
                auxpoz01[x][y] = H[0][1] * poz00[x][y] + H[1][1] * poz01[x][y] + H[2][1] * poz10[x][y] + H[3][1] * poz11[x][y]
                auxpoz10[x][y] = H[0][2] * poz00[x][y] + H[1][2] * poz01[x][y] + H[2][2] * poz10[x][y] + H[3][2] * poz11[x][y]
                auxpoz11[x][y] = H[0][3] * poz00[x][y] + H[1][3] * poz01[x][y] + H[2][3] * poz10[x][y] + H[3][3] * poz11[x][y]
    poz_copy(poz00,poz01,poz10,poz11,auxpoz00,auxpoz01,auxpoz10,auxpoz11)
  

#move
def move(poz00,poz01,poz10,poz11,auxpoz00,auxpoz01,auxpoz10,auxpox11):
    auxpoz00 = Mpls.dot(poz00)
    auxpoz00=auxpoz00.transpose()
    auxpoz01 = Mmin.dot(poz01.transpose())  
    auxpoz01=auxpoz01.transpose()
    auxpoz10 = Mpls.dot(poz10)
    auxpoz11 = Mmin.dot(poz11) 

    poz_copy(poz00,poz01,poz10,poz11,auxpoz00,auxpoz01,auxpoz10,auxpoz11)  



counter= int(input("Enter the number of steps: "))

for step in range(counter):
    toss(poz00,poz01,poz10,poz11,auxpoz00,auxpoz01,auxpoz10,auxpoz11)
    move(poz00,poz01,poz10,poz11,auxpoz00,auxpoz01,auxpoz10,auxpoz11)


probab00 = np.zeros((dim,dim))
probab01 = np.zeros((dim,dim))
probab10 = np.zeros((dim,dim))
probab11 = np.zeros((dim,dim))
tprobab = np.zeros((dim,dim))
totprobab=0

for x in range(dim):
    for y in range(dim):         
            probab00[x][y]=probab00[x][y]+poz00[x][y].real*poz00[x,y].real+poz00[x,y].imag*poz00[x,y].imag
            probab01[x][y]=probab01[x][y]+poz01[x][y].real*poz01[x,y].real+poz01[x,y].imag*poz01[x,y].imag
            probab10[x][y]=probab10[x][y]+poz10[x][y].real*poz10[x,y].real+poz10[x,y].imag*poz10[x,y].imag
            probab11[x][y]=probab11[x][y]+poz11[x][y].real*poz11[x,y].real+poz11[x,y].imag*poz11[x,y].imag
            tprobab[x][y]=probab00[x][y]+probab01[x][y]+probab10[x][y]+probab11[x][y]
            totprobab=totprobab+probab00[x][y]+probab01[x,y]+probab10[x,y]+probab11[x,y]
            
print("pentru orientarea 00 - Est")
print(probab00)
print("pentru orientarea 01 - West")
print(probab01)
print("pentru orientarea 10 - North")
print(probab10)
print("pentru orientarea 11 - South")
print(probab11)

print("probabilitatea de pozitie")
print(tprobab)
    
            
print(totprobab)

f.close

#from qiskit.visualization import plot_state_city
#plot_state_city(tprobab)
    
    


# <h2 style="font-size:2rem;color:orange;">Quantum Discrete Oriented Random Walker on a Sphere - in a file</h2>
# 

# In[11]:


import numpy as np
import math
import random

#from qiskit import *

f = open("out.txt", "w")

#sqare root of 2
DivSqrt2 = 1.0/math.sqrt(2)

#Generalized Coin 4x4 matrix

#H = np.array([[0.5, 0.5, 0.5, 0.5], 
#              [0.5, -0.5, 0.5, -0.5], 
#              [0.5, 0.5, -0.5, -0.5], 
#              [0.5, -0.5, -0.5, 0.5]]) 
H = np.array([[-0.5,  0.5,  0.5,  0.5], 
              [ 0.5, -0.5,  0.5,  0.5], 
              [ 0.5,  0.5, -0.5,  0.5], 
              [ 0.5,  0.5,  0.5, -0.5]]) 

#M+ matrix 8x8 matrix move forward (positive)
Mpls = np.array([[0,0,0,0,0,0,0,1],
               [1,0,0,0,0,0,0,0],
               [0,1,0,0,0,0,0,0],
               [0,0,1,0,0,0,0,0],
               [0,0,0,1,0,0,0,0],
               [0,0,0,0,1,0,0,0],
               [0,0,0,0,0,1,0,0],
               [0,0,0,0,0,0,1,0]])

#M- matrix 8x8 matrix move backward (negative)
Mmin = np.array([[0,1,0,0,0,0,0,0],
               [0,0,1,0,0,0,0,0],
               [0,0,0,1,0,0,0,0],
               [0,0,0,0,1,0,0,0],
               [0,0,0,0,0,1,0,0],
               [0,0,0,0,0,0,1,0],
               [0,0,0,0,0,0,0,1],
               [1,0,0,0,0,0,0,0]])

#position vector
#spatiul Hilbert de dimensiune 2x2x8x8 (2 exp 2 GC  x 2 exp 3 H pozitie pe axa lini1ara X cu 8 puncte x 2 exp 3 H pozitie pe axa liniara Y cu 8 puncte)
#aruncarea de moneda Hadamard influenteaza diferit |00>, |01>, |10> si respectiv |1> si deci si pe coeficientii de pozitie rezultati de aici

dim=8
#cu efect pe X
#pentru orientare |00>
re00 = np.zeros((dim,dim))
im00 = np.zeros((dim,dim))
poz00 = np.empty((dim,dim),complex)
auxpoz00 = np.empty((dim,dim),complex)

#pentru orientare |01>
re01 = np.zeros((dim,dim))
im01 = np.zeros((dim,dim))
poz01 = np.empty((dim,dim),complex)
auxpoz01 = np.empty((dim,dim),complex)

#cu efect pe Y
#pentru orientare |10>
re10 = np.zeros((dim,dim))
im10 = np.zeros((dim,dim))
poz10 = np.empty((dim,dim),complex)
auxpoz10 = np.empty((dim,dim),complex)

#pentru orientare |11>
re11 = np.zeros((dim,dim))
im11 = np.zeros((dim,dim))
poz11 = np.empty((dim,dim),complex)
auxpoz11 = np.empty((dim,dim),complex)

#initial pozition
# moneda (1,0,1,0)
# pozitia X (DivSqrt2+jDivSqrt2,0,0,0,0,0,0)
# pozitia Y (DivSqrt2+jDivSqrt2,0,0,0,0,0,0)

re00[0][0] = 0.5
im00[0][0] = 0 
re01[0][0] = 0.5
im01[0][0] = 0. 
re10[0][0] = 0.5
im10[0][0] = 0  
re11[0][0] = 0.5
im11[0][0] = 0. 

for x in range(dim):
        for y in range(dim):
            poz00[x][y]=complex(re00[x][y],im00[x][y])
            poz01[x][y]=complex(re01[x][y],im01[x][y])
            poz10[x][y]=complex(re10[x][y],im10[x][y])
            poz11[x][y]=complex(re11[x][y],im11[x][y])


#functii
            
def poz_copy(poz00,poz01,poz10,poz11,auxpoz00,auxpoz01,auxpoz10,auxpoz11):
        
#    print("auxpoz00")
#    print(auxpoz00)
#    print("auxpoz01")
#    print(auxpoz01)
#    print("auxpoz10")
#    print(auxpoz10)
#    print("auxpoz11")
#    print(auxpoz11)
    
    for x in range(dim):
        for y in range(dim):
            poz00[x][y]=auxpoz00[x][y]
            poz01[x][y]=auxpoz01[x][y]
            poz10[x][y]=auxpoz10[x][y]
            poz11[x][y]=auxpoz11[x][y]       
    
#    print("poz00")
#    print(poz00)
#    print("poz01")
#    print(poz01)
#    print("poz10")
#    print(poz10)
#    print("poz11")
#    print(poz11)
    

            
#toss
def toss(poz00,poz01,poz10,poz11,auxpoz00,auxpoz01,auxpoz10,auxpoz11):
    for x in range(dim):
        for y in range(dim):
                auxpoz00[x][y] = H[0][0] * poz00[x][y] + H[1][0] * poz01[x][y] + H[2][0] * poz10[x][y] + H[3][0] * poz11[x][y]
                auxpoz01[x][y] = H[0][1] * poz00[x][y] + H[1][1] * poz01[x][y] + H[2][1] * poz10[x][y] + H[3][1] * poz11[x][y]
                auxpoz10[x][y] = H[0][2] * poz00[x][y] + H[1][2] * poz01[x][y] + H[2][2] * poz10[x][y] + H[3][2] * poz11[x][y]
                auxpoz11[x][y] = H[0][3] * poz00[x][y] + H[1][3] * poz01[x][y] + H[2][3] * poz10[x][y] + H[3][3] * poz11[x][y]
    poz_copy(poz00,poz01,poz10,poz11,auxpoz00,auxpoz01,auxpoz10,auxpoz11)
  

#move
def move(poz00,poz01,poz10,poz11,auxpoz00,auxpoz01,auxpoz10,auxpox11):
    auxpoz00 = Mpls.dot(poz00)
    auxpoz00=auxpoz00.transpose()
    auxpoz01 = Mmin.dot(poz01.transpose())  
    auxpoz01=auxpoz01.transpose()
    auxpoz10 = Mpls.dot(poz10)
    auxpoz11 = Mmin.dot(poz11) 

    poz_copy(poz00,poz01,poz10,poz11,auxpoz00,auxpoz01,auxpoz10,auxpoz11)  



counter= int(input("Enter the number of steps: "))

for step in range(counter):
    toss(poz00,poz01,poz10,poz11,auxpoz00,auxpoz01,auxpoz10,auxpoz11)
    move(poz00,poz01,poz10,poz11,auxpoz00,auxpoz01,auxpoz10,auxpoz11)


probab00 = np.zeros((dim,dim))
probab01 = np.zeros((dim,dim))
probab10 = np.zeros((dim,dim))
probab11 = np.zeros((dim,dim))
tprobab = np.zeros((dim,dim))
totprobab=0

for x in range(dim):
    for y in range(dim):         
            probab00[x][y]=probab00[x][y]+poz00[x][y].real*poz00[x,y].real+poz00[x,y].imag*poz00[x,y].imag
            probab01[x][y]=probab01[x][y]+poz01[x][y].real*poz01[x,y].real+poz01[x,y].imag*poz01[x,y].imag
            probab10[x][y]=probab10[x][y]+poz10[x][y].real*poz10[x,y].real+poz10[x,y].imag*poz10[x,y].imag
            probab11[x][y]=probab11[x][y]+poz11[x][y].real*poz11[x,y].real+poz11[x,y].imag*poz11[x,y].imag
            tprobab[x][y]=probab00[x][y]+probab01[x][y]+probab10[x][y]+probab11[x][y]
            totprobab=totprobab+probab00[x][y]+probab01[x,y]+probab10[x,y]+probab11[x,y]
            
print("pentru orientarea 00")
print(probab00)
print("pentru orientarea 01")
print(probab01)
print("pentru orientarea 10")
print(probab10)
print("pentru orientarea 11")
print(probab11)

print("probabilitatea de pozitie")
np.savetxt('test.out', tprobab, fmt='%-5.6f', delimiter=' ')   

            
print(totprobab)

f.close

#from qiskit.visualization import plot_state_city
#plot_state_city(tprobab)
    
    


# <h2 style="font-size:2rem;color:orange;">Quantum Discrete Oriented Random Walker on a Sphere - in a file - output all steps</h2>
# 

# In[14]:


import numpy as np
import math
import random

#from qiskit import *

file1 = open("file1.txt", "w+")
file2 = open("file2.txt", "w+")
csvfile = open('prob.csv','w')

#sqare root of 2
DivSqrt2 = 1.0/math.sqrt(2)

#Generalized Coin 4x4 matrix

#H = np.array([[0.5, 0.5, 0.5, 0.5], 
#              [0.5, -0.5, 0.5, -0.5], 
#              [0.5, 0.5, -0.5, -0.5], 
#              [0.5, -0.5, -0.5, 0.5]]) 
H = np.array([[-0.5,  0.5,  0.5,  0.5], 
              [ 0.5, -0.5,  0.5,  0.5], 
              [ 0.5,  0.5, -0.5,  0.5], 
              [ 0.5,  0.5,  0.5, -0.5]]) 

#M+ matrix 8x8 matrix move forward (positive)
Mpls = np.array([[0,0,0,0,0,0,0,1],
               [1,0,0,0,0,0,0,0],
               [0,1,0,0,0,0,0,0],
               [0,0,1,0,0,0,0,0],
               [0,0,0,1,0,0,0,0],
               [0,0,0,0,1,0,0,0],
               [0,0,0,0,0,1,0,0],
               [0,0,0,0,0,0,1,0]])

#M- matrix 8x8 matrix move backward (negative)
Mmin = np.array([[0,1,0,0,0,0,0,0],
               [0,0,1,0,0,0,0,0],
               [0,0,0,1,0,0,0,0],
               [0,0,0,0,1,0,0,0],
               [0,0,0,0,0,1,0,0],
               [0,0,0,0,0,0,1,0],
               [0,0,0,0,0,0,0,1],
               [1,0,0,0,0,0,0,0]])

#position vector
#spatiul Hilbert de dimensiune 2x2x8x8 (2 exp 2 GC  x 2 exp 3 H pozitie pe axa lini1ara X cu 8 puncte x 2 exp 3 H pozitie pe axa liniara Y cu 8 puncte)
#aruncarea de moneda GC influenteaza diferit |00>, |01>, |10> si respectiv |11> si deci si pe coeficientii de pozitie rezultati de aici

dim=8
#cu efect pe X

#pentru orientare |00>
re00 = np.zeros((dim,dim))
im00 = np.zeros((dim,dim))
poz00 = np.empty((dim,dim),complex)
auxpoz00 = np.empty((dim,dim),complex)

#pentru orientare |01>
re01 = np.zeros((dim,dim))
im01 = np.zeros((dim,dim))
poz01 = np.empty((dim,dim),complex)
auxpoz01 = np.empty((dim,dim),complex)

#cu efect pe Y
#pentru orientare |10>
re10 = np.zeros((dim,dim))
im10 = np.zeros((dim,dim))
poz10 = np.empty((dim,dim),complex)
auxpoz10 = np.empty((dim,dim),complex)

#pentru orientare |11>
re11 = np.zeros((dim,dim))
im11 = np.zeros((dim,dim))
poz11 = np.empty((dim,dim),complex)
auxpoz11 = np.empty((dim,dim),complex)

#initial pozition
# moneda (1,0,1,0)
# pozitia X (DivSqrt2+jDivSqrt2,0,0,0,0,0,0)
# pozitia Y (DivSqrt2+jDivSqrt2,0,0,0,0,0,0)

re00[0][0] = 0.5
im00[0][0] = 0 
re01[0][0] = 0.5
im01[0][0] = 0. 
re10[0][0] = 0.5
im10[0][0] = 0  
re11[0][0] = 0.5
im11[0][0] = 0. 

for x in range(dim):
        for y in range(dim):
            poz00[x][y]=complex(re00[x][y],im00[x][y])
            poz01[x][y]=complex(re01[x][y],im01[x][y])
            poz10[x][y]=complex(re10[x][y],im10[x][y])
            poz11[x][y]=complex(re11[x][y],im11[x][y])


#functii
            
def poz_copy(poz00,poz01,poz10,poz11,auxpoz00,auxpoz01,auxpoz10,auxpoz11):
        
#    print("auxpoz00")
#    print(auxpoz00)
#    print("auxpoz01")
#    print(auxpoz01)
#    print("auxpoz10")
#    print(auxpoz10)
#    print("auxpoz11")
#    print(auxpoz11)
    
    for x in range(dim):
        for y in range(dim):
            poz00[x][y]=auxpoz00[x][y]
            poz01[x][y]=auxpoz01[x][y]
            poz10[x][y]=auxpoz10[x][y]
            poz11[x][y]=auxpoz11[x][y]       
    
#    print("poz00")
#    print(poz00)
#    print("poz01")
#    print(poz01)
#    print("poz10")
#    print(poz10)
#    print("poz11")
#    print(poz11)
    

            
#toss
def toss(poz00,poz01,poz10,poz11,auxpoz00,auxpoz01,auxpoz10,auxpoz11):
    for x in range(dim):
        for y in range(dim):
                auxpoz00[x][y] = H[0][0] * poz00[x][y] + H[1][0] * poz01[x][y] + H[2][0] * poz10[x][y] + H[3][0] * poz11[x][y]
                auxpoz01[x][y] = H[0][1] * poz00[x][y] + H[1][1] * poz01[x][y] + H[2][1] * poz10[x][y] + H[3][1] * poz11[x][y]
                auxpoz10[x][y] = H[0][2] * poz00[x][y] + H[1][2] * poz01[x][y] + H[2][2] * poz10[x][y] + H[3][2] * poz11[x][y]
                auxpoz11[x][y] = H[0][3] * poz00[x][y] + H[1][3] * poz01[x][y] + H[2][3] * poz10[x][y] + H[3][3] * poz11[x][y]
    poz_copy(poz00,poz01,poz10,poz11,auxpoz00,auxpoz01,auxpoz10,auxpoz11)
  

#move
def move(poz00,poz01,poz10,poz11,auxpoz00,auxpoz01,auxpoz10,auxpox11):
    auxpoz00 = Mpls.dot(poz00)
    auxpoz00 = auxpoz00.transpose()
    auxpoz01 = Mmin.dot(poz01.transpose())  
    auxpoz01 = auxpoz01.transpose()
    auxpoz10 = Mpls.dot(poz10)
    auxpoz11 = Mmin.dot(poz11) 

    poz_copy(poz00,poz01,poz10,poz11,auxpoz00,auxpoz01,auxpoz10,auxpoz11)  



counter= int(input("Enter the number of steps: "))
pro = open("prob.csv", "w")

for step in range(counter):
    toss(poz00,poz01,poz10,poz11,auxpoz00,auxpoz01,auxpoz10,auxpoz11)
    move(poz00,poz01,poz10,poz11,auxpoz00,auxpoz01,auxpoz10,auxpoz11)


    probab00 = np.zeros((dim,dim))
    probab01 = np.zeros((dim,dim))
    probab10 = np.zeros((dim,dim))
    probab11 = np.zeros((dim,dim))
    tprobab = np.zeros((dim,dim))
    totprobab=0

    file1 = open("file1.txt", "a")


    for x in range(dim):
        for y in range(dim):         
                probab00[x][y]=probab00[x][y]+poz00[x][y].real*poz00[x,y].real+poz00[x,y].imag*poz00[x,y].imag
                probab01[x][y]=probab01[x][y]+poz01[x][y].real*poz01[x,y].real+poz01[x,y].imag*poz01[x,y].imag
                probab10[x][y]=probab10[x][y]+poz10[x][y].real*poz10[x,y].real+poz10[x,y].imag*poz10[x,y].imag
                probab11[x][y]=probab11[x][y]+poz11[x][y].real*poz11[x,y].real+poz11[x,y].imag*poz11[x,y].imag
                tprobab[x][y]=probab00[x][y]+probab01[x][y]+probab10[x][y]+probab11[x][y]
                totprobab=totprobab+probab00[x][y]+probab01[x,y]+probab10[x,y]+probab11[x,y]
            
    print("pentru orientarea 00")
    print(probab00)
    content = str(probab00)
    file1.write(content)
    file1.write("\n")
    
    np.savetxt('out.txt', probab00, fmt='%-5.6f', delimiter=' ') 


    print("pentru orientarea 01")
    print(probab01)
    content = str(probab01)
    file1.write(content)
    file1.write("\n")    
           
    np.savetxt('out.txt', probab01, fmt='%-5.6f', delimiter=' ')

    print("pentru orientarea 10")
    print(probab10)  
    content = str(probab10)
    file1.write(content)
    file1.write("\n")

    np.savetxt('out.txt', probab10, fmt='%-5.6f', delimiter=' ')  

    print("pentru orientarea 11")
    print(probab11)
    content = str(probab11)
    file1.write(content)
    file1.write("\n")
    
    np.savetxt('out.txt', probab11, fmt='%-5.6f', delimiter=' ')



    
    file2.write("\n step")
    content = str(step)
    file2.write(content)
    file2.write('\n')
    content = str(tprobab)
    file2.write(content)    
    file2.write('\n')
    
    with open('prob.csv','a') as csvfile:
        csvfile.write('step ')
        csvfile.write(str(step))
        csvfile.write('\n')

        np.savetxt(csvfile, tprobab, fmt='%-5.6f', delimiter=' ', newline='\n') 
    
    print("probabilitatea de pozitie")
    print(totprobab)

    
file1.close()
file2.close()
csvfile.close()


#from qiskit.visualization import plot_state_city
#plot_state_city(tprobab)
    
    


# In[ ]:





# In[ ]:




