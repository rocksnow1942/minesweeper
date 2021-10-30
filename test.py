import cv2
import PIL
import numpy as np
import pyautogui
import matplotlib.pyplot as plt
from itertools import product,combinations
FEATURE_PIXELS = [(4,7),(0,7)]
BLOCK_SIZE = (16,16)



def blockClassifier(pixels):
    if pixels[3:].min() == 255:
        return -1
    id = ((pixels[0:3])*[1,2,3]).sum()
    return [1134,  765,  246,  255,  369,  123,  615,    0,  738].index(id)


def boardStateParser(imgArray):    
    bx,by = BLOCK_SIZE
    featurePixels = [imgArray[x::bx,y::by,:] for x,y in FEATURE_PIXELS]
    feature = np.concatenate(featurePixels, axis=2)
    res = np.apply_along_axis(blockClassifier, 2, feature)
    return res

    
def isCloseNeighbor(i,j):
    iUn = findUnopendNeighbor(i)
    jUn = findUnopendNeighbor(j)
    for x in iUn:
        if x in jUn:
            return True
    return False
    

def sortCloseNeighbor(x,X,group):
    neighbors = []
    for i in X:
        if isCloseNeighbor(i,x):
            group.append(i)
            neighbors.append(i)
    for i in neighbors:
        X.remove(i)
    for i in neighbors:
        sortCloseNeighbor(i,X,group)
            
    
def groupCloseNeighbors(X):
    groups=[]
    while X:
        x = X.pop()
        group = [x]
        sortCloseNeighbor(x,X,group)
        groups.append(group)
    return groups

def getNeighborPositions(bp):
    x,y = bp
    p = list(product(range(max(0,x-1),min(x+2,Board.shape[0])), range(max(0,y-1),min(y+2,Board.shape[1]))))
    p.remove(bp)
    return p

    
def getCrossPositions(bp):
    x,y = bp
    p = []
    if x-1>=0:
        p.append((x-1,y))
    if y+1<Board.shape[1]:
        p.append((x,y+1))
    if x+1<Board.shape[0]:
        p.append((x+1,y))
    if y-1>=0:
        p.append((x,y-1))
    return p
    
def findNumberCross(bp):
    ps = getCrossPositions(bp)
    number = []
    for p in ps:
        if Board[p] > 0:
            number.append(p)
    return number
        
    

def findUnopend(bp):
    x,y = bp
    xs = max(x-1 , 0)
    ys = max(y-1 , 0)
    all = Board[xs:x+2,ys:y+2]
    unopened = list((xs+i,ys+j) for i,j in zip(*np.where(all==-1)))
    return unopened
    
    
def findUnopendNeighbor(bp):
    ps = getNeighborPositions(bp)
    unopened = []
    for p in ps:
        if Board[p] == -1:
            unopened.append(p)
    return unopened

def findNumberNeighbor(bp):
    ps = getNeighborPositions(bp)
    number = []
    for p in ps:
        if Board[p] > 0:
            number.append(p)
    return number
    
    
def closeNeighbor(x,i):
    u = findUnopend(x)
    o = findUnopend(i)
    
def findConnectedNeighbor(x):
    uo = findUnopendNeighbor(x)
    xNeighbor = findNumberCross(x)
    closeNeighbor = []
    for i in xNeighbor:
        iUnopen = findUnopendNeighbor(i)
        isClose = False
        for j in iUnopen:
            if j in uo:
                isClose = True
                break
        if isClose:
            closeNeighbor.append(i)
    return closeNeighbor
    
def fillBoard(x,board):
    n = Board[x] # howmany bombos
    uo = findUnopendNeighbor(x) # howmany unopened
    # need to filter out the filled ones in board    
    possibilities = []
    for p in combinations(uo,n):
        conflict = False
        for i in p:            
            if board[i] == 0:
                conflict = True
                break
        if not conflict:
            new = board.copy()
            for i in uo:
                if i in p:                                            
                    new[i] = 1
                else:
                    new[i] = 0                    
            possibilities.append(new)
    
    return possibilities

uo = findUnopendNeighbor((2,22))
n = Board[(2,22)]

    
    
class Game:
    def __init__(self,gameSize=(30,16), boardImg = './board.png'):
        self.gameSize = gameSize
        self.boardImg = boardImg

    def getBoardPosition(self):
        box = pyautogui.locateOnScreen(self.boardImg)
        if box is None:
            raise RuntimeError('Board not found') 
        self.box = box
    
    def readBoardState(self):
        capture = pyautogui.screenshot(region=self.box)
        caparr = np.array(capture,dtype='int16')
        self.state = boardStateParser(caparr)


np.savetxt('s1.txt',g.state,fmt='%2d')
np.savetxt('s3.txt',g.state,fmt='%2d')
g = Game()

g.getBoardPosition()

g.readBoardState()
g.state

s3 = g.state

g.state[g.state>0].sum()

g.state[]
np.where(s3>0)

g.state[g.state>0]

g.state>0


s3 = np.loadtxt('s3.txt',dtype='int8')


np.where(s3>0)
g.state

N = [(i,j) for i,j in zip(*np.where(s3>0))]
N

# group N in to connecting tiles




len(groups)
groups[0]
groups[1]

groups[2]

groups[3]

Board = s3

Board[0,28]

all = Board[0:2,28:31]
all
np.where(all==-1)

list(zip(*np.where(all==-1)))



Board.shape
n = getNeighborPositions((1,29))



    
findUnopend((0,28))


                 
    
                
                
findUnopendNeighbor((1, 19))        
list(combinations([(0, 18), (1, 18), (2, 18)],2))

list(combinations([(1,2),(3,4),(4,5)],5))
        


Board[(1,28)]

def updateState(State, remainX, Possibilites,x=None,callstack=0):        
    if not remainX:
        Possibilites.append(State)
        return
    
    allp = []    
        
    if x is None:
        x = remainX[0]
        p = fillBoard(x,State)
        
        allp.append((x,p))
    else:
    # fist find the neighbors of x that is close(share unopened blocks) 
        connectedNeighbor = findConnectedNeighbor(x)  
        connectedNeighbor = [i for i in connectedNeighbor if i  in remainX]
        if callstack>10:
            print(f'connecte neighbors: {connectedNeighbor}, {x}')
        if not connectedNeighbor:
            x = remainX[0]
            p = fillBoard(x,State)
            allp.append((x,p))
        else:
            for c in connectedNeighbor:
                p = fillBoard(c,State)                
                allp.append((c,p))
    
    for block,P in allp:
        
        for p in P:
            updateState(p,[i for i in remainX if i!=block],Possibilites,block,callstack=callstack+1)
        
        
o = np.full_like(Board,-1)
Possibilites = []        


N = [(i,j) for i,j in zip(*np.where(s3>0))]

Ns = groupCloseNeighbors(N)

len(Ns)
Ns[3]

len(N)
N.pop()
len(N)
N[0]
p = fillBoard((0, 21),o)
p
len(Ns[3])
len(p)
len(N)

Possibilites

Ns[3]

updateState(o,Ns[3],Possibilites)
agg = sum(Possibilites) 

pp= list(zip(*np.where(agg>=0)))
    
for p in pp:
    print(f"At position {p}, p= {agg[p]/5}")

for idx,i in enumerate(Possibilites):
    if i[(1,22)] !=1:
        print(idx)

np.savetxt('wrong.txt',Possibilites[1],fmt='%2d')



len(Possibilites)

len(Possibilites)

    
len(Possibilites)
    
board = np.zeros_like(Board)


np.array(Board)
updateState((1,28),None,None)


connectedNeighbor = findConnectedNeighbor(x)
x
connectedNeighbor

x
len(Ns[3])
remainX = Ns[3]
o = o
State = o
x = remainX[0]
p = fillBoard(x,State)

State = p[0]
remainX = [i for i in remainX if i!=x]

connectedNeighbor = findConnectedNeighbor(x)  
x
connectedNeighbor
for c in connectedNeighbor:
    
[(2, 24), (3, 23)]

c = connectedNeighbor[0]
    
p = fillBoard(c,State)    




len(p)
p[0]


for p in Possibilites:
    print(p[0:3,22:25])





