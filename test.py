import cv2
import PIL
import numpy as np
import pyautogui
import matplotlib.pyplot as plt
from itertools import product,combinations
import time
import random
import platform
from PIL import ImageGrab
from mss import mss
# https://python-mss.readthedocs.io/installation.html


pyautogui.PAUSE = 0.01

if 'darwin' in platform.platform().lower():
    FEATURE_PIXELS = [(8,14),(0,14)]
    BLOCK_SIZE = (32,32)
    PIXEL_FEATURE = [1134, 909, 391, 441, 354, 183, 661, 0, 738]
    PLATFORM = 'mac'
else:
    FEATURE_PIXELS = [(4,7),(0,7)]
    BLOCK_SIZE = (16,16)
    PIXEL_FEATURE = [1134,  765,  246,  255,  369,  123,  615,    0,  738]
    PLATFORM = 'win'
    

def blockClassifier(pixels):
    if pixels[3:].min() == 255:
        return -1
    id = ((pixels[0:3])*[1,2,3]).sum()    
    return PIXEL_FEATURE.index(id)


def boardStateParser(imgArray):    
    bx,by = BLOCK_SIZE
    featurePixels = [imgArray[x::bx,y::by,0:3] for x,y in FEATURE_PIXELS]
    minX = min(i.shape[0] for i in featurePixels )
    minY = min(i.shape[1] for i in featurePixels )
    featurePixels = [i[0:minX,0:minY,:] for i in featurePixels]
    feature = np.concatenate(featurePixels, axis=2)
    res = np.apply_along_axis(blockClassifier, 2, feature)
    return res

def imageCropper(file,row,col):
    m = PIL.Image.open(file)
    w,h = BLOCK_SIZE
    return m.crop((0,0,col*w,row*h))


    
class Game:
    def __init__(self,mode='easy'):
        if mode =='hard':
            row=16
            column=30            
            bombCount = 99
        elif mode == 'easy':
            row=9
            column=9       
            bombCount = 10
        elif mode == 'normal':
            row=16
            column=16
            bombCount = 40
            
        boardImg = f'./{PLATFORM}{mode.capitalize()}.png'
        self.shape=(row,column)
        self.boardImg = boardImg
        self.box = None # box position of game board
        self.state = np.full(self.shape,-1)
        self.sumState = np.zeros(self.shape)
        self.bombCount = bombCount        
        self.playtime = 0
        self.playStat = {'win':[],'lose':[]}
        self.sct = mss()
            
    def __getitem__(self,idx):
        return self.state[idx]
        
    def init(self):
        box = pyautogui.locateOnScreen(self.boardImg)
        if box is None:
            raise RuntimeError('Board not found') 
        self.box = box
        l,t,w,h = box
        self._monitor = {"top":t/2, "left": l/2, "width": w/2, "height": h/2}
    
    def readBoardState(self):        
        # capture = pyautogui.screenshot(region=self.box) # ~0.5s                
        # capture = ImageGrab.grab(bbox=(l,t,l+w,t+h)) @ ~ 0.45s        
        img = self.sct.grab(self._monitor)
        capture = PIL.Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX") # this is very fast! 0.005 second        
        caparr = np.array(capture,dtype='int16')
        self.state = boardStateParser(caparr)
    
    
    def isCloseNeighbor(self,i,j):
        iUn = self.findUnopendNeighbor(i)
        jUn = self.findUnopendNeighbor(j)
        for x in iUn:
            if x in jUn:
                return True
        return False
        
        
    def sortCloseNeighbor(self,x,X,group):
        neighbors = []
        for i in X:
            if self.isCloseNeighbor(i,x):
                group.append(i)
                neighbors.append(i)
        for i in neighbors:
            X.remove(i)
        for i in neighbors:
            self.sortCloseNeighbor(i,X,group)
            
    def groupCloseNeighbors(self,X):
        groups=[]
        while X:
            x = X.pop()
            group = [x]
            self.sortCloseNeighbor(x,X,group)
            groups.append(group)
        return groups
        
    def getCrossPositions(self,bp):
        x,y = bp
        p = []
        if x-1>=0:
            p.append((x-1,y))
        if y+1<self.shape[1]:
            p.append((x,y+1))
        if x+1<self.shape[0]:
            p.append((x+1,y))
        if y-1>=0:
            p.append((x,y-1))
        return p
        
    def findNumberCross(self,bp):
        ps = self.getCrossPositions(bp)
        number = []
        for p in ps:
            if self[p] > 0:
                number.append(p)
        return number
        
    def getNeighborPositions(self,bp):
        x,y = bp
        p = list(product(range(max(0,x-1),min(x+2,self.shape[0])), range(max(0,y-1),min(y+2,self.shape[1]))))
        p.remove(bp)
        return p
    
    def findUnopendNeighbor(self,bp):
        ps = self.getNeighborPositions(bp)
        unopened = []
        for p in ps:
            if self[p] == -1:
                unopened.append(p)
        return unopened
        
    def findConnectedNeighbor(self,x):
        uo = self.findUnopendNeighbor(x)
        xNeighbor = self.findNumberCross(x)
        closeNeighbor = []
        for i in xNeighbor:
            iUnopen = self.findUnopendNeighbor(i)
            isClose = False
            for j in iUnopen:
                if j in uo:
                    isClose = True
                    break
            if isClose:
                closeNeighbor.append(i)
        return closeNeighbor
        
    def fillBoard(self,x,board):
        n = self[x] # howmany bombos
        uo = self.findUnopendNeighbor(x) # howmany unopened
        # need to filter out the filled ones in board    
        possibilities = []
        for p in combinations(uo,n):
            conflict = False
            for i in uo:
                if i in p and board[i]==0:
                    conflict = True
                    break
                elif i not in p and board[i]==1:
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
        
    def updateState(self,State, remainX, Possibilites,x=None,):        
        if not remainX:
            Possibilites.append(State)
            return   
        if len(Possibilites)>500:            
            return
        allp = []                
        if x is None:
            x = remainX[0]
            p = self.fillBoard(x,State)            
            allp.append((x,p))
        else:
        # fist find the neighbors of x that is close(share unopened blocks) 
            connectedNeighbor = self.findConnectedNeighbor(x)  
            connectedNeighbor = [i for i in connectedNeighbor if i  in remainX]        
            if not connectedNeighbor:
                x = remainX[0]
                p = self.fillBoard(x,State)
                allp.append((x,p))
            else:
                for c in connectedNeighbor:
                    p = self.fillBoard(c,State)                
                    allp.append((c,p))        
        for block,P in allp:            
            for p in P:
                self.updateState(p,[i for i in remainX if i!=block],Possibilites,block)
                
    def load(self,file):
        self.state = np.loadtxt(file,dtype=int)
        
    def saveState(self):
        np.savetxt(f"{int(time.monotonic())}.txt",self.state,fmt='%2d')
    
    def calculateSafeBlocks(self):                
        numberBlocks = [(i,j) for i,j in zip(*np.where(self.state>0))]
        Ns = self.groupCloseNeighbors(numberBlocks)        
        cleanBlocks = []  
        sumState = []
        for N in Ns:            
            # N.sort()
            possibilities = []
            state = np.full(self.shape,-1) # maybe remember the state
            self.updateState(state,N,possibilities)
            if not possibilities:
                
                self.saveState()
                continue            
            agg = sum(possibilities)            
            nonBomb = list(zip(*np.where(agg==0)))            
            cleanBlocks.extend(nonBomb)
            agg[agg<0] = 0
            avg = agg.astype(float)/len(possibilities)
            sumState.append(avg)
        if sumState:
            self.sumState = sum(sumState)
        return cleanBlocks
    
    def blockPosition(self,pos):
        row,col = pos
        x,y = (16,16) # weirdly, the x,y position of mouse click doesn't have the scale.
        return self.box.left/2 + (col + 0.5) * (x) , self.box.top/2 + (row + 0.5) * (y)
    
    def click(self,i):
        pyautogui.click(*self.blockPosition(i))        
        
    def randomBlock(self):
        b = self.sumState
        a = self.state
        return random.choice(list(zip(*np.where((b<1 )& (a==-1)))))
            
    def remaining(self):
        return (g.state==-1).sum() - self.bombCount
    
    def clickReset(self,win):
        mode = 'win' if win else 'lose'
        self.playStat[mode].append(time.monotonic()-self.playtime)                    
        pyautogui.click(x=(self.box.left + self.box.width/2)/2, y = self.box.top/2 - 27)
        self.sumState = np.zeros(self.shape)
        self.playtime = time.monotonic()
    
    def printGameSummary(self):
        win = self.playStat['win']
        lose = self.playStat['lose']
        winAvgT = sum(win)/len(win) if win else 0
        loseAvgT = sum(lose)/len(lose) if lose else 0
        print('='*25 +'GAME SUMMARY' + '='*25)
        print(f"Total game won: {len(win)}")
        print(f"Average time to win: {winAvgT:.2f} seconds")
        print(f"Total game lost: {len(lose)}")
        print(f"Average time to lose: {loseAvgT:.2f} seconds")
        print(f"Total time played: {(sum(win) + sum(lose) ) / 60:.2f} minutes.")
        print('='*62)


    def play(self):
        self.playtime = time.monotonic()
        while 1:
            try:                
                self.readBoardState()                            
                if self.remaining()==0:
                    print(f"I win! total win = {len(self.playStat['win'])}")
                    self.clickReset(True)
                    continue
                safeBlocks = self.calculateSafeBlocks()
                if safeBlocks:
                    # for i in safeBlocks[0:1]:
                    self.click(random.choice(safeBlocks))
                else:
                    self.click(self.randomBlock())   
                
                if self.remaining()==1:
                    # click 
                    pyautogui.click(681,246)                    
                    time.sleep(0.5)
            except ValueError:                
                print(f'Im Dead. Reset Game :( Total lose:{len(self.playStat["lose"])}')
                self.clickReset(False)
            except Exception as e:
                self.printGameSummary()
                raise e
                
            
        

if __name__ == '__main__':    
    g = Game(mode='easy')
    g.init()
    g.play()
    
    