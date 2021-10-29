import cv2
import PIL
import numpy as np
import pyautogui
import matplotlib.pyplot as plt

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



class Game:
    def __init__(self,gameSize=(36,16), boardImg = './board.png'):
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
np.savetxt('s2.txt',g.state,fmt='%2d')
g = Game()

g.getBoardPosition()

g.readBoardState()
g.state



g.state[g.state>0].sum()


g.state[g.state>0]




np.loadtxt('s2.txt',dtype='int8').dtype



g.state















