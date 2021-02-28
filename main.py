# Handwritten Numbers Guesser
import sys
import pygame
from pygame.locals import *
import numpy as np
import time
from matplotlib import pyplot as plt
import cv2
from tkinter import *
from tkinter import messagebox
import tensorflow as tf

model2 = tf.keras.models.load_model("model4.model")

pygame.init()
Tk().wm_withdraw() # To hide the main window
scrWidth, scrHeight = 504, 504

screen = pygame.display.set_mode((scrWidth, scrHeight))
pixels = np.zeros((28, 28)) # To store pixel value of the drawing
pygame.display.set_caption("Handwritten Number Detector")

screen.fill((255, 255, 255))

brush = pygame.image.load("./brush/black_dot.png")
brush = pygame.transform.scale(brush, (256, 256))

pygame.display.update()
def divideImage(image):
    col_sum = image.sum(axis=0)

    start = 0
    for i in range(len(col_sum)):
        if col_sum[i] != 0:
            start = i
            break
    
    images = []
    prev = start
    numbersLoc = []
    last = -1
    for i in range(start, len(col_sum)-1):
        if col_sum[i] > 0 and col_sum[i+1] == 0:
            numbersLoc.append([prev, i])
            last = i
        elif col_sum[i] == 0 and col_sum[i+1] > 0:
            prev = i
    
    numbersLoc = [[-start, -start]] + numbersLoc
    numbersLoc.append([56-last, 56-last])
    for i in range(1, len(numbersLoc)-1):
        images.append(image[:, (numbersLoc[i-1][-1] + numbersLoc[i][0])//2: (numbersLoc[i+1][0] + numbersLoc[i][-1])//2])
    for imageInd in range(len(images)):
        extra = (28 - len(images[imageInd][imageInd]))
        imageNew = np.zeros((28, 28))
        imageNew[:, extra//2:28-(extra//2 + (extra%2))] = images[imageInd]
        images[imageInd] = imageNew
    return images


def predictImage():
    global pixels
#     print("Initial Image:")
#     plt.imshow(pixels, cmap=plt.cm.binary)
#     plt.show()
    numberImages = divideImage(pixels)
#     print("Cut Out Image:")
    predictedNumbers = []
    for singleImage in numberImages:
        image28 = np.zeros((28, 28))
        val = model2.predict([np.array([singleImage])])
        predictedNumbers.append(np.argmax(val))
#         print("Predictions -->", np.argmax(val))
#         plt.imshow(singleImage, cmap=plt.cm.binary)
#         plt.show()
    messagebox.showinfo("Prediction","I guess the number is " + "".join(list(map(str, predictedNumbers))))
    # messagebox.showinfo('Continue','OK')

def clearDrawing():
    pixels.fill(0)
    screen.fill((255, 255, 255))
    pygame.display.update()

mouseDown = False
while 1:
    cursorX, cursorY = pygame.mouse.get_pos()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == MOUSEBUTTONDOWN:
            if event.button == 1:
                mouseDown = True
            elif event.button == 3:
                clearDrawing()
        elif event.type == MOUSEBUTTONUP:
            mouseDown = False
            
        if event.type == KEYDOWN:
            if event.key == pygame.K_RETURN:
                predictImage()

            
        if mouseDown:
            pixels[cursorY//18][cursorX//18] = 1
            screen.blit(brush, (cursorX - ((brush.get_width())//2), cursorY - ((brush.get_height())//2)))
            pygame.display.update()
            