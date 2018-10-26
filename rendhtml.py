from flask import Flask,render_template,request
from skimage import io,color,transform
import numpy as np
#import cv2
import sys
import base64
import re
import itertools
import os
sys.path.append(os.path.abspath("./model"))
from load import *
app = Flask(__name__)
global model,graph
letters = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','P','R','T','U','W','X','Y','Z','']
model,graph=init()
def convertImage(imgData1):
    imgstr=re.search(r'base64,(.*)',str(imgData1)).group(1)
    with open('output.png','wb') as output:
        output.write(base64.b64decode(imgstr))
@app.route("/")
def index():
    return render_template("indexReal.html")
@app.route('/predict/',methods=['GET','POST'])
def predict():
    imgData=request.get_data()
    convertImage(imgData)
    #x=cv2.imread('output.png')
    #x=io.imread('output.png')
    #x=cv2.cvtColor(x,cv2.COLOR_BGR2GRAY)
    #x=color.rgb2grey(x)
    #x=cv2.resize(x,(128,64))
    #x=transform.resize(x,(128,64))
    x=x.T
    x=x.astype(np.float32)
    x=x/255
    x=np.expand_dims(x,-1)
    x=np.expand_dims(x,0)
    with graph.as_default():
        out=model.predict(x)
        ret=[]
        for j in range(out.shape[0]):
            out_best=list(np.argmax(out[j,2:],1))
            out_best=[k for k,g in itertools.groupby(out_best)]
            outstr=''
            for c in out_best:
                if c<len(letters):
                    outstr+=letters[c]
        ret.append(outstr)
    return str(ret)
if __name__ == "__main__":
    app.run()
