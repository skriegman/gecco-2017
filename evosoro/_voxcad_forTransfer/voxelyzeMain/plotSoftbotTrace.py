#!/usr/bin/python

import pandas as pd
import numpy as np 
import subprocess as sub
import sys
import glob
from scipy import stats 
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import prettyplotlib as ppl
from prettyplotlib import brewer2mpl
from matplotlib.backends.backend_pdf import PdfPages
import scipy  
import scikits.bootstrap as bootstrap

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

fig = plt.figure()
ax = fig.gca(projection='3d')

for filename in sys.argv[1:]:
	fitnessFile = open(filename)
	traceTime = []
	traceX = []
	traceY = []
	traceZ = []
	for line in fitnessFile:
		if "<Time>" in line:
			traceTime.append( float( line[line.find("<Time>")+len("<Time>"):line.find("</Time>")] ) )
		if "<TraceX>" in line:
			traceX.append( float( line[line.find("<TraceX>")+len("<TraceX>"):line.find("</TraceX>")] ) )
		if "<TraceY>" in line:
			traceY.append( float( line[line.find("<TraceY>")+len("<TraceY>"):line.find("</TraceY>")] ) )
		if "<TraceZ>" in line:
			traceZ.append( float( line[line.find("<TraceZ>")+len("<TraceZ>"):line.find("</TraceZ>")] ) )
	fitnessFile.close()

	dim = 0.01; bodyLength = 5
	if filename == "Example2_testTrace.xml": dim = 0.001; bodyLength = 10;
	# ax.plot(np.array(traceX)/(dim*bodyLength), np.array(traceY)/(dim*bodyLength), np.array(traceZ)/(dim*bodyLength)) # absolute position
	ax.plot((np.array(traceX)-traceX[0])/(dim*bodyLength), (np.array(traceY)-traceY[0])/(dim*bodyLength), (np.array(traceZ)-traceZ[0])/(dim*bodyLength)) # position relative to starting point

ax.set_xlabel("X position (body lengths)")
ax.set_ylabel("Y position (body lengths)")
ax.set_zlabel("Z position (body lengths)")
ax.set_title("Softbot Trace")
# ax.legend()

plt.show()