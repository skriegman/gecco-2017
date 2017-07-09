#!/usr/bin/python

import subprocess as sub
import pandas as pd
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
import prettyplotlib as ppl
from prettyplotlib import brewer2mpl
from matplotlib.backends.backend_pdf import PdfPages
import sys
import scipy  
import scikits.bootstrap as bootstrap  
import os
import time
import numexpr as ne
import bottleneck as bn
import glob

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

fig, ax = plt.subplots(1)

# if "importRun" in sys.argv[1:]:
if True:
	# startTimeImport = time.time()
	# numGens = len(sub.check_output(['ls','allIndividualsData/*.txt']).strip().split("\n"))
	numGens = len(glob.glob('allIndividualsData/*.txt'))
	print "total gens:",numGens
	print

	allData = []
	for i in range(1,numGens):
		thisData = pd.read_csv("allIndividualsData/Gen_%04i.txt"%i,delim_whitespace=True)
		allData.append(thisData.as_matrix())
		# print thisData
		# sys.exit(0)
	tmpData = np.array(allData)
	tmpData = tmpData.reshape((tmpData.shape[0]*tmpData.shape[1],tmpData.shape[2]))

	data = pd.DataFrame(data=tmpData,columns=thisData.columns.values)
	data['lineageID'] = pd.Series(np.zeros(len(data["id"]))*np.nan, index=data.index)
	data['currentBest'] = pd.Series(np.zeros(len(data["id"])), index=data.index)

	for thisGen in range(1,numGens):
		thisGenData = data[data["gen"]==thisGen]
		data.set_value(thisGenData[thisGenData["fitness"]==np.max(thisGenData.fitness.values)].index.values,'currentBest',1)

	data = data.groupby("id").first()
	# data["id"] = data.index.values

maxLineageID = 0
for index, row in data.iterrows():
	if row["parent_id"] < 0:
		data.set_value(index,"lineageID",maxLineageID)
		maxLineageID += 1
	else:
		data.set_value(index,"lineageID",data.get_value(row["parent_id"],"lineageID"))

for thisLineageID in range(int(np.max(data["lineageID"].values))):
	# print data[data["lineageID"]==thisLineageID]
	thisLineageData = data[data["lineageID"]==thisLineageID].groupby("gen").max()
	# print thisLineageData
	thisColor = np.random.rand(3)
	thisAlpha = 0.5 #0.33
	thisLinewidth = 1
	if np.any(thisLineageData["currentBest"].values):
		thisAlpha = 1
		thisLinewidth = 2
		print "once current best lineage:",thisLineageID
	bestLineageFitnessSoFar = 0
	for i in range(len(thisLineageData.index.values)-1):
		# if thisLineageData.fitness.values[i+1] > 0 and thisLineageData.fitness.values[i] > 0:
		# ppl.plot(ax,[thisLineageData.gen.values[i],thisLineageData.gen.values[i+1]],[thisLineageData.fitness.values[i],thisLineageData.fitness.values[i+1]],c=thisColor)
		bestLineageFitnessSoFar = max(bestLineageFitnessSoFar,thisLineageData.fitness.values[i])
		ppl.plot(ax,[thisLineageData.index.values[i],thisLineageData.index.values[i+1]-1],[bestLineageFitnessSoFar,bestLineageFitnessSoFar],c=thisColor, alpha=thisAlpha, linewidth=thisLinewidth)
		ppl.plot(ax,[thisLineageData.index.values[i+1]-1,thisLineageData.index.values[i+1]],[bestLineageFitnessSoFar,max(bestLineageFitnessSoFar,thisLineageData.fitness.values[i+1])],c=thisColor, alpha=thisAlpha, linewidth=thisLinewidth)
		bestLineageFitnessSoFar = max(bestLineageFitnessSoFar,thisLineageData.fitness.values[i+1])
    
# plt.show()
plotName = "afpoTrace.pdf"
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.tight_layout()
pp = PdfPages(plotName)
pp.savefig(fig)
pp.close()
plt.close()
sub.Popen(["gnome-open",plotName])