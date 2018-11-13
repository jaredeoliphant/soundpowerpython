import numpy as np
import sys
from acousticsFunctions import binfileload, weighting
from spectra import autospec,crossspec, fractionalOctave
import matplotlib.pyplot as plt

# path to the files of interest
side = 6

# pressure and power/intensity references
pref = 2e-5
iref = 1e-12

Area1 = 0.15*0.15   # area of one measurement (15 cm distance traveled)
Area2 = (1.2+0.15)**2  # total area of one side

Power = np.zeros((4095,6))
Freq = np.zeros((4095))

# read in frequency array
filename = "Frequency.txt"
fin = open(filename,'r')
for i in range(4095):
    Freq[i] = fin.readline()
fin.close()


# read in all 6 sides of the power data
for side in range(6):
    filename = "PowerSide"+str(side+1)+".txt"
    fin = open(filename,'r')
    for i in range(4095):
        Power[i,side] = fin.readline()
    fin.close()
    plt.semilogx(Freq,10*np.log10(np.abs(Power[:,side])/iref))
print('finished reading the files')



overallSoundPower = np.sum(np.abs(Power),axis=1)
print(overallSoundPower)
plt.semilogx(Freq,10*np.log10(np.abs(overallSoundPower)/iref))

spec, fc = fractionalOctave(Freq,overallSoundPower,flims=[100,10e3],width=3)

plt.figure()
plt.semilogx(fc,10*np.log10(np.abs(spec)/iref))

"""



## convert to a single value to be reported as the A-weighted sound power level
__, Gain = weighting(fc,type='A')  # only save the second output in this case
#Overall Sound power level
Lw_overall = 10*np.log10(np.sum(10**(.1*(Lw+Gain))))   # where C is the A-weighting constant  
print()
print("The A-weighted overall sound power level is: ",Lw_overall)
print()
"""
plt.show()
