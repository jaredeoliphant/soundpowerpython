import numpy as np
import sys
from acousticsFunctions import binfileload, weighting
from spectra import autospec,crossspec, fractionalOctave
import matplotlib.pyplot as plt

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
print('finished reading the files')



overallSoundPower = np.sum(Power,axis=1)
print('overallSoundPower',overallSoundPower)
spec, fc = fractionalOctave(Freq,overallSoundPower,flims=[200,2e3],width=3)

Lw = 10*np.log10(np.abs(spec)/iref)
## convert to a single value to be reported as the A-weighted sound power level
__, Gain = weighting(fc,type='A')  # only save the second output in this case
#Overall Sound power level
Lw_overall = 10*np.log10(np.sum(10**(.1*(Lw+Gain))))   # where C is the A-weighting constant  
print()
print("The A-weighted overall sound power level for intensity method is: ",Lw_overall)





f = np.array([200,250,315,400,500,630,800,1000,1250,1600,2000],dtype=float)

Reverb = np.zeros(len(f))
filename = "reverbsoundpower.txt"
fin = open(filename,'r')
for i in range(len(f)):
    Reverb[i] = fin.readline()
fin.close()


Lw_overall = 10*np.log10(np.sum(10**(.1*(Reverb+Gain))))   # where C is the A-weighting constant  
print()
print("The A-weighted overall sound power level for reverb method is: ",Lw_overall)
print()



fig4, ax4 = plt.subplots()
markerline, stemlines, baseline = ax4.stem(fc-.025*fc,Lw,markerfmt='none',label="Intensity method", basefmt='none') 
markerline1, stemlines1, baseline1 = ax4.stem(fc+.025*fc,Reverb,markerfmt='none',label="Reverb method", basefmt='none')
ax4.set_xscale('log')
ax4.set_xlim((175,2.4e3))
ax4.set_ylim((40,120))
plt.setp(stemlines, color=(0,0,.5), linewidth=7)
plt.setp(stemlines1, color=(0,.5,0), linewidth=7)
ax4.set_xlabel('$1/3$ Octave Frequencies (Hz)')
ax4.set_ylabel('Sound power level, $L_w$ (dB re 1pW)')
ax4.grid(True)
ax4.set_xticks(f.tolist(),minor=False)
ax4.set_xticks([],minor=True)
ax4.set_xticklabels(['200','250','315','400','500','630','800','1k','1.25k','1.6k','2k'])
ax4.legend(loc="upper left")
fig4.savefig("BothSoundPower.png", dpi=1200, bbox_inches='tight')


plt.show()
