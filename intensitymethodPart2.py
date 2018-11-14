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
    # plt.semilogx(Freq,10*np.log10(np.abs(Power[:,side])/iref))
    # plt.semilogx(Freq,np.abs(Power[:,side]))
print('finished reading the files')



overallSoundPower = np.sum(Power,axis=1)
print(overallSoundPower)
# plt.semilogx(Freq,10*np.log10(np.abs(overallSoundPower)/iref))
# plt.xlim((100,10e3))
# plt.ylim((0,0.001))
spec, fc = fractionalOctave(Freq,overallSoundPower,flims=[100,10e3],width=3)

# plt.figure()
# plt.semilogx(fc,10*np.log10(np.abs(spec)/iref))

##
##Don't forget the free-field correction!!!!!
##
freefield_correction = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0.7,0.7,1,1.1,1.6,2.5,3.4,4.8],dtype=float)

Lw = 10*np.log10(np.abs(spec)/iref)+7*freefield_correction
## convert to a single value to be reported as the A-weighted sound power level
__, Gain = weighting(fc,type='A')  # only save the second output in this case
#Overall Sound power level
Lw_overall = 10*np.log10(np.sum(10**(.1*(Lw+Gain))))   # where C is the A-weighting constant  
print()
print("The A-weighted overall sound power level is: ",Lw_overall)
print()




f = np.array([100,125,160,200,250,315,400,500,630,800,1000,1250,1600,2000,2500,3150,4000,5000,6300,8000,10000],dtype=float)

fig4, ax4 = plt.subplots()
markerline, stemlines, baseline = ax4.stem(fc,Lw,markerfmt='none')  #,'color',[.75 .6 0],'linewidth',8,'marker','none')
ax4.set_xscale('log')
ax4.set_xlim((75,14e3))
ax4.set_ylim((40,120))
plt.setp(stemlines, color=(.3,.3,.3), linewidth=7)
ax4.set_xlabel('$1/3$ octave frequencies (Hz)')
ax4.set_ylabel('Sound power level, $L_w$ (dB re 1pW)')
ax4.grid(True)
ax4.set_xticks(f.tolist(),minor=False)
ax4.set_xticks([],minor=True)
ax4.set_xticklabels(['','125','','','250','','','500','','','1000','','','2000','','','4000','','','8000',''])
fig4.savefig("IntensitySoundPower.png", dpi=1200, bbox_inches='tight')





"""
fig5, ax5 = plt.subplots()
ax5.semilogx(f,freefield_correction)
ax5.set_ylim((-15,15))
"""


plt.show()
