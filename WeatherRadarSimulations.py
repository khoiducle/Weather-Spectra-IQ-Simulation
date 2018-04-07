# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 13:58:37 2018
This script follows 
    D.S. Zrnic, Simulation of weather-like Doppler spectra and signals., 
        J. Appl. Meteorol. 14, 619–620. Zrnić, D. S. (1977a).

@author: 
"""
import numpy as np
import matplotlib.pyplot as plt


GLOBAL_VARIABLES ={}
INPUT_FILENAME = 'SimulationParameters.txt'

def main(): 
    local_VARIABLES = {}
    local_VARIABLES.setdefault('Spectrum Width', []).append(2)
    local_VARIABLES.setdefault('Spectrum Width', []).append('m/s')
    local_VARIABLES.setdefault('Vr Mean', []).append(5)
    local_VARIABLES.setdefault('Vr Mean', []).append('m/s')
    
    
    readInput()
    IQ = genIQ(local_VARIABLES)
    estMoM(IQ)
    
    
def estMoM(IQ,*positional_parameters, **keyword_parameters):
    if ('optional' in keyword_parameters):
        print('optional parameter found, it is ', keyword_parameters['optional'])
    
        

    Num = int(GLOBAL_VARIABLES['Num Samples'][0])
    Va = float(GLOBAL_VARIABLES['Vr Aliasing'][0])
    
    R0 = 1/(Num)*np.sum(np.multiply(np.conj(IQ[0:Num]),IQ[0:Num]))
    R1 = 1/(Num-1)*np.sum(np.multiply(np.conj(IQ[0:Num-1]),IQ[1:Num]))
    R2 = 1/(Num-2)*np.sum(np.multiply(np.conj(IQ[0:Num-2]),IQ[2:Num]))
    Vm_est = -Va/np.pi*np.angle(R1)
    print('Estimated Velocity:',Vm_est)

    
def readInput(*positional_parameters, **keyword_parameters):
    if ('optional' in keyword_parameters):
        print('optional parameter found, it is ', keyword_parameters['optional'])
    
    try:
        fid = open(INPUT_FILENAME, 'r')
    except IOError:
        print('Error: __', INPUT_FILENAME, '__ does not appear to exist.')
    else:
        
        for line in fid:
            linesplit = line.split(";")
            GLOBAL_VARIABLES.setdefault(linesplit[0], []).append(linesplit[1])
            GLOBAL_VARIABLES.setdefault(linesplit[0], []).append(linesplit[2])
    
def genIQ(local_VARIABLES,*positional_parameters, **keyword_parameters):
    if ('optional' in keyword_parameters):
        print('optional parameter found, it is ', keyword_parameters['optional'])

    Noi = float(GLOBAL_VARIABLES['Noi Power'][0])
    SNR = float(GLOBAL_VARIABLES['SNR'][0])
    freq = float(GLOBAL_VARIABLES['Frequency'][0])
    Num = int(GLOBAL_VARIABLES['Num Samples'][0])
    Va = float(GLOBAL_VARIABLES['Vr Aliasing'][0])
    
    vm = local_VARIABLES['Vr Mean'][0]
    sig = local_VARIABLES['Spectrum Width'][0]
    
    
    S = Noi*10**(SNR/10)
    vr = np.arange(0,Num,1).reshape(Num,1)
    vrdel = 2*Va/Num
    vr = vr*vrdel-Va
    Nk = Noi/Num
    Sk = 1/(2*sig**2)*np.exp(-abs((-vm+vr))**2/(2*sig**2))+\
        1/(2*sig**2)*np.exp(-abs(-vm+vr+2*Va)**2/(2*sig**2))+\
        1/(2*sig**2)*np.exp(-abs(-vm+vr-2*Va)**2/(2*sig**2))+\
        1/(2*sig**2)*np.exp(-abs(-vm+vr+4*Va)**2/(2*sig**2))+\
        1/(2*sig**2)*np.exp(-abs(-vm+vr-4*Va)**2/(2*sig**2))
    Sk = S*Sk/np.sum(Sk)
    Xk = np.random.rand(Num,1)
    Pk = np.multiply(-(Sk+Nk),np.log(Xk))
    Ak = np.sqrt(Pk)
    Phik = np.pi*np.random.uniform(-1,1,(Num,1))
    Fkvr = Ak*np.exp(1.j*Phik)
    
    
    ind0 = np.where(vr==0)[0]
    v1 = vr[0:int(ind0)+1:1]
    v2 = vr[int(ind0)+1:len(vr):1]
    freq_fft = np.concatenate([np.flipud(v1),np.flipud(v2)])
    f1 = Fkvr[0:int(ind0)+1:1]
    f2 = Fkvr[int(ind0)+1:len(vr):1]
    f_fft = np.concatenate([np.flipud(f1),np.flipud(f2)])

    return np.fft.ifft(f_fft,axis=0)*np.sqrt(Num)
    
    
if __name__== "__main__":
  main()


