import numpy as np
import matplotlib.pyplot as plt

#Parámetro y constantes
fmass = np.array([0.51, 105.66, 1776.99, 4.8, 2.3,  1270., 95., 173210., 4180.])*1e-03 #Masas fermiones en GeV
mv = np.array([80.42, 91.19])  #Masas W y Z en GeV
mh = 125.09 #Masa Higgs
gs = 1.214 #Acople fuerte
v = 246 #VEV
alphas = gs**2/(4.*np.pi) #gs = sqrt(4*pi*alphas)
GF = 1.16*10e-05 #Constante de Fermi
#Matriz PMNS y CKM
PMNS = 0.5*np.transpose(np.array([[0.799+0.844,0.242+0.494,0.284+0.521],[0.516+0.582,0.467+0.678,0.490+0.695],[0.141+0.156,0.639+0.774,0.615+0.754]]))
CKM = np.transpose(np.array([[0.97,0.22,0.0087],[0.22,0.97,0.04],[0.0035,0.041,0.999]]))
thetaW = 28.76*np.pi/180 #Ángulo de Weinberg
cota1 = 4*1e17
cota2 = 1e24


#Acoples débiles g_A
gau = np.array([1./2,1./2]) #Tipo up
gad = -gau #Tipo down
#Acoples débiles g_V
gvu = np.array([1./2.,1./2.-4.*np.sin(thetaW)**2/3.]) #Tipo up
gvd = np.array([-1./2.+2*np.sin(thetaW)**2,-1./2.+2*np.sin(thetaW)**2/3.]) #Tipo down
Mp = 2.4*10e18 #Masa de planck reducida
kappa = 1./Mp #Constante GR
a = 0

#Suma de cuadrados de la CKM y la PMNS
suma = 0
for i in range(0,3):
    for j in range(0,3):
        suma += PMNS[i,j]**2 + CKM[i,j]**2
suma *= 3 #fmass-independent --> *3 flavors

#Suma de los acoples débiles
#suma2 = 3*(np.sum(gau**2) + np.sum(gad**2) + np.sum(gvu**2) + np.sum(gvd**2)) #fmass-independent --> 3 flavors  

#Factores de masa
def xi(m,particle):
    
    if particle == 'h':
        x = mh**2/m**2
    elif particle == 'Z':
        x = mv[1]**2/m**2
    elif particle == 'W':
        x = mv[0]**2/m**2
    elif particle == 'f':
        x = fmass**2/m**2
    return x

#Anchos de decaimiento
def Gamma(m,i):
    gamma = 0

    #DM a 2 higgs
    if i == 1 and m > 2*mh:
        gamma = m**3*(1+2*xi(m,'h'))**2*(1-4*xi(m,'h'))**(0.5)/(32.*np.pi)
        
    #DM a 2 Z   
    elif i == 2 and m > 2*mv[1]:
        gamma = m**3*(1-4*xi(m,'Z')+12*xi(m,'Z')**2)*(1-4*xi(m,'Z'))**(0.5)/(32.*np.pi)
        
    #DM a 2 W    
    elif i == 3 and m > 2*mv[0]:
        gamma = m**3*(1-4*xi(m,'W')+12*xi(m,'W')**2)*(1-4*xi(m,'W'))**(0.5)/(16.*np.pi)
        
    #DM a 2 f y f_bar    
    elif i == 4:
        for j in range(0,9):
            if m > 2*fmass[j]:
                gamma += 9*m**3*xi(m,'f')[j]*(1-4*xi(m,'f')[j])**(1.5)/(8.*np.pi)
                
    #DM a q q_bar y gluón
    elif i == 5:
        for j in range(3,9):
            if m > fmass[j]:
                gamma += alphas*m**3/(4.*np.pi**2)
                
    elif i == 6:
        #DM a f nu_bar y W + h.c
        for j in range(0,3):
            if m > np.max([fmass[j], mv[0]]):
                gamma += 3.*GF*suma*m**5/(2.*np.sqrt(2)*(4.*np.pi)**3)
                
        #DM a q q_bar y W + h.c
        for k in range(3,9,2):
            if m > np.max([fmass[j], mv[0]]):
                gamma += 3.*GF*suma*m**5/(2.*np.sqrt(2)*(4*np.pi)**3)
                
    #DM a f f_bar y Z            
    elif i == 7:
        for j in range(0,9):
            if m > np.max([fmass[j], mv[1]]):
                if j== 3 or j==5 or j==7:
                    suma2 = np.sum(gau**2)+np.sum(gvu**2)
                elif j== 4 or j==6 or j==8:
                    suma2 = np.sum(gad**2)+np.sum(gvd**2)
                else:
                    suma2 = np.sum(gau**2)+np.sum(gvu**2)
                gamma += 3.*GF*suma2*m**5/(8.*np.cos(thetaW)**2*np.sqrt(2)*(4.*np.pi)**3)
    
    #DM a 2 higgs y 2W            
    elif i == 8 and m > 2*mh+2*mv[0]:
        gamma = m**7/(15.*(8*np.pi)**5*v**4)

    #DM a 2 higgs y 2Z    
    elif i == 9 and m > 2*mh+2*mv[1]:
        gamma = m**7/(30.*(8*np.pi)**5*v**4)
        
    return gamma

#Ancho total
def GammaTotal(m):
    sumatoria = 0
    for i in range(1,10):
        sumatoria += Gamma(m,i)
    return sumatoria

#Arreglo masa de DM
DMMass1 = np.arange(1,10**3,0.5)
DMMass2 = np.arange(10**3,10**6,10**3)
DMMass = np.concatenate((DMMass1,DMMass2))
cota1 = cota1*np.ones(len(DMMass))
cota2 = cota2*np.ones(len(DMMass))

#Branching ratios
def BR(particle):
    branching = np.zeros(len(DMMass))
    for i in range(0,len(DMMass)):
        branching[i] = Gamma(DMMass[i],particle)/GammaTotal(DMMass[i])
    return branching

def Gamma_tot_inv(xi):
    f = xi**2*Mp**2*kappa**4
    lifetime = np.zeros(len(DMMass))
    for i in range(0,len(DMMass)):
        lifetime[i] = 1.0/(GammaTotal(DMMass[i]))
    return lifetime/f

plt.figure(figsize = (10,12))
plt.title('Ibarra')
plt.semilogx(DMMass,BR(1),label = 'hh')
plt.semilogx(DMMass,BR(4), label = 'ff')
plt.semilogx(DMMass,BR(2)+BR(3), label = 'WW+ZZ')
plt.semilogx(DMMass,BR(5), label = 'qqg')
plt.semilogx(DMMass,BR(6)+BR(7), label = 'ffW+ffZ')
plt.semilogx(DMMass,BR(8)+BR(9), label = 'hhWW+hhZZ')
#plt.ylim(0.05,1)
plt.legend()
plt.grid()
plt.savefig('DM_Decay_1.1.2.png')
plt.show()

plt.figure(figsize = (10,12))
plt.title('Ibarra')
plt.loglog(DMMass,Gamma_tot_inv(1),label ='$\chi = 1$')
plt.loglog(DMMass,Gamma_tot_inv(10e-8), label = '$\chi = 10^{-8}$')
plt.loglog(DMMass,Gamma_tot_inv(10e-16), label = '$\chi = 10^{-16}$')
plt.loglog(DMMass,cota1, label= 'Age of the universe')
plt.loglog(DMMass,cota2, label = 'Neutrino telescopes')
plt.legend()
plt.grid()
plt.savefig('DM_Decay_1.2.2.png')
plt.show()
        
    
