import numpy as np
import matplotlib.pyplot as plt

#Parámetro y constantes
fmass = np.array([0.51, 0, 105.66, 0, 1776.99, 0, 4.8, 2.3,  1270., 95., 173210., 4180.])*1e-03  #Masas fermiones en GeV
mv = np.array([80.42, 91.19])  #Masas W y Z en GeV
mh = 125.09 #Masa Higgs en GeV
thetaW = 28.76*np.pi/180 #Ángulo de Weinberg
sw2 = np.sin(thetaW)**2  #sin^2 del ángulo de Weinberg

#Acoples elestrodébiles
ga = np.array([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5,])
gv = np.array([0.5, -0.5+2*sw2, 0.5, -0.5+2*sw2, 0.5, -0.5+2*sw2, 0.5-4*sw2/3.,-0.5+2*sw2/3., 0.5-4*sw2/3.,-0.5+2*sw2/3., 0.5-4*sw2/3.,-0.5+2*sw2/3.])

gs = 1.214 #Acople fuerte
v = 246 #VEV
alphas = gs**2/(4.*np.pi) #gs = sqrt(4*pi*alphas)
GF = 1.16*10e-05 #Constante de Fermi

#Matriz PMNS y CKM
PMNS = 0.5*np.transpose(np.array([[0.799+0.844,0.242+0.494,0.284+0.521],[0.516+0.582,0.467+0.678,0.490+0.695],[0.141+0.156,0.639+0.774,0.615+0.754]]))
CKM = np.transpose(np.array([[0.97,0.22,0.0087],[0.22,0.97,0.04],[0.0035,0.041,0.999]]))

Mp = 2.4*10e18 #Masa de planck reducida
kappa = 1./Mp #Constante GR

#Arreglo masa de DM
DMMass1 = np.arange(1,10**3,0.5)
DMMass2 = np.arange(10**3,10**6,10**3)
DMMass = np.concatenate((DMMass1,DMMass2))

#Cotas experimentales
cota1 = 4*1e17*np.ones(len(DMMass))
cota2 = 1e24*np.ones(len(DMMass))

suma = 0
for i in range(0,3):
    for j in range(0,3):
        suma += PMNS[i,j]**2 + CKM[i,j]**2
suma *= 3 #fmass-independent --> *3 flavors

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

Gamma = np.zeros((9,len(DMMass)))

#DM a 2h
print('Calculando phi -> 2h')
for i in range(0,len(DMMass)):
    m = DMMass[i]
    if m > 2*mh:
        Gamma[0,i] = m**3*(1+2*xi(m,'h'))**2*(1-4*xi(m,'h'))**(0.5)/(32.*np.pi)
        
#DM a 2Z
print('Calculando phi -> 2Z')
for i in range(0,len(DMMass)):
    m = DMMass[i]
    if m > 2*mv[1]:
        Gamma[1,i] = m**3*(1-4*xi(m,'Z')+12*xi(m,'Z')**2)*(1-4*xi(m,'Z'))**(0.5)/(32.*np.pi)

#DM a 2W
print('Calculando phi -> 2W')
for i in range(0,len(DMMass)):
    m = DMMass[i]
    if m > 2*mv[0]:
        Gamma[2,i] = m**3*(1-4*xi(m,'W')+12*xi(m,'W')**2)*(1-4*xi(m,'W'))**(0.5)/(16.*np.pi)

#DM a 2 f f_bar
print('Calculando phi -> f f_bar')
for i in range(0,len(DMMass)):
    m = DMMass[i]
    index = np.array([0,2,4,6,7,8,9,10,11])
    for j in index:
            if m > 2*fmass[j]:
                Gamma[3,i] = 9*m**3*xi(m,'f')[j]*(1-4*xi(m,'f')[j])**(1.5)/(8.*np.pi)

#DM a q q_bar g
print('Calculando phi -> q q_bar g')
for i in range(0,len(DMMass)):
    m = DMMass[i]
    for j in range(6,12):
            if m > 2*fmass[j]:
                Gamma[4,i] = alphas*m**3/(4.*np.pi**2)

#DM a f nu_bar y W
print('Calculando phi -> f nu_bar W')
for i in range(0,len(DMMass)):
    m = DMMass[i]
    for j in range(0,12,2):
            if m > np.max([fmass[j], mv[0]]):
                Gamma[5,i] = 3.*GF*suma*mv[0]**2*m**3/(2.*np.sqrt(2)*(4.*np.pi)**3)

#DM a f f_bar y Z 
print('Calculando phi -> f f_bar y Z ')
for i in range(0,len(DMMass)):
    m = DMMass[i]
    for j in range(0,12):
            if m > np.max([fmass[j], mv[1]]):
                Gamma[6,i] = 3.*GF*(ga[j]**2+gv[j]**2)*mv[0]**2*m**3/(2.*np.sqrt(2)*(4.*np.pi)**3)

#DM a 2W 2h
print('Calculando phi -> 2W 2h')
for i in range(0,len(DMMass)):
    m = DMMass[i]
    if m > 2*mv[0]+2*mh:
        Gamma[7,i] = mv[0]**4*m**3/(15.*(8*np.pi)**5*v**4)

#DM a 2Z 2h
print('Calculando phi -> 2Z 2h')
for i in range(0,len(DMMass)):
    m = DMMass[i]
    if m > 2*mv[1]+2*mh:
        Gamma[8,i] = mv[1]**4*m**3/(30.*(8*np.pi)**5*v**4)

Gamma_tot = np.zeros(len(DMMass))

print('Gamma total')
for i in range(0,len(DMMass)):
    Gamma_tot[i] = np.sum(Gamma[:,i],axis=0)

#Branching ratios
print('Branching ratios')
BR = np.zeros((9,len(DMMass)))
for i in range(0,9):
    for j in range(0,len(DMMass)):
        BR[i,j] = Gamma[i,j]/Gamma_tot[j]
        
#Lifetime
print('lifetime')
tau1 = np.zeros(len(DMMass))
xi = 1
f = xi**2*Mp**2*kappa**4
for i in range(0,len(DMMass)):
    tau1[i] = 1/(f*Gamma_tot[i])

tau2 = np.zeros(len(DMMass))
xi = 10e-8
f = xi**2*Mp**2*kappa**4
for i in range(0,len(DMMass)):
    tau2[i] = 1/(f*Gamma_tot[i])

tau3 = np.zeros(len(DMMass))
xi = 10e-16
f = xi**2*Mp**2*kappa**4
for i in range(0,len(DMMass)):
    tau3[i] = 1/(f*Gamma_tot[i])

#Plotting
print('Plotting')
plt.figure(figsize = (10,12))
plt.title('Julio')
plt.semilogx(DMMass, BR[0,:],label = 'hh')
plt.semilogx(DMMass, BR[3,:], label = 'ff')
plt.semilogx(DMMass, BR[1,:]+BR[2,:], label = 'WW+ZZ')
plt.semilogx(DMMass, BR[4,:], label = 'qqg')
plt.semilogx(DMMass, BR[5,:]+BR[6,:], label = 'ffW+ffZ')
plt.semilogx(DMMass, BR[7,:]+BR[8,:], label = 'hhWW+hhZZ')
#plt.ylim(0.05,1)
plt.legend()
plt.grid()
plt.savefig('DM_Decay_2.1.1.png')
plt.show()

plt.figure(figsize = (10,12))
plt.title('Julio')
plt.loglog(DMMass, tau1,label ='$\chi = 1$')
plt.loglog(DMMass, tau2, label = '$\chi = 10^{-8}$')
plt.loglog(DMMass, tau3, label = '$\chi = 10^{-16}$')
plt.loglog(DMMass, cota1, label= 'Age of the universe')
plt.loglog(DMMass, cota2, label = 'Neutrino telescopes')
plt.legend()
plt.grid()
plt.savefig('DM_Decay_2.2.1.png')
plt.show()
        
    
