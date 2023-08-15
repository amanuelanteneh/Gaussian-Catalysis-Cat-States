import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np
from qutip import Qobj, fidelity, destroy
from gym import Env
from gym import spaces
from utils import plotState, getTargetState
import scipy
from scipy.integrate import simpson
from numpy import pi, sqrt, diagonal, triu_indices, arccos, concatenate

class Circuit(Env): # the time-multiplexed optical circuit (the enviornment)

    def __init__(self, dim, initialSqz, 
                 maxTimesteps, exp, sqzMax, phi, 
                 target='fock2', reward='single', seed=None, evaluate=False, loss=0):
        
        if seed != None:
           np.random.seed(seed) # set random seed for env instance
        self.dim = dim # dimension of hilbert space 
        self.rewardMethod = reward
        self.eval = evaluate
        self.Qmax = 7
        self.X = np.arange(-self.Qmax, self.Qmax, 0.1)
        self.eng = sf.Engine("fock", backend_options={"cutoff_dim": self.dim})
        self.prog = sf.Program(2)
        self.exp = exp # penalty exponenet
        self.sqzMag = np.abs(initialSqz)
        self.sqzAngle = np.angle(initialSqz)
        self.sqzMax = sqzMax
        self.successProb = 1
        self.phi = phi
        self.loss = loss
        self.lossy = loss > 0.0
        self.nhat = destroy(self.dim).dag()*destroy(self.dim)
        theta = pi/2
        self.Rpi2 = (1j*theta*self.nhat).expm() # pi/2 rotation operator
        theta = pi
        self.Rpi = (1j*theta*self.nhat).expm()
        theta = pi/4
        self.Rpi4 = (1j*theta*self.nhat).expm()
        
        if target[:8] == 'sqzcatpi':
            params = target.split('%')
            s = float(params[2])
            a = float(params[1])
            self.target1, self.dmTarget1 = getTargetState('sqzcat+', dim, s, a)
            self.target2, self.dmTarget2 = getTargetState('sqzcat-', dim, s, a)
            self.target3, self.dmTarget3 = getTargetState('sqzcat+pi', dim, s, a)
            self.target4, self.dmTarget4 = getTargetState('sqzcat-pi', dim, s, a)            
            self.target = None
        elif target[:6] == 'sqzcat':
            params = target.split('%')
            s = float(params[2])
            a = float(params[1])
            self.target1, self.dmTarget1 = getTargetState('sqzcat+', dim, s, a)
            self.target2, self.dmTarget2 = getTargetState('sqzcat-', dim, s, a)
            self.target = None
        elif target[:5] == 'catpi':
            params = target.split('%')
            a = float(params[1])
            self.target1, self.dmTarget1 = getTargetState('cat+', dim, 0, a)
            self.target2, self.dmTarget2 = getTargetState('cat-', dim, 0, a)
            self.target3, self.dmTarget3 = getTargetState('cat+pi', dim, 0, a)
            self.target4, self.dmTarget4 = getTargetState('cat-pi', dim, 0, a)            
            self.target = None
        elif target[:3] == 'cat':
            params = target.split('%')
            a = float(params[1])
            self.target1, self.dmTarget1 = getTargetState('cat+', dim, 0, a)
            self.target2, self.dmTarget2 = getTargetState('cat-', dim0, a)
            self.target = None
        elif target[:3] == 'gkp':
            params = target.split('%')
            delt = float(params[1])
            self.target1, self.dmTarget1 = getTargetState('gkp+', dim, delt=delt)
            self.target2, self.dmTarget2 = getTargetState('gkp-', dim, delt=delt)
            self.target = None
        elif target[:3] == 'hex':
            params = target.split('%')
            delt = float(params[1])
            self.target1, self.dmTarget1 = getTargetState('hex+', dim, delt=delt)
            self.target2, self.dmTarget2 = getTargetState('hex-', dim, delt=delt)
            self.target = None
        elif target[:6] == 'hex0pi':
            params = target.split('%')
            delt = float(params[1])
            self.target1, self.dmTarget1 = getTargetState('hex0', dim, delt=delt)
            self.target2, self.dmTarget2 = getTargetState('hex0-pi', dim, delt=delt)
            self.target = None
        elif target[:6] == 'gkp0pi':
            self.target1, self.dmTarget1 = getTargetState('gkp0', dim, delt=delt)
            self.target2, self.dmTarget2 = getTargetState('gkp0-pi', dim, delt=delt)
            self.target = None
        elif target[:8] == 'hexnotpi':
            self.target1, self.dmTarget1 = getTargetState('hexnot', dim, delt=delt)
            self.target2, self.dmTarget2 = getTargetState('hexnot-pi', dim, delt=delt)
            self.target = None
        elif target[:5] == 'qunot':
            params = target.split('%')
            delt = 0.4 #float(params[1])
            self.target, self.dmTarget = getTargetState(target, dim, delt=delt)
        else:
            self.target, self.dmTarget = getTargetState(target, dim)
            
        eng = sf.Engine("fock", backend_options={"cutoff_dim": dim})
        
        self.initial = Squeezed(r=self.sqzMag, p=self.sqzAngle)
        
        with self.prog.context as q: #create inital squeezed state
            self.initial | q[0]
        
        result = self.eng.run(self.prog)
        self.psi = result.state
        self.dm = self.psi.reduced_dm([0]) # reduced density matrix
        self.t = 0 # the current time step
        self.T = maxTimesteps # max number of iterations/time steps
        self.steps = []
        
        # state space
        self.observation_space = spaces.Box(low=-1, high=1, shape=( (self.dim**2), ), dtype=np.float32) 
        
        # action space
        minAction = [-1.0, -1.0]
        maxAction = [1.0, 1.0]
        if self.phi:
           minAction.append(-1.0) 
           maxAction.append(1.0)
             
        self.action_space = spaces.Box(low=np.array(minAction).astype(np.float32),\
                                       high=np.array(maxAction).astype(np.float32), dtype=np.float32) 
        
        
    def step(self, action, postselect=None):
        
        transmittivity = (action[0] + 1.0)/2.0 # rescale range from [-1,1] -> [0, 1]
        theta = arccos(transmittivity)
        r = action[1]*self.sqzMax
        if self.phi:
           phi = pi*(action[2] + 1.0)
        else:
           phi = 0
        
        self.prog = sf.Program(2) # create 2 mode circuit
        with self.prog.context as q:
             Squeezed(r, 0) | q[1] 
             DensityMatrix(self.dm) | q[0] # set mode 1 to be output state from prev. step
             BSgate(theta, phi) | (q[0], q[1])
             
        result = self.eng.run(self.prog)
        if self.eval:
           dm0 = result.state.reduced_dm([0]) # density matrix of mode 1 before pnr projection
        tracePNR = result.state.trace() # get trace of simulation step *before* PNR
        
        if postselect == None: # for training/evaluation
            measureProg = sf.Program(self.prog) 
            with measureProg.context as q:
                 if self.lossy:
                    LossChannel(1 - self.loss) | q[0]
                 MeasureFock() | q[0] # stochastic PNR measurement on mode 1
        else: 
            measureProg = sf.Program(self.prog) 
            with measureProg.context as q:
                 if self.lossy:
                    LossChannel(1 - self.loss) | q[0]
                 MeasureFock(select=postselect) | q[0] # postselected PNR measurement on mode 1
        
        result = self.eng.run(measureProg)
        n = result.samples[0][0] # the number of detected photons
        
        if self.eval: # if in evalution mode
            Pn = np.real(dm0[n][n])
            if transmittivity == 1.0: # if agent resets loop
               self.successProb = 1
               Pn = 1
            elif transmittivity == 0.0: # if agent turns off BS
                Pn = 1
            self.successProb *= Pn
        self.t += 1 # increment time step
        self.psi = result.state 
        self.dm = self.psi.reduced_dm([1]) # trace out mode 1
        
        done = self.t == self.T
        
        if self.rewardMethod == 'single':
           F = self.psi.fidelity(self.target.ket(), 1) 
           reward = F**self.exp   
        elif self.rewardMethod == 'dual': # for 2 target states
           F1 = self.psi.fidelity(self.target1.ket(), 1)  
           F2 = self.psi.fidelity(self.target2.ket(), 1)
           F = max(F1, F2)
           reward = F**self.exp
        elif self.rewardMethod == 'four': # for 4 target states
           F1 = self.psi.fidelity(self.target1.ket(), 1)  
           F2 = self.psi.fidelity(self.target2.ket(), 1)
           F3 = self.psi.fidelity(self.target3.ket(), 1)  
           F4 = self.psi.fidelity(self.target4.ket(), 1)
           F = max(F1, F2, F3, F4)
           reward = F**self.exp       
        elif self.rewardMethod == 'sym': 
           halfPiReward, fourPiReward = self.getRotSymReward(self.rewardMethod)
           reward = (halfPiReward * (1-fourPiReward) )**self.exp
           F = reward        

        info = { "Timestep": self.t,
                 "t": transmittivity,
                 "F": F,
                 "phi": phi,
                 "PNR": int(n),
                 "r": r,
                 "Tr-pnr": tracePNR,
                 "P": self.successProb}
        
        if self.eval and done: # for evaluation mode
            W =  result.state.wigner(1, self.X, self.X)
            Wabs = np.absolute(W)
            # calculate negative volume of Wigner function
            info['wigner'] = simpson(simpson(Wabs, self.X), self.X) - simpson(simpson(W, self.X), self.X)
        
            # we dont use displacements or homodyne detection so there should be no displacement in 
            # the quadratures to account for so no need to explicitly center the state
            centeredState = Qobj(self.dm)
            
            stateHalfPi = self.Rpi2 * centeredState * self.Rpi2.dag()
            statePi = self.Rpi * centeredState * self.Rpi.dag()
            stateFourPi = self.Rpi4 * centeredState * self.Rpi4.dag()
        
            piReward = fidelity(statePi, centeredState)**2  
            info['1Pi'] = piReward
            
            halfPiReward = fidelity(stateHalfPi, centeredState)**2
            info['2Pi'] = halfPiReward
            
            fourPiReward = fidelity(stateFourPi, centeredState)**2
            info['4Pi'] = fourPiReward

        self.steps.append(info)
        
        state = self.dm[triu_indices(self.dim, k=1)] # get values above diagonal since dm is hermitian
        diag = diagonal(self.dm) # also get diagonal
        state = concatenate((np.real(state), np.imag(state)), dtype=np.float32, axis=None)
        state = concatenate((state, np.real(diag)), dtype=np.float32, axis=None)

        return(state, reward, done, info)
        
    def render(self, target=True, name="", filename="", steps=None):
        # plot the Wigner function and photon number distribution
        if target and self.target != None:
           plotState(dm=self.dmTarget, Qmax=self.Qmax, points=75, numcut=self.dim,\
                     name=name, fid=-1, filename=filename, step=steps) 
        elif target and self.target == None and self.rewardMethod != 'four':
           plotState(dm=self.dmTarget1, Qmax=self.Qmax, points=75, numcut=self.dim,\
                     name=name+'-0', fid=-1, filename=filename+'-0', step=steps)
           plotState(dm=self.dmTarget2, Qmax=self.Qmax, points=75, numcut=self.dim,\
                     name=name+'-1', fid=-1, filename=filename+'-1', step=steps)
        elif target and self.target == None and self.rewardMethod == 'four':
           plotState(dm=self.dmTarget1, Qmax=self.Qmax, points=75, numcut=self.dim,\
                     name=name+'-0', fid=-1, filename=filename+'-0', step=steps)
           plotState(dm=self.dmTarget2, Qmax=self.Qmax, points=75, numcut=self.dim,\
                     name=name+'-1', fid=-1, filename=filename+'-1', step=steps)
           plotState(dm=self.dmTarget3, Qmax=self.Qmax, points=75, numcut=self.dim,\
                     name=name+'-0-pi2', fid=-1, filename=filename+'-0-pi2', step=steps)
           plotState(dm=self.dmTarget4, Qmax=self.Qmax, points=75, numcut=self.dim,\
                     name=name+'-1-pi2', fid=-1, filename=filename+'-1-pi2', step=steps)

        elif not target and self.target != None:
            F = self.psi.fidelity(self.target.ket(), 1)
            plotState(dm=self.dm, Qmax=self.Qmax, points=75, numcut=self.dim,\
                      name=name, fid=F, filename=filename, step=steps)
        elif not target and self.target == None and self.rewardMethod != 'four':
            F1 = self.psi.fidelity(self.target1.ket(), 1)
            F2 = self.psi.fidelity(self.target2.ket(), 1)
            plotState(dm=self.dm, Qmax=self.Qmax, points=75, numcut=self.dim,\
                      name=name, fid=max(F1,F2), filename=filename, step=steps) 
        elif not target and self.target == None and self.rewardMethod == 'four':
            F1 = self.psi.fidelity(self.target1.ket(), 1)
            F2 = self.psi.fidelity(self.target2.ket(), 1)
            F3 = self.psi.fidelity(self.target3.ket(), 1)
            F4 = self.psi.fidelity(self.target4.ket(), 1)            
            plotState(dm=self.dm, Qmax=self.Qmax, points=75, numcut=self.dim,\
                      name=name, fid=max(F1,F2, F3, F4), filename=filename, step=steps) 

    def reset(self):
        # reset the environment
        self.t = 0
        self.steps = []
        self.eng = sf.Engine("fock", backend_options={"cutoff_dim": self.dim})
        self.prog = sf.Program(2)
        self.successProb = 1
        with self.prog.context as q: #create inital squeezed state
            self.initial | q[0]
        
        result = self.eng.run(self.prog)
        self.psi = result.state  
        self.dm = self.psi.reduced_dm([0])

        state = self.dm[triu_indices(self.dim, k=1)]
        diag = diagonal(self.dm)
        state = np.concatenate((np.real(state), np.imag(state)), dtype=np.float32, axis=None)
        state = np.concatenate((state, np.real(diag)), dtype=np.float32, axis=None)

        return(state)
    
    
    def getRotSymReward(self, rewardMethod):
            # we dont use displacements or homodyne detection so there should be no displacement in 
            # the quadratures to account for so no need to explicitly center the state
            centeredState = Qobj(self.dm)
            
            stateHalfPi = self.Rpi2 * centeredState * self.Rpi2.dag()
            stateFourPi = self.Rpi4 * centeredState * self.Rpi4.dag()             
            halfPiReward = fidelity(stateHalfPi, centeredState)**2  
            fourPiReward = fidelity(stateFourPi, centeredState)**2
            
            return(halfPiReward, fourPiReward)