import strawberryfields as sf
from strawberryfields.ops import Sgate, Dgate, BSgate, MeasureFock, Fock, DensityMatrix, Coherent, Rgate, Squeezed, DisplacedSqueezed
from qutip.tensor import tensor
from qutip.measurement import measure_observable
from qutip.operators import destroy,num, displace, qeye, squeeze
from qutip.states import basis, coherent_dm, ket2dm, fock, coherent, fock_dm
from strawberryfields.backends import BaseFockState
from qutip import  wigner, Qobj, wigner_cmap, isket, fidelity, squeeze, position
import numpy as np
import cmath
import matplotlib.pyplot as plt
from matplotlib import cm
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, EventCallback
from typing import Any, Callable, Dict, List, Optional, Union
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
from stable_baselines3.common import base_class  # pytype: disable=pyi-error
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import TensorBoardOutputFormat
from scipy.special import factorial as fac
import warnings
import os
import cv2
import gym

# various helper functions used by training and evaluation scripts

def sqrGKP(mu, d, delta, cutoff, nmax=400):  
    """Square GKP code state.
    Args:
        d (int): the dimension of the code space.
        mu (int): mu=0,1,...,d-1.
        delta (float): width of the modulating Gaussian envelope.
        cutoff (int): the Fock basis truncation of the returned state vector.
        nmax (int): the Hex GKP state |mu> is calculated by performing the
            sum using n1,n1=-nmax,...,nmax.
    Returns:
        array: a size [cutoff] complex array state vector.
    """
    n1 = np.arange(-nmax, nmax+1)[:, None]
    n2 = np.arange(-nmax, nmax+1)[None, :]

    n1sq = n1**2
    n2sq = n2**2

    sqrt3 = np.sqrt(3)

    arg1 = 1j*np.pi*n2*(d*n1+mu)/d

    amplitude = (np.exp(arg1)).flatten()[:, None]

    alpha = np.sqrt(np.pi/(d)) * ((d*n1+mu - 1j*n2))

    alpha = alpha.flatten()[:, None]
    n = np.arange(cutoff)[None, :]
    coherent = np.exp(-0.5*np.abs(alpha)**2)*alpha**n/np.sqrt(fac(n))
    
    sqr_state = Qobj(np.sum(amplitude*coherent*np.exp(-n*delta**2), axis=0)).unit()
    final = sqr_state
    return final

def hexGKP(mu, d, delta, cutoff, nmax=400):
    r"""Hexagonal GKP code state.
    The Hex GKP state is defined by
    .. math::
        |mu> = \sum_{n_1,n_2=-\infty}^\infty e^{-i(q+\sqrt{3}p)/2}
            \sqrt{4\pi/\sqrt{3}d}(dn_1+\mu) e^{iq\sqrt{4\pi/\sqrt{3}d}n_2}|0>
    where d is the dimension of a code space, \mu=0,1,...,d-1, |0> is the
    vacuum state, and the states are modulated by a Gaussian envelope in the
    case of finite energy:
    ..math:: e^{-\Delta ^2 n}|\mu>
    Args:
        d (int): the dimension of the code space.
        mu (int): mu=0,1,...,d-1.
        delta (float): width of the modulating Gaussian envelope.
        cutoff (int): the Fock basis truncation of the returned state vector.
        nmax (int): the Hex GKP state |mu> is calculated by performing the
            sum using n1,n1=-nmax,...,nmax.
    Returns:
        array: a size [cutoff] complex array state vector.
    """
    n1 = np.arange(-nmax, nmax+1)[:, None]
    n2 = np.arange(-nmax, nmax+1)[None, :]

    n1sq = n1**2
    n2sq = n2**2

    sqrt3 = np.sqrt(3)

    arg1 = -1j*np.pi*n2*(d*n1+mu)/d
    arg2 = -np.pi*(d**2*n1sq+n2sq-d*n1*(n2-2*mu)-n2*mu+mu**2)/(sqrt3*d)
    arg2 *= 1-np.exp(-2*delta**2)

    amplitude = (np.exp(arg1)).flatten()[:, None]

    alpha = np.sqrt(np.pi/(2*sqrt3*d)) * (sqrt3*(d*n1+mu) - 1j*(d*n1-2*n2+mu))

    alpha = alpha.flatten()[:, None]
    n = np.arange(cutoff)[None, :]
    coherent = np.exp(-0.5*np.abs(alpha)**2)*alpha**n/np.sqrt(fac(n))
    
    hex_state = Qobj(np.sum(amplitude*coherent*np.exp(-n*delta**2), axis=0)).unit()
    final = hex_state
    return final


def getTargetState(choice, dim, sqz=0, amp=2, delt=0.4):
    delta = delt # damping strength for gkp
 
    if choice[:4] == 'cat+':
       a = amp
       psi = (coherent(dim, a) + coherent(dim, -a)).unit()
       if choice[4:] == 'pi':
          R = (-1j*(np.pi/2)*destroy(dim).dag()*destroy(dim)).expm()
          psi = R*psi
       ket = np.array(psi).flatten()
       target = BaseFockState(ket, 1, True, dim)
       dm = np.array(ket2dm(psi))

    elif choice[:4] == 'cat-':
       a = amp
       psi = (coherent(dim, a) - coherent(dim, -a)).unit()
       if choice[4:] == 'pi':
          R = (-1j*(np.pi/2)*destroy(dim).dag()*destroy(dim)).expm()
          psi = R*psi
       ket = np.array(psi).flatten()
       target = BaseFockState(ket, 1, True, dim)
       dm = np.array(ket2dm(psi))
    
    elif choice[:7] == 'sqzcat+':
       a = amp
       r = sqz
       psi = (coherent(dim, a) + coherent(dim, -a)).unit()
       psi = (squeeze(dim, r) * psi).unit()
       if choice[7:] == 'pi':
          R = (-1j*(np.pi/2)*destroy(dim).dag()*destroy(dim)).expm()
          psi = R*psi
       ket = np.array(psi).flatten()
       target = BaseFockState(ket, 1, True, dim)
       dm = np.array(ket2dm(psi))

    elif choice[:7] == 'sqzcat-':
       a = amp
       r = sqz
       psi = (coherent(dim, a) - coherent(dim, -a)).unit()
       psi = (squeeze(dim, r) * psi).unit()
       if choice[7:] == 'pi':
          R = (-1j*(np.pi/2)*destroy(dim).dag()*destroy(dim)).expm()
          psi = R*psi
       ket = np.array(psi).flatten()
       target = BaseFockState(ket, 1, True, dim)
       dm = np.array(ket2dm(psi))
    
    elif choice == 'gkp0':
        d = 2
        mu = 0
        psi = sqrGKP(mu, d , delta, dim)
        ket = np.array(psi).flatten()
        target = BaseFockState(ket, 1, True, dim)
        dm = np.array(ket2dm(psi))

    elif choice == 'gkp1':
        d = 2
        mu = 1
        psi = sqrGKP(mu, d , delta, dim)
        ket = np.array(psi).flatten()
        target = BaseFockState(ket, 1, True, dim)
        dm = np.array(ket2dm(psi))
    
    elif choice == 'qunot':
        d = 1
        mu = 0
        psi = sqrGKP(mu, d , delta, dim)
        ket = np.array(psi).flatten()
        target = BaseFockState(ket, 1, True, dim)
        dm = np.array(ket2dm(psi))
        
    elif choice == 'hexnot':
        d = 1
        mu = 0
        R = (-1j*(2*np.pi/3)*destroy(dim).dag()*destroy(dim)).expm()
        psi = R*hexGKP(mu, d , delta, dim)
        ket = np.array(psi).flatten()
        target = BaseFockState(ket, 1, True, dim)
        dm = np.array(ket2dm(psi))

    elif choice == 'hexnot-pi':
        d = 1
        mu = 0
        R1 = (-1j*(2*np.pi/3)*destroy(dim).dag()*destroy(dim)).expm()
        R2 = (-1j*(np.pi/2)*destroy(dim).dag()*destroy(dim)).expm()
        psi = R2*R1*hexGKP(mu, d , delta, dim)
        ket = np.array(psi).flatten()
        target = BaseFockState(ket, 1, True, dim)
        dm = np.array(ket2dm(psi))

    elif choice == 'hex0':
        d = 2
        mu = 0
        psi = hexGKP(mu, d , delta, dim)
        ket = np.array(psi).flatten()
        target = BaseFockState(ket, 1, True, dim)
        dm = np.array(ket2dm(psi))
        
    elif choice == 'hex0-pi':
        d = 2
        mu = 0
        R = (-1j*(np.pi/2)*destroy(dim).dag()*destroy(dim)).expm()
        psi = hexGKP(mu, d , delta, dim)
        psi = R*psi
        ket = np.array(psi).flatten()
        target = BaseFockState(ket, 1, True, dim)
        dm = np.array(ket2dm(psi))
        
    elif choice == 'gkp0-pi':
        d = 2
        mu = 0
        R = (-1j*(np.pi/2)*destroy(dim).dag()*destroy(dim)).expm()
        psi = sqrGKP(mu, d , delta, dim)
        psi = R*psi
        ket = np.array(psi).flatten()
        target = BaseFockState(ket, 1, True, dim)
        dm = np.array(ket2dm(psi))

    elif choice == 'hex1':
        d = 2
        mu = 1
        psi = hexGKP(mu, d , delta, dim)
        ket = np.array(psi).flatten()
        target = BaseFockState(ket, 1, True, dim)
        dm = np.array(ket2dm(psi))

    elif choice[:4] == 'fock':
       psi = fock(dim, int(choice[4]))
       ket = np.array(psi).flatten()
       target = BaseFockState(ket, 1, True, dim)
       dm = np.array(ket2dm(psi))
        
    elif choice[:5] == 'super':
       psi = (fock(dim, int(choice[5])) + fock(dim, int(choice[6]))).unit()
       ket = np.array(psi).flatten()
       target = BaseFockState(ket, 1, True, dim)
       dm = np.array(ket2dm(psi))

    elif choice == 'bin':
       psi = 0.5*(fock(dim, 0) + np.sqrt(3)*fock(dim, 4))
       ket = np.array(psi).flatten()
       target = BaseFockState(ket, 1, True, dim)
       dm = np.array(ket2dm(psi))

       
    return(target, dm)


# helper function to plot Wigner funcs and photon number dists
def plotState(dm, Qmax, points, numcut, name, fid=-1, filename='', step=None): 
        # Qmax: highest quadrature value at the plot edge
        # points: number of points in one axis from 0 out
        # numcut: define the maximum photon number to show for the probability distribution
        dx=Qmax/points
        xvec = np.arange(-points,points+1)*dx
        X,Y = np.meshgrid(xvec, xvec)  
        W = wigner(Qobj(dm), xvec, xvec)
    
        fig = plt.figure(figsize=(16, 8), dpi=180 )
        plt.rcParams.update({'font.size': 10})
        
        ax = fig.add_subplot(2, 3, 1)
        font = 15
        if step != None:
            if len(step) > 10:
                font = 9.5
            else:
                font = 15
        #define a surface plot and use vmin and vmax to set the limits of the colorscale to ensure the middle is at the origin
        p = ax.contourf(X, Y, W, 60, cmap=cm.RdBu, vmin=-1/np.pi,vmax=1/np.pi)
        ax.set_ylabel('P', fontsize=20)
        if step == None:
           ax.set_xlabel('Q \n\n' + r'$\rho_{'+name+'}$', fontsize=20)
        else:
            label = "Q \n"
            for i in range(len(step)):
                st = str(i+1)
                label += '\n'+r'$t_{'+st+'}$: ' + str(round(step[i]['t'], 2))
                label += r'  $r_{'+st+'}$: ' + str(round(step[i]['r'], 2))
                label += r'  $n_{'+st+'}$: ' + str(step[i]['PNR'])
                label += r'  $φ_{'+st+'}$: ' + str(round(step[i]['phi'], 2))
                label += r'  $F_{'+st+'}$: ' + str(round(step[i]['F'], 3))
                
            ax.set_xlabel(label, fontsize=font)
        if fid == -1:
           ax.set_title("Wigner function of "+name+" state")
        else:
           ax.set_title("Wigner function of "+name+" state\n Fidelity: "+str(round(fid*100, 2)) + "%" )
        cb = fig.colorbar(p, shrink = 0.8) # add a colorbar
        
        # surface_plot with color grading and color bar
        ax = fig.add_subplot(2, 3, 2, projection='3d')
        p = ax.plot_surface(X, Y, W, rstride=1, cstride=1, cmap=cm.RdBu, vmin=-1/np.pi,vmax=1/np.pi, linewidth=0.5)
        ax.set_ylabel('P')
        ax.set_xlabel('Q')
        ax.set_title("Wigner function of "+name+" state")
        cb = fig.colorbar(p, shrink=.66, pad=0.05)
        ax = fig.add_subplot(2, 3, 3)
        probs = np.real(dm.diagonal()[0:numcut])
        avg = 0
        for i in range(numcut):
            avg += i*(probs[i])
        tr = sum(probs)
        p = ax.bar(range(numcut), probs)
        if numcut > 30:
           ax.set_xticks(range(0, numcut+1, 5))
        else:
            ax.set_xticks(range(0, numcut+1, 2))
        ax.set_ylabel('P(n)')
        if step == None:
           ax.set_xlabel('n', fontsize=font)
        else:
            label = "n \n" 
            for i in range(len(step)):
                st = str(i+1)
                label += '\n'+r'tr$(\rho_{'+st+'})$: ' + str(round(step[i]['Tr-pnr'], 4))
                label +=  r'  $P(\vec{n}_{'+st+'})$: '  + str(round(100*step[i]['P'], 3)) + '%'
            ax.set_xlabel(label, fontsize=font)
        ax.set_title(f"Photon number distribution of {name} state.\n" r"$\bar{n}$" + f": {round(avg,2)}" + r"    tr$(\rho)$" + f": {round(tr, 4)}" )

        plt.savefig(filename+".png", dpi=120)
        plt.clf()
        plt.close()
        #return(W, fig)

# to create the animations of the episodes
def animate(imgFolder, vidName):
        image_folder = imgFolder
        video_name = vidName+'.avi'

        images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
        images = sorted(images, key = lambda x: int(x.split('-')[0])) # sort images by step number
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape
        # save at 0.08 frames per second
        video = cv2.VideoWriter(image_folder+video_name, 0, 0.08, (width,height))

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        cv2.destroyAllWindows()
        video.release()        


def makeHist(numBins, finalPis, finalhalfPis, finalWigs, finalFourPis, \
             finalFids, numEvalEpisodes, modelName, sqzMax, step1BSt, step2BSt, step3BSt, \
            step1r, step2r, step3r):
    plt.rcParams.update({'font.size': 12})
    fig = plt.figure(figsize=(16, 8), dpi=180)
    
    ax = fig.add_subplot(4, 2, 1)
    bins = np.linspace(0, 1., numBins)
    ax.axis(xmin=-0.1, xmax=1.1) # xlim([0, 1])
    
    bins = np.linspace(-0.1, 1.1, numBins)
    ax.axis(xmin=-0.1, xmax=1.1) # xlim([0, 1])
    ax.hist(finalFourPis, bins=bins, alpha=0.8)
    ax.set_title(f'Final state $\pi/4$ fidelities for {numEvalEpisodes} evaluation episodes.')
    ax.set_xlabel(f'Final state fidelity with itself rotated by $\pi/4$ using {numBins} bins')
    ax.set_ylabel('count')
    
    ax = fig.add_subplot(4, 2, 2)
    bins = np.linspace(-0.1, 1.1, numBins)
    ax.axis(xmin=-0.1, xmax=1.1) # xlim([0, 1])
    ax.hist(finalhalfPis, bins=bins, alpha=0.8)
    ax.set_title(f'Final state $\pi/2$ fidelities for {numEvalEpisodes} evaluation episodes.')
    ax.set_xlabel(f'Final state fidelity with itself rotated by $\pi/2$ using {numBins} bins')
    ax.set_ylabel('count')
    
    ax = fig.add_subplot(4, 2, 3)
    bins = np.linspace(-0.1, 1.1, numBins)
    ax.axis(xmin=-0.1, xmax=1.1) # xlim([0, 1])
    ax.hist(finalPis, bins=bins, alpha=0.8)
    ax.set_title(f'Final state $\pi$ fidelities for {numEvalEpisodes} evaluation episodes.')
    ax.set_xlabel(f'Final state fidelity with itself rotated by $\pi$ using {numBins} bins')
    ax.set_ylabel('count')
    
    ax = fig.add_subplot(4, 2, 4)
    bins = np.linspace(-0.1, 3., numBins)
    ax.axis(xmin=-0.1, xmax=3.) # xlim([0, 1])
    ax.hist(finalWigs, bins=bins, alpha=0.8)
    ax.set_title(f'Negative volume of Wigner function of final state for {numEvalEpisodes} evaluation episodes.')
    ax.set_xlabel(f'Negative volume of Wigner function of final state using {numBins} bins')
    ax.set_ylabel('count')
    
    
    ax = fig.add_subplot(4, 2, 5)
    bins = np.linspace(-0.1, 1.1, numBins)
    ax.axis(xmin=-0.1, xmax=1.1)
    ax.hist(finalFids, bins=bins, alpha=0.8)
    ax.set_xticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9, 1.0])
    ax.set_title(f'Fidelity of final state with target state for {numEvalEpisodes} evaluation episodes.')
    ax.set_xlabel(f'Final state fidelity with target state using {numBins} bins')
    ax.set_ylabel('count')
    
    ax = fig.add_subplot(4, 2, 6)
    bins = np.linspace(-0.1, 1.1, numBins)
    ax.axis(xmin=-0.1, xmax=1.1)
    ax.hist(step1BSt, bins=bins, alpha=0.8, label='step 1')
    ax.hist(step2BSt, bins=bins, alpha=0.8, label='step 2')
    ax.hist(step3BSt, bins=bins, alpha=0.8, label='step 3')
    ax.set_title(f'Transmittivity for steps 1-3 for {numEvalEpisodes} evaluation episodes.')
    ax.set_xlabel(f'Transmittivity value of steps 1-3 using {numBins} bins')
    ax.set_ylabel('count')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    ax = fig.add_subplot(4, 2, 7)
    bins = np.linspace(-sqzMax-0.1, sqzMax+0.1, numBins)
    ax.axis(xmin=-sqzMax-0.1, xmax=sqzMax+0.1)
    ax.hist(step1r, bins=bins, alpha=0.8, label='step 1')
    ax.hist(step2r, bins=bins, alpha=0.8, label='step 2')
    ax.hist(step3r, bins=bins, alpha=0.8, label='step 3')
    ax.set_title(f'Squeezing for steps 1-3 for {numEvalEpisodes} evaluation episodes.')
    ax.set_xlabel(f'Squeezing value of steps 1-3 using {numBins} bins')
    ax.set_ylabel('count')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    fig.tight_layout()
    plt.savefig('evals/'+modelName+"/eval-histogram.png", dpi=180)
    
    plt.clf()
    plt.close()
    


class TimestepCallbackMulti(BaseCallback):
    """
    Callback used for logging timestep data.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        # Log scalar values 
        infos = self.locals["infos"]
        t = []
        r = []
        F = []
        n = []
        phi = []
        tr = []
        for i in range(len(infos)):

            t.append(infos[i]['t'])

            
            r.append(infos[i]['r'])


            F.append(infos[i]['F'])
           
            tr.append(infos[i]['Tr-pnr'])
            phi.append(infos[i]['phi'])
                        
            n.append(infos[i]['PNR'])

        
        self.logger.record(f"timestep/avg_transmitivity", np.mean(t))

        self.logger.record(f"timestep/avg_squeezing", np.mean(r))

        self.logger.record(f"timestep/avg_trace", np.mean(tr))
        self.logger.record(f"timestep/avg_phi", np.mean(phi))

        self.logger.record(f"timestep/avg_fidelity", np.mean(F))
        self.logger.record(f"timestep/avg_PNR", np.mean(n))
        self.logger.record(f"timestep/trace_env0", infos[0]['Tr-pnr'])
        self.logger.record(f"timestep/trace_env1", infos[1]['Tr-pnr'])
       
        self.logger.dump(self.num_timesteps)

        return(True)

    
class TimestepCallback(BaseCallback):
    """
    Callback used for logging timestep data.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        # Log scalar values 
        
        t = self.training_env.buf_infos[0]['t']
        n = self.training_env.buf_infos[0]['PNR']
        F = self.training_env.buf_infos[0]['F']
        r = self.training_env.buf_infos[0]['r']
        phi = self.training_env.buf_infos[0]['phi']
        Tr = self.training_env.buf_infos[0]['Tr-pnr']
        
        self.logger.record(f"timestep/transmitivity", t)
        
        self.logger.record(f"timestep/squeezing", r)
        
        self.logger.record(f"timestep/fidelity", F)
        
        self.logger.record(f"timestep/PNR", n)
        
        self.logger.record(f"timestep/trace", Tr)
        
        self.logger.record(f"timestep/phi", phi)
        
        self.logger.dump(self.num_timesteps)
        
        return(True)
    

class EpisodeCallbackMulti(BaseCallback):
    """
    Callback used for logging episode data.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        # Custom counter to reports stats
        # (and avoid reporting multiple values for the same step)
        self._episode_counter = 0
        self._tensorboard_writer = None

    def _init_callback(self) -> None:
        assert self.logger is not None
        # Retrieve tensorboard writer to not flood the logger output
        for out_format in self.logger.output_formats:
            if isinstance(out_format, TensorBoardOutputFormat):
                self._tensorboard_writer = out_format
        assert self._tensorboard_writer is not None, "You must activate tensorboard logging when using RawStatisticsCallback"

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        for i in range(len(infos)):
            if self.locals['dones'][i]:
                #self.logger.record("episode/episode_return", infos[i]["r"])
                #self.logger.record("episode/episode_length", infos[i]["l"])
                self.logger.record(f"episode/final_fidelity", infos[i]["F"])
                self.logger.record(f"episode/success_probability", infos[i]["P"])
                self._episode_counter += 1

        self.logger.dump(self._episode_counter)                

        return True


class EpisodeCallback(BaseCallback):
    """
    Callback used for logging episode data.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        # Custom counter to reports stats
        # (and avoid reporting multiple values for the same step)
        self._episode_counter = 0
        self._tensorboard_writer = None

    def _init_callback(self) -> None:
        assert self.logger is not None
        # Retrieve tensorboard writer to not flood the logger output
        for out_format in self.logger.output_formats:
            if isinstance(out_format, TensorBoardOutputFormat):
                self._tensorboard_writer = out_format
        assert self._tensorboard_writer is not None, "You must activate tensorboard logging when using RawStatisticsCallback"

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" in info: # if episode ends
                logger_dict = {
                    "episode/episode_return": info["episode"]["r"],
                    "episode/episode_length": info["episode"]["l"],
                    "episode/final_fidelity": info["F"],
                    "episode/success_probability": info["P"]
                }
                exclude_dict = {key: None for key in logger_dict.keys()}
                self._episode_counter += 1
                self._tensorboard_writer.write(logger_dict, exclude_dict, self._episode_counter)

        return True
    
 
class CheckpointCallback(BaseCallback):
    """
    Callback for saving a model every ``save_freq`` calls
    to ``env.step()``.
    By default, it only saves model checkpoints,
    you need to pass ``save_replay_buffer=True``,
    and ``save_vecnormalize=True`` to also save replay buffer checkpoints
    and normalization statistics checkpoints.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``

    :param save_freq: Save checkpoints every ``save_freq`` call of the callback.
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param save_replay_buffer: Save the model replay buffer
    :param save_vecnormalize: Save the ``VecNormalize`` statistics
    :param verbose: Verbosity level: 0 for no output, 2 for indicating when saving model checkpoint
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "rl_model",
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.save_replay_buffer = save_replay_buffer
        self.save_vecnormalize = save_vecnormalize

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _checkpoint_path(self, checkpoint_type: str = "", extension: str = "") -> str:
        """
        Helper to get checkpoint path for each type of checkpoint.

        :param checkpoint_type: empty for the model, "replay_buffer_"
            or "vecnormalize_" for the other checkpoints.
        :param extension: Checkpoint file extension (zip for model, pkl for others)
        :return: Path to the checkpoint
        """
        return os.path.join(self.save_path, f"{self.name_prefix}.{extension}")

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = self._checkpoint_path(extension="zip")
            self.model.save(model_path)
            if self.verbose >= 2:
                print(f"Saving model checkpoint to {model_path}")

            if self.save_replay_buffer and hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
                # If model has a replay buffer, save it too
                replay_buffer_path = self._checkpoint_path("replay_buffer_", extension="pkl")
                self.model.save_replay_buffer(replay_buffer_path)
                if self.verbose > 1:
                    print(f"Saving model replay buffer checkpoint to {replay_buffer_path}")

            if self.save_vecnormalize and self.model.get_vec_normalize_env() is not None:
                # Save the VecNormalize statistics
                vec_normalize_path = self._checkpoint_path("vecnormalize_", extension="pkl")
                self.model.get_vec_normalize_env().save(vec_normalize_path)
                if self.verbose >= 2:
                    print(f"Saving model VecNormalize to {vec_normalize_path}")

        return True
    


def schedule(initial_lr: float) -> Callable[[float], float]:
    """
    Step-function learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progressRemaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        if progressRemaining > 0.9:
           learningRate = initial_lr
        
        else:
           learningRate = initial_lr/2.0
        
        
        return learningRate

    return func