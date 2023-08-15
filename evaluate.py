import numpy as np
import json
import os
import sys
import pprint
from circuit import Circuit
from stable_baselines3 import PPO
from utils import *

modelName = sys.argv[3]

print("Model name:", modelName, flush=True)
params = modelName.split('_')[:24]
seed = int(sys.argv[2])

maxSteps = int(params[3]) 
dim = int(params[1])
exp = float(params[7]) 
reward = params[9]
initialSqz = float(params[5])
sqzMax = float(params[11])
target = params[13]
lstm = params[15] == 't'
phi = params[17] == 't' 
loss = float(params[19])

numEvalEpisodes = int(sys.argv[1])
deterministic = True 
params = {params[i]: params[i + 1] for i in range(0, len(params), 2)}

# create evaluation env
env = Circuit(dim, initialSqz, maxSteps, exp, sqzMax, phi, target,\
                                   reward, seed, evaluate=True, loss=loss)


model = PPO.load('models/'+modelName+"/rl_model", env=env)
evalmodelname = modelName+"_seed_"+str(seed)
os.makedirs('evals/'+evalmodelname, exist_ok=True)


finalPis = np.array([])
finalhalfPis = np.array([])
finalfourPis = np.array([])
finalWigs = np.array([])
finalFids = np.array([])
#finalProbs = np.array([])
step1BSt = np.array([])
step2BSt = np.array([])
step3BSt = np.array([])
step1r = np.array([])
step2r = np.array([])
step3r = np.array([])


lstm_states = None
episode_starts = np.ones((1,), dtype=bool)

e = 0

while e < numEvalEpisodes:
  obs = env.reset()
  i = 0 
  done = False
  while not done:
      if lstm:
          action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
          action, _states = model.predict(obs, deterministic=deterministic)
          obs, reward, done, info = env.step(action)
          episode_starts = done
      else:
          action, _ = model.predict(obs, deterministic=deterministic)
          obs, reward, done, info = env.step(action)
            
      i += 1

    
  finalfourPis = np.append(finalfourPis, env.steps[-1]['4Pi'])
  finalPis = np.append(finalPis, env.steps[-1]['1Pi'])
  finalhalfPis = np.append(finalhalfPis, env.steps[-1]['2Pi'])
  finalWigs = np.append(finalWigs, env.steps[-1]['wigner'])
  #finalProbs = np.append(finalProbs, env.steps[-1]['P'])
  finalFids = np.append(finalFids, env.steps[-1]['F'])
  step1BSt = np.append(step1BSt, env.steps[0]['t'])
  step2BSt = np.append(step2BSt, env.steps[1]['t']) 
  step3BSt = np.append(step3BSt, env.steps[2]['t'])
  step1r = np.append(step1r, env.steps[0]['r'])
  step2r = np.append(step2r, env.steps[1]['r']) 
  step3r = np.append(step3r, env.steps[2]['r'])
  e += 1

  if e < 300:      
     env.render(False, name='episode '+str(e), filename='evals/'+evalmodelname+'/plot-'+str(e), steps=env.steps)

print("Random seed:", seed, "\n")
print("Max steps:", maxSteps, "\n")
print("Total number of episodes:", e, "\n")
print("Number of episodes with final fidelity 90% of more:", (finalFids >= .90).sum())
print("Number of episodes with final fidelity 95% of more:", (finalFids >= .95).sum())
print("Number of episodes with final fidelity 96% of more:", (finalFids >= .96).sum())
print("Number of episodes with final fidelity 97% of more:", (finalFids >= .97).sum())
print("Number of episodes with final fidelity 98% of more:", (finalFids >= .98).sum())
print("Number of episodes with final fidelity 99% of more:", (finalFids >= .99).sum())

numBins = 200
makeHist(numBins, finalPis, finalhalfPis, finalWigs, finalfourPis, finalFids,\
         numEvalEpisodes, evalmodelname, sqzMax, step1BSt, step2BSt, step3BSt, step1r, step2r, step3r)
