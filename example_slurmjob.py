'''
example_slurmjob.py

An example of a parallelizable numpyro job for use on slurm

Meant to be used as a test of multi-node usage on the cluster

-HM 6/8
'''

#===========================
import numpyro as npy
import numpy as np
from jax.random import PRNGKey
import jax

rkey = PRNGKey(1)

num_chains = 2
npy.set_platform('cpu')
npy.set_host_device_count(num_chains)

print("Starting program with %i cores registering from jax.local_device_count()" %jax.local_device_count())

#===========================

DATA = np.loadtxt('./data.dat')

X, Y, E = DATA[:,0],DATA[:,1],DATA[:,2]

def model(X,Y,E):
    
    m = npy.sample('m', npy.distributions.Uniform(-5,5))
    c = npy.sample('c', npy.distributions.Uniform(-5,5))

    Y_pred = m*X + c
    with npy.plate('data',len(X)):
        npy.sample('y', npy.distributions.Normal(Y_pred,E), obs=Y)
        

#===========================

print("-"*79)
print("Doing MCMC")

sampler = npy.infer.MCMC(npy.infer.NUTS(model=model),
                         num_warmup = 300,
                         num_samples = 500,
                         num_chains = num_chains)
sampler.run(rkey, X,Y,E)

print("Done")
print("-"*79)

sampler.print_summary()


#==========================
