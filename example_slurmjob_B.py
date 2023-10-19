'''
example_slurmjob.py

An example of a parallelizable numpyro job for use on slurm

Meant to be used as a test of multi-node usage on the cluster

-HM 6/8, cont 10/8
'''

#===========================
print("-"*79)
print("Doing Imports")
import numpyro as npy
import numpy as np
from jax.random import PRNGKey
import jax
import os
from numpyro.contrib.nested_sampling import NestedSampler

    #===========================
def main():
    print("-"*79)
    print("Doing NumPyro setup")
          
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
    
    #===========================
    print("-"*79)
    print("Doing Nested Sampling")
    NS = NestedSampler(model=model, 
                       constructor_kwargs={'num_live_points': 5000, 'max_samples': 50000, 'num_parallel_samplers': jax.local_device_count()},
                       termination_kwargs={'live_evidence_frac': 0.01}
                      )
    NS.run(rkey, X,Y,E)

    print("Done")
    print("-"*79)

    sampler.print_summary()
    NS.print_summary()
    print("-"*79)


#==========================

if __name__ == "__main__":
    os.environ["XLA_FLAGS"] = (
        "--xla_force_host_platform_device_count=2 "
        "--xla_cpu_multi_thread_eigen=true"
    )
    main()
