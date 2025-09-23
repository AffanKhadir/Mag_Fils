data_params = {
    'seed': 2646,  # Can be None
    'data_name': 'test',  # The arrays will be saved in ./Data/<data_name>. Te inference script needs this name
    'noise_sigma': 5,   # THe noise standard deviation in unuits of rad/m2
    'do_plot': True,  # if set to true, several plots illustrating groundtruths and the data willbe produced
    'n_data': 640,  # number od data points, which will be disributed randomly over the domain
}

domain_params = {
    'full_sphere': False,
    # only necessary for full sky
    'nside': 128,
    # only necessary for cutout
    'nx': 100,  # number of pixels in x
    'ny': 100,  # number of pixels in y
    'dx': .05,  # resolution in x
    'dy': .05,  # resolution in x
    'center': [0, 0],  # coordinate of the central pixels on the sky
}

# Parameters for the log-amplitude sky map
amplitude_params = {#  Prior parameters on the shape of the power spectrum ([mean, std])
                    'asperity': [.001, .001], 'flexibility': [1., .001], 'fluctuations': [2., .5], 'loglogavgslope': [-4., .1],
                    # Parameters on the mean of the sign field
                    'offset_mean': 2., 'offset_std': [.1, .1]
                    }

# Parameters for the sign sky map
sign_params = {#  Prior parameters on the shape of the power spectrum ([mean, std])
               'asperity': [.001, .001], 'flexibility': [1., .001], 'fluctuations': [2., 1.], 'loglogavgslope': [-4, .1],
               # Parameters on the mean of the sign field
               'offset_mean': 0, 'offset_std': [.5, .1],
              }

extragal_params = {'model': 'MultivariateGaussian',  # Model for the generation of the extragalactic component, choices are
                                                     # MultivariateGaussian, UnivariateGaussian or <None>
                   'alpha': 2., 'q': 1.,  # Parameters for the eta factor in the MultivariateGaussian case
                   'extragal_sigma': 7, # The standard deviation for the extraglactic RMs (might be modified again )
                   }
