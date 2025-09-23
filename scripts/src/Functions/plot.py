import nifty8 as ift
import matplotlib.pyplot as pl
from matplotlib import cm
import numpy as np
import os
from scipy.linalg import LinAlgError

from .helpers import density_estimation


def progress_plot(path, kl, sky_models, power_models, noise_models, data_std, iteration_number):
    for name, sky in sky_models.items():
        sky_plot(path + '/sky/' + name + '/', name + '_' + str(iteration_number), sky, kl)
        power_from_sky_plot(path + '/power/' + name + '_(sky)/', name + '_' + str(iteration_number), sky, kl)
    for name, amplitude in power_models.items():
        power_from_model_plot(path + '/power/' + name + '/', name + '_' + str(iteration_number), amplitude, kl)
    for name, noise in noise_models.items():
        noise_labels = True
        try:
            noise_plot(path + '/noise/' + name + '/', name + '_' + str(iteration_number), data_std[name], noise, kl,
                       noise=noise_labels)
        except LinAlgError:
            print('Warning: Kernel density estimation for ' + name + ' scatter plot not possible due to LinAlgError')
            noise_plot(path + '/noise/' + name + '/', name + '_' + str(iteration_number), data_std[name], noise, kl,
                       kde=False, noise=noise_labels)
            continue
    return


def data_and_prior_plot(path, data_adjoint_dict, data_dict, model_dict, n_pictures):
    prior_plot(path + '/prior/', model_dict, n_pictures)
    data_plot(path + '/data/', data_adjoint_dict, data_dict)


def joint_hist(path, hist_dict, iteration_number):
    if not os.path.exists(path):
        os.makedirs(path)
    pl.figure()
    fcs = [(0, 0, 1, 0.2), (1, 0, 1, 0.2), (0, 1, 0, 0.2), (1, 1, 0, 0.2)]
    i = 0
    for name, data in hist_dict.items():
        if isinstance(data, ift.Field):
            data = data.val
        pl.hist(data, bins=100, label=name, density=True, range=(0, 3000), fc=fcs[i])
        i += 1
    pl.legend()
    pl.savefig(path + 'joint_hists_' + str(iteration_number) + '.png')
    pl.figure()
    fcs = [(0, 0, 1, 0.5), (1, 0, 1, 0.5), (0, 1, 0, 0.5), (1, 1, 0, 0.5)]
    i = 0
    for name, data in hist_dict.items():
        if isinstance(data, ift.Field):
            data = data.val
        pl.hist(data, bins=100, label=name, density=True, range=(0, 3000), fc=fcs[i])
        i += 1
    pl.yscale('log')
    pl.legend()
    pl.savefig(path + 'joint_hists_log_' + str(iteration_number) + '.png')


def hist(path, hist_dict, iteration_number):
    if not os.path.exists(path):
        os.makedirs(path)
    for name, data in hist_dict.items():
        if isinstance(data, ift.Field):
            data = data.val
        pl.figure()
        pl.hist(data, bins=100, label=name, density=True, range=(0, 1000))
        pl.savefig(path + name + 'hist_' + str(iteration_number) + '.png')


def scatter(path, scatter_dict, iteration_number):
    if not os.path.exists(path):
        os.makedirs(path)
    for name, values in scatter_dict.items():
        pl.figure()
        pl.scatter(values[1], values[0])
        pl.savefig(path + name + 'scatter_' + str(iteration_number) + '.png')
    for name, values in scatter_dict.items():
        pl.figure()
        pl.scatter(values[1], values[0])
        pl.yscale('log')
        pl.savefig(path + name + 'log_scatter_' + str(iteration_number) + '.png')
    pl.close('all')


def prior_plot(path, model_dict, n_pictures):
    if not os.path.exists(path):
        os.makedirs(path)
    for name, model in model_dict.items():
        plot = ift.Plot()
        for i in range(n_pictures):
            plot.add(model(ift.from_random(model.domain, 'normal')))
        plot.output(name=path + name + '_prior_samples.png')


def data_plot(path, data_adjoint_dict, data_dict):
    if not os.path.exists(path):
        os.makedirs(path)
    for name, data_adjoint in data_adjoint_dict.items():
        plot = ift.Plot()
        plot.add(data_adjoint)
        plot.output(name=path + name + '_data_projection.png')
    for name, data in data_dict.items():
        pl.figure()
        pl.hist(data.val, bins=100)
        pl.xlabel(name)
        pl.savefig(path + name + '_data_hist.png')



def sky_plot(path, name, sky_model, kl):
    if not os.path.exists(path):
        os.makedirs(path)
    sc = ift.StatCalculator()
    for s in kl.samples.iterator(sky_model):
        sc.add(s)
    plot = ift.Plot()
    plot.add(sky_model.force(kl.position), title=name + '_mean')
    plot.add(sc.var.sqrt(), title=name + '_std')
    plot.output(ny=1, nx=2, figsize=(6, 3), name=path + name + '.png')


def power_from_model_plot(path, name, amplitude_model, kl,):
    if not os.path.exists(path):
        os.makedirs(path)
    amp_model_samples = [s for s in kl.samples.iterator(amplitude_model)]
    plot = ift.Plot()
    linewidth = [1.] * len(amp_model_samples) + [3., ]
    plot.add(amp_model_samples,
             title="Sampled Posterior Power Spectrum, " + name, linewidth=linewidth)
    plot.output(name=path + name + ".png")


def power_from_sky_plot(path, name, sky_model, kl,):
    if not os.path.exists(path):
        os.makedirs(path)
    plot = ift.Plot()
    ht = ift.HarmonicTransformOperator(sky_model.target[0].get_default_codomain(),
                                       sky_model.target[0])
    plot.add(
        [ift.power_analyze(ht.adjoint(s)) for s in kl.samples.iterator(sky_model)],
        title="Power Spectrum of Signal Posterior Samples, " + name)
    plot.output(name=path + name + ".png")


def noise_plot(path, name, data_std, noise_model, kl, kde=True, noise=True):
    if not os.path.exists(path):
        os.makedirs(path)
    item_1 = np.log10(data_std.val)
    item_2 = np.log10(noise_model.force(kl.position).val)
    if noise:
        item_2 += 0.5
    xmin, xmax, ymin, ymax = 10 ** -5, 10 ** 5, 10 ** -8, 10 ** 8
    pl.figure()
    pl.scatter(item_1, item_2, marker=',', s=0.5, color='black')
    if kde:
        xxx, yyy, zzz = density_estimation(item_1, item_2, np.log10(xmin), np.log10(xmax),
                                           np.log10(ymin), np.log10(ymax))
        xx = np.linspace(np.log10(xmin), np.log10(xmax), 10)
        yy = np.linspace(np.log10(xmin), np.log10(xmax), 10)

        pl.contour(xxx, yyy, np.log10(zzz + 1), cmap=cm.cool, linewidths=0.9, levels=np.linspace(0.01, 1, 10))
        c1 = pl.contourf(xxx, yyy, np.log10(zzz + 1), cmap=cm.cool, levels=np.linspace(0.01, 1, 10))
        col = pl.colorbar(c1)
        pl.plot(xx, yy, '--', c='red', linewidth=0.5)
        col.set_label(r'$\log\left(1+\mathcal{P}\right)$')
    if noise:
        pl.xlabel(r'measured $\sigma$')
        pl.ylabel(r'inferred $\sigma$')
    # pl.xlim([np.log10(xmin), np.log10(xmax), ])
    # pl.ylim([np.log10(ymin), np.log10(ymax), ])
    pl.savefig(path + name + '.png', format='png', dpi=800)
    pl.close()
    return
