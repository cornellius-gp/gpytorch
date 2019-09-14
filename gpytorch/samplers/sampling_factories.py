import torch
import gpytorch

from . import MeanEllipticalSlice, SGD

# defining ESS factory
def ess_factory(nsamples, data_mod, data_lh, idx=None):
    # pull out latent model and spectrum from the data model
    omega = data_mod.covar_module.get_omega(idx)
    g_init = data_mod.covar_module.get_latent_params(idx)
    latent_lh = data_mod.covar_module.get_latent_lh(idx)
    latent_mod = data_mod.covar_module.get_latent_mod(idx)

    # update training data
    latent_lh.train()
    latent_mod.train()
    latent_mod.set_train_data(inputs = omega, targets = None, strict = False)

    # draw prior prior distribution
    prior_dist = latent_lh(latent_mod(omega))

    # define a function of the model and log density
    def ess_ell_builder(demeaned_logdens, data_mod, data_lh):
        with torch.no_grad(), gpytorch.settings.fast_computations(log_prob=False):

            data_mod.covar_module.set_latent_params(demeaned_logdens, idx)
            data_mod.prediction_strategy = None

            loss = data_lh(data_mod(*data_mod.train_inputs)).log_prob(data_mod.train_targets).sum()
            return loss

    # creating model
    return MeanEllipticalSlice(g_init, prior_dist, ess_ell_builder, nsamples, pdf_params=(data_mod, data_lh))

# defining slice sampler factory
def ss_factory(nsamples, data_mod, data_lh, idx = None):
    if isinstance(data_mod, list):
        data_mod = data_mod[0]
        data_lh = data_lh[0]

    # defining log-likelihood function
    data_mod.train()
    data_lh.train()

    # pull out latent model and spectrum from the data model
    latent_lh = data_mod.covar_module.get_latent_lh(idx)
    latent_mod = data_mod.covar_module.get_latent_mod(idx)
    omega = data_mod.covar_module.get_omega(idx)
    demeaned_logdens = data_mod.covar_module.get_latent_params(idx)

    # update the training inputs
    latent_mod.set_train_data(inputs=omega, targets=demeaned_logdens.detach(), strict=False)

    data_mll = gpytorch.ExactMarginalLogLikelihood(data_lh, data_mod)

    def ss_ell_builder(latent_mod, latent_lh, data_mod, data_lh):

        latent_lh.train()
        latent_mod.train()

        with gpytorch.settings.max_preconditioner_size(15), gpytorch.settings.cg_tolerance(1e-3), gpytorch.settings.max_cg_iterations(1000):
            loss = data_mll(data_mod(*data_mod.train_inputs), data_mod.train_targets)
            print('Loss is: ', loss)
            #num_y = len(data_mod.train_targets)
            #print('P_y is: ', data_lh(data_mod(*data_mod.train_inputs)).log_prob(data_mod.train_targets)/num_y)
            #print('p_nu is: ', data_mod.covar_module.latent_prior.log_prob(data_mod.covar_module.latent_params)/num_y)
            return loss

    ell_func = lambda h: ss_ell_builder(latent_mod, latent_lh, data_mod, data_lh)

    pars_for_optimizer = list(data_mod.parameters())

    return SGD(pars_for_optimizer, ell_func, n_samples = nsamples, lr=1e-2)

def ss_multmodel_factory(nsamples, data_mods, data_lhs, idx=None):
    for dm in data_mods:
        dm.train()
    for dlh in data_lhs:
        dlh.train()

    mll_list = [gpytorch.ExactMarginalLogLikelihood(dlh, dm) for dlh, dm in zip(data_lhs, data_mods)]

    latent_lh = data_mods[0].covar_module.latent_lh
    latent_mod = data_mods[0].covar_module.latent_mod
    # print(list(latent_mod.named_parameters()))

    def ss_ell_builder(latent_mod, latent_lh, data_mods, data_lhs):

        latent_lh.train()
        latent_mod.train()
        # compute prob #
        loss = 0.
        for i in range(len(data_mods)):
            # pull out latent GP and omega
            demeaned_logdens = data_mods[i].covar_module.latent_params
            omega = data_mods[i].covar_module.omega

            # update latent model
            latent_mod.set_train_data(inputs=omega, targets=demeaned_logdens.detach(), strict=False)

            # compute loss
            loss = loss + mll_list[i](data_mods[i](*mll_list[i].model.train_inputs), mll_list[i].model.train_targets)

        return loss

    ell_func = lambda h: ss_ell_builder(latent_mod, latent_lh, data_mods, data_lhs)

    data_par_list = [list(dm.parameters()) for dm in data_mods]
    optim_pars = [par for sublist in data_par_list for par in sublist]
    return SGD(optim_pars, ell_func, n_samples = nsamples, lr =1e-1)
