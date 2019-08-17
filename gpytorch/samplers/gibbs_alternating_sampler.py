import torch
import copy

class GibbsAlternatingSampler:
        def __init__(self, outer_sampler_factory, inner_fact_list, totalSamples,
                 numInnerSamples, numOuterSamples, n_stn, n_omg=100, **kwargs):
            self.outer_sampler_factory = outer_sampler_factory
            self.inner_fact_list = inner_fact_list
            self.totalSamples = totalSamples
            self.numInnerSamples = numInnerSamples
            self.numOuterSamples = numOuterSamples
            self.n_omg = n_omg
            self.n_stn = n_stn

            self.inner_samples_list = [[] for _ in range(n_stn)]

        def run(self):
            outer_samples = []

            for step in range(self.totalSamples):
                print("running iteration ", step, " of ", self.totalSamples)
                curr_inner_samples = torch.zeros(self.n_stn, self.n_omg, self.numInnerSamples)
                ## run outer sampler ##
                curr_outer_samples, _ = self.outer_sampler_factory(self.numOuterSamples).run()

                ## run inner samplers ##
                for stn in range(self.n_stn):
                    curr_inner_samples[stn, :, :], _ = self.inner_fact_list[stn](self.numInnerSamples).run()

                outer_samples.append(copy.deepcopy(curr_outer_samples))
                if step == 0:
                    inner_samples = curr_inner_samples
                else:
                    inner_samples = torch.cat([inner_samples, curr_inner_samples], 2)
                    # print("inner samples:", inner_samples.shape)

            self.hsampled = torch.cat(outer_samples, dim=-1)
            self.gsampled = inner_samples

            # return self.hsampled, self.gsampled
