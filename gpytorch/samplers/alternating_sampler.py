import torch
import copy
import time

class AlternatingSampler:
    def __init__(self, model, likelihood, outer_sampler_factory, inner_sampler_factory, totalSamples,
                numInnerSamples, numOuterSamples, num_dims=1, num_tasks=1):

        self.model = model
        self.likelihood = likelihood

        self.outer_sampler_factory = outer_sampler_factory
        self.inner_sampler_factory = inner_sampler_factory

        self.totalSamples = totalSamples
        self.numInnerSamples = numInnerSamples
        self.numOuterSamples = numOuterSamples

        self.num_dims = num_dims
        self.num_tasks = num_tasks

    def run(self):
        total_time = 0.0

        outer_samples = [[] for x in range(self.num_dims)]
        inner_samples = [[] for x in range(self.num_dims)]

        for step in range(self.totalSamples):
            ts =  time.time()
            for in_dim in range(self.num_dims):

                if self.num_dims == 1:
                    idx = None
                else:
                    idx = in_dim
                    print('Step: ', step, 'Dimension: ', in_dim)

                # run outer sampler factory
                curr_outer_samples, _ = self.outer_sampler_factory(self.numOuterSamples,
                                self.model, self.likelihood, idx).run()

                # loop through every task
                curr_task_list = []
                for task in range(self.num_tasks):
                    print('Task:', task, "; Iteration", step)
                    # run inner sampler factory

                    curr_task_samples, _ = self.inner_sampler_factory[task](self.numInnerSamples,
                            self.model[task], self.likelihood[task], idx).run()

                    curr_task_list.append(copy.deepcopy(curr_task_samples.detach().unsqueeze(0)))

                curr_inner_samples = torch.cat(curr_task_list, dim=0)

                outer_samples[in_dim].append(copy.deepcopy(curr_outer_samples.detach()))
                inner_samples[in_dim].append(copy.deepcopy(curr_inner_samples.detach()))

            ts_d = torch.abs(torch.tensor(ts - time.time()))

            total_time += ts_d
            print("Seconds for Iteration {} : {}".format(step,ts_d))

        self.hsampled = [torch.cat(outer_samples[id], dim=-1) for id in range(self.num_dims)]
        self.gsampled = [torch.cat(inner_samples[id], dim=-1) for id in range(self.num_dims)]
        # return self.hsampled, self.gsampled # don't return anything, makes notebooks icky
