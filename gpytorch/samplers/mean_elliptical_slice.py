import torch

from .elliptical_slice import EllipticalSliceSampler


class MeanEllipticalSlice(EllipticalSliceSampler):
    def __init__(self, f_init, dist, lnpdf, nsamples, pdf_params=()):

        mean_vector = dist.mean

        demeaned_lnpdf = lambda g: lnpdf(g + mean_vector, *pdf_params)

        demeaned_init = f_init - mean_vector

        samples = dist.sample(sample_shape = torch.Size((nsamples,))).t()
        demeaned_samples = samples - mean_vector.unsqueeze(1)

        super(MeanEllipticalSlice, self).__init__(demeaned_init, demeaned_samples, demeaned_lnpdf, nsamples, pdf_params=())

        self.mean_vector = mean_vector

    def run(self):
        self.f_sampled, self.ell = super().run()

        #add means back into f_sampled
        self.f_sampled = self.f_sampled + self.mean_vector.unsqueeze(1)

        return self.f_sampled, self.ell