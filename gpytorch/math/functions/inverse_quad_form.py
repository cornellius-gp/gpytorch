from gpytorch.math.functions import Invmv

# Function for computing quadratic forms using the inverse kernel matrix, i.e.,
# y^{\top}K^{-1}y for a vector y and p.d. square kernel matrix K.
class InverseQuadForm(Invmv):
    def __call__(self, input_mat, input_vec):
        return super(InverseQuadForm, self).__call__(input_mat, input_vec).dot(input_vec)
