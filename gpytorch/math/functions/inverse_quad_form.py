import torch
from torch.autograd import Function, Variable


# Function for computing quadratic forms using the inverse kernel matrix, i.e.,
# y^{\top}K^{-1}y for a vector y and p.d. square kernel matrix K.
class InverseQuadForm(Function):
    def __init__(self, y):
        if isinstance(y, Variable):
            y = y.data
        self.y = y


    # Computes y^T K^{-1} y
    def forward(self, chol_matrix): 
        res = chol_matrix.new().resize_(1)
        
        k_inv_y = self.y.potrs(chol_matrix)
        output = k_inv_y.dot(self.y)

        res.fill_(output)
        self.k_inv_y = k_inv_y.squeeze()
        return res


    # Let \alpha = K^{-1}y, then d(y^T K^{-1} y)/dK is the
    # outer product \alpha \alpha^T.
    def backward(self, grad_output):
        grad_output_val = grad_output[0]
        k_inv_y = self.k_inv_y
        outer_prod = torch.ger(k_inv_y, k_inv_y)
        return outer_prod.mul_(grad_output_val)


    def __call__(self, input_var):
        if not hasattr(input_var, 'chol_data'):
            input_var.chol_data = input_var.data.potrf()

        # Switch the variable data with cholesky data, for computation
        orig_data = input_var.data
        input_var.data = input_var.chol_data
        res = super(InverseQuadForm, self).__call__(input_var)

        # Revert back to original data
        input_var.data = orig_data
        return res
