import torch
from torch.autograd import Function, Variable


# Function for computing quadratic forms using the inverse kernel matrix, i.e.,
# y^{\top}K^{-1}y for a vector y and p.d. square kernel matrix K.
class InverseQuadForm(Function):
    # Computes y^T K^{-1} y
    def forward(self, chol_matrix, y): 
        res = chol_matrix.new().resize_(1)
        
        k_inv_y = y.potrs(chol_matrix)
        output = k_inv_y.dot(y)

        res.fill_(output)
        self.k_inv_y = k_inv_y.squeeze()
        return res


    # Let \alpha = K^{-1}y, then d(y^T K^{-1} y)/dK is the
    # outer product \alpha \alpha^T.
    def backward(self, grad_output):
        grad_output_val = grad_output[0]
        k_inv_y = self.k_inv_y
        grad_input_mat = None
        grad_input_vec = None

        if self.needs_input_grad[0]:
            outer_prod = torch.ger(k_inv_y, k_inv_y)
            grad_input_mat = outer_prod.mul_(-grad_output_val)

        if self.needs_input_grad[0]:
            grad_input_vec = k_inv_y.mul_(2 * grad_output_val)

        return grad_input_mat, grad_input_vec


    def __call__(self, input_mat, input_vec):
        if not hasattr(input_mat, 'chol_data'):
            input_mat.chol_data = input_mat.data.potrf()

        # Switch the variable data with cholesky data, for computation
        orig_data = input_mat.data
        input_mat.data = input_mat.chol_data
        res = super(InverseQuadForm, self).__call__(input_mat, input_vec)

        # Revert back to original data
        input_mat.data = orig_data
        return res
