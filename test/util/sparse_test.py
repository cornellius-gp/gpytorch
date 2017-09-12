import torch
from gpytorch.utils import sparse_eye, sparse_getitem, sparse_repeat


def test_sparse_eye():
    res = sparse_eye(5)
    actual = torch.eye(5)
    assert torch.equal(res.to_dense(), actual)


indices = torch.LongTensor([[0, 1, 2, 3, 4], [2, 1, 0, 0, 1]])
values = torch.FloatTensor([3, 4, 5, 2, 6])
sparse = torch.sparse.FloatTensor(indices, values, torch.Size((5, 3)))
dense = sparse.to_dense()


def test_sparse_getitem_one_dim_int():
    actual = dense[3]
    res = sparse_getitem(sparse, 3)
    assert torch.equal(actual, res.to_dense())


def test_sparse_getitem_one_dim_slice():
    actual = dense[2:4]
    res = sparse_getitem(sparse, slice(2, 4))
    assert torch.equal(actual, res.to_dense())


def test_sparse_getitem_two_dim_int():
    actual = dense[2, 1]
    res = sparse_getitem(sparse, (2, 1))
    assert actual == res


def test_sparse_getitem_two_dim_int_slice():
    actual = dense[:, 1]
    res = sparse_getitem(sparse, (slice(None, None, None), 1))
    assert torch.equal(actual, res.to_dense())

    actual = dense[1, :]
    res = sparse_getitem(sparse, (1, slice(None, None, None)))
    assert torch.equal(actual, res.to_dense())


def test_sparse_getitem_two_dim_slice():
    actual = dense[2:4, 1:3]
    res = sparse_getitem(sparse, (slice(2, 4), slice(1, 3)))
    assert torch.equal(actual, res.to_dense())


def test_sparse_repeat_1d():
    sparse_1d = sparse_getitem(sparse, 1)
    actual = sparse_1d.to_dense().repeat(3, 1)
    res = sparse_repeat(sparse_1d, 3, 1)
    assert torch.equal(actual, res.to_dense())

    actual = sparse_1d.to_dense().repeat(2, 3)
    res = sparse_repeat(sparse_1d, 2, 3)
    assert torch.equal(actual, res.to_dense())


def test_sparse_repeat_2d():
    actual = sparse.to_dense().repeat(3, 2)
    res = sparse_repeat(sparse, 3, 2)
    assert torch.equal(actual, res.to_dense())

    actual = sparse.to_dense().repeat(1, 2)
    res = sparse_repeat(sparse, 1, 2)
    assert torch.equal(actual, res.to_dense())

    actual = sparse.to_dense().repeat(3, 1)
    res = sparse_repeat(sparse, 3, 1)
    assert torch.equal(actual, res.to_dense())
