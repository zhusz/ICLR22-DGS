import torch


def vertices_to_faces(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 2or3]  # zhuzhu
    :param faces: [batch size, number of faces, 3)
    :return: [batch size, number of faces, 3, 2or3]  # zhuzhu
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    # assert (vertices.shape[2] == 3)  # zhuzhu
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    if nv <= 0 or nf <= 0 or faces.min() < 0 or faces.max() >= nv:
        import ipdb
        ipdb.set_trace()
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]

    # vertices = vertices.reshape((bs * nv, 3))
    vertices = vertices.reshape((bs * nv, -1))  # zhuzhu

    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]
