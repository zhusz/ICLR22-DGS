import torch
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np


def showPoint3D(point, existing_handles=None, c='r', marker='o'):
    if existing_handles is None:
        ax = plt.axes(projection='3d')
    else:
        ax = existing_handles
    if type(point) is torch.Tensor:
        point = point.detach().cpu().numpy()
    if type(point) is list:
        point = np.array(point)
    point = point.reshape((-1, 3))
    ax.scatter(point[:, 0], point[:, 1], point[:, 2], c=c, marker=marker)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z axis')
    return ax, plt


def showPoint3D4(point, existing_handles=None, c='r', marker='o', s=5):
    # Note the s is only for 2D plots
    if existing_handles is None:
        axlist = []
        axlist.append(plt.subplot(221, projection='3d'))
        axlist.append(plt.subplot(222))
        axlist.append(plt.subplot(223))
        axlist.append(plt.subplot(224))
    else:
        axlist = existing_handles
    if type(point) is torch.Tensor:
        point = point.detach().cpu().numpy()
    if type(point) is list:
        point = np.array(point)
    point = point.reshape((-1, 3))

    axlist[0].scatter(point[:, 0], point[:, 1], point[:, 2], c=c, marker=marker)
    axlist[0].set_xlabel('X Axis')
    axlist[0].set_ylabel('Y Axis')
    axlist[0].set_zlabel('Z axis')

    axlist[1].grid(b=True)
    axlist[1].scatter(point[:, 0], point[:, 2], c=c, marker=marker, s=s)  # xz
    axlist[1].set_xlabel('X Axis')
    axlist[1].set_ylabel('Z Axis')

    axlist[2].grid(b=True)
    axlist[2].scatter(point[:, 2], point[:, 1], c=c, marker=marker, s=s)  # zy
    axlist[2].set_xlabel('Z Axis')
    axlist[2].set_ylabel('Y Axis')

    axlist[3].grid(b=True)
    axlist[3].scatter(point[:, 0], point[:, 1], c=c, marker=marker, s=s)  # xy
    axlist[3].set_xlabel('X Axis')
    axlist[3].set_ylabel('Y Axis')

    return axlist, plt


def demoShowBothTwo():
    # If you wish to use both the two functions above and shown into 2 figure, use this:
    pr = np.random.rand(100, 3).astype(np.float32)
    pb = np.random.rand(100, 3).astype(np.float32)

    # from codes_py.toolbox_show_draw.show3D_v1 import showPoint3D4, showPoint3D
    import matplotlib.pyplot as plt
    plt.figure(1)
    ax, _ = showPoint3D4(pb, None, 'b', 'o')
    ax, plt1 = showPoint3D4(pr, ax, 'r', 'o')
    plt.figure(2)
    ax, _ = showPoint3D(pb, None, 'b', 'o')
    ax, plt2 = showPoint3D(pr, ax, 'r', 'o')
    plt.show()


def checkShowPoint3D():
    pr = np.random.rand(100, 3).astype(np.float32)
    pb = np.random.rand(100, 3).astype(np.float32)
    ax, _ = showPoint3D(pb, 'b', 'o')
    ax, plt = showPoint3D(pr, 'r', 'x', ax)
    plt.show()


def showMesh3D(vert, face=None, existing_handles=None, linewidths=0.2):
    raise NotImplementedError('This function is not yet OK')
    if existing_handles is None:
        ax = plt.axes(projection='3d')
    else:
        ax = existing_handles
    if type(vert) is torch.Tensor:
        vert = vert.detach().cpu().numpy()
    if type(face) is torch.Tensor:
        face = face.detach().cpu().numpy()
    vert = vert.reshape((-1, 3))
    if face is not None:
        face = face.reshape((-1, 3))
    ax.plot_trisurf(vert[:, 0], vert[:, 1], vert[:, 2],
                    triangles=face,
                    cmap='viridis', linewidths=linewidths)
    return ax, plt


if __name__ == '__main__':
    demoShowBothTwo()
