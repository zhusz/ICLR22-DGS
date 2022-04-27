# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# PYOPENGL_PLATFORM=osmesa
import pyrender
import trimesh
import numpy as np


# def render_single_mesh_pyrender_backend(vert0, face0, fxywxy, lights=None, renderer=None):
#     # If you render for the first time,
#     #       set lights to be None (if you wish the default otherwise set your values)
#     #       and set the renderer to be None
#
#     # If you render not for the first time,
#     #       set lights to be [] or None if you wish to keep the original lighting otherwise
#     #       your new lights (None in here would not add new lights)
#
#     # If you wish to do scene.clear, do it outside
#
#     assert len(fxywxy) == 4


class PyrenderManager(object):
    def __init__(self, winWidth, winHeight):
        self.renderer = pyrender.OffscreenRenderer(winWidth, winHeight)
        self.scene = pyrender.Scene()

    def clear(self):
        self.scene.clear()

    @staticmethod
    def fix_pose(pose):
        # 3D Rotation about the x-axis.
        t = np.pi
        c = np.cos(t)
        s = np.sin(t)
        R = np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s, c]])
        axis_transform = np.eye(4)
        axis_transform[:3, :3] = R
        return pose @ axis_transform

    def add_camera(self, fxywxy, camInv0):
        assert len(fxywxy) == 4
        assert camInv0.shape == (4, 4)
        focalLenthWidth = fxywxy[0]
        focalLenthHeight = fxywxy[1]
        winWidth = fxywxy[2]
        winHeight = fxywxy[3]

        camObj = pyrender.IntrinsicsCamera(
            cx=winWidth / 2., cy=winHeight / 2.,
            fx=focalLenthWidth, fy=focalLenthHeight,
        )
        self.scene.add(camObj, pose=self.fix_pose(camInv0))

    def add_point_light(self, pointLoc=[2., 2., 2.], intensity=0.8, color=[25., 50., 200.]):
        assert len(pointLoc) == 3
        pl = pyrender.PointLight(color=color, intensity=intensity)
        light_matrix = np.eye(4).astype(np.float32)
        light_matrix[0, 3] = pointLoc[0]
        light_matrix[1, 3] = pointLoc[1]
        light_matrix[2, 3] = pointLoc[2]

        self.scene.add(pl, pose=self.fix_pose(light_matrix))

    def add_plain_mesh(self, vert0, face0):
        self.scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(
            vertices=vert0, faces=face0,
        ), smooth=True))

    def add_vertRgb_mesh(self, vert0, face0, vertRgb0):
        mesh = trimesh.Trimesh(
            vertices=vert0, faces=face0,
        )
        assert vertRgb0.shape[0] == vert0.shape[0]
        assert vertRgb0.shape[1] == 3
        assert vertRgb0.min() >= -0.0001
        assert vertRgb0.max() <= 1.0001
        mesh.visual.vertex_colors = np.concatenate([
            (vertRgb0 * 255.).astype(np.uint8),
            255 * np.ones((vert0.shape[0], 1), dtype=np.uint8)
        ], 1)
        self.scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))

    def add_vertRgb_mesh_via_faceRgb(self, vert0, face0, faceRgb0):
        mesh = trimesh.Trimesh(
            vertices=vert0, faces=face0,
        )
        assert faceRgb0.shape[0] == face0.shape[0]
        assert faceRgb0.shape[1] == 3
        assert faceRgb0.min() >= -0.0001
        assert faceRgb0.max() <= 1.0001
        face_colors = np.concatenate([
            (faceRgb0 * 255.).astype(np.uint8),
            255 * np.ones((face0.shape[0], 1), dtype=np.uint8)
        ], 1)
        mesh.visual.vertex_colors = trimesh.visual.color.face_to_vertex_color(
            mesh, face_colors)
        self.scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))

    def add_faceRgb_mesh(self, vert0, face0, faceRgb0):
        mesh = trimesh.Trimesh(
            vertices=vert0, faces=face0,
        )
        assert faceRgb0.shape[0] == face0.shape[0]
        assert faceRgb0.shape[1] == 3
        assert faceRgb0.min() >= -0.0001
        assert faceRgb0.max() <= 1.0001
        mesh.visual.face_colors = np.concatenate([
            (faceRgb0 * 255.).astype(np.uint8),
            255 * np.ones((face0.shape[0], 1), dtype=np.uint8)
        ], 1)
        self.scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))

    def render(self):
        return self.renderer.render(self.scene)


