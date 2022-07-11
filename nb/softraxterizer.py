import jax
import jax.numpy as np

'''
JAX implementation of Soft Rasterizer (softras)
(c) 2021 Kartik Chandra; see MIT license attached

Soft Rasterizer: A Differentiable Renderer for Image-based 3D Reasoning
Shichen Liu, Tianye Li, Weikai Chen, and Hao Li (ICCV 2019)
https://arxiv.org/abs/1904.01786
https://github.com/ShichenLiu/SoftRas
'''

def get_pixel(left=-1, top=1, right=1.5, bottom=-1.5, xres=50, yres=50):
    '''
    Generates a grid of pixel samples in a given viewport, at a given resolution.
    
    Parameters:
        left (float): left edge of viewport
        top (float): top edge of viewport
        right (float): right edge of viewport
        bottom (float): bottom edge of viewport
        xres (int): number of samples along horizontal axis
        yres (int): number of samples along vertical axis
    Returns:
        pixel (N x 3): array of sample coordinates,
        shape (tuple): shape to reshape softras'ed outputs to get an image
    '''
    Xs = np.linspace(left, right, xres)
    Ys = np.linspace(top, bottom, yres)
    X, Y = np.meshgrid(Xs, Ys)
    Z = np.zeros_like(X)
    pixel = np.stack((X, Y, Z), axis=2).reshape(-1, 1, 3)
    return pixel, Z.shape

eps = 1e-8

def softras(mesh, pixel, C, SIGMA=1e-1, GAMMA=1e-1):
    '''
    Differentiably rasterizes a mesh using the SoftRas algorithm.
    
    Parameters:
        mesh (T x 3[face] x 3[xyz]): mesh, as list of triples of vertices
        pixel (N x 3[xyz]): pixel locations at which to render
        C (T x 3[face]): texture brightness at each face
        SIGMA (float): parameter from softras paper
        GAMMA (float): parameter from softras paper
    Returns:
        image (N): rendered pixel values, should be reshaped to form image
    '''
    Zbuf = mesh[:, :, 2]
    proj = mesh.at[:, :, 2].set(0)
    
    def dot(a, b):
        return (a * b).sum(axis=-1, keepdims=True)

    def d2_point_to_finite_edge(i):
        A = proj[:, i, :]
        B = proj[:, (i + 1) % 3, :]
        Va = B - A
        Vp = pixel - A
        projln = dot(Vp, Va) / (dot(Va, Va) + eps)
        projpt = np.clip(projln, 0, 1) * Va[None, :, :]
        out = dot(Vp - projpt, Vp - projpt)
        return out[:, :, 0]

    d2 = np.minimum(
        np.minimum(d2_point_to_finite_edge(0), d2_point_to_finite_edge(1)),
        d2_point_to_finite_edge(2)
    )
    
    def signed_area_to_point(i):
        A = proj[:, i, :]
        B = proj[:, (i + 1) % 3, :]
        Va = B - A
        area = np.cross(Va, pixel - A)[:, :, 2] / 2
        return area

    Aa = signed_area_to_point(0)
    Ab = signed_area_to_point(1)
    Ac = signed_area_to_point(2)
    Aabc = Aa + Ab + Ac + eps
    in_triangle =\
        np.equal(np.sign(Aa), np.sign(Ab)).astype('float32') *\
        np.equal(np.sign(Aa), np.sign(Ac)).astype('float32') * 2 - 1

    D = jax.nn.sigmoid(in_triangle * d2 / SIGMA)

    bary = np.stack([Aa, Ab, Ac], axis=2) / Aabc[:, :, None]
    bary_clipped = np.clip(bary, 0, 1)
    bary_clipped = bary_clipped / (bary_clipped.sum(axis=2, keepdims=True) + eps)

    Zb = (bary_clipped * np.roll(Zbuf, 2, axis=1)).sum(axis=2)
    Zb = (Zb.max() - Zb) / (Zb.max() - Zb.min())

    Zbe = np.exp(np.clip(Zb / GAMMA, -20., 20.))
    DZbe = D * Zbe
    w = DZbe / (DZbe.sum(axis=1, keepdims=True) + np.exp(eps / GAMMA))
    return (w * C).sum(axis=1)
