import numpy as np


cos = np.cos
sin = np.sin
sqrt = np.sqrt
atan2 = np.arctan2
acos = np.arccos
pi = np.pi
dot = np.dot


def quat_to_cart(qx, qy, qz, qw):
    x = 2 * qx * qz + 2 * qy * qw
    y = 2 * qy * qz - 2 * qx * qw
    z = 1 - 2 * qx * qx - 2 * qy * qy
    v = [x, y, z]
    return v / np.linalg.norm(v)


def cart_to_spher(x, y, z):
    phi = atan2(y, x)
    if phi < 0:
        phi += 2 * pi
    theta = acos(z)
    return phi, theta


def spher_to_cart(phi, theta):
    x = sin(theta) * cos(phi)
    y = sin(theta) * sin(phi)
    z = cos(theta)
    return x, y, z


def arc_dist(phi_1, theta_1, phi_2, theta_2):
    t_1 = cos(theta_1) * cos(theta_2)
    t_2 = sin(theta_1) * sin(theta_2) * cos(phi_1 - phi_2)
    return acos(t_1 + t_2)


def spher_to_tile(phi, theta, t_hor, t_vert):
    i = int(theta / pi * t_vert)
    j = int(phi / 2 / pi * t_hor)
    return i * t_hor + j


def angle(v_1, v_2):
    v_1_u = v_1 / np.linalg.norm(v_1)
    v_2_u = v_2 / np.linalg.norm(v_2)
    return np.arccos(np.clip(dot(v_1_u, v_2_u), -1.0, 1.0))


def walk_on_sphere(phi_1, theta_1, phi_2, theta_2, d_1, d_2):
    """Returns the spherical coordinates of the end of a spherical walk

    Parameters
    ----------
    phi_1, theta_1 : float, float
        The spherical coordinates of P_1
    phi_2, theta_2 : float, float
        The spherical coordinates of P_2
    d_1 : float
        The time between P_1 and P_2
    d_2 : float
        The time between P_2 and P_3

    Returns
    -------
    float, float
        The spherical coordinates of P_3
    """

    # If P_1 equals P_2, simply return P_1
    if phi_1 == phi_2 and theta_1 == theta_2:
        return phi_1, theta_1

    # Convert to cartesian coordinates
    p_1 = spher_to_cart(phi_1, theta_1)
    p_2 = spher_to_cart(phi_2, theta_2)

    # Determine rotation matrix to rotate all vectors onto the XY-plane
    t = np.cross(p_1, p_2)
    u = np.cross(t, [0, 0, 1])
    if np.linalg.norm(u) > 0:
        u = u / np.linalg.norm(u)
    a = angle(t, [0, 0, 1])
    k = [[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]]
    r = np.identity(3) + dot(sin(a), k) + dot(1 - cos(a), np.matmul(k, k))

    # Rotate vectors
    p_1_r = dot(r, p_1)
    p_2_r = dot(r, p_2)

    # Extract phi values
    phi_1 = atan2(p_1_r[1], p_1_r[0])
    if phi_1 < 0:
        phi_1 += 2 * pi
    phi_2 = atan2(p_2_r[1], p_2_r[0])
    if phi_2 < 0:
        phi_2 += 2 * pi

    # Determine the rotated version of P_3
    d_phi = phi_2 - phi_1
    if d_phi < 0:
        d_phi += 2 * pi
    phi_3 = phi_2 + d_2 * d_phi / d_1
    p_3_r = spher_to_cart(phi_3, pi / 2)

    # Inverse rotation to retrieve P_3
    p_3 = dot(np.linalg.inv(r), p_3_r)

    # Convert to spherical coordinates
    phi, theta = cart_to_spher(*p_3)

    return phi, theta
