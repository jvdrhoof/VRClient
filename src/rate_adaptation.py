import numpy as np

import help_functions as hf


class RateAdaptation(object):
    def __init__(self, buffer_size, seg_dur, t_hor, t_vert, n_qual, vp_rad):
        self.buffer_size = buffer_size
        self.seg_dur = seg_dur
        self.t_hor = t_hor
        self.t_vert = t_vert
        self.n_qual = n_qual
        self.vp_rad = vp_rad

    def __repr__(self):
        raise NotImplementedError("Object representation not implemented")

    def __str__(self):
        raise NotImplementedError("String representation not implemented")

    def adapt(self, s_id, file_sizes, budget_bits, buffered, phi_1, theta_1,
              phi_2=None, theta_2=None):
        raise NotImplementedError("Rate adaptation heuristic not implemented")


class UVP(RateAdaptation):
    def __init__(self, buffer_size, seg_dur, t_hor, t_vert, n_qual, vp_rad):
        super(UVP, self).__init__(buffer_size, seg_dur, t_hor, t_vert, n_qual,
                                  vp_rad)

    def __repr__(self):
        return "UVP"

    def __str__(self):
        return "UVP"

    def adapt(self, s_id, file_sizes, budget_bits, buffered, phi_vp, theta_vp,
              phi_2=None, theta_2=None):
        """Adapts the quality of each tile, respecting the total bandwidth.

           Initially, only the quality of tiles within the viewport is
           increased. If sufficient bandwidth is available, the quality of
           other tiles is increased as well.

           Returns a list of quality levels (one for each tile), along with the
           order of tiles (in terms of distance).
        """

        n_tiles = self.t_hor * self.t_vert
        tiles = [t for t in range(2, n_tiles + 2) if buffered[t - 2] < 1]

        # Sort tiles according to distance
        d_hor = 2 * np.pi / self.t_hor
        d_vert = np.pi / self.t_vert
        distances = []
        for i in range(self.t_vert):
            for j in range(self.t_hor):
                index = i * self.t_hor + j + 2
                if index in tiles:
                    phi_c = d_hor * (j + 0.5)
                    theta_c = d_vert * (i + 0.5)
                    a = hf.arc_dist(phi_vp, theta_vp, phi_c, theta_c)
                    distances.append((index, a))
        distances.sort(key=lambda tup: tup[1])
        order = [x[0] for x in distances]

        # Transferrable bits
        n_bits_max = budget_bits

        # Correct for already downloaded bits
        n_bits_max -= sum(file_sizes[t][buffered[t - 2]] for t in
                          range(2, n_tiles + 2) if buffered[t - 2] > 0)

        # Determine minimum and maximum number of bytes to transfer
        n_bits_low = sum(file_sizes[t][1] for t in tiles)
        n_bits_high = sum(file_sizes[t][self.n_qual] for t in tiles)

        # Initialize quality representations
        qualities = [0] * n_tiles
        for t in tiles:
            qualities[t - 2] = 1
        n_bits = n_bits_low

        # Handle straightforward cases
        if n_bits_low >= n_bits_max:
            return qualities, order
        elif n_bits_high <= n_bits_max:
            for t in tiles:
                qualities[t - 2] = self.n_qual
            return qualities, order

        # Increase quality for all tiles with center in viewport
        tiles_in = [x[0] for x in distances if x[1] <= self.vp_rad / 2]
        for q in range(2, self.n_qual + 1):
            for t in tiles_in:
                fs_bits = file_sizes[t][q] - file_sizes[t][q - 1]
                if n_bits + fs_bits <= n_bits_max:
                    qualities[t - 2] = q
                    n_bits += fs_bits
                else:
                    return qualities, order

        # Increase quality for all other tiles
        tiles_out = [x[0] for x in distances if x[1] > self.vp_rad / 2]
        for q in range(2, self.n_qual + 1):
            for t in tiles_out:
                fs_bits = file_sizes[t][q] - file_sizes[t][q - 1]
                if n_bits + fs_bits <= n_bits_max:
                    qualities[t - 2] = q
                    n_bits += fs_bits
                else:
                    return qualities, order

        # The following should never be executed
        return qualities, order


class CTF(RateAdaptation):
    def __init__(self, buffer_size, seg_dur, t_hor, t_vert, n_qual):
        super(CTF, self).__init__(buffer_size, seg_dur, t_hor, t_vert, n_qual,
                                  0)

    def __repr__(self):
        return "CTF"

    def __str__(self):
        return "CTF"

    def adapt(self, s_id, file_sizes, budget_bits, buffered, phi_vp, theta_vp,
              phi_2=None, theta_2=None):
        """Adapts the quality of each tile, respecting the total bandwidth.

           Tile per tile, the quality is increased to the maximum value.

           Returns a list of quality levels (one for each tile), along with the
           order of tiles (in terms of distance)
        """

        n_tiles = self.t_hor * self.t_vert
        tiles = [t for t in range(2, n_tiles + 2) if buffered[t - 2] < 1]

        # Sort tiles according to distance
        d_hor = 2 * np.pi / self.t_hor
        d_vert = np.pi / self.t_vert
        distances = []
        for i in range(self.t_vert):
            for j in range(self.t_hor):
                index = i * self.t_hor + j + 2
                if index in tiles:
                    phi_c = d_hor * (j + 0.5)
                    theta_c = d_vert * (i + 0.5)
                    a = hf.arc_dist(phi_vp, theta_vp, phi_c, theta_c)
                    distances.append((index, a))
        distances.sort(key=lambda tup: tup[1])
        order = [x[0] for x in distances]

        # Transferrable bits
        n_bits_max = budget_bits

        # Correct for already downloaded bits
        n_bits_max -= sum(file_sizes[t][buffered[t - 2]] for t in
                          range(2, n_tiles + 2) if buffered[t - 2] > 0)

        # Determine minimum and maximum number of bytes to transfer
        n_bits_low = sum(file_sizes[t][1] for t in tiles)
        n_bits_high = sum(file_sizes[t][self.n_qual] for t in tiles)

        # Initialize quality representations
        qualities = [0] * n_tiles
        for t in tiles:
            qualities[t - 2] = 1
        n_bits = n_bits_low

        # Handle straightforward cases
        if n_bits_low >= n_bits_max:
            return qualities, order
        elif n_bits_high <= n_bits_max:
            for t in tiles:
                qualities[t - 2] = self.n_qual
            return qualities, order

        # Increase quality for all tiles with center in viewport
        tiles = [x[0] for x in distances]
        for t in tiles:
            for q in range(2, self.n_qual + 1):
                fs_bits = file_sizes[t][q] - file_sizes[t][q - 1]
                if n_bits + fs_bits <= n_bits_max:
                    qualities[t - 2] = q
                    n_bits += fs_bits
                else:
                    return qualities, order

        # The following should never be executed
        return qualities, order


class Petrangeli(RateAdaptation):
    def __init__(self, buffer_size, seg_dur, t_hor, t_vert, n_qual):
        super(Petrangeli, self).__init__(buffer_size, seg_dur, t_hor, t_vert,
                                         n_qual, 0)

    def _get_polar_index(self, phi, theta):
        if theta < np.pi / 4:
            return 0
        elif np.pi / 4 <= theta <= np.pi * 3 / 4:
            return 1 + int(phi * 2 / np.pi)
        else:
            return 5

    def __repr__(self):
        return "Petrangeli"

    def __str__(self):
        return "Petrangeli"

    def adapt(self, s_id, file_sizes, budget_bits, buffered, phi_1, theta_1,
              phi_2, theta_2):
        """Adapts the quality of each tile, respecting the total bandwidth.

           Three zones are used in total: viewport, adjacent and other.

           Returns a list of quality levels (one for each tile), along with the
           (estimated) total file size.
        """

        # WARNING: buffered is not used in this rate adaptation heuristic
        #          and therefore assumes a one-time download only

        n_tiles = self.t_hor * self.t_vert
        tiles = list(range(2, n_tiles + 2))

        # Sort tiles according to distance
        d_hor = 2 * np.pi / self.t_hor
        d_vert = np.pi / self.t_vert
        distances = []
        for i in range(self.t_vert):
            for j in range(self.t_hor):
                phi_c = d_hor * (j + 0.5)
                theta_c = d_vert * (i + 0.5)
                a = hf.arc_dist(phi_1, theta_1, phi_c, theta_c)
                distances.append((i * self.t_hor + j + 2, a))
        distances.sort(key=lambda tup: tup[1])
        order = [x[0] for x in distances]

        # Transferrable bits
        n_bits_max = budget_bits

        # Correct for already downloaded bits
        n_bits_max -= sum(file_sizes[t][buffered[t - 2]] for t in tiles if
                          buffered[t - 2] > 0)

        # Determine minimum and maximum number of bytes to transfer
        n_bits_low = sum(file_sizes[t][1] for t in tiles)
        n_bits_high = sum(file_sizes[t][self.n_qual] for t in tiles)

        # Handle straightforward cases
        if n_bits_low >= n_bits_max:
            return [1] * n_tiles, order
        elif n_bits_high <= n_bits_max:
            return [self.n_qual] * n_tiles, order

        # Initialize quality representations
        qualities = [1] * n_tiles
        n_bits = n_bits_low

        # Map tiles on polar tiles
        tiles = [[], [], [], [], [], []]
        for i in range(self.t_vert):
            for j in range(self.t_hor):
                phi = d_hor * (j + 0.5)
                theta = d_vert * (i + 0.5)
                index = i * self.t_hor + j
                tiles[self._get_polar_index(phi, theta)].append(index)

        # Assign tile categories
        i_1 = self._get_polar_index(phi_1, theta_1)
        i_2 = self._get_polar_index(phi_2, theta_2)
        if i_1 == i_2:
            viewport = [i_1]
            if i_1 == 0 or i_1 == 5:
                adjacent = [1, 2, 3, 4]
                other = [5 - i_1]
            else:
                adjacent = [(i_1 + 1) % 4, (i_1 + 3) % 4]
                other = [0, 5, 1 + (i_1 + 1) % 4]
        else:
            viewport = [i_1, i_2]
            adjacent = [x for x in [1, 2, 3, 4] if x not in viewport]
            other = [x for x in [0, 1, 2, 3, 4, 5] if x not in viewport and
                     x not in adjacent]
        viewport = sum([[x for x in tiles[i]] for i in viewport], [])
        adjacent = sum([[x for x in tiles[i]] for i in adjacent], [])
        other = sum([[x for x in tiles[i]] for i in other], [])

        # Increase quality in a per-category fashion
        n_budget = n_bits_max - n_bits
        for tiles in [viewport, adjacent, other]:
            for q in range(2, self.n_qual + 1):
                n_cost = sum(file_sizes[i + 2][q] - file_sizes[i + 2][q - 1]
                             for i in tiles)
                if n_cost < n_budget:
                    for i in tiles:
                        qualities[i] = q
                    n_budget -= n_cost
                else:
                    return qualities, order

        # The following should never be executed
        return qualities, order


class Hosseini(RateAdaptation):
    def __init__(self, buffer_size, seg_dur, t_hor, t_vert, n_qual):
        super(Hosseini, self).__init__(buffer_size, seg_dur, t_hor, t_vert,
                                       n_qual, 0)

    def __repr__(self):
        return "Hosseini"

    def __str__(self):
        return "Hosseini"

    def adapt(self, s_id, file_sizes, budget_bits, buffered, phi_vp, theta_vp,
              phi_2=None, theta_2=None):
        """Adapts the quality of each tile, respecting the total bandwidth.

           Tile per tile, the quality is increased to the maximum value.

           Returns a list of quality levels (one for each tile), along with the
           (estimated) total file size.
        """

        n_tiles = self.t_hor * self.t_vert
        tiles = [t for t in range(2, n_tiles + 2) if buffered[t - 2] < 1]

        # Transferrable bits
        n_bits_max = budget_bits

        # Correct for already downloaded bits
        n_bits_max -= sum(file_sizes[t][buffered[t - 2]] for t in
                          range(2, n_tiles + 2) if buffered[t - 2] > 0)

        # Determine minimum and maximum number of bytes to transfer
        n_bits_low = sum(file_sizes[t][1] for t in tiles)
        n_bits_high = sum(file_sizes[t][self.n_qual] for t in tiles)

        # Initialize quality representations
        qualities = [0] * n_tiles
        for t in tiles:
            qualities[t - 2] = 1
        n_bits = n_bits_low

        # Handle straightforward cases
        if n_bits_low >= n_bits_max:
            return qualities, list(range(2, n_tiles + 2))
        elif n_bits_high <= n_bits_max:
            for t in tiles:
                qualities[t - 2] = self.n_qual
            return qualities, list(range(2, n_tiles + 2))

        # Define different zones
        d_hor = 2 * np.pi / self.t_hor
        d_vert = np.pi / self.t_vert
        i = int(theta_vp / d_vert)
        j = int(phi_vp / d_hor)
        z_1 = [i * self.t_hor + j + 2]
        z_2 = set()
        j_1 = (j + self.t_hor - 1) % self.t_hor
        j_2 = (j + 1) % self.t_hor
        if i == 0:
            z_2.add(j_1 + 2)
            z_2.add(j_2 + 2)
            if self.t_vert > 1:
                z_2.add(self.t_hor + j_1 + 2)
                z_2.add(self.t_hor + j + 2)
                z_2.add(self.t_hor + j_2 + 2)
        elif i == self.t_vert - 1:
            z_2.add(i * self.t_hor + j_1 + 2)
            z_2.add(i * self.t_hor + j_2 + 2)
            z_2.add((i - 1) * self.t_hor + j_1 + 2)
            z_2.add((i - 1) * self.t_hor + j + 2)
            z_2.add((i - 1) * self.t_hor + j_2 + 2)
        else:
            z_2.add(i * self.t_hor + j_1 + 2)
            z_2.add(i * self.t_hor + j_2 + 2)
            z_2.add((i - 1) * self.t_hor + j_1 + 2)
            z_2.add((i - 1) * self.t_hor + j + 2)
            z_2.add((i - 1) * self.t_hor + j_2 + 2)
            z_2.add((i + 1) * self.t_hor + j_1 + 2)
            z_2.add((i + 1) * self.t_hor + j + 2)
            z_2.add((i + 1) * self.t_hor + j_2 + 2)
        z_2 = list(z_2)
        z_1 = sorted([t for t in z_1 if t in tiles])
        z_2 = sorted([t for t in z_2 if t in tiles and t not in z_1])
        z_3 = sorted([t for t in tiles if t not in z_1 and t not in z_2])
        order = z_1 + z_2 + z_3

        # Increase quality zone by zone, tile by tile
        for z in [z_1, z_2, z_3]:
            for t in z:
                for q in range(2, self.n_qual + 1):
                    fs_bits = file_sizes[t][q] - file_sizes[t][q - 1]
                    if n_bits + fs_bits <= n_bits_max:
                        qualities[t - 2] = q
                        n_bits += fs_bits
                    else:
                        return qualities, order

        # The following should never be executed
        return qualities, order
