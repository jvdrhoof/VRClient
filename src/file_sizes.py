import numpy as np


def read(DATA_DIR, v_id, t_hor, t_vert, n_qual, n_seg):
    """Reads the file sizes for a given video. Note that this could be done
       much more efficiently through dataframes!

    Parameters
    ----------
    v_id : int
        The video's ID in Wu's dataset (0-8)
    t_hor, t_vert : int, int
        Number of horizontal and vertical tiles
    n_qual : int
        Number of quality representations
    n_seg : int
        Number of video segments

    Returns
    -------
    dictionary
        A dictionary containing all file sizes
    """

    file_sizes = {}

    n_tiles = t_hor * t_vert
    d_vert = np.pi / t_vert
    d_hor = 2 * np.pi / t_hor

    for s in range(1, n_seg + 1):
        file_sizes[s] = {}
        for t in range(1, n_tiles + 2):
            file_sizes[s][t] = {}

    directory = "%s/video/%i/%ix%i/" % (DATA_DIR, v_id, t_hor, t_vert)
    for q in range(1, n_qual + 1):
        s = 1
        t = 2
        with open("%s/size_%i.dat" % (directory, q), 'r') as f:
            for line in f:
                file_sizes[s][t][q] = int(line.split('\t')[0]) * 8
                t += 1
                if t > n_tiles + 1:
                    t = 2
                    s += 1
                    if s > n_seg:
                        break

    return file_sizes
