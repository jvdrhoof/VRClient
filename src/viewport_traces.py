import help_functions as hf


def read_trace(DATA_DIR, u_id, v_id):
    """Reads the viewport locations for a given user and video

    Parameters
    ----------
    u_id : int
        The user's ID in Wu's dataset (1-49)
    v_id : int
        The video's ID in Wu's dataset (0-8)

    Returns
    -------
    list
        A list of tuples, containing a timestamp, phi and theta
    """

    trace = []
    reset = skipped = False

    with open("%s/traces/%i/video_%i.csv" % (DATA_DIR, u_id, v_id), 'r') as f:
        for line in f:
            # Skip first line
            if not skipped:
                skipped = True
                continue

            l = line.strip().split(',')
            time = float(l[1])

            # Remove first few samples if needed
            if trace and time + 1 < trace[-1][0] and not reset:
                trace = []
                reset = True

            # Avoid double entries
            if trace and time == trace[-1][0]:
                continue

            # Convert coordinates
            qx, qy, qz, qw = [float(l[i]) for i in range(2, 6)]
            x, y, z = hf.quat_to_cart(qx, qy, qz, qw)
            phi, theta = hf.cart_to_spher(x, y, z)

            # Append to list
            trace.append((time, phi, theta))

    return trace
