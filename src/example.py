import numpy as np

import file_sizes
import player
import rate_adaptation as ra
import viewport_traces


# Data directory
DATA_DIR = "/home/user/VR/"

# User properties
u_id = 1                # User ID in the dataset collected by Wu et al.

# Video properties
v_id = 0                # Video ID in the dataset collected by Wu et al.
t_hor = 4               # Number of horizontal tiles
t_vert = 4              # Number of vertical tiles
n_qual = 5              # Number of quality representations
n_seg = 30              # Number of video segments
seg_dur = 1.065         # Segment duration [s]

# Player properties
buffer_size = 2.130     # Buffer size [s]
vp_deg = 110            # Viewport size [deg]

# Server properties
host = "10.0.0.1"       # Host IP
port = 8080             # Host port

# Configurations
rah = 3                 # 0: UVP, 1: UVQ, 2: CTF, 3: Petrangeli, 4: Hosseini
reorder = 0             # 0: no reassignment, 1: reassignment
predict = 1             # 0: last known, 1: spherical walk, 2: perfect
n_conn = 1              # Number of parallel TCP connections

# Read file sizes for the given video and tiling scheme
file_sizes = file_sizes.read(DATA_DIR, v_id, t_hor, t_vert, n_qual, n_seg)

# Read timestamps and viewport locations for the given user and video
trace = viewport_traces.read_trace(DATA_DIR, u_id, v_id)

# Initialize rate adaptation heuristic
if rah == 0:
    vp_rad = vp_deg * np.pi / 180
    rate_adapter = ra.UVP(buffer_size, seg_dur, t_hor, t_vert, n_qual, vp_rad)
elif rah == 1:
    vp_rad = 2 * np.pi
    rate_adapter = ra.UVP(buffer_size, seg_dur, t_hor, t_vert, n_qual, vp_rad)
elif rah == 2:
    rate_adapter = ra.CTF(buffer_size, seg_dur, t_hor, t_vert, n_qual)
elif rah == 3:
    rate_adapter = ra.Petrangeli(buffer_size, seg_dur, t_hor, t_vert, n_qual)
else:
    rate_adapter = ra.Hosseini(buffer_size, seg_dur, t_hor, t_vert, n_qual)

# Initiate video player
p = player.Player(host, port, buffer_size, seg_dur, v_id, n_seg, t_hor, t_vert,
                  file_sizes, rate_adapter, reorder, predict, n_conn, trace)

# Run the video session
p.run()
