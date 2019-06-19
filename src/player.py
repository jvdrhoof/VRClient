import httplib2
import numpy as np
import queue
import time
import threading

import help_functions as hf


class Player(object):
    """
    A Python-based VR player

    ...

    Methods
    -------
    run()
        Run the VR player
    """

    def __init__(self, host, port, buffer_size, seg_dur, v_id, n_seg, t_hor,
                 t_vert, file_sizes, rate_adapter, reassignment, predict,
                 n_conn, trace):
        """
        Parameters
        ----------
        host : str
            IP address
        port : int
            Port number
        buffer_size : float
            Buffer size [s]
        seg_dur : float
            Segment duration [s]
        v_id : int
            The video's ID in Wu's dataset (0-8)
        n_seg : int
            Number of video segments
        t_hor, t_vert : int, int
            Number of horizontal and vertical tiles
        file_sizes : dictionary
            File sizes for all segments, tiles and quality representations
        rate_adapter : RateAdaptation
            Rate adaptation heuristic
        reassignment : bool
            Whether the quality representations can be reassigned
        predict : int
            What type of viewport prediction should be used
        n_conn : int
            Number of parallel TCP connections
        trace : list
            A list of tuples, containing a timestamp, phi and theta
        """

        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.seg_dur = seg_dur
        self.v_id = v_id
        self.n_seg = n_seg
        self.t_hor = t_hor
        self.t_vert = t_vert
        self.file_sizes = file_sizes
        self.rate_adapter = rate_adapter
        self.reassignment = reassignment
        self.predict = predict
        self.n_conn = n_conn
        self.trace = trace

        self.download_queue = DownloadQueue()
        self.seg_queue = queue.Queue()
        self.play_queue = queue.Queue()
        self.time_start_playing = -1
        self.iterator = 0

        self.freeze_freq = 0
        self.freeze_dur = 0

    def _play(self):
        """Simulates playout of video content

        """

        # Run until the video has finished playing
        n_played = 0
        freezing = False

        while n_played < self.n_seg:
            # Check if content is available
            if n_played > 0 and self.seg_queue.qsize() == 0:
                freezing = True

            # Get segment from queue
            qualities = self.seg_queue.get()
            time_curr = time.time()
            if n_played == 0:
                self.time_start_playing = time_curr

            # Update freeze statistics
            if freezing:
                print("Freeze of length %f observed for segment %i" %
                      (time_curr - time_last_played - self.seg_dur,
                       n_played + 1))
                self.freeze_freq += 1
                self.freeze_dur += time_curr - time_last_played - self.seg_dur

            # Determine viewport coordinates and forward to quality queue
            time_start = n_played * self.seg_dur
            while self.iterator < len(self.trace) - 1 and \
                    time_start > self.trace[self.iterator + 1][0]:
                self.iterator += 1
            phi, theta = self.trace[self.iterator][1:]
            self.play_queue.put((n_played + 1, time_curr))

            # Pass one segment duration, while updating the viewport iterator
            time_now = time.time()
            time_remaining = max(self.seg_dur - (time_now - time_curr), 0)
            time_it = time_now
            while time_it - time_now < time_remaining:
                while self.iterator < len(self.trace) - 1 and time_start \
                      + time_it - time_curr > self.trace[self.iterator + 1][0]:
                    self.iterator += 1
                time.sleep(0.020)
                time_it = time.time()

            # End current task
            self.seg_queue.task_done()
            time_last_played = time_curr
            freezing = False
            n_played += 1

    def _predict_viewport(self, delta_t, s_id):
        """Predicts future viewport location P_3

        Parameters
        ----------
        delta_t : float
            Time before the predicted point, multiple of 0.250
        s_id : int
            Segment number

        Returns
        -------
        float, float, float, float
            The spherical coordinates of P_3 and P_2
        """

        # Last known location
        if self.predict == 0 or self.iterator < 5:
            phi_2, theta_2 = self.trace[self.iterator][1:]
            phi_3, theta_3 = self.trace[self.iterator][1:]

        # Spherical walk
        elif self.predict == 1:
            d = {0: 0, 0.25: 0.1875, 0.5: 0.3, 0.75: 0.375, 1: 0.4,
                 1.25: 0.4375, 1.5: 0.45, 1.75: 0.4375, 2: 0.4}
            phi_1, theta_1 = self.trace[self.iterator - 5][1:]
            phi_2, theta_2 = self.trace[self.iterator][1:]
            d_1 = self.trace[self.iterator][0] \
                - self.trace[self.iterator - 5][0]
            d_2 = d[delta_t]
            phi_3, theta_3 = hf.walk_on_sphere(phi_1, theta_1, phi_2, theta_2,
                                               d_1, d_2)

        # Perfect prediction
        else:
            t_playout = (s_id - 1) * self.seg_dur
            i = self.iterator
            while i < len(self.trace) - 1 and t_playout > self.trace[i + 1][0]:
                i += 1
            phi_2, theta_2 = self.trace[self.iterator][1:]
            phi_3, theta_3 = self.trace[i][1:]

        return phi_3, theta_3, phi_2, theta_2

    def _do_work(self):
        """Initiates a new HTTP connection and retrieves video content

        """

        # Initiate new HTTP connection
        h = httplib2.Http()

        while True:
            # Retrieve tuple containing segment number, tile number and quality
            tup = self.download_queue.get()
            s_id, t_id, quality = tup[1], tup[2], tup[3]

            # If segment number smaller than 0: stop running
            if s_id < 0:
                self.download_queue.task_done()
                return

            # If the tile is not the base layer, log the quality
            if t_id > 1:
                self.qualities[t_id - 2] = quality

            # Generate URL and send the request
            url = self._generate_url(s_id, t_id, quality)
            print(url)
            (resp_headers, content) = h.request(url, "GET")

            # End task
            self.download_queue.task_done()

    def _generate_url(self, s_id, t_id, quality):
        """Generates URL based on segment number, tile number and quality

        Parameters
        ----------
        s_id : int
            Segment number
        t_id : int
            Tile number
        quality : int
            Quality representation

        Returns
        -------
        str
            Generated URL
        """

        h = "http://%s:%s" % (self.host, self.port)
        d = "%s/%sx%s" % (self.v_id, self.t_hor, self.t_vert)

        # If quality is zero, initialization files are requested
        if quality == 0:
            # Non-tiled video
            if self.t_hor == self.t_vert == 1:
                f = "1/seg_dashinit.mp4"
            # Base tile for tiled video
            elif t_id == -1:
                f = "1/seg_dash_set1_init.mp4"
            # Regular tile for tiled video
            else:
                f = "1/seg_dash_track%i_init.mp4" % (-t_id)
        # If not, regular video content is requested
        else:
            # Non-tiled video
            if self.t_hor == self.t_vert == 1:
                f = "%i/seg_dash%i.m4s" % (quality, s_id)
            # Tiled video
            else:
                f = "%i/seg_dash_track%i_%i.m4s" % (quality, t_id, s_id)

        return "%s/%s/%s" % (h, d, f)

    def _buffer(self):
        """Buffers VR content

        """

        # Initialize settings
        bandwidth = 0
        n_tiles = self.t_hor * self.t_vert

        # Start workers
        for _ in range(self.n_conn):
            t = threading.Thread(target=self._do_work)
            t.daemon = True
            t.start()

        # Loop over all segments, in order
        for s_id in range(1, self.n_seg + 1):

            print("Buffering segment %i" % s_id)

            # Reset downloaded qualities
            self.qualities = [0] * n_tiles

            # If this is the first segment, send requests to download queue
            if s_id == 1:
                if self.t_hor == self.t_vert == 1:
                    self.download_queue.put((0, 1, -1, 0))
                else:
                    for t_id in range(1, n_tiles + 2):
                        self.download_queue.put((0, 1, -t_id, 0))

            # Set time since last update
            time_last_update = time.time()

            # If playout has not started yet, default values are assigned
            if self.time_start_playing < 0:
                phi_2 = phi_3 = np.pi
                theta_2 = theta_3 = np.pi / 2
                time_pred = time_dl = 0
                time_dl = self.seg_dur

            # Otherwise, viewport prediction is used
            else:
                # Round delta_t to a multiple of 0.250 s
                delta_t = min(0.25 * max(round(time_pred / 0.25), 0), 2)

                # Predict future location
                phi_3, theta_3, phi_2, theta_2 = \
                    self._predict_viewport(delta_t, s_id)

            # Determine bitrate budget
            budget_bits = bandwidth * time_dl

            # Rate adaptation
            qualities, order = self.rate_adapter.adapt(s_id,
                                                       self.file_sizes[s_id],
                                                       budget_bits,
                                                       self.qualities, phi_3,
                                                       theta_3, phi_2, theta_2)

            print(qualities)

            # Time before downloading current segment
            time_start = time.time()

            # Send required resources to download queue
            if self.t_hor > 1 or self.t_vert > 1:
                self.download_queue.put((1, s_id, 1, 1))
            d = 2
            for t_id in order[::-1]:
                quality = qualities[t_id - 2]
                self.download_queue.put((d, s_id, t_id, quality))
                d += 1

            # If reassignments are possible, values are frequently recalculated
            if self.reassignment:

                while self.download_queue.qsize() > 1:

                    # Viewport prediction
                    if self.time_start_playing > 0:
                        time_delta = time.time() - time_last_update
                        time_last_update = time.time()
                        time_pred -= time_delta
                        time_dl -= time_delta
                        delta_t = min(0.25 * max(round(time_pred / 0.25), 0),
                                      2)
                        phi_3, theta_3, phi_2, theta_2 = \
                            self._predict_viewport(delta_t, s_id)

                    # Rate adaptation
                    fs = self.file_sizes[s_id]
                    qualities, order = self.rate_adapter.adapt(s_id, fs,
                                                               budget_bits,
                                                               self.qualities,
                                                               phi_3, theta_3,
                                                               phi_2, theta_2)

                    # Update required resources in download queue
                    d = 2
                    tuples = []
                    for t_id in order[::-1]:
                        quality = qualities[t_id - 2]
                        if quality > 0:
                            tuples.append((d, s_id, t_id, quality))
                            d += 1
                    self.download_queue.replace(tuples)

                    # Sleep (+-47Hz)
                    time.sleep(0.020)

            # Wait until all tiles are downloaded
            self.download_queue.join()

            print(self.qualities)

            # Total download time
            time_passed = time.time() - time_start

            # Determine bandwidth
            bits = 0
            for t_id in range(2, n_tiles + 2):
                quality = self.qualities[t_id - 2]
                bits += self.file_sizes[s_id][t_id][quality]
            bandwidth = bits / time_passed
            print(bandwidth / 1000000, time_passed)

            # Push segment to playout queue
            self.seg_queue.put(self.qualities[:])

            # If the buffer is not full, continue
            if self.seg_queue.qsize() < self.buffer_size / self.seg_dur:
                if self.play_queue.qsize() == 0:
                    time_pred = time_dl = 0
                else:
                    while self.play_queue.qsize() > 0:
                        last_id, last_played = self.play_queue.get()
                    next_id = s_id + 1
                    next_played = last_played + (next_id - last_id) \
                        * self.seg_dur
                    time_pred = next_played - time.time()
                    time_dl = self.seg_dur - (self.buffer_size - time_pred)

            # Otherwise, wait till a buffered segment is consumed
            else:
                limit = self.buffer_size / self.seg_dur
                while self.seg_queue.qsize() >= limit:
                    time.sleep(0.001)
                time_pred = self.buffer_size
                time_dl = self.seg_dur

            print("----------------------------------------------------------")

        # Terminate workers
        for _ in range(self.n_conn):
            self.download_queue.put((0, -1, 0, 0))

    def run(self):
        """Run the VR player

        """

        t = threading.Thread(target=self._play)
        t.daemon = True
        t.start()
        self._buffer()
        self.seg_queue.join()
        t.join()


class DownloadQueue:
    """
    A queue for prioritized tile downloads

    ...

    Methods
    -------
    put(tup)
        Put a tuple in the queue

    get()
        Get the next tuple in the queue

    replace(tups)
        Replace tuples, possibly containing new decisions

    qsize()
        Number of tuples remaining

    task_done()
        Indicate finishing a task

    join()
        Wait for the queue to be empty
    """

    def __init__(self):
        self.d = {}
        self.i = 0
        self.mutex = threading.Lock()

    def put(self, tup):
        self.mutex.acquire()
        self.d[tup[2]] = tup
        self.i += 1
        self.mutex.release()

    def get(self):
        while True:
            self.mutex.acquire()
            if len(self.d) > 0:
                k = min(self.d.items(), key=lambda x: x[1][0])[0]
                tup = self.d[k]
                del self.d[k]
                self.mutex.release()
                return tup
            self.mutex.release()

    def replace(self, tups):
        self.mutex.acquire()
        for tup in tups:
            if tup[2] in self.d:
                self.d[tup[2]] = tup
        self.mutex.release()

    def qsize(self):
        return len(self.d)

    def task_done(self):
        self.mutex.acquire()
        self.i -= 1
        self.mutex.release()

    def join(self):
        while self.i > 0:
            time.sleep(0.001)
