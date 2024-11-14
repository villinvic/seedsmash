import zmq
import pyglet


class WindowWorker:
    def __init__(
            self,
            window,
            update_interval_s=5,
            pipe_name="namedpipe",
            **update_args
    ):
        self.window = window

        context = zmq.Context()
        self.pull_pipe = context.socket(zmq.PULL)
        self.pull_pipe.connect(f"ipc://{pipe_name}")

        pyglet.clock.schedule_interval(self.update_window, update_interval_s, **update_args)
        pyglet.app.run()

    def update_window(self, dt, **update_args):
        data = None
        try:
            while True:
                data = self.pull_pipe.recv_pyobj(flags=zmq.NOBLOCK)
        except Exception as e:
            pass

        return data