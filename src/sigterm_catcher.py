import signal


class SIGTERMCatcher:

    def __init__(self):
        self.was_killed = False
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        self.was_killed = True