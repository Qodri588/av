import json
import subprocess


class VideoFrameSource:
    """Sequential FFmpeg-backed RGB frame decoder with automatic looping."""

    def __init__(self, path: str, fps: int = 60):
        self.path = path
        self.fps = max(1, int(fps))
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "json", path],
            check=True, capture_output=True, text=True)
        stream = json.loads(probe.stdout)["streams"][0]
        self.width, self.height = int(stream["width"]), int(stream["height"])
        self.frame_bytes = self.width * self.height * 3
        self._proc = None
        self._index = -1
        self._start(reset_index=True)

    def _start(self, reset_index: bool):
        self.close()
        self._proc = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-i", self.path,
             "-vf", f"fps={self.fps},vflip", "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL)
        if reset_index:
            self._index = -1

    def _read_exact(self):
        data = bytearray()
        while len(data) < self.frame_bytes:
            chunk = self._proc.stdout.read(self.frame_bytes - len(data))
            if not chunk:
                return None
            data.extend(chunk)
        return bytes(data)

    def frame_at(self, time_seconds: float):
        target = max(0, int(time_seconds * self.fps))
        if target < self._index:
            self._start(reset_index=True)
        frame = None
        while self._index < target:
            frame = self._read_exact()
            if frame is None:
                self._start(reset_index=False)
                frame = self._read_exact()
                if frame is None:
                    return None
            self._index += 1
        return frame

    def close(self):
        if self._proc:
            self._proc.kill()
            self._proc.wait()
            self._proc = None
