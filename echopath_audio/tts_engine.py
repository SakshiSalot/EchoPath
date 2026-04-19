import pyttsx3
import threading
import queue
import time


class TTSEngine:
    """
    Thread-safe, non-blocking TTS engine.
    Runs in background thread so it never freezes the main script.
    """

    def __init__(self, rate=150, volume=1.0, cooldown=2.0):
        self.rate     = rate
        self.volume   = volume
        self.cooldown = cooldown

        self._queue        = queue.Queue()
        self._last_spoken  = {}
        self._last_message = ""
        self._running      = True

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        """Background thread — initializes engine and processes queue."""
        engine = pyttsx3.init()
        engine.setProperty('rate', self.rate)
        engine.setProperty('volume', self.volume)

        # Pick English voice
        voices = engine.getProperty('voices')
        for voice in voices:
            if 'english' in voice.name.lower() or 'en' in voice.id.lower():
                engine.setProperty('voice', voice.id)
                break

        while self._running:
            try:
                item = self._queue.get(timeout=0.1)
                if item:
                    text, out_path = item
                    if out_path:
                        # Save to WAV file
                        engine.save_to_file(text, out_path)
                        engine.runAndWait()
                    else:
                        # Live playback
                        engine.say(text)
                        engine.runAndWait()
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[TTS Error] {e}")

    def speak(self, message: str, force: bool = False):
        """Queue a message for live audio playback."""
        if not message:
            return
        now = time.time()
        if message == self._last_message and not force:
            return
        if (now - self._last_spoken.get(message, 0)) < self.cooldown and not force:
            return
        self._last_spoken[message] = now
        self._last_message = message
        self._queue.put((message, None))

    def save(self, message: str, out_path: str):
        """Save a spoken phrase to a WAV file."""
        self._queue.put((message, out_path))
        self._queue.join()  # wait until file is saved before continuing

    def stop(self):
        """Cleanly shut down the TTS thread."""
        self._running = False
        self._thread.join(timeout=2)