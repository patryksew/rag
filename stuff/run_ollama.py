import os
import subprocess
import time
import socket
from typing import Optional, Any
import fire
import ollama


def is_ollama_running(host: str = "127.0.0.1", port: int = 11435) -> bool:
    """Check if the Ollama server is already listening on the given host/port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        try:
            s.connect((host, port))
            return True
        except (socket.timeout, ConnectionRefusedError):
            return False


def start_ollama(host: str = "127.0.0.1", port: int = 11435) -> Optional[subprocess.Popen[Any]]:
    """Starts the Ollama server in the background if it's not already running."""
    if is_ollama_running(host, port):
        print(">" * 20, "Ollama server is already running.", "<" * 20, sep="")
        return None

    env = os.environ.copy()
    env["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
    env["OLLAMA_HOST"] = f"{host}:{port}"

    process = subprocess.Popen(
        ["ollama", "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid
    )

    retries = 15
    while retries > 0:
        if is_ollama_running(host, port):
            return process
        time.sleep(1)
        retries -= 1

    raise RuntimeError("Ollama server failed to start in the allotted time.")


class Dupa:
    """Class to manage Ollama interaction."""

    def __init__(self) -> None:
        """Initialize the Dupa manager and start Ollama server."""
        self.process = start_ollama()
        self.client = ollama.Client(
            host="127.0.0.1:11435"
        )

    def dupa(self, prompt: str) -> str:
        """Generate a response from the model for the given prompt.

        Args:
            prompt: Prompt string.

        Returns:
            Generated response string.
        """
        return str(self.client.generate(model="qwen3:0.6b", prompt=prompt).response)

    def __del__(self) -> None:
        """Clean up by terminating the Ollama server process."""
        if hasattr(self, 'process') and self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                print("Error terminating the Ollama server.")
                pass


if __name__ == '__main__':
    fire.Fire(Dupa)
