import fire
import ollama

def main():
    clent = ollama.Client(
        host="127.0.0.1:11435"
    )
    msg = clent.generate(model="qwen3:0.6b", prompt="Write a poem")
    print(msg.response)

class Dupa:
    def __init__(self):
        self.clent = ollama.Client(
            host="127.0.0.1:11435"
        )

    def dupa(self, prompt: str) -> str:
        return self.clent.generate(model="qwen3:0.6b", prompt=prompt).response

if __name__ == '__main__':
    fire.Fire(Dupa)
    # main()
