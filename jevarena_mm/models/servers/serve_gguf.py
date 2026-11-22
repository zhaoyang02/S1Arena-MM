"""qev serve --llama-url ..., with one change: qev's LlamaCppEngine sends every branch of a request to llama-server in a
single /completion call, each branch carrying the base64 image; a request with many questions (privacy: 64 labels on a
large photo) exceeds llama-server's HTTP payload limit (413). Branches are sent in chunks of 8 instead; each branch is
an independent completion, so the probabilities are unchanged.

    python -m jevarena_mm.models.servers.serve_gguf --llama-url http://127.0.0.1:8141 --tokenizer <OneJev snapshot> --name X --port 8041
"""
import sys

CHUNK = 8


def main():
    from qev import cli, llamacpp

    orig = llamacpp.LlamaCppEngine._complete

    def complete(self, prompts, extra):
        out = []
        for i in range(0, len(prompts), CHUNK):
            out += orig(self, prompts[i:i + CHUNK], extra)
        return out

    llamacpp.LlamaCppEngine._complete = complete
    sys.argv = ["qev", "serve", *sys.argv[1:]]
    cli.main()


if __name__ == "__main__":
    main()
