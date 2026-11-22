# Changelog

## 1.0.0

First public release of S1Arena-MM.

- 116 tasks with 54,514 items over image, short video, long video and audio, and 106 closed-loop environments (102
  games and 4 spoken-conversation environments).
- The media are in the gated Hugging Face dataset [Swindl/S1Arena-MM](https://huggingface.co/datasets/Swindl/S1Arena-MM);
  `jevarena-mm fetch` reads its tag v1.0.0, gets the sources it does not host from their publishers, and
  `jevarena-mm build` rebuilds the items whose annotations may not be redistributed.
- Model adapters for the System-1 decision models and multimodal LLMs of the leaderboard, with their input policies.
- Results of 33 models on the video tasks and closed loops, and on the image and audio tasks.
