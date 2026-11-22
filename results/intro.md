Video results of S1Arena-MM: every model below on the 41 video tasks (`vid`, `lvid`; 17,500 items) and the 102
closed-loop game environments (`loop.game`: GameWorld, BALROG, VS-Bench), one sample per item, each model with the
input policy of its model card or official preprocessing ([docs/models.md](../docs/models.md)). The runs follow the
plans in [`sweeps/`](../sweeps/). Latency is each model's batch-1 measurement on an H100 80 GB PCIe (two
for the models that do not fit on one; [docs/latency.md](../docs/latency.md)), and GameWorld realtime replays the GameWorld games at that latency.

Nemotron-3-Nano-Omni-30B-A3B hears the first 600 s of audio and sees at most 64 frames (its card allows up to an hour
and 128 frames), as longer inputs did not fit on 40 GB GPUs.
