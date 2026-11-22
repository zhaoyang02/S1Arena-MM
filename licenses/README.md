# Licenses and attribution

What some sources ask to keep with every copy of their data, and credits for the photos and videos hosted in the
media dataset that keep their own Creative Commons licenses. The dataset card lists each folder's source and license; this folder adds
what those licenses ask for beyond that listing.

## License texts kept with the data

| File | Covers | Why it is here |
|---|---|---|
| `ARKitScenes-LICENSE.txt` | `vsi-bench/**/arkitscenes_*.mp4` (the ARKitScenes scenes of VSI-Bench) | Apple's ARKitScenes license allows redistribution for non-commercial purposes if this notice is kept with it (LICENSE of apple-aiml-research/ARKitScenes, commit 7283761) |
| `AgentRewardBench-TERMS.md` | `agentrewardbench/` | Its terms of use: "Derivatives must also include the terms of use above." |
| `BVCC/LICENSE.txt` and the files beside it | the BVCC clips of `audio_delivery/media/` (aud.judge/naturalness/bvcc) | BVCC's own files (Zenodo record 10691660): its ratings are BSD 3-Clause, whose notice must be kept; the clips come from VCC 2016 and 2018 (CC BY 4.0), VCC 2020 (ODbL 1.0, DbCL) and ESPnet-TTS (from public-domain LJSpeech), each file listing its clips and citations; no Blizzard Challenge samples are used. VCC 2016 and 2018 list DAPS (CC BY-NC 4.0) among their sources, so treat those clips as non-commercial |
| `Nexar-LICENSE.txt` | `nexar/` | The Nexar Open Data License asks redistributions to keep its copyright notice, conditions and disclaimers (LICENSE of empiricearth/nexar_collision_prediction, revision 7535d06) |

## Multimodal-Mind2Web

`jevarena_img/img.step_gui_action/` holds screenshots built from Multimodal-Mind2Web
(osunlp/Multimodal-Mind2Web, revision 1b4c6a8; Zheng et al., 2024), which its card tags `openrail`; SeeAct's README
says its data is under an OpenRAIL-D license (https://huggingface.co/blog/open_rail) and publishes no other text.
Multimodal-Mind2Web is built on Mind2Web (Deng et al., 2023), licensed CC BY 4.0 by the OSU NLP Group. Use these
files under both.

## Credits for Flickr photos and videos

`attribution/<folder>.csv` lists, for every hosted file of the folder, its Flickr page and, where the source records
it, its license (columns `file`, `source_page`, `license`, `license_url`). Each Flickr page names the author and the
license. Written by `python scripts/hf_media.py attribution` from each source's own records.

| File | Folder | Source of the credits |
|---|---|---|
| `attribution/koniq10k.csv` | `koniq10k/` | KonIQ-10k names its YFCC100M photos by Flickr id; YFCC100M holds Creative Commons media only |
| `attribution/konvid-1k.csv` | `konvid-1k/` | KoNViD-1k names its YFCC100M videos by Flickr id |
| `attribution/nextqa.csv` | `nextqa/` | NExT-QA's VidOR videos come from YFCC100M, named by Flickr id |
| `attribution/okvqa.csv` | `okvqa/` | COCO's image records (`flickr_url`, `license`) for the val2014 images |
| `attribution/aokvqa.csv` | `aokvqa/` | A-OKVQA's question to COCO image id, then COCO's image records |
| `attribution/vispr.csv` | `vispr/` | VISPR's `source_url` per image, from Open Images, which lists its images as CC BY 2.0; VISPR records no source for 20 of these files |
