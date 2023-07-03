
# smZNodes
A selection of custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI). There is only one at the moment.

## CLIP Text Encode++

<p align="center">
<img width="290" alt="CLIP Text Encode++" src="https://github.com/shiimizu/ComfyUI_smZNodes/assets/54494639/fdb011a9-238e-47d0-a6ca-1188ef0d99f4">
</p>

Achieve identical embeddings from [`stable-diffusion-webui`](https://github.com/AUTOMATIC1111/stable-diffusion-webui) for [`ComfyUI`](https://github.com/comfyanonymous/ComfyUI).

This means you can reproduce (or come close) to the same images generated on `stable-diffusion-webui` (and its forks) as on `ComfyUI`.

Simple prompts generate _identical_ images. More complex prompts with complex attention/emphasis/weighting generate _very similar_ images. I suspect the slight variations are due to how `ComfyUI` samples. `ComfyUI` also seems to generate slightly more saturated images.

### Installation
Navigate to the **_ComfyUI/custom_nodes_** directory, and run:

```
git clone https://github.com/shiimizu/ComfyUI_smZNodes.git
```

### Info

| Options              | Explanation                                                                                                                                                                                                                                                                                                                                                                                        |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `parser`             | The parser selected to parse prompts into tokens and then transformed (encoded) into embeddings.                                                                                                                                                                                                                                                                                                   |
| `mean_normalization` | `true` or `false`. Whether to take the mean or not.<br>This is done according to `stable-diffusion-webui`. They say that it's probably not the correct way to take the mean.                                                                                                                                                                                                                       |
| `multi_conditioning` | This is usually set to `true` for your positive prompt and `false` for your negative prompt. <blockquote> For each prompt, the list is obtained by splitting the prompt using the `AND` separator. <br>See [Compositional Visual Generation with Composable Diffusion Models](https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/) <blockquote> |

| Parser            | Explanation                                                                      |
| ----------------- | -------------------------------------------------------------------------------- |
| `comfy`           | The default way `ComfyUI` handles everything                                     |
| `comfy++ `        | Uses `ComfyUI`'s parser but encodes tokens the way `stable-diffusion-webui` does |
| `A1111`           | The default parser used in `stable-diffusion-webui`                              |
| `full`            | Same as `A1111` but whitespaces and newlines are stripped                        |
| `compel`          | Uses [`compel`](https://github.com/damian0815/compel)                            |
| `fixed attention` | Prompt is untampered with                                                        |

>**Note**
>Every `parser` except `comfy` uses `stable-diffusion-webui`'s encoding pipeline.

### Tips to get reproducible results on both UIs
- Use the CPU to generate noise on `stable-diffusion-webui`. See [this](https://github.com/comfyanonymous/ComfyUI/discussions/118) discussion.
- Use non-ancestral samplers.
- If you're using `DDIM` as your sampler, use the `ddim_uniform` scheduler.
- There are different `unipc` configurations. Adjust accordingly on both UIs.

### FAQs
- How does this differ from [`ComfyUI_ADV_CLIP_emb`](https://github.com/BlenderNeko/ComfyUI_ADV_CLIP_emb)?
    - In regards to `stable-diffusion-webui`:
      - Mine parses prompts using their parser.
      -  Mine takes the mean exactly as they do. `ComfyUI_ADV_CLIP_emb` probably takes the correct mean but hey, this is for the purpose of reproducible images.
