
# smZNodes
A selection of custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

1. [CLIP Text Encode++](#clip-text-encode)
2. [Settings](#settings)

Contents

* [Tips to get reproducible results on both UIs](#tips-to-get-reproducible-results-on-both-uis)
* [FAQs](#faqs)
* [Installation](#installation)


## CLIP Text Encode++

<p align="center">
    <div class="row">
        <p align="center">
            <img width="1255" alt="Clip Text Encode++ – Default settings on stable-diffusion-webui" src="https://github.com/shiimizu/ComfyUI_smZNodes/assets/54494639/ec85fd20-2b83-43cd-9f19-5aba34034e2a">
    </div>
</p>

CLIP Text Encode++ can generate identical embeddings from [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

This means you can reproduce the same images generated from `stable-diffusion-webui` on `ComfyUI`.

Simple prompts generate _identical_ images. More complex prompts with complex attention/emphasis/weighting may generate images with slight differences. In that case, you can try using the `Settings` node to match outputs.

### Features

- [Prompt editing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#prompt-editing)
    - [Alternating words](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#alternating-words)
- [`AND`](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#composable-diffusion) keyword (similar to the ConditioningCombine node)
- [`BREAK`](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#break-keyword) keyword (same as the ConditioningConcat node)
- Weight normalization
- Optional `embedding:` identifier


### Comparisons
These images can be dragged into ComfyUI to load their workflows. Each image is done using the [Silicon29](https://huggingface.co/Xynon/SD-Silicon) (in SD v1.5) checkpoint with 18 steps using the Heun sampler.

|stable-diffusion-webui|A1111 parser|Comfy parser|
|:---:|:---:|:---:|
| ![00008-0-cinematic wide shot of the ocean, beach, (palmtrees_1 5), at sunset, milkyway](https://github.com/shiimizu/ComfyUI_smZNodes/assets/54494639/719457d8-96fc-495e-aabc-48c4fe4d648d) | ![A1111 parser comparison 1](https://github.com/shiimizu/ComfyUI_smZNodes/assets/54494639/e446b4ab-6f11-4194-b708-f7bdd1cb8fa8) | ![Comfy parser comparison 1](https://github.com/shiimizu/ComfyUI_smZNodes/assets/54494639/e2e04235-02cc-433a-a2f0-7d58be14d6f5) |
| ![00007-0-a photo of an astronaut riding a horse on mars, ((palmtrees_1 2) on water)](https://github.com/shiimizu/ComfyUI_smZNodes/assets/54494639/9ad8b569-8c6d-4a09-bf36-288d81ce4cf9) | ![A1111 parser comparison 2](https://github.com/shiimizu/ComfyUI_smZNodes/assets/54494639/81767441-c286-41db-a59a-4c69603d84d7) | ![Comfy parser comparison 2](https://github.com/shiimizu/ComfyUI_smZNodes/assets/54494639/ed62c23c-c9bd-41cf-9a37-eab4f9d5e12b) |

Image slider links:
- https://imgsli.com/MTkxMjE0
- https://imgsli.com/MTkxMjEy

### Options

|Name|Description|
| --- | --- |
| `parser` | The parser to parse prompts into tokens and then transformed (encoded) into embeddings. Taken from [SD.Next](https://github.com/vladmandic/automatic/discussions/99#discussioncomment-5931014). |
| `mean_normalization` | Whether to take the mean of your prompt weights. It's `true` by default on `stable-diffusion-webui`.<br>This is implemented according to how it is in `stable-diffusion-webui`. |
| `multi_conditioning` | <blockquote> For each prompt, the list is obtained by splitting the prompt using the `AND` separator. <br>See: [Compositional Visual Generation with Composable Diffusion Models](https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/) </blockquote> <ul><li>a way to use multiple prompts at once</li><li>allows `AND` in the negative prompt as well</li><li>supports weights for prompts: `a cat :1.2 AND a dog AND a penguin :2.2`. The weights default to 1</li><li>each prompt gets a cfg value of `cfg * weight / N`, where `N` is the number of positive prompts. In `stable-diffusion-webui`, each prompt gets a cfg value of `cfg * weight`. To match their behaviour, you can add a weight of `:N` to every prompt _or_ simply set a cfg value of `cfg * N`</li></ul>This uses `CFGDenoiser` internally, so if it's disabled in the `Settings` node, the prompts will act like it came from the ConditioningCombine node and go through the default behaviour in ComfyUI. |
|`use_old_emphasis_implementation`| <blockquote>Use old emphasis implementation. Can be useful to reproduce old seeds.</blockquote>|

> [!TIP]  
> You can right click the node to show/hide some of the widgets. E.g. the `with_SDXL` option.

<br>

| Parser            | Description                                                                      |
| ----------------- | -------------------------------------------------------------------------------- |
| `comfy`           | The default way `ComfyUI` handles everything                                     |
| `comfy++`         | Uses `ComfyUI`'s parser but encodes tokens the way `stable-diffusion-webui` does, allowing to take the mean as they do. |
| `A1111`           | The default parser used in `stable-diffusion-webui`                              |
| `full`            | Same as `A1111` but whitespaces, newlines, and special characters are stripped                        |
| `compel`          | Uses [`compel`](https://github.com/damian0815/compel)                            |
| `fixed attention` | Prompt is untampered with                                                        |

> [!IMPORTANT]  
> Every `parser` except `comfy` uses `stable-diffusion-webui`'s encoding pipeline.

> [!WARNING]  
> LoRA syntax (`<lora:name:1.0>`) is not suppprted.

## Settings

<div align="center">
    <img width="1262" alt="Settings-node-showcase" src="https://github.com/shiimizu/ComfyUI_smZNodes/assets/54494639/37b50faa-81f4-41b1-88ec-2b28e9c5708e">
    <p>Settings node showcase</p>
</div>


The `Settings` node is a dynamic node functioning similar to the Reroute node and is used to fine-tune results during sampling or tokenization. The inputs can be replaced with another input type even after it's been connected. `CLIP` inputs only apply settings to CLIP Text Encode++. Settings apply locally based on its links just like nodes that do model patches. I made this node to explore the various settings found in `stable-diffusion-webui`.

This node can change whenever it is updated, so you may have to **recreate** it to prevent issues. Settings can be overridden by using another `Settings` node somewhere past a previous one. Right click the node for the `Hide/show all descriptions` menu option.


## Tips to get reproducible results on both UIs
- Use the same seed, sampler settings, RNG (CPU or GPU), clip skip (CLIP Set Last Layer), etc.
- Ancestral and SDE samplers may not be deterministic.
- If you're using `DDIM` as your sampler, use the `ddim_uniform` scheduler.
- There are different `unipc` configurations. Adjust accordingly on both UIs.

## FAQs
- How does this differ from [`ComfyUI_ADV_CLIP_emb`](https://github.com/BlenderNeko/ComfyUI_ADV_CLIP_emb)?
    - While the weights are normalized in the same manner, the tokenization and encoding pipeline that's taken from stable-diffusion-webui differs from ComfyUI's. These small changes add up and ultimately produces different results.
- Where can I learn more about how ComfyUI interprets weights?
    - https://comfyanonymous.github.io/ComfyUI_examples/faq/
    - https://blenderneko.github.io/ComfyUI-docs/Interface/Textprompts/
    - [comfyui.creamlab.net)](https://comfyui.creamlab.net/guides/f_text_prompt)


## Installation

Three methods are available for installation:

1. Load via [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager)
2. Clone the repository directly into the extensions directory.
3. Download the project manually.


### Load via ComfyUI Manager


<div align="center">
    <img width="1207" alt="ComfyUI Manager" src="https://github.com/shiimizu/ComfyUI_smZNodes/assets/54494639/310d934d-c8db-4c4a-af2a-7a26938eb751">
    <p>Install via ComfyUI Manager</p>
</div>

### Clone Repository

```shell
cd path/to/your/ComfyUI/custom_nodes
git clone https://github.com/shiimizu/ComfyUI_smZNodes.git
```

### Download Manually

1. Download the project archive from [here](https://github.com/shiimizu/ComfyUI_smZNodes/archive/refs/heads/main.zip).
2. Extract the downloaded zip file.
3. Move the extracted files to `path/to/your/ComfyUI/custom_nodes`.
4. Restart ComfyUI

The folder structure should resemble: `path/to/your/ComfyUI/custom_nodes/ComfyUI_smZNodes`.


### Update

To update the extension, update via [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) or pull the latest changes from the repository:

```shell
cd path/to/your/ComfyUI/custom_nodes/ComfyUI_smZNodes
git pull
```

## Credits

* [AUTOMATIC1111](https://github.com/AUTOMATIC1111) / [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
* [comfyanonymous](https://github.com/comfyanonymous) / [ComfyUI](https://github.com/comfyanonymous/ComfyUI)