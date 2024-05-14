import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { getPngMetadata } from "../../../scripts/pnginfo.js";
let _EXIF = null

app.registerExtension({
	name: "Comfy.smZ.WorkflowImage",

	/**
	 * Set up the app on the page
	 */
    async setup() {
        // exif.js is wrapped around a try{} clause because ComfyUI tries to import the script directly
        // when we have to load it through the browser.
        const externalScript = document.createElement('script');
        // Get the current script's file location by using an Error object and inspecting its stack trace.
        let exifPath = null;
        try {
            throw new Error();
        } catch (error) {
            // Extract the stack trace as a string
            const stackTrace = error.stack || error.stacktrace;

            // Split the stack trace into lines
            const stackLines = stackTrace.split('\n');

            // Use the second line, which contains information about the caller
            const callerLine = stackLines[1];

            // Extract the file location from the caller line
            const scriptLocation = callerLine.match(/http.*\.js:\d+:\d+/)[0];
            const url = new URL(scriptLocation);
            const relativePath = url.pathname
            exifPath = relativePath.substring(0, relativePath.lastIndexOf('/')) + '/exif.js';
        }
        const _exifPath = exifPath? exifPath.toLowerCase() : ""
        if (!exifPath || !_exifPath.includes('ComfyUI') || !_exifPath.includes('smz') || !_exifPath.includes('exif.js'))
            exifPath = '/extensions/ComfyUI_smZNodes/js/exif.js'
        externalScript.src = exifPath;
        externalScript.onload = function(e) {
            _EXIF = EXIF
            const handleFile = app.handleFile
            app.handleFile = async function(file) {
                let r = null
                if (file.type === "image/png") {
                    const pngInfo = await getPngMetadata(file);
                    if (pngInfo?.workflow) {
                        r = await app.loadGraphData(JSON.parse(pngInfo.workflow));
                    } else if (pngInfo?.prompt) {
                        r = app.loadApiJson(JSON.parse(pngInfo.prompt));
                    } else if (pngInfo?.parameters) {
                        r = await importA1111(app.graph, pngInfo.parameters);
                    } else {
                        this.showErrorOnFileLoad(file)
                    }
                } else if (file.type === "image/jpeg" || file.type === "image/jpg") {
                    let jpegMetadata = await getJpegMetadataA111(app, file)
                    if (!jpegMetadata) {
                        jpegMetadata = await getPngMetadata(file);
                        if (jpegMetadata) {
                            r = await importA1111(app.graph, jpegMetadata.parameters);
                        } else {
                            this.showErrorOnFileLoad(file)
                        }
                    }
                } else {
                    r = handleFile.apply(this, arguments);
                }
                return r
            }
        };
        document.head.appendChild(externalScript);
    },
})

export async function getJpegMetadataA111(app, file) {
    return await new Promise((resolve) => {
        try {
            const reader = new FileReader();
            reader.onload = (event) => {
                try {
                    if (_EXIF) {
                        let rawJpegMetdata = _EXIF.readFromBinaryFile(event.target.result)
                        if (Object.keys(rawJpegMetdata).includes('UserComment')) {
                            const jpegMetadata = String.fromCharCode(...rawJpegMetdata.UserComment.slice(9).filter((value) => value !== 0));
                            importA1111(app.graph, jpegMetadata);
                            resolve(jpegMetadata);
                            return jpegMetadata
                        }
                    }
                    resolve(false);
                } catch (error) {
                    console.error('[smZ.WorkflowImage]', error)
                    resolve(false);
                }
            };
            reader.onerror = () => resolve(false);
            reader.readAsArrayBuffer(file);
        } catch (error) {
            resolve(false);
        }
    })
}

export async function importA1111(graph, parameters) {
    const p = parameters.lastIndexOf("\nSteps:");
    if (p > -1) {
        const embeddings = await api.getEmbeddings();
        const regexp = /(.+?): (?:"(.+?)",?|(.+?)(?:,|$))/gm;
        let opts = [...parameters.substr(p).matchAll(regexp)]
            .reduce((a, m) => ({ ...a, [m[1].trim().toLowerCase()]: m[3] ? m[3] : m[2] }), {})
        const p2 = parameters.lastIndexOf("\nNegative prompt:", p);
        if (p2 > -1) {
            let positive = parameters.substr(0, p2).trim();
            let negative = parameters.substring(p2 + 18, p).trim();

            const ckptNode = LiteGraph.createNode("CheckpointLoaderSimple");
            const clipSkipNode = LiteGraph.createNode("CLIPSetLastLayer");
            const positiveNode = LiteGraph.createNode("smZ CLIPTextEncode");
            const negativeNode = LiteGraph.createNode("smZ CLIPTextEncode");
            const settingsNode = LiteGraph.createNode("smZ Settings");
            const samplerNode = LiteGraph.createNode("KSampler");
            const imageNode = LiteGraph.createNode("EmptyLatentImage");
            const vaeNode = LiteGraph.createNode("VAEDecode");
            const vaeLoaderNode = LiteGraph.createNode("VAELoader");
            const saveNode = LiteGraph.createNode("SaveImage");
            let hrSamplerNode = null;

            setWidgetValue(positiveNode, "parser", "A1111");
            setWidgetValue(negativeNode, "parser", "A1111");
            setWidgetValue(settingsNode, "RNG", "gpu");
            setWidgetValue(settingsNode, "sgm_noise_multiplier", false);


            const ceiln = (v, n) => Math.ceil(v / n) * n;

            function getWidget(node, name) {
                return node.widgets.find((w) => w.name === name);
            }

            function setWidgetValue(node, name, value, isOptionPrefix) {
                const w = getWidget(node, name);
                if (isOptionPrefix) {
                    const o = w.options.values.find((w) => w.startsWith(value));
                    if (o) {
                        w.value = o;
                    } else {
                        console.warn(`Unknown value '${value}' for widget '${name}'`, node);
                        w.value = value;
                    }
                } else {
                    w.value = value;
                }
            }

            // Fuzzy search. Hash checking would be better.
            const similarityThreshold = 0.4;

            function stringSimilarity(str1, str2, gramSize = 2) {
                function getNGrams(s, len) {
                    s = ' '.repeat(len - 1) + s.toLowerCase() + ' '.repeat(len - 1);
                    let v = new Array(s.length - len + 1);
                    for (let i = 0; i < v.length; i++) {
                        v[i] = s.slice(i, i + len);
                    }
                    return v;
                }

                if (!str1?.length || !str2?.length) { return 0.0; }

                let s1 = str1.length < str2.length ? str1 : str2;
                let s2 = str1.length < str2.length ? str2 : str1;

                let pairs1 = getNGrams(s1, gramSize);
                let pairs2 = getNGrams(s2, gramSize);
                let set = new Set(pairs1);

                let total = pairs2.length;
                let hits = 0;
                for (let item of pairs2) {
                    if (set.delete(item)) {
                        hits++;
                    }
                }
                return hits / total;
            }

            function createLoraNodes(clipNode, text, prevClip, prevModel) {
                const loras = [];
                text = text.replace(/<(?:lora|lyco):([^:]+:[^>]+)>/g, function(m, c) {
                    const s = c.split(":");
                    const weight = parseFloat(s[1]);
                    if (isNaN(weight)) {
                        console.warn("Invalid LORA", m);
                    } else {
                        loras.push({ name: s[0], weight });
                    }
                    return "";
                });

                for (const l of loras) {
                    const loraNode = LiteGraph.createNode("LoraLoader");
                    graph.add(loraNode);

                    // Fuzzy search
                    const w = getWidget(loraNode, "lora_name");
                    const o = w.options.values.map((w) => ({ name: w, similarity: stringSimilarity(l.name, basename(w)) }));
                    o.sort((a, b) => b.similarity - a.similarity)
                    setWidgetValue(loraNode, "lora_name", o?.[0]?.name ? (o[0].similarity > similarityThreshold ? o[0].name : l.name) : l.name, true);

                    // setWidgetValue(loraNode, "lora_name", l.name, true);
                    setWidgetValue(loraNode, "strength_model", l.weight);
                    setWidgetValue(loraNode, "strength_clip", l.weight);
                    prevModel.node.connect(prevModel.index, loraNode, 0);
                    prevClip.node.connect(prevClip.index, loraNode, 1);
                    prevModel = { node: loraNode, index: 0 };
                    prevClip = { node: loraNode, index: 1 };
                }

                prevClip.node.connect(1, clipNode, 0);
                prevModel.node.connect(0, samplerNode, 0);
                if (hrSamplerNode) {
                    prevModel.node.connect(0, hrSamplerNode, 0);
                }

                return { text, prevModel, prevClip };
            }

            function replaceEmbeddings(text) {
                if (!embeddings.length) return text;
                return text.replaceAll(
                    new RegExp(
                        "\\b(" + embeddings.map((e) => e.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("\\b|\\b") + ")\\b",
                        "ig"
                    ),
                    "embedding:$1"
                );
            }

            function popOpt(name) {
                const v = opts[name];
                delete opts[name];
                return v;
            }

            function basename(n) {
                const i = n.lastIndexOf('.')
                if (i !== -1)
                    return n.substr(0, i)
                else
                    return n
            }

            graph.clear();
            graph.add(ckptNode);
            graph.add(clipSkipNode);
            graph.add(positiveNode);
            graph.add(negativeNode);
            graph.add(settingsNode);
            graph.add(samplerNode);
            graph.add(imageNode);
            graph.add(vaeNode);
            graph.add(vaeLoaderNode);
            graph.add(saveNode);

            ckptNode.connect(1, clipSkipNode, 0);
            clipSkipNode.connect(0, positiveNode, 0);
            clipSkipNode.connect(0, negativeNode, 0);
            ckptNode.connect(0, samplerNode, 0);
            positiveNode.connect(0, samplerNode, 1);
            negativeNode.connect(0, samplerNode, 2);
            imageNode.connect(0, samplerNode, 3);
            vaeNode.connect(0, saveNode, 0);
            samplerNode.connect(0, vaeNode, 0);
            vaeLoaderNode.connect(0, vaeNode, 1);

            const handlers = {
                model(v) {
                    const vbasename = basename(v)
                    const w = getWidget(ckptNode, "ckpt_name");
                    const o = w.options.values.map((w) => ({ name: w, similarity: stringSimilarity(vbasename, basename(w)) }));
                    o.sort((a, b) => b.similarity - a.similarity)
                    setWidgetValue(ckptNode, "ckpt_name", o?.[0]?.name ? (o[0].similarity > similarityThreshold ? o[0].name : v) : v, true);
                },
                "cfg scale"(v) {
                    setWidgetValue(samplerNode, "cfg", +v);
                },
                "clip skip"(v) {
                    setWidgetValue(clipSkipNode, "stop_at_clip_layer", -v);
                },
                sampler(v) {
                    // SDE Heun solver is found in the SamplerCustom node
                    let name = v.toLowerCase().replace("++", "pp").replaceAll(" ", "_").replace("_heun", "").replace("_sde", "_sde_gpu");
                    if (name.includes("karras")) {
                        name = name.replace("karras", "").replace(/_+$/, "").replace(/_a$/, "_ancestral");
                        setWidgetValue(samplerNode, "scheduler", "karras");
                    } else if (name.includes("exponential")) {
                        name = name.replace("exponential", "").replace(/_+$/, "").replace(/_a$/, "_ancestral");
                        setWidgetValue(samplerNode, "scheduler", "exponential");
                    } else {
                        name = name.replace(/_+$/, "").replace(/_a$/, "_ancestral");
                        setWidgetValue(samplerNode, "scheduler", "normal");
                    }
                    const w = getWidget(samplerNode, "sampler_name");
                    const o = w.options.values.find((w) => w === name || w === "sample_" + name);
                    if (o) {
                        setWidgetValue(samplerNode, "sampler_name", o);
                    }
                },
                size(v) {
                    const wxh = v.split("x");
                    const w = ceiln(+wxh[0], 8);
                    const h = ceiln(+wxh[1], 8);
                    const hrUp = popOpt("hires upscale");
                    const hrSz = popOpt("hires resize");
                    let hrMethod = popOpt("hires upscaler");

                    setWidgetValue(imageNode, "width", w);
                    setWidgetValue(imageNode, "height", h);

                    if (hrUp || hrSz) {
                        let uw, uh;
                        if (hrUp) {
                            uw = w * hrUp;
                            uh = h * hrUp;
                        } else {
                            const s = hrSz.split("x");
                            uw = +s[0];
                            uh = +s[1];
                        }

                        let upscaleNode;
                        let latentNode;

                        if (hrMethod.startsWith("Latent")) {
                            latentNode = upscaleNode = LiteGraph.createNode("LatentUpscale");
                            graph.add(upscaleNode);
                            samplerNode.connect(0, upscaleNode, 0);

                            switch (hrMethod) {
                                case "Latent":
                                    hrMethod = "bilinear";
                                    break;
                                case "Latent (nearest-exact)":
                                    hrMethod = "nearest-exact";
                                    break;
                            }
                            setWidgetValue(upscaleNode, "upscale_method", hrMethod, true);
                        } else {
                            const decode = LiteGraph.createNode("VAEDecode");
                            graph.add(decode);
                            samplerNode.connect(0, decode, 0);
                            vaeLoaderNode.connect(0, decode, 1);

                            const upscaleLoaderNode = LiteGraph.createNode("UpscaleModelLoader");
                            graph.add(upscaleLoaderNode);
                            setWidgetValue(upscaleLoaderNode, "model_name", hrMethod, true);

                            const modelUpscaleNode = LiteGraph.createNode("ImageUpscaleWithModel");
                            graph.add(modelUpscaleNode);
                            decode.connect(0, modelUpscaleNode, 1);
                            upscaleLoaderNode.connect(0, modelUpscaleNode, 0);

                            upscaleNode = LiteGraph.createNode("ImageScale");
                            graph.add(upscaleNode);
                            modelUpscaleNode.connect(0, upscaleNode, 0);

                            const vaeEncodeNode = (latentNode = LiteGraph.createNode("VAEEncode"));
                            graph.add(vaeEncodeNode);
                            upscaleNode.connect(0, vaeEncodeNode, 0);
                            vaeLoaderNode.connect(0, vaeEncodeNode, 1);

                            const previewNode = LiteGraph.createNode("PreviewImage");
                            graph.add(previewNode);
                            decode.connect(0, previewNode, 0);
                        }

                        setWidgetValue(upscaleNode, "width", ceiln(uw, 8));
                        setWidgetValue(upscaleNode, "height", ceiln(uh, 8));

                        hrSamplerNode = LiteGraph.createNode("KSampler");
                        graph.add(hrSamplerNode);
                        ckptNode.connect(0, hrSamplerNode, 0);
                        positiveNode.connect(0, hrSamplerNode, 1);
                        negativeNode.connect(0, hrSamplerNode, 2);
                        latentNode.connect(0, hrSamplerNode, 3);
                        hrSamplerNode.connect(0, vaeNode, 0);

                    }
                },
                steps(v) {
                    setWidgetValue(samplerNode, "steps", +v);
                },
                seed(v) {
                    setWidgetValue(samplerNode, "seed", +v);
                },
                vae(v) {
                    const vbasename = basename(v)
                    const w = getWidget(vaeLoaderNode, "vae_name");
                    const o = w.options.values.map((w) => ({ name: w, similarity: stringSimilarity(vbasename, basename(w)) }));
                    o.sort((a, b) => b.similarity - a.similarity)
                    setWidgetValue(vaeLoaderNode, "vae_name", o?.[0]?.name ? (o[0].similarity > similarityThreshold ? o[0].name : v) : v, true);
                },
                ["hires steps"](v) {
                    const o = +v
                    if (o) setWidgetValue(hrSamplerNode, "steps", o);
                },
                eta(v) {
                    const o = +v
                    if (o) setWidgetValue(settingsNode, "eta", o);
                },
                s_churn(v) {
                    const o = +v
                    if (o) setWidgetValue(settingsNode, "s_churn", o);
                },
                s_tmin(v) {
                    const o = +v
                    if (o) setWidgetValue(settingsNode, "s_tmin", o);
                },
                s_tmax(v) {
                    const o = +v
                    if (o) setWidgetValue(settingsNode, "s_tmax", o);
                },
                s_noise(v) {
                    const o = +v
                    if (o) setWidgetValue(settingsNode, "s_noise", o);
                },
                ensd(v) {
                    const o = +v
                    if (o) setWidgetValue(settingsNode, "ENSD", o);
                },
                rng(v) {
                    const oo = v.toLowerCase()
                    const w = getWidget(settingsNode, "RNG");
                    const o = w.options.values.find((w) => w === oo);
                    if (o)
                        setWidgetValue(settingsNode, "RNG", o);
                },
                ["sgm noise multiplier"](v) {
                    let o = v
                    if (typeof o === "string") o = v.toLowerCase()
                    setWidgetValue(settingsNode, "sgm_noise_multiplier", o || o === "true");
                },
                ["pad conds"](v) {
                    let o = v
                    if (typeof o === "string") o = v.toLowerCase()
                    setWidgetValue(settingsNode, "pad_cond_uncond", o || o === "true");
                },
            };

            if (hrSamplerNode) {
                setWidgetValue(hrSamplerNode, "steps", getWidget(samplerNode, "steps").value);
            }

            for (const opt in opts) {
                if (opt in handlers) {
                    handlers[opt](popOpt(opt));
                }
            }

            if (hrSamplerNode) {
                setWidgetValue(hrSamplerNode, "seed", getWidget(samplerNode, "seed").value);
                setWidgetValue(hrSamplerNode, "cfg", getWidget(samplerNode, "cfg").value);
                setWidgetValue(hrSamplerNode, "scheduler", getWidget(samplerNode, "scheduler").value);
                setWidgetValue(hrSamplerNode, "sampler_name", getWidget(samplerNode, "sampler_name").value);
                setWidgetValue(hrSamplerNode, "denoise", +(popOpt("denoising strength") || "1"));
            }

            let n = createLoraNodes(positiveNode, positive, { node: clipSkipNode, index: 0 }, { node: ckptNode, index: 0 });
            positive = n.text;
            n = createLoraNodes(negativeNode, negative, n.prevClip, n.prevModel);
            negative = n.text;

            setWidgetValue(positiveNode, "text", replaceEmbeddings(positive));
            setWidgetValue(negativeNode, "text", replaceEmbeddings(negative));

            n.prevModel.node.connect(0, settingsNode, 0);
            settingsNode.connect(0, samplerNode, 0);
            if (hrSamplerNode)
                settingsNode.connect(0, hrSamplerNode, 0);

            graph.arrange();

            for (const opt of ["model hash"]) {
                delete opts[opt];
            }

            console.warn("Unhandled parameters:", opts);
        }
    }
}