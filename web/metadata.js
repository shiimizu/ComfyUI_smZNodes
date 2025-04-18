import { app as _app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { getPngMetadata } from "../../scripts/pnginfo.js";

let _EXIF;

_app.registerExtension({
    name: "Comfy.smZ.WorkflowImage",

    /**
     * Set up the app on the page
     */
    async setup(app) {
        // exif.js is wrapped around a try{} clause to prevent runtime exceptions.
        // The script is loaded through the browser.
        const script = document.createElement("script");
        const scriptLocation = new URL("exif.js", import.meta.url);
        script.src = scriptLocation.pathname;

        function removeExt(f) {
            if (!f) return f
            const p = f.lastIndexOf('.')
            if (p === -1) return f
            return f.substring(0, p)
        }

        function getFileNameFromUrl(url) {
            const urlParts = new URL(url);
            const pathParts = urlParts.pathname.split('/');
            return pathParts[pathParts.length - 1];
        }

        function fetchToFile(url) {
            return fetch(url)
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    return response.blob();
                })
                .then(blob => {
                    const fileName = getFileNameFromUrl(url);
                    const mimeType = blob.type;
                    return new File([blob], removeExt(fileName) + "." + mimeType.split("/")[1], { type: mimeType });
                });
        }

        /**
         * Loads workflow data from the specified file
         * @param {File} file
         */
        const handleFile = app.handleFile;
        let _handleFile = async function (file) {
            if (!file) return;
            if (!file.name)
                file.name = `${Date.now()}.${file.type.split("/")[1]}`
            if (file.name.startsWith("http")) {
                return fetchToFile(file.name)
                    .then(_file => {
                        return _handleFile(_file);
                    })
                    .catch(error => {
                        console.error('Error:', error);
                    });
            }
            const fileName = removeExt(file.name)
            let r = null;
            if (file.type === "image/png" || file.type === "image/webp") {
                const pngInfo = file.type === "image/png" ? await getPngMetadata(file) : await getWebpMetadata(file);
                const workflow = pngInfo?.workflow || pngInfo?.Workflow;
                const prompt = pngInfo?.prompt || pngInfo?.Prompt;
                const parameters = pngInfo?.parameters || pngInfo?.Parameters;
                if (workflow) {
                    r = await app.loadGraphData(JSON.parse(workflow), true, true, fileName);
                } else if (prompt) {
                    r = app.loadApiJson(JSON.parse(prompt), fileName);
                } else if (parameters) {
                    // Note: Not putting this in `importA1111` as it is mostly not used
                    // by external callers, and `importA1111` has no access to `app`.
                    // useWorkflowService().beforeLoadNewGraph()
                    r = await importA1111(app.graph, parameters)
                    // useWorkflowService().afterLoadNewGraph(fileName, app.graph.serialize())
                } else {
                    app.showErrorOnFileLoad(file);
                }
            } else if (file.type === "image/jpeg" || file.type === "image/jpg") {
                let value = await getJpegMetadataA111(app, file);
                if (value) {
                    try {
                        const isObj = o => o?.constructor === Object;
                        let obj = JSON.parse(value);
                        if (isObj(obj)) {
                            // image2image workflow
                            console.info("Workflow from CivitAI Generator:", obj)
                            let extra = obj.extra;
                            let extraMetadata = obj.extraMetadata;
                            delete obj["extra"]
                            delete obj["extraMetadata"]
                            for (const key in obj) {
                                if (obj[key].class_type === "LoadImage") {
                                    console.info("Load the original workflow from:", obj[key].inputs.image);
                                    break;
                                }
                            }
                            r = app.loadApiJson(obj, fileName);
                            obj["extra"] = extra;
                            obj["extraMetadata"] = extraMetadata;
                            return r;
                        }
                    } catch (error) { }
                } else {
                    value = await getPngMetadata(file);
                    if (value)
                        r = await importA1111(app.graph, value);
                }
                if (!value)
                    app.showErrorOnFileLoad(file);
            } else {
                r = handleFile.apply(this, arguments);
            }
            return r;
        };

        script.onload = function (ev) {
            try {
                _EXIF = {};
                _EXIF.readExif = EXIF.readFromBinaryFile;
            } catch (e) {
                console.error(e);
            }
            app.handleFile = _handleFile;
        };
        document.head.appendChild(script);
    }
});

export async function getJpegMetadataA111(app, file) {
    return await new Promise((resolve) => {
        try {
            const reader = new FileReader();
            reader.onload = (event) => {
                try {
                    let value = null;
                    let obj = _EXIF?.readExif(event.target.result);
                    if (obj?.UserComment) {
                        let val = obj.UserComment.slice(9).filter(it => it !== 0 && it !== 0x0B);
                        value = new TextDecoder('utf-16').decode(new Uint16Array(val));
                    }
                    if (value) {
                        importA1111(app.graph, value);
                        resolve(value);
                        return value;
                    } else {
                        console.warn("[smZ.WorkflowImage] Metadata not found.");
                        resolve(false);
                    }
                } catch (error) {
                    console.error("[smZ.WorkflowImage]", error);
                    resolve(false);
                }
            };
            reader.onerror = () => resolve(false);
            reader.readAsArrayBuffer(file);
        } catch (error) {
            resolve(false);
        }
    });
}


function parseExifData(exifData) {
    const isLittleEndian = String.fromCharCode(...exifData.slice(0, 2)) === "II";

    function readInt(offset, littleEndian, length) {
        const view = new DataView(exifData.buffer, exifData.byteOffset + offset, length);
        if (length === 2) return view.getUint16(0, littleEndian);
        if (length === 4) return view.getUint32(0, littleEndian);
    }

    const tiffHeaderOffset = 0;
    const ifd0Offset = readInt(tiffHeaderOffset + 4, isLittleEndian, 4);

    function parseIFD(offset) {
        const numEntries = readInt(offset, isLittleEndian, 2);
        const result = {};

        for (let i = 0; i < numEntries; i++) {
            const entryOffset = offset + 2 + i * 12;
            const tag = readInt(entryOffset, isLittleEndian, 2);
            const type = readInt(entryOffset + 2, isLittleEndian, 2);
            const count = readInt(entryOffset + 4, isLittleEndian, 4);
            const valueOffset = readInt(entryOffset + 8, isLittleEndian, 4);

            let value = null;

            if (type === 2) { // ASCII string
                const strBytes = exifData.slice(valueOffset, valueOffset + count - 1);
                value = new TextDecoder("utf-8").decode(strBytes);
            } else if (type === 7 && tag === 0x9286) { // UNDEFINED, UserComment
                const data = exifData.slice(valueOffset, valueOffset + count);
                const encodingHeader = new TextDecoder("ascii").decode(data.slice(0, 8));
                if (encodingHeader.startsWith("UNICODE")) {
                    const utf16Data = data.slice(8);
                    value = new TextDecoder("utf-16be").decode(utf16Data);
                } else {
                    value = "[Unsupported encoding]";
                }
            } else if (type === 4 && count === 1) { // LONG (for Exif IFD pointer)
                const subIFDOffset = valueOffset;
                const subIFD = parseIFD(subIFDOffset);
                result[tag] = subIFD;
                continue;
            }

            result[tag] = value;
        }

        return result;
    }

    return parseIFD(ifd0Offset);
}

function getWebpMetadata(file) {
    return new Promise((r2) => {
        const reader = new FileReader();
        reader.onload = (event) => {
            const webp = new Uint8Array(event.target.result);
            const dataView = new DataView(webp.buffer);
            if (dataView.getUint32(0) !== 1380533830 || dataView.getUint32(8) !== 1464156752) {
                console.error("Not a valid WEBP file");
                r2({});
                return;
            }
            let offset = 12;
            let txt_chunks = {};

            try {
                while (offset < webp.length) {
                    const chunk_length = dataView.getUint32(offset + 4, true);
                    const chunk_type = String.fromCharCode(
                        ...webp.slice(offset, offset + 4)
                    );
                    if (chunk_type === "\0EXI") offset++;
                    if (chunk_type === "EXIF" || chunk_type === "\0EXI") {
                        let data30 = parseExifData(webp.slice(offset + 8, offset + 8 + chunk_length));
                        for (const key in data30) {
                            const value3 = data30[key];
                            if (typeof value3 === "string") {
                                const index2 = value3.indexOf(":");
                                txt_chunks[value3.slice(0, index2)] = value3.slice(index2 + 1);
                            } else if (typeof value3 === "object") {
                                for (const k in value3) {
                                    const value2 = value3[k];
                                    txt_chunks["parameters"] = value2;
                                }
                            }
                        }
                        break;
                    }
                    offset += 8 + chunk_length;
                }
            } catch (e) {
                r2({});
                return;
            }
            r2(txt_chunks);
        };
        reader.readAsArrayBuffer(file);
    });
}

export async function importA1111(graph, parameters) {
    parameters = parameters.replace(/[\u0000\u000B]/gm, "");
    const p = parameters.lastIndexOf("\nSteps:");
    if (p > -1) {
        const embeddings = await api.getEmbeddings();
        const inputList = parameters.substring(p + 1).split(", ");

        function splitOnFirst(str, sep, idx) {
            const index = idx ?? str.indexOf(sep);
            return index < 0 ? [str] : [str.slice(0, index), str.slice(index + sep.length)];
        }

        function parseParameters(inputList) {
            const result = [];
            for (let i = 0; i < inputList.length; i++) {
                const current = inputList[i];
                let endDelimiter = null;

                if (current.includes("[{")) {
                    endDelimiter = "}]";
                } else if (current.includes("{")) {
                    endDelimiter = "}";
                } else if (current.includes(`"`)) {
                    endDelimiter = `"`;
                }

                if (endDelimiter) {
                    const escapedEnd = [...endDelimiter].map(char => `\\${char}`).join('');
                    let j = i;

                    for (; j < inputList.length; j++) {
                        const lookahead = inputList[j];

                        if (lookahead !== endDelimiter &&
                            !lookahead.endsWith(escapedEnd) &&
                            lookahead.endsWith(endDelimiter)
                        ) {
                            break;
                        }
                    }

                    if (i === j) {
                        result.push(current);
                    } else {
                        result.push(inputList.slice(i, j + 1).join(", "));
                        i = j;
                    }
                } else {
                    result.push(current);
                }
            }
            return result.reduce((acc, part) => {
                const [key, value] = splitOnFirst(part, ": ");
                acc[key.trim().toLowerCase()] = value;
                return acc;
            }, {});
        }
        const opts = parseParameters(inputList);
        const p2 = parameters.lastIndexOf("\nNegative prompt:", p);
        const has_neg = p2 > -1;
        if (true) {
            let positive = parameters.substring(0, has_neg ? p2 : p).trim();
            let negative = has_neg ? parameters.substring(p2 + 18, p).trim() : "";

            try {
                for (const o of ["civitai resources", "civitai metadata"]) {
                    if (opts[o])
                        opts[o] = JSON.parse(opts[o])
                    if (Array.isArray(opts[o])) {
                        for (const it of opts[o]) {
                            if (it.modelName) {
                                if (it.type === "checkpoint")
                                    opts.model = it.modelName + `_${it.modelVersionName ?? ""}`
                                else if (it.type === "lora" || it.type.includes("lyco"))
                                    positive += `<${it.type}:${it.modelName}_${it.modelVersionName ?? ""}:${it.weight}>`
                            }
                        }
                    }
                }
            } catch (e) {
                console.error(e);
            }

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
            let nodes = {};

            setWidgetValue(positiveNode, "parser", "A1111");
            setWidgetValue(negativeNode, "parser", "A1111");
            setWidgetValue(settingsNode, "RNG", "gpu");
            setWidgetValue(settingsNode, "sgm_noise_multiplier", false);

            const ceiln = (v, n) => Math.ceil(v / n) * n;

            function getWidget(node, name) {
                return node.widgets.find((w) => w.name === name);
            }

            function setWidgetValue(node, name, value, isOptionPrefix) {
                if (!node) return;
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

            function initNode(key, name) {
                if (!nodes[key]) {
                    const _name = name ? name : (key[0].toUpperCase() + key.slice(1)).split("Node")[0];
                    const node = LiteGraph.createNode(_name);
                    if (node) {
                        nodes[key] = node;
                        graph.add(nodes[key]);
                    }
                }
                return nodes[key];
            }

            // Fuzzy search. Hash checking would be better.
            const similarityThreshold = 0.4;

            function stringSimilarity(str1, str2, gramSize = 2) {
                function getNGrams(s, len) {
                    s = " ".repeat(len - 1) + s.toLowerCase() + " ".repeat(len - 1);
                    let v = new Array(s.length - len + 1);
                    for (let i = 0; i < v.length; i++) {
                        v[i] = s.slice(i, i + len);
                    }
                    return v;
                }

                if (!str1?.length || !str2?.length) {
                    return 0.0;
                }

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
                text = text.replace(/<(?:lora|lyco|lycoris):([^:]+:[^>]+)>/g, function (m, c) {
                    const s = splitOnFirst(c, ":", c.lastIndexOf(":"));
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
                    o.sort((a, b) => b.similarity - a.similarity);
                    setWidgetValue(
                        loraNode,
                        "lora_name",
                        o?.[0]?.name ? (o[0].similarity > similarityThreshold ? o[0].name : l.name) : l.name,
                        true
                    );

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
                return text.replaceAll(new RegExp("\\b(" + embeddings.map((e) => e.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("\\b|\\b") + ")\\b", "ig"), "embedding:$1");
            }

            function popOpt(name, get) {
                const v = opts[name];
                if (!get) delete opts[name];
                return v;
            }

            function basename(s) {
                let i = s.lastIndexOf(".");
                i = i !== -1 ? i : s.length;
                let z = s.replaceAll("\\", "/").lastIndexOf("/");
                z = z !== -1 ? z : 0;
                return s.substring(z, i);
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
            clipSkipNode.mode = 4;
            ckptNode.connect(0, samplerNode, 0);
            positiveNode.connect(0, samplerNode, 1);
            negativeNode.connect(0, samplerNode, 2);
            imageNode.connect(0, samplerNode, 3);
            vaeNode.connect(0, saveNode, 0);
            samplerNode.connect(0, vaeNode, 0);
            vaeLoaderNode.connect(0, vaeNode, 1);

            const handlers = {
                model(v) {
                    const vbasename = basename(v);
                    const w = getWidget(ckptNode, "ckpt_name");
                    const o = w.options.values.map((w) => ({ name: w, similarity: stringSimilarity(vbasename, basename(w)) }));
                    o.sort((a, b) => b.similarity - a.similarity);
                    setWidgetValue(
                        ckptNode,
                        "ckpt_name",
                        o?.[0]?.name ? (o[0].similarity > similarityThreshold ? o[0].name : v) : v,
                        true
                    );
                },
                "cfg scale"(v) {
                    setWidgetValue(samplerNode, "cfg", +v);
                },
                "clip skip"(v) {
                    setWidgetValue(clipSkipNode, "stop_at_clip_layer", -v);
                    clipSkipNode.mode = 0;
                },
                "schedule type"(v) {
                    // This is called after sampler(v)
                    let name = v.toLowerCase().replaceAll(" ", "_");
                    let scheduler_map = {
                        Automatic: "normal",
                        // "Uniform": "",
                        // "Polyexponential": "",
                        // "SGM Uniform": "",
                        // "KL Optimal": "",
                        "Align Your Steps": "ays",
                        DDIM: "ddim_uniform",
                        // "Turbo": "",
                        "Align Your Steps GITS": "gits",
                        "Align Your Steps 11": "ays_30",
                        "Align Your Steps 32": "ays_30+"
                    };
                    for (const k in scheduler_map) {
                        if (v === k) {
                            name = scheduler_map[k];
                            break;
                        }
                    }
                    setWidgetValue(samplerNode, "scheduler", name);
                },
                sampler(v, _sn) {
                    let sn = samplerNode;
                    if (_sn)
                        sn = _sn;
                    // SDE Heun solver is found in the SamplerCustom node
                    let name = v.toLowerCase().replace("cfg++", "cfg_pp").replace("++", "pp").replace("dpm2", "dpm_2").replaceAll(" ", "_").replace("_heun", "").replace("_sde", "_sde_gpu");
                    if (name.includes("karras")) {
                        name = name.replace("karras", "");
                        setWidgetValue(sn, "scheduler", "karras");
                    } else if (name.includes("exponential")) {
                        name = name.replace("exponential", "");
                        setWidgetValue(sn, "scheduler", "exponential");
                    } else {
                        setWidgetValue(sn, "scheduler", "normal");
                    }
                    name = name.replace(/_+$/, "").replace(/_a$/, "_ancestral");
                    // const w = getWidget(sn, "sampler_name");
                    // const o = w.options.values.find((w) => w === name || w === "sample_" + name);
                    const o = name;
                    if (o) {
                        setWidgetValue(sn, "sampler_name", o);
                    }
                    return o;
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

                        if (hrMethod?.startsWith("Latent")) {
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
                    setWidgetValue(samplerNode, "seed", +(popOpt("global seed", true) || v));
                },
                vae(v) {
                    const vbasename = basename(v);
                    const w = getWidget(vaeLoaderNode, "vae_name");
                    const o = w.options.values.map((w) => ({ name: w, similarity: stringSimilarity(vbasename, basename(w)) }));
                    o.sort((a, b) => b.similarity - a.similarity);
                    setWidgetValue(
                        vaeLoaderNode,
                        "vae_name",
                        o?.[0]?.name ? (o[0].similarity > similarityThreshold ? o[0].name : v) : v,
                        true
                    );
                },
                "noise schedule"(v) {
                    const node = initNode("modelSamplingDiscreteNode", "ModelSamplingDiscrete");
                    const w = getWidget(ckptNode, "ckpt_name");
                    if (v === "Zero Terminal SNR") {
                        setWidgetValue(node, "zsnr", true);
                        if (w.value && w.value.toLowerCase().includes("vpred"))
                            setWidgetValue(node, "sampling", "v_prediction");
                    }
                },
                "rescale_cfg_enabled"(v) {
                    const node = initNode("rescaleCFGNode", "RescaleCFG");
                    if (v.toLowerCase() !== "true")
                        node.mode = 4;
                },
                "rescale_cfg_multiplier"(v) {
                    const node = initNode("rescaleCFGNode", "RescaleCFG");
                    setWidgetValue(node, "multiplier", +v);
                },
                "module 1"(v) {
                    this.vae(v);
                },
                ["hires steps"](v) {
                    const o = +v;
                    if (o) setWidgetValue(hrSamplerNode, "steps", o);
                },
                eta(v) {
                    const o = +v;
                    if (o) setWidgetValue(settingsNode, "eta", o);
                },
                s_churn(v) {
                    const o = +v;
                    if (o) setWidgetValue(settingsNode, "s_churn", o);
                },
                s_tmin(v) {
                    const o = +v;
                    if (o) setWidgetValue(settingsNode, "s_tmin", o);
                },
                s_tmax(v) {
                    const o = +v;
                    if (o) setWidgetValue(settingsNode, "s_tmax", o);
                },
                s_noise(v) {
                    const o = +v;
                    if (o) setWidgetValue(settingsNode, "s_noise", o);
                },
                ensd(v) {
                    const o = +v;
                    if (o) setWidgetValue(settingsNode, "ENSD", o);
                },
                rng(v) {
                    const oo = v.toLowerCase();
                    const w = getWidget(settingsNode, "RNG");
                    const o = w.options.values.find((w) => w === oo);
                    if (o) setWidgetValue(settingsNode, "RNG", o);
                },
                ["sgm noise multiplier"](v) {
                    let o = v;
                    if (typeof o === "string") o = v.toLowerCase();
                    setWidgetValue(settingsNode, "sgm_noise_multiplier", o || o === "true");
                },
                ["pad conds"](v) {
                    let o = v;
                    if (typeof o === "string") o = v.toLowerCase();
                    setWidgetValue(settingsNode, "pad_cond_uncond", o || o === "true");
                },
                ["skip early cfg"](v) {
                    setWidgetValue(settingsNode, "skip_early_cond", +v);
                },
                "sd upscale upscaler"(v) {
                    const node = initNode("SDUpscaleNode", "UpscaleModelLoader");
                    setWidgetValue(node, "model_name", v, true);
                },
                ["token merging ratio"](v) {
                    const node = initNode("tomePatchModelNode", "TomePatchModel");
                    setWidgetValue(node, "ratio", +v);
                },
                ["token merging ratio hr"](v) {
                    const node = initNode("tomePatchModelHrNode", "TomePatchModel");
                    setWidgetValue(node, "ratio", +v);
                },
                "freeu_enabled"(v) {
                    const node = initNode("FreeU_V2Node");
                    if (v.toLowerCase() !== "true")
                        node.mode = 4;
                },
                "freeu_b1"(v) {
                    const node = initNode("FreeU_V2Node");
                    setWidgetValue(node, "b1", +v);
                },
                "freeu_b2"(v) {
                    const node = initNode("FreeU_V2Node");
                    setWidgetValue(node, "b2", +v);
                },
                "freeu_s1"(v) {
                    const node = initNode("FreeU_V2Node");
                    setWidgetValue(node, "s1", +v);
                },
                "freeu_s2"(v) {
                    const node = initNode("FreeU_V2Node");
                    setWidgetValue(node, "s2", +v);
                },
                "sag_enabled"(v) {
                    const node = initNode("selfAttentionGuidanceNode");
                    if (v.toLowerCase() !== "true")
                        node.mode = 4;
                },
                "sag_scale"(v) {
                    const node = initNode("selfAttentionGuidanceNode");
                    setWidgetValue(node, "scale", +v);
                },
                "sag_blur_sigma"(v) {
                    const node = initNode("selfAttentionGuidanceNode");
                    setWidgetValue(node, "blur_sigma", +v);
                },
                "sag_blur_sigma"(v) {
                    const node = initNode("selfAttentionGuidanceNode");
                    setWidgetValue(node, "blur_sigma", +v);
                },

                emphasis(v) {
                    if (v && v.toLowerCase() === "no norm") {
                        setWidgetValue(positiveNode, "mean_normalization", false);
                        setWidgetValue(negativeNode, "mean_normalization", false);
                    } else {
                        opts["emphasis"] = v;
                    }
                },
                "multidiffusion_enabled"(v) {
                    const node = initNode("tiledDiffusionNode");
                    if (v.toLowerCase() !== "true")
                        node.mode = 4;
                },
                "multidiffusion_method"(v) {
                    const node = initNode("tiledDiffusionNode");
                    setWidgetValue(node, "method", "Mixture of Diffusers", true);
                },
                "multidiffusion_tile_width"(v) {
                    const node = initNode("tiledDiffusionNode");
                    setWidgetValue(node, "tile_width", +v);
                },
                "multidiffusion_tile_height"(v) {
                    const node = initNode("tiledDiffusionNode");
                    setWidgetValue(node, "tile_height", +v);
                },
                "multidiffusion_tile_overlap"(v) {
                    const node = initNode("tiledDiffusionNode");
                    setWidgetValue(node, "tile_overlap", +v);
                },
                "multidiffusion_tile_batch_size"(v) {
                    const node = initNode("tiledDiffusionNode");
                    setWidgetValue(node, "tile_batch_size", +v);
                },
                "ultimate sd upscale upscaler"(v) {
                    const node = initNode(v === "None" ? "UltimateSDUpscaleNoUpscaleNode" : "UltimateSDUpscaleNode");
                    for (const k of ["seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"]) {
                        let val = getWidget(samplerNode, k).value;
                        if (k === "denoise")
                            val = opts["denoising strength"] ?? val;
                        setWidgetValue(node, k, val);
                    }
                    positiveNode.connect(0, node, 2);
                    negativeNode.connect(0, node, 3);
                    vaeLoaderNode.connect(0, node, 4);
                    const USDUpscaleSaveImageNode = initNode("USDUpscaleSaveImageNode", "SaveImage");
                    node.connect(0, USDUpscaleSaveImageNode, 0);
                },
                "ultimate sd upscale tile_width"(v) {
                    const node = nodes.UltimateSDUpscaleNoUpscaleNode ?? nodes.UltimateSDUpscaleNode;
                    setWidgetValue(node, "tile_width", +v);
                },
                "ultimate sd upscale tile_height"(v) {
                    const node = nodes.UltimateSDUpscaleNoUpscaleNode ?? nodes.UltimateSDUpscaleNode;
                    setWidgetValue(node, "tile_height", +v);
                },
                "ultimate sd upscale mask_blur"(v) {
                    const node = nodes.UltimateSDUpscaleNoUpscaleNode ?? nodes.UltimateSDUpscaleNode;
                    setWidgetValue(node, "mask_blur", +v);
                },
                "ultimate sd upscale padding"(v) {
                    const node = nodes.UltimateSDUpscaleNoUpscaleNode ?? nodes.UltimateSDUpscaleNode;
                    setWidgetValue(node, "tile_padding", +v);
                },
            };

            if (hrSamplerNode) {
                setWidgetValue(hrSamplerNode, "steps", getWidget(samplerNode, "steps").value);
            }

            for (const opt in opts) {
                handlers[opt]?.(popOpt(opt));
            }

            if (hrSamplerNode) {
                const hr_sampler = popOpt("hires sampler");
                setWidgetValue(hrSamplerNode, "seed", +popOpt("global seed", true) || getWidget(samplerNode, "seed").value);
                setWidgetValue(hrSamplerNode, "cfg", +popOpt("hires cfg scale") || getWidget(samplerNode, "cfg").value);
                setWidgetValue(hrSamplerNode, "scheduler", getWidget(samplerNode, "scheduler").value);
                setWidgetValue(hrSamplerNode, "sampler_name", hr_sampler ? handlers.sampler(hr_sampler, hrSamplerNode) : getWidget(samplerNode, "sampler_name").value);
                setWidgetValue(hrSamplerNode, "denoise", +(popOpt("denoising strength") || "1"));
            } else {
                if (opts["denoising strength"]) {
                    setWidgetValue(samplerNode, "denoise", +popOpt("denoising strength"));
                }
            }

            let n = createLoraNodes(positiveNode, positive, { node: clipSkipNode, index: 0 }, { node: ckptNode, index: 0 });
            positive = n.text;
            n = createLoraNodes(negativeNode, negative, n.prevClip, n.prevModel);
            negative = n.text;

            setWidgetValue(positiveNode, "text", replaceEmbeddings(positive));
            setWidgetValue(negativeNode, "text", replaceEmbeddings(negative));

            let currentModelNode = settingsNode;
            let prevModelNode = currentModelNode;

            n.prevModel.node.connect(0, currentModelNode, 0);

            function connectModelNode(key, updatePrevious) {
                if (nodes[key]) {
                    currentModelNode.connect(0, nodes[key], 0);
                    currentModelNode = nodes[key];
                    if (updatePrevious)
                        prevModelNode = currentModelNode;
                }
            }

            connectModelNode("modelSamplingDiscreteNode", true);
            connectModelNode("rescaleCFGNode", true);
            connectModelNode("FreeU_V2Node", true);
            connectModelNode("selfAttentionGuidanceNode", true);
            connectModelNode("tiledDiffusionNode", true);
            connectModelNode("tomePatchModelNode");
            const USDUNode = nodes.UltimateSDUpscaleNoUpscaleNode ?? nodes.UltimateSDUpscaleNode;
            if (USDUNode)
                currentModelNode.connect(0, USDUNode, 0);

            currentModelNode.connect(0, samplerNode, 0);
            currentModelNode = prevModelNode;

            connectModelNode("tomePatchModelHrNode");

            if (hrSamplerNode)
                currentModelNode.connect(0, hrSamplerNode, 0);

            graph.arrange();

            for (const opt of ["model hash"]) {
                delete opts[opt];
            }

            console.warn("Unhandled parameters:", opts);
        }
    }
}
