import { app as _app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

let _EXIF;

const NON_FINITE_REGEX = /"(?:\\.|[^"\\])*"|(?<![\w.-])(-?Infinity|NaN)(?![\w.])/g;

function replaceNonFiniteTokens(str) {
    let warned = false;
    return str.replace(NON_FINITE_REGEX, (match, token) => {
        if (token) {
            if (!warned) {
                console.warn("JSON contained non-finite numeric tokens (NaN/Infinity); they were replaced with null.");
                warned = true;
            }
            return "null";
        }
        return match;
    });
}

export function parseJsonWithNonFinite(data) {
    if (typeof data !== "string") return data;
    try {
        return JSON.parse(data);
    } catch {
        try {
            return JSON.parse(replaceNonFiniteTokens(data));
        } catch {
            return null;
        }
    }
}

const A1111_NEGATIVE_PROMPT_PREFIX = "\nNegative prompt:";

function normalizeA1111Parameters(parameters) {
    const stepsIndex = parameters.lastIndexOf("\nSteps:");
    if (
        stepsIndex === -1 ||
        parameters.lastIndexOf(A1111_NEGATIVE_PROMPT_PREFIX, stepsIndex) > -1
    ) {
        return parameters;
    }
    return `${parameters.slice(0, stepsIndex)}${A1111_NEGATIVE_PROMPT_PREFIX}${parameters.slice(stepsIndex)}`;
}

export async function getPngMetadata(file) {
    if (window.comfyAPI?.pnginfo?.getPngMetadata) {
        return await window.comfyAPI.pnginfo.getPngMetadata(file);
    }
    if (window.comfyAPI?.png?.getFromPngFile) {
        return await window.comfyAPI.png.getFromPngFile(file);
    }
    try {
        const mod = await import("../../scripts/pnginfo.js");
        if (mod?.getPngMetadata) {
            return await mod.getPngMetadata(file);
        }
    } catch {}
    return {};
}

_app.registerExtension({
    name: "Comfy.smZ.WorkflowImage",

    async setup(app) {
        const script = document.createElement("script");
        const scriptLocation = new URL("exif.js", import.meta.url);
        script.src = scriptLocation.href;

        function removeExt(f) {
            if (!f) return f;
            const p = f.lastIndexOf(".");
            if (p === -1) return f;
            return f.substring(0, p);
        }

        function getFileNameFromUrl(url) {
            const urlParts = new URL(url);
            const pathParts = urlParts.pathname.split("/");
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
                    return new File([blob], removeExt(fileName) + "." + (mimeType.split("/")[1] || "png"), { type: mimeType });
                });
        }

        const handleFile = app.handleFile;
        const _handleFile = async function (file) {
            if (!file) return;
            if (!file.name) {
                file.name = `${Date.now()}.${file.type ? file.type.split("/")[1] || "png" : "png"}`;
            }
            if (file.name.startsWith("http")) {
                return fetchToFile(file.name)
                    .then(_file => _handleFile(_file))
                    .catch(error => {
                        console.error("[smZ.WorkflowImage] Error fetching URL:", error);
                    });
            }

            const fileName = removeExt(file.name);

            async function processWorkflowInfo(info) {
                if (!info) return null;
                const workflow = info.workflow || info.Workflow;
                const prompt = info.prompt || info.Prompt;
                const parameters = info.parameters || info.Parameters;

                if (workflow) {
                    const parsedWorkflow = parseJsonWithNonFinite(workflow);
                    if (parsedWorkflow) {
                        return await app.loadGraphData(parsedWorkflow, true, true, fileName);
                    }
                } else if (prompt) {
                    const parsedPrompt = parseJsonWithNonFinite(prompt);
                    if (parsedPrompt) {
                        return app.loadApiJson(parsedPrompt, fileName);
                    }
                } else if (parameters) {
                    return await importA1111(app.graph, parameters);
                }
                return null;
            }

            let r = null;

            if (file.type === "image/jpeg" || file.type === "image/jpg" || file.name.endsWith(".jpg") || file.name.endsWith(".jpeg")) {
                const info = await getJpegMetadataA111(app, file);
                if (info) {
                    const parsed = parseJsonWithNonFinite(info);
                    if (parsed && typeof parsed === "object") {
                        const extra = parsed.extra;
                        const extraMetadata = parsed.extraMetadata;
                        delete parsed["extra"];
                        delete parsed["extraMetadata"];
                        r = app.loadApiJson(parsed, fileName);
                        parsed["extra"] = extra;
                        parsed["extraMetadata"] = extraMetadata;
                        return r;
                    } else if (typeof info === "string") {
                        r = await importA1111(app.graph, info);
                        if (r === "imported" || r === "imported-without-embeddings") {
                            return r;
                        }
                    }
                } else {
                    const pngInfo = await getPngMetadata(file);
                    r = await processWorkflowInfo(pngInfo);
                    if (r) return r;
                }
                return handleFile.apply(app, arguments);
            }

            if (file.type === "image/png" || file.name.endsWith(".png")) {
                const pngInfo = await getPngMetadata(file);
                r = await processWorkflowInfo(pngInfo);
                if (r) return r;
                return handleFile.apply(app, arguments);
            }

            if (file.type === "image/webp" || file.name.endsWith(".webp")) {
                const webpInfo = await getWebpMetadata(file);
                r = await processWorkflowInfo(webpInfo);
                if (r) return r;
                return handleFile.apply(app, arguments);
            }

            if (window.comfyAPI?.parser?.getWorkflowDataFromFile) {
                try {
                    const parsedData = await window.comfyAPI.parser.getWorkflowDataFromFile(file);
                    if (parsedData) {
                        r = await processWorkflowInfo(parsedData);
                        if (r) return r;
                    }
                } catch (e) {
                    console.error("[smZ.WorkflowImage] Error parsing file metadata:", e);
                }
            }

            return handleFile.apply(app, arguments);
        };

        script.onload = function () {
            try {
                _EXIF = {};
                _EXIF.readExif = EXIF.readFromBinaryFile;
            } catch (e) {
                console.error("[smZ.WorkflowImage] Failed to initialize EXIF reader:", e);
            }
            app.handleFile = _handleFile;
        };

        script.onerror = function () {
            console.warn("[smZ.WorkflowImage] exif.js failed to load; using native fallback.");
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
                    if (_EXIF?.readExif) {
                        const obj = _EXIF.readExif(event.target.result);
                        if (obj?.UserComment) {
                            const val = obj.UserComment.slice(9).filter(it => it !== 0 && it !== 0x0B);
                            value = new TextDecoder("utf-16").decode(new Uint16Array(val));
                        }
                    }

                    if (!value && event.target.result) {
                        const uint8 = new Uint8Array(event.target.result);
                        for (let i = 0; i < uint8.length - 4; i++) {
                            if (uint8[i] === 0xFF && uint8[i + 1] === 0xE1) {
                                const exifStart = i + 4;
                                if (String.fromCharCode(...uint8.slice(exifStart, exifStart + 6)) === "Exif\0\0") {
                                    const parsedExif = parseExifData(uint8.subarray(exifStart + 6));
                                    if (parsedExif?.[0x9286]) {
                                        value = parsedExif[0x9286];
                                    }
                                }
                                break;
                            }
                        }
                    }

                    resolve(value || false);
                } catch (error) {
                    console.error("[smZ.WorkflowImage]", error);
                    resolve(false);
                }
            };
            reader.onerror = () => resolve(false);
            reader.onabort = () => resolve(false);
            reader.readAsArrayBuffer(file);
        } catch (error) {
            resolve(false);
        }
    });
}

export function parseExifData(exifData) {
    const isLittleEndian = String.fromCharCode(...exifData.slice(0, 2)) === "II";

    function readInt(offset, littleEndian, length) {
        if (offset + length > exifData.length) return 0;
        const view = new DataView(exifData.buffer, exifData.byteOffset + offset, length);
        if (length === 2) return view.getUint16(0, littleEndian);
        if (length === 4) return view.getUint32(0, littleEndian);
        return 0;
    }

    const tiffHeaderOffset = 0;
    const ifd0Offset = readInt(tiffHeaderOffset + 4, isLittleEndian, 4);
    if (!ifd0Offset || ifd0Offset >= exifData.length) return {};

    function parseIFD(offset) {
        if (!offset || offset + 2 > exifData.length) return {};
        const numEntries = readInt(offset, isLittleEndian, 2);
        const result = {};

        for (let i = 0; i < numEntries; i++) {
            const entryOffset = offset + 2 + i * 12;
            if (entryOffset + 12 > exifData.length) break;
            const tag = readInt(entryOffset, isLittleEndian, 2);
            const type = readInt(entryOffset + 2, isLittleEndian, 2);
            const count = readInt(entryOffset + 4, isLittleEndian, 4);
            const valueOffset = readInt(entryOffset + 8, isLittleEndian, 4);

            let value = null;

            if (type === 2) {
                if (valueOffset < exifData.length) {
                    const len = Math.max(0, Math.min(count - 1, exifData.length - valueOffset));
                    const strBytes = exifData.slice(valueOffset, valueOffset + len);
                    value = new TextDecoder("utf-8").decode(strBytes);
                }
            } else if (type === 7 && tag === 0x9286) {
                if (valueOffset < exifData.length) {
                    const data = exifData.slice(valueOffset, Math.min(valueOffset + count, exifData.length));
                    const encodingHeader = new TextDecoder("ascii").decode(data.slice(0, 8));
                    if (encodingHeader.startsWith("UNICODE")) {
                        const utf16Data = data.slice(8);
                        try {
                            value = new TextDecoder(isLittleEndian ? "utf-16le" : "utf-16be").decode(utf16Data);
                        } catch {
                            value = new TextDecoder("utf-16").decode(utf16Data);
                        }
                    } else if (encodingHeader.startsWith("ASCII\0\0\0")) {
                        value = new TextDecoder("ascii").decode(data.slice(8));
                    } else {
                        value = new TextDecoder("utf-8").decode(data);
                    }
                }
            } else if (type === 4 && count === 1) {
                const subIFD = parseIFD(valueOffset);
                result[tag] = subIFD;
                continue;
            }

            result[tag] = value;
        }

        return result;
    }

    return parseIFD(ifd0Offset);
}

export function getWebpMetadata(file) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (event) => {
            const webp = new Uint8Array(event.target.result);
            const dataView = new DataView(webp.buffer);
            if (dataView.getUint32(0) !== 0x52494646 || dataView.getUint32(8) !== 0x57454250) {
                console.error("Not a valid WEBP file");
                resolve({});
                return;
            }
            let offset = 12;
            const txt_chunks = {};

            try {
                while (offset < webp.length) {
                    const chunk_length = dataView.getUint32(offset + 4, true);
                    const chunk_type = String.fromCharCode(...webp.slice(offset, offset + 4));

                    if (chunk_type === "EXIF") {
                        let exifOffset = offset + 8;
                        let exifLength = chunk_length;
                        if (String.fromCharCode(...webp.slice(exifOffset, exifOffset + 6)) === "Exif\0\0") {
                            exifOffset += 6;
                            exifLength -= 6;
                        }
                        const data = parseExifData(webp.slice(exifOffset, exifOffset + exifLength));
                        for (const key in data) {
                            const value = data[key];
                            if (typeof value === "string") {
                                const idx = value.indexOf(":");
                                if (idx > -1) {
                                    txt_chunks[value.slice(0, idx)] = value.slice(idx + 1);
                                } else {
                                    txt_chunks[key] = value;
                                }
                            } else if (typeof value === "object" && value !== null) {
                                for (const k in value) {
                                    if (typeof value[k] === "string") {
                                        txt_chunks["parameters"] = value[k];
                                    }
                                }
                            }
                        }
                        break;
                    }

                    offset += 8 + chunk_length + (chunk_length % 2);
                }
            } catch (e) {
                console.error("[smZ.WorkflowImage] Error parsing WebP chunks:", e);
                resolve({});
                return;
            }
            resolve(txt_chunks);
        };
        reader.onerror = () => resolve({});
        reader.onabort = () => resolve({});
        reader.readAsArrayBuffer(file);
    });
}

function groupObjectPropertiesInPlace(originalObject, groupKeys) {
    const groupedObject = {};
    groupKeys.forEach(groupKey => {
        const group = {};
        for (const key in originalObject) {
            for (const delimiter of ["_", " "]) {
                if (key.startsWith(groupKey + delimiter)) {
                    const newKey = key.replace(groupKey + delimiter, "");
                    group[newKey] = originalObject[key];
                    delete originalObject[key];
                }
            }
        }
        if (Object.keys(group).length > 0) {
            groupedObject[groupKey] = group;
        }
    });
    Object.assign(originalObject, groupedObject);
}

export async function importA1111(graph, parameters, beforeGraphClear) {
    if (typeof parameters !== "string") return "not-a1111";
    parameters = parameters.replace(/[\u0000\u000B]/gm, "");
    const normalizedParameters = normalizeA1111Parameters(parameters);
    const p = normalizedParameters.lastIndexOf("\nSteps:");
    if (p === -1) return "not-a1111";

    const p2 = normalizedParameters.lastIndexOf(A1111_NEGATIVE_PROMPT_PREFIX, p);
    if (p2 === -1) return "not-a1111";

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
            } else if (current.includes('"{')) {
                endDelimiter = '}"';
            } else if (current.includes("{")) {
                endDelimiter = "}";
            } else if (current.includes(`"`)) {
                endDelimiter = `"`;
            }

            if (endDelimiter) {
                const escapedEnd = [...endDelimiter].map(char => `\\${char}`).join("");
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
            if (key && value !== undefined) {
                acc[key.trim().toLowerCase()] = value.trim();
            }
            return acc;
        }, {});
    }

    const inputList = normalizedParameters.substring(p + 1).split(", ");
    const opts = parseParameters(inputList);
    if (!Object.keys(opts).length) return "not-a1111";

    let positive = normalizedParameters.substring(0, p2).trim();
    let negative = normalizedParameters.substring(p2 + A1111_NEGATIVE_PROMPT_PREFIX.length, p).trim();

    try {
        for (const o of ["civitai resources", "civitai metadata"]) {
            if (opts[o]) {
                opts[o] = parseJsonWithNonFinite(opts[o]);
            }
            if (Array.isArray(opts[o])) {
                for (const it of opts[o]) {
                    if (it.modelName) {
                        if (it.type === "checkpoint")
                            opts.model = it.modelName + `_${it.modelVersionName ?? ""}`;
                        else if (it.type === "lora" || it.type.includes("lyco"))
                            positive += `<${it.type}:${it.modelName}_${it.modelVersionName ?? ""}:${it.weight}>`;
                    }
                }
            }
        }
    } catch (e) {
        console.error("[smZ.WorkflowImage] Error parsing CivitAI metadata:", e);
    }

    const ckptNode = LiteGraph.createNode("CheckpointLoaderSimple");
    const clipSkipNode = LiteGraph.createNode("CLIPSetLastLayer");
    const positiveNode = LiteGraph.createNode("smZ CLIPTextEncode") || LiteGraph.createNode("CLIPTextEncode");
    const negativeNode = LiteGraph.createNode("smZ CLIPTextEncode") || LiteGraph.createNode("CLIPTextEncode");
    const settingsNode = LiteGraph.createNode("smZ Settings");
    const samplerNode = LiteGraph.createNode("KSampler");
    const imageNode = LiteGraph.createNode("EmptyLatentImage");
    const vaeNode = LiteGraph.createNode("VAEDecode");
    const vaeLoaderNode = LiteGraph.createNode("VAELoader");
    const saveNode = LiteGraph.createNode("SaveImage");

    if (
        !ckptNode ||
        !clipSkipNode ||
        !positiveNode ||
        !negativeNode ||
        !samplerNode ||
        !imageNode ||
        !vaeNode ||
        !saveNode
    ) {
        console.error("Failed to create required nodes for A1111 import");
        return "core-nodes-unavailable";
    }

    const hires = { samplerNode: null, steps: null };
    const nodes = {};

    function getWidget(node, name) {
        return node?.widgets?.find((w) => w.name === name || w.name?.endsWith(`.${name}`));
    }

    function setWidgetValue(node, name, value, isOptionPrefix) {
        if (!node) return;
        const w = getWidget(node, name);
        if (!w) return;
        if (isOptionPrefix) {
            const values = w.options?.values;
            const o = values?.find((val) => String(val).startsWith(String(value)));
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

    setWidgetValue(positiveNode, "parser", "A1111");
    setWidgetValue(negativeNode, "parser", "A1111");
    if (settingsNode) {
        setWidgetValue(settingsNode, "RNG", "gpu");
        setWidgetValue(settingsNode, "sgm_noise_multiplier", false);
    }

    const ceil64 = (v) => Math.ceil(v / 64) * 64;

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

    function initPropertiesNode(v, groupKey, nodeKey, nodeName) {
        if (typeof v !== "object" && Object.keys(v || {}).length !== 0) {
            opts[groupKey] = v;
            return;
        }
        const node = initNode(nodeKey, nodeName);
        if (!node) return;
        if (v.enabled) {
            if (String(v.enabled).toLowerCase() !== "true")
                node.mode = 4;
            delete v["enabled"];
        }
        for (const [key, value] of Object.entries(v)) {
            let vv = value;
            if (value === "True") vv = true;
            else if (value === "False") vv = false;
            setWidgetValue(node, key, vv);
        }
        return node;
    }

    const similarityThreshold = 0.4;

    function stringSimilarity(str1, str2, gramSize = 2) {
        function getNGrams(s, len) {
            s = " ".repeat(len - 1) + s.toLowerCase() + " ".repeat(len - 1);
            const v = new Array(s.length - len + 1);
            for (let i = 0; i < v.length; i++) {
                v[i] = s.slice(i, i + len);
            }
            return v;
        }

        if (!str1?.length || !str2?.length) return 0.0;

        const s1 = str1.length < str2.length ? str1 : str2;
        const s2 = str1.length < str2.length ? str2 : str1;

        const pairs1 = getNGrams(s1, gramSize);
        const pairs2 = getNGrams(s2, gramSize);
        const set = new Set(pairs1);

        let hits = 0;
        for (const item of pairs2) {
            if (set.delete(item)) hits++;
        }
        return hits / pairs2.length;
    }

    function basename(s) {
        if (!s) return "";
        let i = s.lastIndexOf(".");
        i = i !== -1 ? i : s.length;
        let z = s.replaceAll("\\", "/").lastIndexOf("/");
        z = z !== -1 ? z + 1 : 0;
        return s.substring(z, i);
    }

    function fuzzySearch(node, widget_name, v) {
        const vbasename = basename(v);
        const w = getWidget(node, widget_name);
        const values = w?.options?.values || [];
        const o = values.map((n) => ({ name: n, similarity: stringSimilarity(vbasename, basename(n)) }));
        o.sort((a, b) => b.similarity - a.similarity);
        return o?.[0]?.name ? (o[0].similarity > similarityThreshold ? o[0].name : v) : v;
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
            if (!loraNode) continue;
            graph.add(loraNode);

            const w = getWidget(loraNode, "lora_name");
            const values = w?.options?.values || [];
            const o = values.map((wVal) => ({ name: wVal, similarity: stringSimilarity(l.name, basename(wVal)) }));
            o.sort((a, b) => b.similarity - a.similarity);
            setWidgetValue(
                loraNode,
                "lora_name",
                o?.[0]?.name ? (o[0].similarity > similarityThreshold ? o[0].name : l.name) : l.name,
                true
            );

            setWidgetValue(loraNode, "strength_model", l.weight);
            setWidgetValue(loraNode, "strength_clip", l.weight);
            prevModel.node.connect(prevModel.index, loraNode, 0);
            prevClip.node.connect(prevClip.index, loraNode, 1);
            prevModel = { node: loraNode, index: 0 };
            prevClip = { node: loraNode, index: 1 };
        }

        prevClip.node.connect(1, clipNode, 0);
        prevModel.node.connect(0, samplerNode, 0);
        if (hires.samplerNode) {
            prevModel.node.connect(0, hires.samplerNode, 0);
        }

        return { text, prevModel, prevClip };
    }

    function replaceEmbeddings(text) {
        if (!embeddings?.length) return text;
        return text.replaceAll(
            new RegExp(
                "\\b(" +
                embeddings.map((e) => e.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("\\b|\\b") +
                ")\\b",
                "ig"
            ),
            "embedding:$1"
        );
    }

    function popOpt(name, get) {
        const v = opts[name];
        if (!get) delete opts[name];
        return v;
    }

    const comfyApi = api || window.comfyAPI?.api || _app?.api;
    const { embeddings, embeddingsLoaded } = await (comfyApi?.getEmbeddings
        ? comfyApi.getEmbeddings().then((embs) => ({ embeddings: embs || [], embeddingsLoaded: true }))
        : Promise.resolve({ embeddings: [], embeddingsLoaded: false })
    ).catch((error) => {
        console.error("Failed to load embeddings for A1111 import:", error);
        return { embeddings: [], embeddingsLoaded: false };
    });

    await beforeGraphClear?.();

    const buildGraph = () => {
        graph.clear();
        graph.add(ckptNode);
        graph.add(clipSkipNode);
        graph.add(positiveNode);
        graph.add(negativeNode);
        if (settingsNode) graph.add(settingsNode);
        graph.add(samplerNode);
        graph.add(imageNode);
        graph.add(vaeNode);
        if (vaeLoaderNode) graph.add(vaeLoaderNode);
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

        if (vaeLoaderNode) {
            vaeLoaderNode.connect(0, vaeNode, 1);
        } else {
            ckptNode.connect(2, vaeNode, 1);
        }

        const handlers = {
            model(v) {
                setWidgetValue(ckptNode, "ckpt_name", fuzzySearch(ckptNode, "ckpt_name", v), true);
            },
            "cfg scale"(v) {
                setWidgetValue(samplerNode, "cfg", +v);
            },
            "clip skip"(v) {
                setWidgetValue(clipSkipNode, "stop_at_clip_layer", -Number(v));
                clipSkipNode.mode = 0;
            },
            "schedule type"(v) {
                let name = v.toLowerCase().replaceAll(" ", "_");
                const scheduler_map = {
                    Automatic: "normal",
                    "Align Your Steps": "ays",
                    DDIM: "ddim_uniform",
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
                const sn = _sn || samplerNode;
                let name = v
                    .toLowerCase()
                    .replace("cfg++", "cfg_pp")
                    .replace("++", "pp")
                    .replace("dpm2", "dpm_2")
                    .replaceAll(" ", "_")
                    .replace("_heun", "")
                    .replace("_sde", "_sde_gpu");

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
                setWidgetValue(sn, "sampler_name", name);
                return name;
            },
            size(v) {
                const wxh = v.split("x");
                const w = ceil64(+wxh[0]);
                const h = ceil64(+wxh[1]);
                const hrUp = popOpt("hires upscale");
                const hrSz = popOpt("hires resize");
                hires.steps = popOpt("hires steps");
                let hrMethod = popOpt("hires upscaler");

                setWidgetValue(imageNode, "width", w);
                setWidgetValue(imageNode, "height", h);

                if (hrUp || hrSz) {
                    let uw, uh;
                    if (hrUp) {
                        uw = w * Number(hrUp);
                        uh = h * Number(hrUp);
                    } else if (hrSz) {
                        const s = hrSz.split("x");
                        uw = +s[0];
                        uh = +s[1];
                    } else {
                        return;
                    }

                    let upscaleNode = null;
                    let latentNode = null;

                    if (hrMethod?.startsWith("Latent")) {
                        latentNode = upscaleNode = LiteGraph.createNode("LatentUpscale");
                        if (!upscaleNode) return;
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
                        const decode = LiteGraph.createNode("VAEDecodeTiled") || LiteGraph.createNode("VAEDecode");
                        if (!decode) return;
                        graph.add(decode);
                        samplerNode.connect(0, decode, 0);
                        if (vaeLoaderNode) vaeLoaderNode.connect(0, decode, 1);
                        else ckptNode.connect(2, decode, 1);

                        const upscaleLoaderNode = LiteGraph.createNode("UpscaleModelLoader");
                        if (!upscaleLoaderNode) return;
                        graph.add(upscaleLoaderNode);
                        setWidgetValue(upscaleLoaderNode, "model_name", hrMethod ?? "", true);

                        const modelUpscaleNode = LiteGraph.createNode("ImageUpscaleWithModel");
                        if (!modelUpscaleNode) return;
                        graph.add(modelUpscaleNode);
                        decode.connect(0, modelUpscaleNode, 1);
                        upscaleLoaderNode.connect(0, modelUpscaleNode, 0);

                        upscaleNode = LiteGraph.createNode("ImageScale");
                        if (!upscaleNode) return;
                        graph.add(upscaleNode);
                        modelUpscaleNode.connect(0, upscaleNode, 0);

                        const vaeEncodeNode = LiteGraph.createNode("VAEEncodeTiled") || LiteGraph.createNode("VAEEncode");
                        if (!vaeEncodeNode) return;
                        latentNode = vaeEncodeNode;
                        graph.add(vaeEncodeNode);
                        upscaleNode.connect(0, vaeEncodeNode, 0);
                        if (vaeLoaderNode) vaeLoaderNode.connect(0, vaeEncodeNode, 1);
                        else ckptNode.connect(2, vaeEncodeNode, 1);

                        const previewNode = LiteGraph.createNode("PreviewImage");
                        if (previewNode) {
                            graph.add(previewNode);
                            decode.connect(0, previewNode, 0);
                        }
                    }

                    setWidgetValue(upscaleNode, "width", ceil64(uw));
                    setWidgetValue(upscaleNode, "height", ceil64(uh));

                    const hrSampler = LiteGraph.createNode("KSampler");
                    if (!hrSampler || !latentNode) return;
                    hires.samplerNode = hrSampler;
                    graph.add(hrSampler);
                    ckptNode.connect(0, hrSampler, 0);
                    positiveNode.connect(0, hrSampler, 1);
                    negativeNode.connect(0, hrSampler, 2);
                    latentNode.connect(0, hrSampler, 3);
                    hrSampler.connect(0, vaeNode, 0);
                }
            },
            steps(v) {
                setWidgetValue(samplerNode, "steps", +v);
            },
            seed(v) {
                setWidgetValue(samplerNode, "seed", +(popOpt("global seed", true) || v));
            },
            vae(v) {
                if (vaeLoaderNode) {
                    setWidgetValue(vaeLoaderNode, "vae_name", fuzzySearch(vaeLoaderNode, "vae_name", v), true);
                }
            },
            "module 1"(v) {
                this.vae(v);
            },
            "module 2"(v) {
                const w = getWidget(ckptNode, "ckpt_name");
                if (w?.value && String(w.value).toLowerCase().includes("flux") && opts["module 3"]) {
                    const node = initNode("DualCLIPLoaderNode", "DualCLIPLoader");
                    if (node) {
                        setWidgetValue(node, "clip_name1", fuzzySearch(node, "clip_name1", v), true);
                        setWidgetValue(node, "clip_name2", fuzzySearch(node, "clip_name2", opts["module 3"]), true);
                        setWidgetValue(node, "type", "flux", true);
                        node.connect(0, clipSkipNode, 0);
                    }
                } else {
                    opts["module 2"] = v;
                }
            },
            ["hires steps"](v) {
                hires.steps = v;
            },
            "noise schedule"(v) {
                const node = initNode("modelSamplingDiscreteNode", "ModelSamplingDiscrete");
                if (node && v === "Zero Terminal SNR") {
                    setWidgetValue(node, "zsnr", true);
                    setWidgetValue(node, "sampling", "v_prediction");
                }
            },
            discrete(v) {
                const node = initPropertiesNode(v, "discrete", "modelSamplingDiscrete2Node", "ModelSamplingDiscrete");
                if (opts.advanced_sampling_mode === "Discrete")
                    delete opts.advanced_sampling_mode;
                if (opts.advanced_sampling_enabled) {
                    if (String(opts.advanced_sampling_enabled).toLowerCase() !== "true" && node)
                        node.mode = 4;
                    delete opts.advanced_sampling_enabled;
                }
            },
            rescale_cfg_enabled(v) {
                const node = initNode("rescaleCFGNode", "RescaleCFG");
                if (node && String(v).toLowerCase() !== "true")
                    node.mode = 4;
            },
            rescale_cfg_multiplier(v) {
                const node = initNode("rescaleCFGNode", "RescaleCFG");
                if (node) setWidgetValue(node, "multiplier", +v);
            },
            eta(v) {
                if (settingsNode) setWidgetValue(settingsNode, "eta", +v);
            },
            s_churn(v) {
                if (settingsNode) setWidgetValue(settingsNode, "s_churn", +v);
            },
            s_tmin(v) {
                if (settingsNode) setWidgetValue(settingsNode, "s_tmin", +v);
            },
            s_tmax(v) {
                if (settingsNode) setWidgetValue(settingsNode, "s_tmax", +v);
            },
            s_noise(v) {
                if (settingsNode) setWidgetValue(settingsNode, "s_noise", +v);
            },
            ensd(v) {
                if (settingsNode) setWidgetValue(settingsNode, "ENSD", +v);
            },
            rng(v) {
                if (settingsNode) setWidgetValue(settingsNode, "RNG", v, true);
            },
            ["sgm noise multiplier"](v) {
                if (settingsNode) setWidgetValue(settingsNode, "sgm_noise_multiplier", String(v).toLowerCase() === "true");
            },
            ["pad conds"](v) {
                if (settingsNode) setWidgetValue(settingsNode, "pad_cond_uncond", String(v).toLowerCase() === "true");
            },
            ["skip early cfg"](v) {
                if (settingsNode) setWidgetValue(settingsNode, "skip_early_cond", +v);
            },
            ngms(v) {
                if (settingsNode) setWidgetValue(settingsNode, "NGMS", +v);
            },
            emphasis(v) {
                if (v && String(v).toLowerCase() === "no norm") {
                    setWidgetValue(positiveNode, "mean_normalization", false);
                    setWidgetValue(negativeNode, "mean_normalization", false);
                } else {
                    opts["emphasis"] = v;
                }
            },
            "sd upscale upscaler"(v) {
                const node = initNode("SDUpscaleNode", "UpscaleModelLoader");
                if (node) setWidgetValue(node, "model_name", v, true);
            },
            ["token merging ratio"](v) {
                const node = initNode("tomePatchModelNode", "TomePatchModel");
                if (node) setWidgetValue(node, "ratio", +v);
            },
            ["token merging ratio hr"](v) {
                const node = initNode("tomePatchModelHrNode", "TomePatchModel");
                if (node) setWidgetValue(node, "ratio", +v);
            },
            freeu(v) {
                initPropertiesNode(v, "freeu", "FreeU_V2Node");
            },
            sag(v) {
                initPropertiesNode(v, "sag", "selfAttentionGuidanceNode");
            },
            pag(v) {
                initPropertiesNode(v, "pag", "perturbedAttentionNode");
            },
            multidiffusion(v) {
                initPropertiesNode(v, "multidiffusion", "tiledDiffusionNode");
            },
            "tiled diffusion"(v) {
                const vv = v.replaceAll("'", '"').replaceAll("True", "true").replaceAll("False", "false").slice(1, v.length - 1);
                let obj;
                try {
                    obj = JSON.parse(vv);
                } catch (e) {}
                if (!obj) {
                    opts["tiled diffusion"] = v;
                    return;
                }
                obj["scale factor"] = popOpt("tiled diffusion scale factor");
                obj["upscaler"] = popOpt("tiled diffusion upscaler");
                const node = initNode("tiledDiffusion2Node", "TiledDiffusion");
                const cf = 8;
                setWidgetValue(node, "method", obj["Method"], true);
                setWidgetValue(node, "tile_width", +obj["Latent tile width"] * cf);
                setWidgetValue(node, "tile_height", +obj["Latent tile height"] * cf);
                setWidgetValue(node, "tile_overlap", +obj["Overlap"] * cf);
                setWidgetValue(node, "tile_batch_size", +obj["Tile batch size"]);
                if (obj["upscaler"] && obj["scale factor"]) {
                    const upscalerNode = initNode("TDUpscaleModelLoaderNode", "UpscaleModelLoader");
                    setWidgetValue(upscalerNode, "model_name", fuzzySearch(upscalerNode, "model_name", obj["upscaler"]), true);
                    const w = getWidget(imageNode, "width")?.value || 512;
                    const h = getWidget(imageNode, "height")?.value || 512;
                    setWidgetValue(imageNode, "width", w / +obj["scale factor"]);
                    setWidgetValue(imageNode, "height", h / +obj["scale factor"]);
                    const ImageUpscaleWithModelTDNode = initNode("ImageUpscaleWithModelTDNode", "ImageUpscaleWithModel");
                    const LoadImageTDNode = initNode("LoadImageTDNode", "LoadImage");
                    const ImageScaleTDNode = initNode("ImageScaleTDNode", "ImageScale");
                    const VAEEncodeTDNode = initNode("VAEEncodeTDNode", "VAEEncode");
                    upscalerNode.connect(0, ImageUpscaleWithModelTDNode, 0);
                    LoadImageTDNode.connect(0, ImageUpscaleWithModelTDNode, 1);
                    ImageUpscaleWithModelTDNode.connect(0, ImageScaleTDNode, 0);
                    setWidgetValue(ImageScaleTDNode, "width", w);
                    setWidgetValue(ImageScaleTDNode, "height", h);
                    ImageScaleTDNode.connect(0, VAEEncodeTDNode, 0);
                    if (vaeLoaderNode) vaeLoaderNode.connect(0, VAEEncodeTDNode, 1);
                    else ckptNode.connect(2, VAEEncodeTDNode, 1);
                    VAEEncodeTDNode.connect(0, samplerNode, 3);
                }
            },
            "ultimate sd upscale"(v) {
                const nodeKey = v.upscaler === "None" ? "UltimateSDUpscaleNoUpscaleNode" : "UltimateSDUpscaleNode";
                const node = initPropertiesNode(v, "ultimate sd upscale", nodeKey);
                if (v.upscaler && node) delete v["upscaler"];
                for (const k of ["seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"]) {
                    let val = getWidget(samplerNode, k)?.value;
                    if (k === "denoise") val = opts["denoising strength"] ?? val;
                    setWidgetValue(node, k, val);
                }
                positiveNode.connect(0, node, 2);
                negativeNode.connect(0, node, 3);
                if (vaeLoaderNode) vaeLoaderNode.connect(0, node, 4);
                else ckptNode.connect(2, node, 4);
                const _saveImageNode = initNode("USDUpscaleSaveImageNode", "SaveImage");
                node.connect(0, _saveImageNode, 0);
            },
            latent_modifier(v) {
                const node = initPropertiesNode(v, "latent_modifier", "latentModifierNode", "Latent Diffusion Mega Modifier");
                const seed = getWidget(samplerNode, "seed")?.value;
                setWidgetValue(node, "seed", seed);
            }
        };

        const groupKeys = [
            "freeu",
            "multidiffusion",
            "ultimate sd upscale",
            "latent_modifier",
            "sag",
            "pag",
            "discrete"
        ];
        groupObjectPropertiesInPlace(opts, groupKeys);

        for (const opt in opts) {
            handlers[opt]?.(popOpt(opt));
        }

        const hrSamplerNode = hires.samplerNode;
        if (hrSamplerNode) {
            const hr_sampler = popOpt("hires sampler");
            const hrSteps = hires.steps || popOpt("hires steps");
            setWidgetValue(
                hrSamplerNode,
                "steps",
                hrSteps ? +hrSteps : getWidget(samplerNode, "steps")?.value
            );
            setWidgetValue(
                hrSamplerNode,
                "seed",
                +popOpt("global seed", true) || getWidget(samplerNode, "seed")?.value
            );
            setWidgetValue(
                hrSamplerNode,
                "cfg",
                +popOpt("hires cfg scale") || getWidget(samplerNode, "cfg")?.value
            );
            setWidgetValue(
                hrSamplerNode,
                "scheduler",
                getWidget(samplerNode, "scheduler")?.value
            );
            setWidgetValue(
                hrSamplerNode,
                "sampler_name",
                hr_sampler ? handlers.sampler(hr_sampler, hrSamplerNode) : getWidget(samplerNode, "sampler_name")?.value
            );
            setWidgetValue(
                hrSamplerNode,
                "denoise",
                +(popOpt("denoising strength") || "1")
            );
        } else if (opts["denoising strength"]) {
            setWidgetValue(samplerNode, "denoise", +popOpt("denoising strength"));
        }

        let n = createLoraNodes(positiveNode, positive, { node: clipSkipNode, index: 0 }, { node: ckptNode, index: 0 });
        positive = n.text;
        n = createLoraNodes(negativeNode, negative, n.prevClip, n.prevModel);
        negative = n.text;

        setWidgetValue(positiveNode, "text", replaceEmbeddings(positive));
        setWidgetValue(negativeNode, "text", replaceEmbeddings(negative));
        setWidgetValue(positiveNode, "mode.text", replaceEmbeddings(positive));
        setWidgetValue(negativeNode, "mode.text", replaceEmbeddings(negative));

        let currentModelNode = settingsNode || ckptNode;
        let prevModelNode = currentModelNode;

        if (settingsNode) {
            n.prevModel.node.connect(0, currentModelNode, 0);
        }

        function connectModelNode(key, updatePrevious) {
            if (nodes[key]) {
                currentModelNode.connect(0, nodes[key], 0);
                currentModelNode = nodes[key];
                if (updatePrevious) prevModelNode = currentModelNode;
            }
        }

        connectModelNode("latentModifierNode", true);
        connectModelNode("modelSamplingDiscreteNode", true);
        connectModelNode("modelSamplingDiscrete2Node", true);
        connectModelNode("rescaleCFGNode", true);
        connectModelNode("FreeU_V2Node", true);
        connectModelNode("selfAttentionGuidanceNode", true);
        connectModelNode("perturbedAttentionNode", true);
        connectModelNode("tiledDiffusionNode", true);
        connectModelNode("tiledDiffusion2Node", true);
        connectModelNode("tomePatchModelNode");

        const USDUNode = nodes.UltimateSDUpscaleNoUpscaleNode ?? nodes.UltimateSDUpscaleNode;
        if (USDUNode) currentModelNode.connect(0, USDUNode, 1);

        currentModelNode.connect(0, samplerNode, 0);
        currentModelNode = prevModelNode;

        connectModelNode("tomePatchModelHrNode");

        if (hrSamplerNode) {
            currentModelNode.connect(0, hrSamplerNode, 0);
        }

        // 1. Force synchronous size and visibility updates BEFORE arranging
        if (settingsNode?.applyWidgetLogic) {
            settingsNode.applyWidgetLogic(false);
        }
        if (positiveNode?.applyWidgetLogic) {
            positiveNode.applyWidgetLogic(false);
        }
        if (negativeNode?.applyWidgetLogic) {
            negativeNode.applyWidgetLogic(false);
        }

        // 2. Temporarily unmute clipSkipNode so LiteGraph calculates distinct column levels
        const prevClipSkipMode = clipSkipNode.mode;
        clipSkipNode.mode = 0;

        // 3. Arrange with adequate margin (80px)
        graph.arrange(80);

        // 4. Restore original bypass/mute mode
        clipSkipNode.mode = prevClipSkipMode;

        graph.setDirtyCanvas(true, true);

        for (const opt of [
            "model hash",
            "ensd",
            "version",
            "vae hash",
            "ti hashes",
            "lora hashes",
            "hashes"
        ]) {
            delete opts[opt];
        }

        if (Object.keys(opts).length) {
            console.warn("Unhandled parameters:", opts);
        }
    };

    if (typeof LiteGraph?.withGraphIntentSource === "function") {
        LiteGraph.withGraphIntentSource("load", buildGraph);
    } else if (typeof window?.comfyAPI?.graphIntents?.withGraphIntentSource === "function") {
        window.comfyAPI.graphIntents.withGraphIntentSource("load", buildGraph);
    } else if (typeof window?.withGraphIntentSource === "function") {
        window.withGraphIntentSource("load", buildGraph);
    } else {
        buildGraph();
    }

    return embeddingsLoaded ? "imported" : "imported-without-embeddings";
}

if (typeof window !== "undefined") {
    window.comfyAPI = window.comfyAPI || {};
    window.comfyAPI.pnginfo = window.comfyAPI.pnginfo || {};
    window.comfyAPI.pnginfo.importA1111 = importA1111;
}