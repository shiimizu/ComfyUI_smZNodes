import { app as _app } from "../../scripts/app.js";

export const ids1 = new Set(["smZ CLIPTextEncode"]);
export const ids2 = new Set(["smZ Settings"]);

export const CONVERTED_TYPE = "converted-widget";
export const HEADING_IDENTIFIER = "\u3164"; // Hangul Filler (ㅤ)

export const KNOWN_HEADINGS = [
    "Stable Diffusion",
    "Compute Settings",
    "Sampler parameters",
    "Optimizations",
    "Compatibility",
    "Experimental"
];

export const findWidgetByName = (node, name) => node.widgets?.find((w) => w.name === name);

const HIDDEN_SIZE_FN = () => [0, -4];

// --- Minimum Height Calculator ---

function getRequiredHeight(node) {
    let hasVisible = false;
    let h = LiteGraph.NODE_TITLE_HEIGHT || 30;

    for (const w of node.widgets || []) {
        if (w.type === CONVERTED_TYPE || w.hidden) continue;
        hasVisible = true;

        let widgetH = LiteGraph.NODE_WIDGET_HEIGHT || 20;
        // Only use custom computeSize if it returns a positive height
        if (w.computeSize && w.computeSize !== HIDDEN_SIZE_FN) {
            const sz = w.computeSize(node.size[0]);
            if (sz && sz[1] > 0) {
                widgetH = sz[1];
            }
        }
        h += widgetH + 4;
    }

    if (!hasVisible) {
        return (LiteGraph.NODE_TITLE_HEIGHT || 30) + 8;
    }
    return h + 14;
}

// --- Modern Heading Banner Canvas Renderer ---

function drawPrettyHeadingBanner(ctx, title, w, y, height) {
    ctx.save();
    ctx.globalAlpha = 1.0;
    
    const text = String(title || "").trim();
    if (!text) {
        ctx.restore();
        return;
    }

    const canvas = ctx.canvas;
    const isLegacyCanvas = canvas === _app?.canvas?.canvas || canvas === window?.app?.canvas?.canvas;

    // Detect zoom magnification in Nodes 2.0 (WidgetLegacy)
    // We normalize sizes instead of mutating canvas.width (which caused the blur).
    let scale = 1.0;
    if (!isLegacyCanvas && ctx.getTransform) {
        const t = ctx.getTransform();
        const ctxScale = Math.hypot(t.a, t.b) || 1.0;
        const cssStretch = (canvas.clientWidth && canvas.width) ? (canvas.clientWidth / canvas.width) : 1.0;
        scale = ctxScale * cssStretch;
        if (!scale || scale <= 0 || isNaN(scale)) scale = 1.0;
    }

    const yOffset = y || 0;
    const bannerWidth = w || 200;
    const slotH = height && height > 0 ? height : 28;

    // Top padding: shifts the banner down to provide comfortable breathing room on top
    const topPadding = 4;
    const centerY = yOffset + topPadding + Math.floor((slotH - topPadding) / 2);

    // Normalize dimensions if inside Nodes 2.0 zoomed WidgetLegacy
    const norm = 1.0 / scale;
    const baseFontSize = 11;
    const fontSize = isLegacyCanvas ? baseFontSize : Math.max(6, baseFontSize * norm);
    const pillH = isLegacyCanvas ? 16 : Math.max(8, 16 * norm);
    const pillPad = isLegacyCanvas ? 10 : 10 * norm;
    const radius = Math.floor(pillH / 2);

    // 1. Divider line with smooth fade-out gradient
    const lineMargin = 12;
    ctx.lineWidth = isLegacyCanvas ? 1 : Math.max(0.5, 1 * norm);
    const grad = ctx.createLinearGradient(lineMargin, 0, bannerWidth - lineMargin, 0);
    grad.addColorStop(0, "rgba(255, 255, 255, 0.04)");
    grad.addColorStop(0.2, "rgba(255, 255, 255, 0.16)");
    grad.addColorStop(0.5, "rgba(255, 255, 255, 0.22)");
    grad.addColorStop(0.8, "rgba(255, 255, 255, 0.16)");
    grad.addColorStop(1, "rgba(255, 255, 255, 0.04)");
    ctx.strokeStyle = grad;

    ctx.beginPath();
    ctx.moveTo(lineMargin, centerY);
    ctx.lineTo(bannerWidth - lineMargin, centerY);
    ctx.stroke();

    // 2. Measure text
    ctx.font = `bold ${fontSize}px sans-serif`;
    const textW = ctx.measureText(text).width;

    // 3. Pill badge dimensions
    const pillW = Math.min(textW + pillPad * 2, bannerWidth - 24);
    const pillX = (bannerWidth - pillW) / 2;
    const pillY = centerY - Math.floor(pillH / 2);

    // 4. Draw pill badge background
    ctx.beginPath();
    if (typeof ctx.roundRect === "function") {
        ctx.roundRect(pillX, pillY, pillW, pillH, radius);
    } else {
        ctx.rect(pillX, pillY, pillW, pillH);
    }
    
    ctx.fillStyle = "#1e1f26";
    ctx.fill();

    // Subtle border
    ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
    ctx.stroke();

    // 5. Draw text label
    ctx.fillStyle = "#94a3b8";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, bannerWidth / 2, centerY + 0.5);

    ctx.restore();
}

/**
 * Cache storage for removed widgets so their instances and values are preserved.
 * Keyed by node ID or instance.
 */
function getWidgetStore(node) {
    if (!node.__smZ_widgetStore) {
        node.__smZ_widgetStore = {
            hiddenWidgets: new Map(), // name -> widget instance
            originalOrder: [],        // array of widget names
        };
        // Record default ordering once
        if (node.widgets) {
            node.__smZ_widgetStore.originalOrder = node.widgets.map((w) => w.name);
        }
    }
    return node.__smZ_widgetStore;
}

// --- Safe Non-Destructive Widget Hiding Engine ---

export function toggleWidget(node, widgetOrName, forceHide) {
    if (!node) return;
    const widget = typeof widgetOrName === "string" 
        ? findWidgetByName(node, widgetOrName) 
        : widgetOrName;

    if (!widget) return;

    if (!widget.__smZ_origComputeSize && widget.computeSize && widget.computeSize !== HIDDEN_SIZE_FN) {
        widget.__smZ_origComputeSize = widget.computeSize;
    }

    const shouldHide = forceHide ?? !widget.hidden;

    widget.hidden = Boolean(shouldHide);
    widget.options ??= {};

    if (shouldHide) {
        widget.computeSize = HIDDEN_SIZE_FN;
        widget.options.hidden = true;
        widget.options.canvasOnly = true; // When hidden, skip Vue completely
        widget.options.hideInPanel = true;

        for (const el of ["inputEl", "input", "element", "parentEl"]) {
            if (widget[el]?.style) {
                widget[el].style.display = "none";
            }
        }
    } else {
        delete widget.options.hidden;

        if (widget.heading) {
            widget.computeSize = () => [0, 32]; // Allocate 32px
            delete widget.options.canvasOnly;  // MUST be deleted so WidgetLegacy mounts in Vue!
            widget.options.hideInPanel = true; // Keeps dummy text out of the parameters sidebar
        } else if (widget.__smZ_origComputeSize) {
            widget.computeSize = widget.__smZ_origComputeSize;
            delete widget.options.canvasOnly;
            delete widget.options.hideInPanel;
        } else {
            delete widget.computeSize;
            delete widget.options.canvasOnly;
            delete widget.options.hideInPanel;
        }

        for (const el of ["inputEl", "input", "element", "parentEl"]) {
            if (widget[el]?.style) {
                if (widget.heading && (el === "inputEl" || el === "input")) {
                    widget[el].style.display = "none";
                } else {
                    widget[el].style.display = "";
                }
            }
        }
    }

    // Trigger Nodes 2.0 Vue Proxy commit
    if (Array.isArray(node.widgets)) {
        node.widgets = [...node.widgets];
    }

    fitNode(node);
}

export function fitNode(node, forceShrink = false) {
    if (!node) return;

    for (const w of node.widgets || []) {
        if (!w.hidden) {
            delete w.width;
        }
    }

    const minRequired = getRequiredHeight(node);
    const hasVisible = (node.widgets || []).some(w => !w.hidden && !w.options?.hidden);

    const newHeight = forceShrink || !hasVisible
        ? minRequired
        : Math.max(node.size?.[1] || 0, minRequired);

    if (node.setSize) {
        node.setSize([node.size[0], newHeight]);
    } else if (node.size) {
        node.size[1] = newHeight;
    }
    node.setDirtyCanvas?.(true, true);
    _app.graph?.setDirtyCanvas?.(true, true);
}

// --- Connection & Link Type Detection ---

function getConnectedInputType(node, slotIndex = 0) {
    const input = node?.inputs?.[slotIndex];
    if (!input || input.link == null) return null;

    // Use node.graph to support modern subgraphs and nested workflows
    const graph = node.graph || _app.graph || _app.rootGraph;
    let link = graph?.links?.[input.link];
    while (link) {
        const originNode = graph?.getNodeById?.(link.origin_id);
        if (!originNode) break;

        // Traverse through Reroutes or chained Settings nodes
        if (originNode.type === "Reroute" || ids2.has(originNode.type) || ids2.has(originNode.comfyClass)) {
            const rerouteInput = originNode.inputs?.[0];
            if (rerouteInput?.link != null) {
                link = graph?.links?.[rerouteInput.link];
                continue;
            }
        }
        const originOutput = originNode.outputs?.[link.origin_slot];
        const type = originOutput?.type || link.type;
        if (type && type !== "*" && type !== "target") {
            return String(type).toUpperCase();
        }
        break;
    }
    return null;
}

// Matches both V3 IDs and legacy V1 names for CLIP-specific settings
function isClipSetting(w) {
    if (!w) return false;
    const name = (w.name || "").toLowerCase();
    const label = (w.label || w.options?.display_name || "").toLowerCase();

    return (
        name.includes("prompt_word_wrap") ||
        name.includes("comma_padding_backtrack") ||
        name.includes("enable_emphasis") ||
        name.includes("use_old_scheduling") ||
        name.includes("use_prev_scheduling") ||
        label.includes("prompt word wrap") ||
        label.includes("enable emphasis") ||
        label.includes("prompt editing timelines")
    );
}

export function getHeadingTitle(w) {
    if (!w) return "";
    let val = typeof w.value === "string" ? w.value.trim() : "";
    if (val && KNOWN_HEADINGS.includes(val)) return val;
    if (typeof w.options?.default === "string" && w.options.default.trim()) {
        val = w.options.default.trim();
        if (KNOWN_HEADINGS.includes(val)) return val;
    }
    // Fallback: match by the count of Hangul filler characters (\u3164) in the name
    const name = String(w.name || "");
    const fillerCount = (name.match(/\u3164/g) || []).length;
    if (fillerCount >= 1 && fillerCount <= KNOWN_HEADINGS.length) {
        return KNOWN_HEADINGS[fillerCount - 1];
    }
    return val || name;
}

export function isHeadingWidget(w) {
    if (!w) return false;
    if (w.heading) return true;
    const name = String(w.name || "");
    if (name.includes(HEADING_IDENTIFIER) || name.includes("\u3164")) return true;
    const val = typeof w.value === "string" ? w.value.trim() : (typeof w.options?.default === "string" ? w.options.default.trim() : "");
    if (KNOWN_HEADINGS.includes(val)) return true;
    return false;
}

function isInfoWidget(w) {
    if (!w) return false;
    return Boolean(w.info || w.name?.startsWith("info_") || w.name?.includes("info_"));
}

// --- Logic for CLIPTextEncode++ ---

function applyClipWidgetLogic(node, isInteractive = false) {
    if (!node?.widgets?.length) return;

    node.properties ??= {};
    node.properties.hide_mode ??= true;
    node.properties.hide_steps ??= true;
    node.properties.hide_multi_conditioning ??= false;
    node.properties.hide_mean_normalization ??= false;
    node.properties.hide_use_old_emphasis_implementation ??= true;

    toggleWidget(node, "mode", Boolean(node.properties.hide_mode));
    toggleWidget(node, "with_SDXL", Boolean(node.properties.hide_mode));
    toggleWidget(node, "smZ_steps", Boolean(node.properties.hide_steps));

    const parserWidget = findWidgetByName(node, "parser");
    const parserVal = parserWidget?.value || "comfy";

    if (parserVal === "comfy") {
        toggleWidget(node, "multi_conditioning", Boolean(node.properties.hide_multi_conditioning));
        toggleWidget(node, "mean_normalization", true);
        toggleWidget(node, "use_old_emphasis_implementation", true);
    } else if (parserVal === "comfy++") {
        toggleWidget(node, "mean_normalization", Boolean(node.properties.hide_mean_normalization));
        toggleWidget(node, "multi_conditioning", Boolean(node.properties.hide_multi_conditioning));
        toggleWidget(node, "use_old_emphasis_implementation", true);
    } else {
        toggleWidget(node, "mean_normalization", Boolean(node.properties.hide_mean_normalization));
        toggleWidget(node, "multi_conditioning", Boolean(node.properties.hide_multi_conditioning));
        toggleWidget(node, "use_old_emphasis_implementation", Boolean(node.properties.hide_use_old_emphasis_implementation));
    }

    fitNode(node, false);
}

function setupClipNode(node, isLoad = false) {
    // Expose synchronous method callable from metadata.js
    node.applyWidgetLogic = function (isInteractive = false) {
        applyClipWidgetLogic(this, isInteractive);
    };

    node.properties ??= {};
    node.properties.hide_mode ??= true;
    node.properties.hide_steps ??= true;
    node.properties.hide_multi_conditioning ??= false;
    node.properties.hide_mean_normalization ??= false;
    node.properties.hide_use_old_emphasis_implementation ??= true;

    // Constantly purge frozen widget widths before every frame render
    const origDrawForeground = node.onDrawForeground;
    node.onDrawForeground = function (ctx) {
        if (this.widgets) {
            for (const w of this.widgets) {
                if (w.type !== CONVERTED_TYPE && !w.hidden) delete w.width;
            }
        }
        return origDrawForeground ? origDrawForeground.apply(this, arguments) : undefined;
    };

    const origOnResize = node.onResize;
    node.onResize = function (size) {
        const minH = getRequiredHeight(this);
        if (size[1] < minH) size[1] = minH;

        const res = origOnResize ? origOnResize.apply(this, arguments) : undefined;
        if (this.widgets) {
            for (const w of this.widgets) {
                if (w.type !== CONVERTED_TYPE && !w.hidden) delete w.width;
            }
        }
        return res;
    };

    const parserWidget = findWidgetByName(node, "parser");
    if (parserWidget && !parserWidget.__smZHooked) {
        parserWidget.__smZHooked = true;
        const origCb = parserWidget.callback;
        parserWidget.callback = function (val) {
            const res = origCb ? origCb.apply(this, arguments) : undefined;
            applyClipWidgetLogic(node, true);
            return res;
        };
    }

    // Hook mode widget callback so switching options expands widgets properly
    const modeWidget = findWidgetByName(node, "mode") || findWidgetByName(node, "with_SDXL");
    if (modeWidget && !modeWidget.__smZHooked) {
        modeWidget.__smZHooked = true;
        const origCb = modeWidget.callback;
        modeWidget.callback = function (val) {
            const res = origCb ? origCb.apply(this, arguments) : undefined;
            for (const w of node.widgets || []) {
                if (w.type !== CONVERTED_TYPE && !w.hidden) delete w.width;
            }
            fitNode(node, false);
            return res;
        };
    }

    applyClipWidgetLogic(node, !isLoad);
}

// --- Logic for Settings (smZ) ---

function applySettingsWidgetLogic(node, isInteractive = false) {
    if (!node?.widgets?.length) return;

    node.properties ??= {};
    const showHeadings = node.properties.show_headings ?? true;
    const connType = getConnectedInputType(node, 0);
    const isModel = connType?.includes?.("MODEL");
    const isClip = connType?.includes?.("CLIP");

    // Disconnected: hide everything completely
    if (!connType || !(isModel || isClip)) {
        for (const w of node.widgets) {
            toggleWidget(node, w, true);
        }
        fitNode(node, true);
        return;
    }

    for (const w of node.widgets) {
        // Heal any corrupted widgets with negative height stubs
        if (w.type !== CONVERTED_TYPE && !w.hidden && w.computeSize && !w.heading) {
            const sz = w.computeSize(node.size?.[0] || 100);
            if (!sz || sz[1] <= 0) {
                delete w.computeSize;
                delete w.__smZ_origComputeSize;
            }
        }

        // 1. Headings
        if (isHeadingWidget(w)) {
            w.heading = true;
            w.disabled = true;
            w.options ??= {};
            w.options.hideInPanel = true;
            w.computeSize = () => [0, 32]; // Set height to 32px

            const headingTitle = getHeadingTitle(w);
            if (!w.value) w.value = headingTitle;

            if (!w.__smZ_styledHeading) {
                w.__smZ_styledHeading = true;
                w.draw = function (ctx, n, width, y, height) {
                    if (this.hidden || this.options?.hidden) return;
                    const text = getHeadingTitle(this);
                    const renderW = width && width > 0 ? width : (n?.size?.[0] ? n.size[0] - 20 : 200);
                    drawPrettyHeadingBanner(ctx, text, renderW, y || 0, height);
                };
            }

            if (!showHeadings) {
                toggleWidget(node, w, true);
            } else {
                const isCompatibility = headingTitle.includes("Compatibility");
                if (isModel) {
                    toggleWidget(node, w, isCompatibility);
                } else if (isClip) {
                    const isClipHeading = headingTitle.includes("Stable Diffusion") || headingTitle.includes("Compatibility");
                    toggleWidget(node, w, !isClipHeading);
                }
            }
            continue;
        }

        // 2. Info widgets
        if (isInfoWidget(w)) {
            toggleWidget(node, w, true);
            continue;
        }

        // 3. Debug widget: MODEL & CLIP input only
        if (w.name === "debug") {
            toggleWidget(node, w, !(isModel || isClip));
            continue;
        }

        // 4. Regular settings
        const isClipParam = isClipSetting(w);
        if (isModel) {
            toggleWidget(node, w, isClipParam);
        } else if (isClip) {
            toggleWidget(node, w, !isClipParam);
        }
    }

    fitNode(node, isInteractive);
}

function setupSettingsNode(node, isLoad = false) {
    // Expose synchronous method callable from metadata.js
    node.applyWidgetLogic = function (isInteractive = false) {
        applySettingsWidgetLogic(this, isInteractive);
    };

    // Ensure adequate width for long settings names
    if (!node.size || node.size[0] < 360) {
        node.size = [360, node.size?.[1] || 100];
    }

    node.properties ??= {};
    if (node.properties.show_headings === undefined) node.properties.show_headings = true;

    const origDrawForeground = node.onDrawForeground;
    node.onDrawForeground = function (ctx) {
        if (this.widgets) {
            for (const w of this.widgets) {
                if (w.type !== CONVERTED_TYPE && !w.hidden) delete w.width;
            }
        }
        return origDrawForeground ? origDrawForeground.apply(this, arguments) : undefined;
    };

    const origOnResize = node.onResize;
    node.onResize = function (size) {
        const minH = getRequiredHeight(this);
        if (size[1] < minH) size[1] = minH;

        const res = origOnResize ? origOnResize.apply(this, arguments) : undefined;
        if (this.widgets) {
            for (const w of this.widgets) {
                if (w.type !== CONVERTED_TYPE && !w.hidden) delete w.width;
            }
        }
        return res;
    };

    const origOnConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function (type, index, connected, link_info) {
        const res = origOnConnectionsChange ? origOnConnectionsChange.apply(this, arguments) : undefined;
        requestAnimationFrame(() => {
            applySettingsWidgetLogic(node, true);
        });
        return res;
    };

    applySettingsWidgetLogic(node, !isLoad);
}

// --- Modern Extension Registration ---

_app.registerExtension({
    name: "Comfy.smZ.dynamicWidgets",

    init() {
        const style = document.createElement("style");
        style.appendChild(document.createTextNode(`
            .${CONVERTED_TYPE} {
                display: none !important;
                visibility: hidden !important;
                height: 0px !important;
            }
            .smZ-custom-textarea {
                color: inherit;
                opacity: 0.75;
            }
        `));
        document.head.appendChild(style);
    },

    nodeCreated(node) {
        const nodeType = node.comfyClass || node.type;
        if (ids1.has(nodeType)) setupClipNode(node, false);
        if (ids2.has(nodeType)) setupSettingsNode(node, false);
    },

    loadedGraphNode(node) {
        const nodeType = node.comfyClass || node.type;
        if (ids1.has(nodeType)) setupClipNode(node, true);
        if (ids2.has(nodeType)) setupSettingsNode(node, true);
    },

    getNodeMenuItems(node) {
        const items = [];
        const nodeType = node.comfyClass || node.type;

        // Context menu for CLIPTextEncode++
        if (ids1.has(nodeType)) {
            items.push(
                {
                    content: node.properties?.hide_mode ? "Show Mode Widget" : "Hide Mode Widget",
                    callback: () => {
                        node.properties.hide_mode = !node.properties.hide_mode;
                        applyClipWidgetLogic(node, true);
                    }
                },
                {
                    content: node.properties?.hide_steps ? "Show Steps Widget" : "Hide Steps Widget",
                    callback: () => {
                        node.properties.hide_steps = !node.properties.hide_steps;
                        applyClipWidgetLogic(node, true);
                    }
                }
            );

            const parserWidget = findWidgetByName(node, "parser");
            const parserVal = parserWidget?.value || "comfy";

            if (parserVal === "comfy") {
                items.push({
                    content: node.properties?.hide_multi_conditioning ? "Show Multi Conditioning" : "Hide Multi Conditioning",
                    callback: () => {
                        node.properties.hide_multi_conditioning = !node.properties.hide_multi_conditioning;
                        applyClipWidgetLogic(node, true);
                    }
                });
            } else if (parserVal === "comfy++") {
                items.push(
                    {
                        content: node.properties?.hide_mean_normalization ? "Show Mean Normalization" : "Hide Mean Normalization",
                        callback: () => {
                            node.properties.hide_mean_normalization = !node.properties.hide_mean_normalization;
                            applyClipWidgetLogic(node, true);
                        }
                    },
                    {
                        content: node.properties?.hide_multi_conditioning ? "Show Multi Conditioning" : "Hide Multi Conditioning",
                        callback: () => {
                            node.properties.hide_multi_conditioning = !node.properties.hide_multi_conditioning;
                            applyClipWidgetLogic(node, true);
                        }
                    }
                );
            } else {
                items.push(
                    {
                        content: node.properties?.hide_mean_normalization ? "Show Mean Normalization" : "Hide Mean Normalization",
                        callback: () => {
                            node.properties.hide_mean_normalization = !node.properties.hide_mean_normalization;
                            applyClipWidgetLogic(node, true);
                        }
                    },
                    {
                        content: node.properties?.hide_multi_conditioning ? "Show Multi Conditioning" : "Hide Multi Conditioning",
                        callback: () => {
                            node.properties.hide_multi_conditioning = !node.properties.hide_multi_conditioning;
                            applyClipWidgetLogic(node, true);
                        }
                    },
                    {
                        content: node.properties?.hide_use_old_emphasis_implementation ? "Show Old Emphasis" : "Hide Old Emphasis",
                        callback: () => {
                            node.properties.hide_use_old_emphasis_implementation = !node.properties.hide_use_old_emphasis_implementation;
                            applyClipWidgetLogic(node, true);
                        }
                    }
                );
            }
        }

        // Context menu for Settings
        if (ids2.has(nodeType)) {
            items.push({
                content: node.properties?.show_headings ? "Hide Headings" : "Show Headings",
                callback: () => {
                    node.properties.show_headings = !node.properties.show_headings;
                    applySettingsWidgetLogic(node, true);
                }
            });
        }

        return items;
    }
});


const OLD_KEYS = [
    "text", "parser", "mean_normalization", "multi_conditioning",
    "use_old_emphasis_implementation", "with_SDXL", "ascore",
    "width", "height", "crop_w", "crop_h", "target_width",
    "target_height", "text_g", "text_l", "smZ_steps"
];

const SDXL_KEYS = [
    "text_g", "text_l", "ascore", "width", "height",
    "crop_w", "crop_h", "target_width", "target_height"
];

// Added 'multi_conditioning' here in its natural place
const COMMON_KEYS = [
    "parser", "mean_normalization", "multi_conditioning",
    "use_old_emphasis_implementation", "smZ_steps"
];

_app.registerExtension({
    name: "smZ.CLIPTextEncode.Migration",
    beforeConfigureGraph(graphData) {
        if (!graphData?.nodes) return;

        for (const node of graphData.nodes) {
            if (node.type === "smZ CLIPTextEncode") {
                const val = node.widgets_values;

                if (Array.isArray(val) && val[0] !== "Standard" && val[0] !== "SDXL") {
                    const v = Object.fromEntries(OLD_KEYS.map((k, i) => [k, val[i]]));
                    const common = COMMON_KEYS.map(k => v[k]);

                    node.widgets_values = v.with_SDXL
                        ? ["SDXL", ...SDXL_KEYS.map(k => v[k]), ...common]
                        : ["Standard", v.text, ...common];
                }
            }
        }
    }
});