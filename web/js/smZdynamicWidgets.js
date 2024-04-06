import { app as _app } from "../../../scripts/app.js";
import { mergeIfValid, getWidgetConfig, setWidgetConfig } from "../../../extensions/core/widgetInputs.js";
// import { ComfyWidgets } from "/scripts/widgets.js";

export const ids1 = new Set(["smZ CLIPTextEncode"])
export const ids2 = new Set(["smZ Settings"])
export const widgets = ['mean_normalization', 'multi_conditioning', 'use_old_emphasis_implementation', 'with_SDXL']
export const widgets_sdxl = ['ascore', 'width', 'height', 'crop_w', 'crop_h', 'target_width', 'target_height', 'text_g', 'text_l']
export const getSetWidgets = new Set(['parser', 'with_SDXL'])

export let origProps = {};
export const HIDDEN_TAG = "smZhidden"

export const findWidgetByName = (node, name) => node.widgets.find((w) => w.name === name);
export const findWidgetsByName = (node, name) => node.widgets.filter((w) => w.name.endsWith(name));

export const doesInputWithNameExist = (node, name) => node.inputs ? node.inputs.some((input) => input.name === name) : false;

// round in increments of n, with an offset
export function round(number, increment = 10, offset = 0) {
    return Math.ceil((number - offset) / increment ) * increment + offset;
}

export function toggleWidget(node, widget, show = false, suffix = "") {
    if (!widget || doesInputWithNameExist(node, widget.name)) return;
    if (!origProps[widget.name]) {
        origProps[widget.name] = { origType: widget.type, origComputeSize: widget.computeSize};
    }
    const origSize = node.size;

    widget.type = show ? origProps[widget.name].origType : HIDDEN_TAG + suffix;
    widget.computeSize = show ? origProps[widget.name].origComputeSize : () => [0, -3.3];

    widget.linkedWidgets?.forEach(w => toggleWidget(node, w, ":" + widget.name, show));

    const height = show ? Math.max(node.computeSize()[1], origSize[1]) : node.size[1];
    node.setSize([node.size[0], height]);
    if (show)
        delete widget.computedHeight;
    else
        widget.computedHeight = 0;
}

// Passing an array with a companion_widget_name means
// the toggle will happen for every name that looks like it.
// Useful for duplicates created in group nodes.
export function toggleMenuOption(node, widget_arr, _show = null, perform_action = true) {
    const [widget_name, companion_widget_name] = Array.isArray(widget_arr) ? widget_arr : [widget_arr]
    let nwname = widget_name
    // companion_widget_name to use the new name assigned in a group node to get the correct widget 
    if (companion_widget_name) {
        for (const gnc of getGroupNodeConfig(node)) {
            const omap = Object.values(gnc.oldToNewWidgetMap).find(x => Object.values(x).find(z => z === companion_widget_name))
            const n = omap[widget_name]
            if(n) nwname = n;
        }
    }
    const widgets = companion_widget_name ? [findWidgetByName(node, nwname)] : findWidgetsByName(node, nwname)
    for (const widget of widgets)
        toggleMenuOption0(node, widget, _show, perform_action)
}

function toggleMenuOption0(node, widget, _show = null, perform_action = true) {
    if (!widget || doesInputWithNameExist(node, widget.name)) return;
    if (!origProps[widget.name]) {
        origProps[widget.name] = { origType: widget.type, origComputeSize: widget.computeSize};
    }
    const show = (widget.type === origProps[widget.name].origType)
    if (perform_action) {
        toggleWidget(node, widget, _show !== null ? _show : !show)
        node.setDirtyCanvas(true);
    }
}

export function getGroupNodeConfig(node) {
    let ls = []
    let nodeData = node.constructor?.nodeData
    if (nodeData) {
        for(const sym of Object.getOwnPropertySymbols(nodeData) ) {
            const o = nodeData[sym]; 
            if (o) ls.push(o)
        }
    }
    return ls
}

export function widgetLogic(node, widget) {
    const wname = widget.name
    if (wname.endsWith("parser")) {
        const in_comfy = widget?.value?.includes?.("comfy")
        toggleMenuOption(node, ['multi_conditioning', wname], !in_comfy)
        toggleMenuOption(node, ['mean_normalization', wname], widget.value !== "comfy")
        const uoei = 'use_old_emphasis_implementation'
        if (findWidgetsByName(node, uoei)?.some(x => x?.value))
            toggleMenuOption(node, [uoei, wname], true)
        if (in_comfy)
            toggleMenuOption(node, [uoei, wname], false)
    } else if (wname.endsWith("with_SDXL")) {
        toggleMenuOption(node, ['text', wname], !widget.value)

        // Resize node when widget is set to false
        if (!widget.value) {
            // Prevents resizing on init/webpage reload
            if(widget.init === false) {
                // Resize when set to false
                node.setSize([node.size[0], Math.max(100, round(node.size[1]/1.5))])
            }
        } else {
            // When enabled, set init to false
            widget.init = false
        }

        // Toggle sdxl widgets if sdxl widget value is true/false
        for (const w of widgets_sdxl) {
            toggleMenuOption(node, [w, wname], widget.value)
        }

        // Keep showing the widget if it's enabled
        if (widget.value && widget.type === HIDDEN_TAG) {
            toggleMenuOption(node, [widget.name, wname], true)
        }
    }
}

// Specfic to cliptextencode
function applyWidgetLogic(node) {
    if (!node.widgets || (node.widgets && !node.widgets.length)) return
    if (node.widgets) {
        let gncl = getGroupNodeConfig(node)
        for (const w of node.widgets) {
            for (const gsw of [...getSetWidgets]) {
                if (!w.name.endsWith(gsw)) continue;
                // Possibly uneeded
                /*let shouldBreak = false
                for (const gnc of gncl) {
                    const nwmap = gnc.newToOldWidgetMap[w.name]
                    console.log('=== gnc.newToOldWidgetMap',gnc.newToOldWidgetMap,'w.name',w.name,'nwmap',nwmap)
                    // nwmap.inputName: resolved, actual widget name.
                    if (nwmap && !(ids1.has(nwmap.node.type) && nwmap.inputName === gsw))
                        shouldBreak = true
                }
                if (shouldBreak) break;*/
                widgetLogic(node, w);

				let val = w.value;
                Object.defineProperty(w, 'value', {
                    get() {
                        return val;
                    },
                    set(newVal) {
						if (newVal !== val) {
                            val = newVal
                            widgetLogic(node, w);
                        }
                    }
                });

                // Hide SDXL widget on init
                // Doing it in nodeCreated fixes its toggling for some reason
                if (w.name.endsWith('with_SDXL')) {
                    toggleMenuOption(node, ['with_SDXL', w.name])
                    w.init = true
                    
                    // Hide steps
                    toggleMenuOption(node, ['smZ_steps', w.name] , false)
                }
            }
        }

        // Reduce initial node size cause of SDXL widgets
        // node.setSize([node.size[0], Math.max(node.size[1]/1.5, 220)])
        node.setSize([node.size[0], 220])
    }
}

function toggle_all_settings_desc_widgets(node, _show = null) {
    let found_widgets = node.widgets.filter((w) => w.name.includes('info'));
    let is_showing = _show !== null ? _show : null
    found_widgets.forEach(w => {
        toggleMenuOption(node, [w.name, w.name], _show)
        is_showing = _show !== null ? _show : w.type === origProps[w.name].origType
    });

    let w = node.widgets.find((w) => w.name === 'extra');
    if (w) {
        let value = null;
        try {
            value =JSON.parse(w.value);
        } catch (error) {
            // when node definitions change due to an update or some other error
            value = {"show":true}
        }
        value.show = is_showing;
        w.value = JSON.stringify(value);
    }

    // Collapse the node if the widgets aren't showing
    if (!is_showing) {
        node.setSize([node.size[0], node.computeSize()[1]])
    }
}

function create_custom_option(content, _callback) {
    return {
        content: content,
        callback: () => _callback(),
    }
};

function widgetLogicSettings(node) {
    const supported_types = {MODEL: 'MODEL', CLIP: 'CLIP'}
    const clip_headings = ['Stable Diffusion', 'Compatibility']
    const clip_entries = 
    ["info_comma_padding_backtrack",
    "Prompt word wrap length limit",
    "enable_emphasis",
    "info_use_prev_scheduling",
    "Use previous prompt editing timelines"]

    if(!node.widgets) return;

    const index = node.index || 0
    
    const extra = node.widgets.find(w => w.name.endsWith('extra'))
    let extra_data = extra._value
    toggleMenuOption(node, extra.name, false)
    const condition = (sup) => node.outputs?.[index] && (node.outputs[index].name === sup || node.outputs[index].type === sup) || 
                                    node.inputs?.[index] && node.inputs[index].type === sup;
    // w._name: to make sure it's from our node. though, it should have a better name for the variable
    for (const w of node.widgets) {
        if (w.name.endsWith('extra')) continue;
        if (node.inputs?.find(i => i.name === w.name)) {
            w.type = 'converted-widget'
        }
        if(w.type === 'converted-widget') continue;
        if(!w._name) continue;

        // heading `values` won't get duplicated names due to group nodes
        // So we won't need to do `clip_headings.some()`
        // though, it should be read only...
        if (condition(supported_types.MODEL)) {
            const flag=(clip_entries.some(str => (typeof str === 'string' && str.includes(w.name))) || (typeof w.value === 'string' && w.heading && w.value.includes('Compatibility')))
            if (w.info && !clip_entries.some(str => str.includes(w.name)))
                toggleMenuOption(node, [w.name, w.name], extra_data.show_descriptions)
            else if (w.heading && typeof w.value === 'string' && !w.value.includes('Compatibility'))
                toggleMenuOption(node, [w.name, w.name], extra_data.show_headings)
            else
                toggleMenuOption(node, [w.name, w.name], !flag)
                // toggleMenuOption(node, w.name, flag ? false : true) //doesn't work?
        } else if (condition(supported_types.CLIP)) {
            // if w.name in list -> enable, else disable
            const flag = clip_entries.some(str => str.includes(w.name))
            if (w.info && flag)
                toggleMenuOption(node, [w.name, w.name], extra_data.show_descriptions)
            else if (w.heading && typeof w.value === 'string' && clip_headings.includes(w.value))
                toggleMenuOption(node, [w.name, w.name], extra_data.show_headings)
            else
                toggleMenuOption(node, [w.name, w.name], flag)
        } else {
            toggleMenuOption(node, w.name, false)
        }
    }

    node.setSize([node.size[0], node.computeSize()[1]])
    node.onResize?.(node.size)
    node.setDirtyCanvas(true);
    _app.graph.setDirtyCanvas(true, true);
}

_app.registerExtension({
    name: "Comfy.smZ.dynamicWidgets",

    /**
     * Called when a node is created. Used to add menu options to nodes.
     * @param node The node that was created.
     * @param app The app.
     */
    nodeCreated(node, app) {
        const nodeType = node.type || node.constructor?.type
        let inGroupNode = false
        let inGroupNode2 = false
        let nodeData = node.constructor?.nodeData
        if (nodeData) {
            for(let sym of Object.getOwnPropertySymbols(nodeData) ) {
                const nds = nodeData[sym];
                const nodes = nds?.nodeData?.nodes
                if (nodes) {
                    for (const _node of nodes) {
                        const _nodeType = _node.type
                        if (ids1.has(_nodeType))
                            inGroupNode = true
                        if (inGroupNode)
                            ids1.add(nodeType) // GroupNode's type
                        if (ids2.has(_nodeType))
                            inGroupNode2 = true
                        if (inGroupNode2)
                            ids2.add(nodeType) // GroupNode's type
                    }
                }
            }
        }

        // ClipTextEncode++ node
        if (ids1.has(nodeType) || inGroupNode) {
            applyWidgetLogic(node)
        }
        // Settings node
        if (ids2.has(nodeType) || inGroupNode2) {
            if(!node.properties)
                node.properties = {}
            node.properties.showOutputText = true
            

            // allows bypass (in conjunction with below's `allows bypass`)
            // by setting node.inputs[0].type to a concrete type, instead of '*'. ComfyUI will complain otherwise.
            node.onBeforeConnectInput = function(inputIndex) {
                if (inputIndex !== node.index) return inputIndex

                // so we can connect to reroutes
                const tp = 'Reroute'
                node.type = tp
                if (node.constructor) node.constructor.type=tp
                this.type = tp
                if (this.constructor) this.constructor.type=tp
                Object.assign(node.inputs[inputIndex], {...node.inputs[inputIndex], name: '*', type: '*'});
                Object.assign(this.inputs[inputIndex], {...this.inputs[inputIndex], name: '*', type: '*'});
                node.beforeConnectInput = true
                this.beforeConnectInput = true
                return inputIndex;
            }
            
            // Call once on node creation
            node.setupWidgetLogic = function() {
                let nt = nodeType // JSON.parse(JSON.stringify(nodeType))
                node.type = nodeType
                if (node.constructor) node.constructor.type=nodeType
                node.applyOrientation = function() {
                    node.type = nodeType
                    if (node.constructor) node.constructor.type=nodeType
                }
                node.index = 0
                let i = 0
                const innerNode = node.getInnerNodes?.().find(n => {const r = ids2.has(n.type); if (r) node.index = i; ++i; return r }  )
                const innerNodeWidgets = innerNode?.widgets
                i = 0
                node.widgets.forEach(function(w) {
                    if (innerNodeWidgets) {
                        if(innerNodeWidgets.some(iw => w.name.endsWith(iw.name)))
                            w._name = w.name
                    } else
                        w._name = w.name
                    
                    // Styling.
                    if (w.name.includes('ㅤ')) {
                        w.heading = true
                        // w.disabled = true
                    } else if (w.name.includes('info')) {
                        w.info = true
                        w.inputEl.disabled = true;
                        w.inputEl.readOnly = true;
                        w.inputEl.style.alignContent = 'center';
                        w.inputEl.style.textAlign = 'center';
                        if (!w.inputEl.classList.contains('smZ-custom-textarea'))
                            w.inputEl.classList.add('smZ-custom-textarea')
                    }
                })
                const extra_widget = node.widgets.find(w => w.name.endsWith('extra'))
                if (extra_widget) {
                    let extra_data = null
                    try {
                        extra_data = JSON.parse(extra_widget.value);
                    } catch (error) {
                        // when node definitions change due to an update or some other error
                        extra_data = {show_headings: true, show_descriptions: false, mode: '*'}
                    }
                    extra_widget._value = extra_data
                    Object.defineProperty(extra_widget, '_value', {
                        get() {
                            return extra_data;
                        },
                        set(newVal) {
                            extra_data = newVal
                            extra_widget.value = JSON.stringify(extra_data)
                        }
                    });
                }

                widgetLogicSettings(node);
                // Hijack getting our node type so we can work with Reroutes
                Object.defineProperty(node.constructor, 'type', {
                    get() {
                        let s = new Error().stack
                        const rr = ['rerouteNode.js']
                        // const rr = ['rerouteNode.js',  'reroutePrimitive.js']
                        // const rr = ['rerouteNode.js', 'groupNode.js', 'reroutePrimitive.js']
                        if (rr.some(rx => s.includes(rx))) {
                            return 'Reroute'
                        }
                        return nt;
                    },
                    set(newVal) {
                        if (newVal !== nt) {
                            nt = newVal
                        }
                    }
                });

                if (node.outputs[node.index]) {
                    let val = node.outputs[node.index].type;
                    // Hijacks getting/setting type
                    Object.defineProperty(node.outputs[node.index], 'type', {
                        get() {
                            return val;
                        },
                        set(newVal) {
                            if (newVal !== val) {
                                val = newVal
                                // console.log('====== group node test. node', node, 'group', node.getInnerNodes?.())
                                if (node.inputs && node.inputs[node.index]) node.inputs[node.index].type = newVal
                                if (node.outputs && node.outputs[node.index]) node.outputs[node.index].name  = newVal || '*';
                                node.properties.showOutputText = true // Reroute node accesses this
                                node.type = nodeType
                                if (node.constructor) node.constructor.type=nodeType
                                // this.type = nodeType
                                // if (this.constructor) this.constructor.type=nodeType
                                // console.log('==== setupWidgetLogic', `val: '${val}' newval: '${newVal}' '${node.outputs[node.index].name}'`)
                                widgetLogicSettings(node);
                            }
                        }
                    });
                }
            }
            node.setupWidgetLogic()

            const onConfigure = node.onConfigure;
            node.onConfigure = function(o) {
                // Call again after the node is created since there might be link
                // For example: if we reload the graph
                node.setupWidgetLogic()
                const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
                return r;
            }

            // ================= Adapted from rerouteNode.js =================
            node.onAfterGraphConfigured = function () {
                    requestAnimationFrame(() => {
                        node.onConnectionsChange(LiteGraph.INPUT, null, true, null);
                });
            };
            node.onConnectionsChange = function (type, index, connected, link_info) {
                // node.index = index || 0
                // index = node.index || 0
                if (index !== node.index) return
                node.setupWidgetLogic()
                // if (index === node.index) node.setupWidgetLogic()

                const type_map = {
                    [LiteGraph.OUTPUT] : 'OUTPUT',
                    [LiteGraph.INPUT] : 'INPUT',
                }

                // console.log("======== onConnectionsChange type", type, "connected", connected, 'app.graph.links[l]',app.graph.links)
                // console.log('=== app.graph.links',app.graph.links, 'node.inputs[index]',node.inputs[index],'node.outputs[index]',node.outputs[index])
                // console.log('===  onConnectionsChange type', type_map[type], 'index',index,'connected',connected,'node.inputs', node.inputs,'node.outputs', node.outputs,'link_info',link_info, 'node', node)

                node.type = nodeType
                if (node.constructor) node.constructor.type=nodeType
                this.type = nodeType
                if (this.constructor) this.constructor.type=nodeType
            

                // Prevent multiple connections to different types when we have no input
                if (connected && type === LiteGraph.OUTPUT) {
                    // Ignore wildcard nodes as these will be updated to real types
                    const types = this.outputs?.[index]?.links ? new Set(this.outputs[index].links.map((l) => app.graph.links[l]?.type)?.filter((t) => t !== "*")) : new Set()
                    if (types?.size > 1) {
                        const linksToDisconnect = [];
                        for (let i = 0; i < this.outputs[index].links.length - 1; i++) {
                            const linkId = this.outputs[index].links[i];
                            const link = app.graph.links[linkId];
                            linksToDisconnect.push(link);
                        }
                        for (const link of linksToDisconnect) {
                            const node = app.graph.getNodeById(link.target_id);
                            node.disconnectInput(link.target_slot);
                        }
                    }
                }

                // Find root input
                let currentNode = this;
                let updateNodes = [];
                let inputType = null;
                let inputNode = null;
                while (currentNode) {
                    updateNodes.unshift(currentNode);
                    const linkId = currentNode?.inputs?.[index]?.link;
                    if (linkId) {
                        const link = app.graph.links[linkId];
                        if (!link) return;
                        const node = app.graph.getNodeById(link.origin_id);
                        const type = node.constructor.type || node.type;
                        if (type === "Reroute" || ids2.has(type)) {
                            if (node === this) {
                                // We've found a circle
                                currentNode.disconnectInput(link.target_slot);
                                currentNode = null;
                            } else {
                                // Move the previous node
                                currentNode = node;
                            }
                        } else {
                            // We've found the end
                            inputNode = currentNode;
                            inputType = node.outputs[link.origin_slot]?.type ?? null;
                            break;
                        }
                    } else {
                        // This path has no input node
                        currentNode = null;
                        break;
                    }
                }

                // Find all outputs
                const nodes = [this];
                // const nodes = [link_info?.origin_id ? app.graph.getNodeById(link_info.origin_id).outputs ? 
                //     app.graph.getNodeById(link_info.origin_id): this : this,this];
                let outputType = null;
                while (nodes.length) {
                    currentNode = nodes.pop();
                    // if (currentNode.outputs)
                    const outputs = (currentNode && currentNode.outputs?.[index]?.links ? currentNode.outputs?.[index]?.links : []) || [];
                    // console.log('=== .outputs',outputs,'currentNode',currentNode)
                    if (outputs.length) {
                        for (const linkId of outputs) {
                            const link = app.graph.links[linkId];

                            // When disconnecting sometimes the link is still registered
                            if (!link) continue;

                            const node = app.graph.getNodeById(link.target_id);
                            const type = node.constructor.type || node.type;
                            
                            if (type === "Reroute" || ids2.has(type)) {
                                // Follow reroute nodes
                                nodes.push(node);
                                updateNodes.push(node);
                            } else {
                                // We've found an output
                                const nodeOutType =
                                    node.inputs && node.inputs[link?.target_slot] && node.inputs[link.target_slot].type
                                        ? node.inputs[link.target_slot].type
                                        : null;
                                if (inputType && inputType !== "*" && nodeOutType !== inputType) {
                                    // The output doesnt match our input so disconnect it
                                    node.disconnectInput(link.target_slot);
                                } else {
                                    outputType = nodeOutType;
                                }
                            }
                        }
                    } else {
                        // No more outputs for this path
                    }
                }

                const displayType = inputType || outputType || "*";
                const color = LGraphCanvas.link_type_colors[displayType];

                let widgetConfig;
                let targetWidget;
                let widgetType;
                // Update the types of each node
                for (const node of updateNodes) {
                    // If we dont have an input type we are always wildcard but we'll show the output type
                    // This lets you change the output link to a different type and all nodes will update
                    if (!(node.outputs && node.outputs[index])) continue
                    node.outputs[index].type = inputType || "*";
                    node.__outputType = displayType;
                    node.outputs[index].name = node.properties.showOutputText ? displayType : "";
                    // node.size = node.computeSize();
                    // node.applyOrientation();

                    for (const l of node.outputs[index].links || []) {
                        const link = app.graph.links[l];
                        if (link) {
                            link.color = color;

                            if (app.configuringGraph) continue;
                            const targetNode = app.graph.getNodeById(link.target_id);
                            const targetInput = targetNode.inputs?.[link.target_slot];
                            if (targetInput?.widget) {
                                const config = getWidgetConfig(targetInput);
                                if (!widgetConfig) {
                                    widgetConfig = config[1] ?? {};
                                    widgetType = config[0];
                                }
                                if (!targetWidget) {
                                    targetWidget = targetNode.widgets?.find((w) => w.name === targetInput.widget.name);
                                }

                                const merged = mergeIfValid(targetInput, [config[0], widgetConfig]);
                                if (merged.customConfig) {
                                    widgetConfig = merged.customConfig;
                                }
                            }
                        }
                    }
                }

                for (const node of updateNodes) {
                    if (!(node.inputs && node.inputs[index])) continue
                    if (widgetConfig && outputType) {
                        node.inputs[index].widget = { name: "value" };
                        setWidgetConfig(node.inputs[index], [widgetType ?? displayType, widgetConfig], targetWidget);
                    } else {
                        setWidgetConfig(node.inputs[index], null);
                    }
                }

                if (inputNode && inputNode.inputs[index]) {
                    const link = app.graph.links[inputNode.inputs[index].link];
                    if (link) {
                        link.color = color;
                        if (node.outputs?.[index]) {
                            node.outputs[index].name = inputNode.__outputType || inputNode.outputs[index].type;

                            // allows bypass
                            node.inputs[index] = Object.assign(node.inputs[0], {...node.inputs[index], name: '*', type: node.outputs[index].name});
                        }
                    }
                }
            }
            // ================= Adapted from rerouteNode.js =================
        }

        // Add extra MenuOptions for
        // ClipTextEncode++ and Settings node
        if (ids1.has(nodeType) || inGroupNode || ids2.has(nodeType) || inGroupNode2) {
            // Save the original options
            const getExtraMenuOptions = node.getExtraMenuOptions;

            node.getExtraMenuOptions = function (_, options) {
                // Call the original function for the default menu options
                const r = getExtraMenuOptions ? getExtraMenuOptions.apply(this, arguments) : undefined;
                let customOptions = []
                node.setDirtyCanvas(true, true);
                // if (!r) return r;

                if (ids2.has(nodeType) || inGroupNode2) {
                    // Clean up MenuOption
                    const hiddenWidgets = node.widgets.filter(w => w.type === HIDDEN_TAG || w.heading || w.info)
                    let filtered = false
                    let wo = options.filter(o => o === null || (o && !hiddenWidgets.some(w => {const i = o.content.includes(`Convert ${w.name} to input`); if(i) filtered = i; return i} )))
                    options.splice(0, options.length, ...wo);

                    if(hiddenWidgets.length !== node.widgets.length) {
                        customOptions.push(null) // seperator
                        const d=function(_node) {
                            const extra = _node.widgets.find(w => w.name.endsWith('extra'))
                            let extra_data = extra._value
                            // extra_data.show_descriptions = !extra_data.show_descriptions
                            extra._value = {...extra._value, show_descriptions: !extra_data.show_descriptions}
                            widgetLogicSettings(_node)
                        }
                        const h=function(_node) {
                            const extra = _node.widgets.find(w => w.name.endsWith('extra'))
                            let extra_data = extra._value
                            extra._value = {...extra._value, show_headings: !extra_data.show_headings}
                            // extra_data.show_headings = !extra_data.show_headings
                            widgetLogicSettings(_node)
                        }
                        customOptions.push(create_custom_option("Hide/show all headings", h.bind(this, node)))
                        customOptions.push(create_custom_option("Hide/show all descriptions", d.bind(this, node)))
                    }
                }

                if (ids1.has(nodeType) || inGroupNode) {
                    // Dynamic MenuOption depending on the widgets
                    const content_hide_show = "Hide/show ";
                    // const whWidgets = node.widgets.filter(w => w.name === 'width' || w.name === 'height')
                    const hiddenWidgets = node.widgets.filter(w => w.type === HIDDEN_TAG)
                    // doesn't take GroupNode into account
                    // const with_SDXL = node.widgets.find(w => w.name.endsWith('with_SDXL'))
                    const parser = node.widgets.find(w => w.name.endsWith('parser'))
                    const in_comfy = parser?.value?.includes?.("comfy")
                    let ws = widgets.map(widget_name => create_custom_option(content_hide_show + widget_name, toggleMenuOption.bind(this, node, widget_name)))
                    ws = ws.filter((w) => (in_comfy && parser.value !== 'comfy' && w.content.includes('mean_normalization')) || (in_comfy && w.content.includes('with_SDXL')) || !in_comfy )
                    // customOptions.push(null) // seperator
                    customOptions.push(...ws)

                    let wo = options.filter(o => o === null || (o && !hiddenWidgets.some(w => o.content.includes(`Convert ${w.name} to input`))))
                    const width = node.widgets.find(w => w.name.endsWith('width'))
                    const height = node.widgets.find(w => w.name.endsWith('height'))
                    if (width && height) {
                        const width_type = width.type.toLowerCase()
                        const height_type = height.type.toLowerCase()
                        if (!(width_type.includes('number') || width_type.includes('int') || width_type.includes('float') ||
                        height_type.includes('number') || height_type.includes('int') || height_type.includes('float')))
                            wo = wo.filter(o => o === null || (o && !o.content.includes('Swap width/height')))
                    }
                    options.splice(0, options.length, ...wo);
                }
                // options.unshift(...customOptions); // top
                options.splice(options.length - 1, 0, ...customOptions)
                // return r;
            }
        }
    }
});

const placeholder_opacity = '0.75'
const style = document.createElement('style');
const createCssText = (x) => `.smZ-custom-textarea::-${x} {color: inherit; opacity: ${placeholder_opacity}}`
style.appendChild(document.createTextNode(createCssText('webkit-input-placeholder'))) // Only WebKit
style.appendChild(document.createTextNode(createCssText('ms-input-placeholder'))) // Only IE
style.appendChild(document.createTextNode(createCssText('moz-placeholder'))) // Only Firefox
document.head.appendChild(style);