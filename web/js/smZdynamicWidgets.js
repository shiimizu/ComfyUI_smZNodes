import { app } from "/scripts/app.js";
// import { app } from "../../../scripts/app.js";
// import { ComfyWidgets } from "../../../scripts/widgets.js";

const ids1 = new Set(["smZ CLIPTextEncode"])
const ids2 = new Set(["smZ Settings"])
const widgets = ['mean_normalization', 'multi_conditioning', 'use_old_emphasis_implementation', 'with_SDXL']
const widgets_sdxl = ['ascore', 'width', 'height', 'crop_w', 'crop_h', 'target_width', 'target_height', 'text_g', 'text_l']
const getSetWidgets = new Set(['parser', 'with_SDXL'])

let origProps = {};
const HIDDEN_TAG = "smZhidden"

const findWidgetByName = (node, name) => node.widgets.find((w) => w.name === name);
const findWidgetsByName = (node, name) => node.widgets.filter((w) => w.name.endsWith(name));

const doesInputWithNameExist = (node, name) => node.inputs ? node.inputs.some((input) => input.name === name) : false;

// round in increments of x, with an offset
function round(number, increment = 10, offset = 0) {
    return Math.ceil((number - offset) / increment ) * increment + offset;
}

function toggleWidget(node, widget, show = false, suffix = "") {
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

function widgetLogic(node, widget) {
    const wname = widget.name
    if (wname.endsWith("parser")) {
        const in_comfy = widget.value.includes("comfy")
        toggleMenuOption(node, ['multi_conditioning', wname], !in_comfy)
        toggleMenuOption(node, ['mean_normalization', wname], widget.value !== "comfy")
        if (in_comfy) {
            toggleMenuOption(node, ['use_old_emphasis_implementation', wname], false)
        }
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

function getGroupNodeConfig(node) {
    let ls = []
    let nodeData = node.constructor?.nodeData
    if (nodeData) {
        for(let sym of Object.getOwnPropertySymbols(nodeData) ) {
            const o = nodeData[sym]; 
            if (!o) continue;
            ls.push(o)
        }
    }
    return ls
}

function getSetters(node) {
    if (node.widgets) {
        let gncl = getGroupNodeConfig(node)
        for (const w of node.widgets) {
            for (const gsw of [...getSetWidgets]) {
                if (!w.name.endsWith(gsw)) continue;
                let shouldBreak = false
                for (const gnc of gncl) {
                    const nwmap = gnc.newToOldWidgetMap[w.name]
                    if (nwmap && !(nwmap.node.type === [...ids1][0] && nwmap.inputName === gsw))
                        shouldBreak = true
                }
                if (shouldBreak) break;
                widgetLogic(node, w);
                w._value = w.value

                Object.defineProperty(w, 'value', {
                    get() {
                        return w._value;
                    },
                    set(newVal) {
                        w._value = newVal
                        widgetLogic(node, w);
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
    }
}

function toggleMenuOption(node, widget_arr, _show = null, perform_action = true) {
    const gncl = getGroupNodeConfig(node)
    const [widget_name, companion_widget_name] = Array.isArray(widget_arr) ? widget_arr : [widget_arr]
    let nwname = widget_name
    // Use companion_widget_name to get the correct widget with the new name
    if (companion_widget_name) {
        for (const gnc of gncl) {
            const omap = Object.values(gnc.oldToNewWidgetMap).find(x => Object.values(x).find(z => z === companion_widget_name))
            const tmp2 = omap[widget_name]
            if (tmp2) nwname = tmp2;
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

app.registerExtension({
    name: "comfy.smZ.dynamicWidgets",

    /**
     * Called when a node is created. Used to add menu options to nodes.
     * @param node The node that was created.
     * @param app The app.
     */
    nodeCreated(node) {
        const nodeType = node.type || node.constructor?.type
        let inGroupNode = false
        let inGroupNode2 = false
        let innerNodes = node.getInnerNodes?.()
        if (innerNodes) {
            for (const inode of innerNodes) {
                const _nodeType = inode.type || inode.constructor?.type
                if (ids1.has(_nodeType))
                    inGroupNode = ids1.has(_nodeType)
                if (inGroupNode)
                    ids1.add(nodeType) // GroupNode's type
                if (ids2.has(_nodeType))
                    inGroupNode2 = ids2.has(_nodeType)
                if (inGroupNode2)
                    ids2.add(nodeType) // GroupNode's type
            }
        }
        // let nodeData = node.constructor?.nodeData
        // if (nodeData) {
        //     for(let sym of Object.getOwnPropertySymbols(nodeData) ) {
        //     const nds = nodeData[sym];
        //     if (nds) {
        //         inGroupNode=true
        //         inGroupNode2=true
        //         break
        //     }
        //     }
        // }
        // ClipTextEncode++ node
        if (ids1.has(nodeType) || inGroupNode) {
            node.widgets.forEach(w => w._name = w.name)

            getSetters(node)

            // Reduce initial node size cause of SDXL widgets
            // node.setSize([node.size[0], Math.max(node.size[1]/1.5, 220)])
            node.setSize([node.size[0], 220])
        }
        // Settings node
        if (ids2.has(nodeType) || inGroupNode2) {
            node.serialize_widgets = true

            const onConfigure = node.onConfigure;
            node.onConfigure = function(o) {
                const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
                const w = this.widgets.find(w => w.name === 'extra')
                let value = null
                try {
                    value = JSON.parse(w.value);
                } catch (error) {
                    // when node definitions change due to an update or some other error
                    value = {"show":true}
                }
                toggle_all_settings_desc_widgets(this, value.show)
                return r;
            }

            // Styling.
            node.widgets.forEach(function(w) {
                w._name = w.name
                if (w.name.includes('ㅤ')) {
                    w.heading = true
                } else if (w.name.includes('info')) {
                    w.info = true
                    w.inputEl.disabled = true;
                    w.inputEl.readOnly = true;
                    w.inputEl.style.opacity = 0.75;
                    w.inputEl.style.alignContent = 'center';
                    w.inputEl.style.textAlign = 'center';
                }
            })
            // Hide `extra` widget
            toggleMenuOption(node, 'extra', false)
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
                    const content_hide_show = "Hide/show all descriptions";
                    customOptions.push(null) // seperator
                    customOptions.push(create_custom_option(content_hide_show, toggle_all_settings_desc_widgets.bind(this, node)))
    
                    // Alternate way to cleanup MenuOption
                    const toHideWidgets = node.widgets.filter(w => w.name.includes('ㅤ') || w.name.includes('info') || w.name.includes('extra'))
                    const wo = options.filter(o => o === null || (o && !toHideWidgets.some(w => o.content.includes(`Convert ${w.name} to input`))))
                    options.splice(0, options.length, ...wo);
                }

                if (ids1.has(nodeType) || inGroupNode) {
                    // Dynamic MenuOption depending on the widgets
                    const content_hide_show = "Hide/show ";
                    // const whWidgets = node.widgets.filter(w => w.name === 'width' || w.name === 'height')
                    const hiddenWidgets = node.widgets.filter(w => w.type === HIDDEN_TAG)
                    // doesn't take GroupNode into account
                    const with_SDXL = node.widgets.find(w => w.name === 'with_SDXL')
                    const parser = node.widgets.find(w => w.name === 'parser')
                    const in_comfy = parser.value.includes("comfy")
                    let ws = widgets.map(widget_name => create_custom_option(content_hide_show + widget_name, toggleMenuOption.bind(this, node, widget_name)))
                    ws = ws.filter((w) => (in_comfy && parser.value !== 'comfy' && w.content.includes('mean_normalization')) || (in_comfy && w.content.includes('with_SDXL')) || !in_comfy )
                    // customOptions.push(null) // seperator
                    customOptions.push(...ws)

                    let wo = options.filter(o => o === null || (o && !hiddenWidgets.some(w => o.content.includes(`Convert ${w.name} to input`))))
                    const width = node.widgets.find(w => w.name === 'width')
                    const height = node.widgets.find(w => w.name === 'height')
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
