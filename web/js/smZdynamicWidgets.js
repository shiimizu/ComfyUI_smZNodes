import { app } from "/scripts/app.js";
// import { app } from "../../../scripts/app.js";
// import { ComfyWidgets } from "../../../scripts/widgets.js";

const ids = ["CLIP Text Encode++", "Settings (smZ)"]
const widgets = ['parser', 'mean_normalization', 'multi_conditioning', 'use_old_emphasis_implementation', 'with_SDXL']
const getSetWidgets = ['parser', 'with_SDXL']

let origProps = {};
const HIDDEN_TAG = "smZhidden"

const findWidgetByName = (node, name) => node.widgets.find((w) => (w._name !== undefined && w._name ? w._name : w.name) === name);

const doesInputWithNameExist = (node, name) => node.inputs ? node.inputs.some((input) => input.name === name) : false;

function toggleWidget(node, widget, show = false, suffix = "") {
    if (!widget || doesInputWithNameExist(node, widget.name)) return;
    if (!origProps[widget.name]) {
        origProps[widget.name] = { origType: widget.type, origComputeSize: widget.computeSize};
    }
    const origSize = node.size;

    widget.type = show ? origProps[widget.name].origType : HIDDEN_TAG + suffix;
    widget.computeSize = show ? origProps[widget.name].origComputeSize : () => [0, -4];

    widget.linkedWidgets?.forEach(w => toggleWidget(node, w, ":" + widget.name, show));

    const height = show ? Math.max(node.computeSize()[1], origSize[1]) : node.size[1];
    node.setSize([node.size[0], height]);
    if (show)
        delete widget.computedHeight;
    else
        widget.computedHeight = 0;
}

function widgetLogic(node, widget) {
    if (widget.name === 'parser') {
        const in_comfy = widget.value.includes("comfy")
        toggleMenuOption(node, 'multi_conditioning', !in_comfy)
        toggleMenuOption(node, 'mean_normalization', widget.value !== "comfy")
        if (in_comfy) {
            toggleMenuOption(node, 'use_old_emphasis_implementation', false)
        }
    } else if (widget.name === 'with_SDXL') {
        const ws = ['ascore', 'width', 'height', 'crop_w', 'crop_h', 'target_width', 'target_height', 'text_g', 'text_l']
        toggleMenuOption(node, 'text', !widget.value)

        // Resize node when widget is set to false
        if (!widget.value) {
            // Prevents resizing on init/webpage reload
            if(widget.init === false) {
                // Resize when set to false
                node.setSize([node.size[0], Math.max(102, node.size[1]/1.5)])
            }
        } else {
            // When enabled, set init to false
            widget.init = false
        }

        // Toggle sdxl widgets if sdxl widget value is true/false
        for (const w of ws) {
            toggleMenuOption(node, w, widget.value)
        }

        // Keep showing the widget if it's enabled
        if (widget.value && widget.type === HIDDEN_TAG) {
            toggleMenuOption(node, widget.name, true)
        }
    }
}


function getSetters(node) {
    if (node.widgets)
        for (const w of node.widgets) {
            if (getSetWidgets.includes(w.name)) {
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
                if (w.name === 'with_SDXL' && w) {
                    toggleMenuOption(node, 'with_SDXL')
                    w.init = true
                }
            }
        }
}

function toggleMenuOption(node, widget_name, _show = null, perform_action = true) {
    let widget = findWidgetByName(node, widget_name)
    const name_backup = widget.name
    // Empty names result in clean MenuOptions
    // Our real name could be in _name
    widget.name = widget_name ? widget_name : (widget._name ? widget._name : widget.name)
    if (!widget || doesInputWithNameExist(node, widget.name)) return;
    if (!origProps[widget.name]) {
        origProps[widget.name] = { origType: widget.type, origComputeSize: widget.computeSize};
    }
    const show = (widget.type === origProps[widget.name].origType)
    if (perform_action) {
        toggleWidget(node, widget, _show !== null ? _show : !show)
        node.setDirtyCanvas(true);
    }
    widget.name = name_backup
}

function toggle_all_settings_desc_widgets(node, widget_name = '', _show = null) {
    let found_widgets = node.widgets.filter((w) => (w._name ? w._name : w.name).includes('info'));
    let is_showing = true
    found_widgets.forEach(w => {
        if (w) {
            const name_backup = w.name
            w.name = w._name ? w._name : w.name
            toggleMenuOption(node, w.name, _show)
            is_showing = (w.type === origProps[w.name].origType)
            w.name = name_backup
        }
    });

    let w = node.widgets.find((w) => (w._name ? w._name : w.name) === 'extra');
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
        node.setSize([node.size[0], 0])
    }
}

function handleSettings(node) {
    if (!node.widgets) return;
    const w = node.widgets.find(w => (w._name ? w._name : w.name) === 'extra')
    w._name = (w._name ? w._name : w.name)
    w.name = ''
    // Hide `extra` widget
    toggleMenuOption(node, 'extra', false)
}

app.registerExtension({
    name: "comfy.smZ.dynamicWidgets",

    /**
     * Called after inputs, outputs, widgets, menus are added to the node given the node definition.
     * Used to add methods to nested nodes.
     * @param nodeType The ComfyNode object to be registered with LiteGraph.
     * @param nodeData The node definition.
     * @param app The app.
     * @returns {Promise<void>}
     */
    beforeRegisterNodeDef(nodeType, nodeData, app) {
        /**
         * Called after the node has been configured.
         * On initial configure of nodes hide all converted widgets
         * See web/extensions/core/widgetInputs.js
         * See litegraph.core.js
         * @param node The node that was created.
         */
        if (nodeType.title === ids[1]) {
            const origOnConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function() {
                const r = origOnConfigure ? origOnConfigure.apply(this, arguments) : undefined;
                    const w = this.widgets.find(w => (w._name ? w._name : w.name) === 'extra')
                    let value = null
                    try {
                        value =JSON.parse(w.value);
                    } catch (error) {
                        // when node definitions change due to an update or some other error
                        value = {"show":true}
                    }
                    toggle_all_settings_desc_widgets(this, null, value.show)
                    const ww = this.widgets.find(w => (w._name ? w._name : w.name) === 'extra')
                    return r;
                }
        }
        if (nodeType.title === ids[0] || nodeType.title === ids[1]) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                let that = this;

                let customOptions = []
                function create_custom_option(content, _callback) {
                    return {
                        content: content,
                        callback: () => _callback(),
                    }
                };

                if (nodeType.title === ids[1]) {
                    // Styling. Setting `name` cleans up the MenuOption
                    that.widgets.forEach(function(w, i) {
                        if (w.name.includes('ㅤ')) {
                            w._name = w.name
                            w.name = ''
                            w.heading = true
                        } else if (w.name.includes('info')) {
                            w._name = w.name
                            w.name = 'text'
                            w.info = true
                            w.inputEl.disabled = true;
                            w.inputEl.readOnly = true;
                            w.inputEl.style.opacity = 0.75;
                            w.inputEl.style.alignContent = 'center';
                            w.inputEl.style.textAlign = 'center';
                        }
                    })
                    const content_hide_show = "Hide/show all descriptions";
                    customOptions.length = 0
                    customOptions.push(create_custom_option(content_hide_show, toggle_all_settings_desc_widgets.bind(this, that)))
                    customOptions.push(null) // seperator

                } else if (nodeType.title === ids[0]) {
                    // Hide steps
                    toggleMenuOption(that, 'steps', false)

                    // Reduce initial node size cause of SDXL widgets
                    that.setSize([that.size[0], 220])
                    customOptions.length = 0
                    const content_hide_show = "Hide/show ";
                    customOptions.push(...widgets.map(widget_name => create_custom_option(content_hide_show + widget_name, toggleMenuOption.bind(this, that, widget_name))))
                    customOptions.push(null) // seperator
                };
                // Save the original options
                const getBaseMenuOptions = that.getExtraMenuOptions;
                // Add new options
                that.getExtraMenuOptions = function (_, options) {
                    // Call the original function for the default menu options
                    getBaseMenuOptions.call(this, _, options);
                    options.unshift(...customOptions);
                }
                return r;
            };
        }
    },

    /**
     * Called when a node is created. Used to add menu options to nodes.
     * @param node The node that was created.
     * @param app The app.
     */
    nodeCreated(node) {
        if (node.getTitle() == ids[0]) {
            getSetters(node)
        } 
        else if (node.getTitle() == ids[1]) {
            handleSettings(node)
        }
    }
});