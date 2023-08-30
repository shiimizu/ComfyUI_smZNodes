import { app } from "/scripts/app.js";

const id = "CLIP Text Encode++"
let origProps = {};

const findWidgetByName = (node, name) => node.widgets.find((w) => w.name === name);

const doesInputWithNameExist = (node, name) => node.inputs ? node.inputs.some((input) => input.name === name) : false;

function toggleWidget(node, widget, show = false, suffix = "") {
	if (!widget || doesInputWithNameExist(node, widget.name)) return;
	if (!origProps[widget.name]) {
		origProps[widget.name] = { origType: widget.type, origComputeSize: widget.computeSize };
	}
	const origSize = node.size;

	widget.type = show ? origProps[widget.name].origType : "smZhidden" + suffix;
	widget.computeSize = show ? origProps[widget.name].origComputeSize : () => [0, -4];

	widget.linkedWidgets?.forEach(w => toggleWidget(node, w, ":" + widget.name, show));

	const height = show ? Math.max(node.computeSize()[1], origSize[1]) : node.size[1];
	node.setSize([node.size[0], height]);
}

function widgetLogic(node, widget) {
    if (widget.name === 'parser') {
        const in_comfy = widget.value.includes("comfy")
        toggleMenuOption(node, 'multi_conditioning', !in_comfy)
        toggleMenuOption(node, 'use_CFGDenoiser', !in_comfy)
        toggleMenuOption(node, 'mean_normalization', widget.value !== "comfy")
        if (in_comfy)
            toggleMenuOption(node, 'use_old_emphasis_implementation', false)
	} else if (widget.name === 'with_SDXL') {
        const ws = ['ascore', 'width', 'height', 'crop_w', 'crop_h', 'target_width', 'target_height', 'text_g', 'text_l']
        toggleMenuOption(node, 'text', !widget.value)

        // Resize node to half when widget is set to false
        if (!widget.value){
            node.setSize([node.size[0], node.size[0]/2.0])
        }

        // Show list of widgets if sdxl widget value is true and vice-versa
        for (const w of ws) {
            toggleMenuOption(node, w, widget.value)
        }

        // Keep showing the widget if it's enabled
        if (widget.value) {
            toggleMenuOption(node, widget.name, true)
        }
	}
}

const widgets = ['parser', 'mean_normalization', 'multi_conditioning', 'use_old_emphasis_implementation', 'use_CFGDenoiser', 'with_SDXL']
const getSetWidgets = ['parser', 'with_SDXL']

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
                // For some reason this fixes its toggling
                if (w.name === 'with_SDXL' && w) {
                    toggleMenuOption(node, 'with_SDXL')
                    w.init = true
                }
			}
		}
}

function toggleMenuOption(node, widget_name, val=null, perform_action = true) {
    let widget = findWidgetByName(node, widget_name)
    if (!widget || doesInputWithNameExist(node, widget.name)) return;
    if (!origProps[widget.name]) {
        origProps[widget.name] = { origType: widget.type, origComputeSize: widget.computeSize };
    }
    const show = (widget.type === origProps[widget.name].origType)
    if (perform_action) {
        toggleWidget(node, widget, val !== null ? val : !show)
        node.setDirtyCanvas(true);
    }
    return show
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
		if (nodeType.title == id) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                let that = this;
                const r = onNodeCreated ? onNodeCreated.apply(that, arguments) : undefined;

                // Reduce node size cause of SDXL widgets
                that.setSize([that.size[0], that.size[0]/2.0])

                // Save the original options
                const getBaseMenuOptions = that.getExtraMenuOptions;
                // Add new options
                that.getExtraMenuOptions = function(_, options) {
                    // Call the original function for the default menu options
                    getBaseMenuOptions.call(this, _, options);

                    let create_custom_option = (content, widget_name) => ({
                        content: content,
                        callback: () => {
                            toggleMenuOption(this, widget_name)
                        },
                    })

                    const content_hide_show = "Hide/show ";
                    const customOptions = widgets.map(widget_name => create_custom_option(content_hide_show + widget_name, widget_name))

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
		if (node.getTitle() == id) {
			getSetters(node)
		}
	}
});
