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
		if (widget.value.includes("comfy")) {
			toggleWidget(node, findWidgetByName(node, 'multi_conditioning'))
			toggleWidget(node, findWidgetByName(node, 'use_old_emphasis_implementation'))
			toggleWidget(node, findWidgetByName(node, 'use_CFGDenoiser'))
		} else {
			toggleWidget(node, findWidgetByName(node, 'multi_conditioning'), true)
			// toggleWidget(node, findWidgetByName(node, 'use_old_emphasis_implementation'), true)
			toggleWidget(node, findWidgetByName(node, 'use_CFGDenoiser'), true)
		}
		if (widget.value === "comfy") {
            toggleWidget(node, findWidgetByName(node, 'mean_normalization'))
		} else {
			toggleWidget(node, findWidgetByName(node, 'mean_normalization'), true)
		}
	} else if (widget.name === 'with_SDXL') {
        if (!widget.init) {
            widget.init = true
            toggleWidget(node, findWidgetByName(node, 'with_SDXL'))
        }
        if (!widget.value) {
			toggleWidget(node, findWidgetByName(node, 'ascore'))
			toggleWidget(node, findWidgetByName(node, 'width'))
			toggleWidget(node, findWidgetByName(node, 'height'))
			toggleWidget(node, findWidgetByName(node, 'crop_w'))
			toggleWidget(node, findWidgetByName(node, 'crop_h'))
			toggleWidget(node, findWidgetByName(node, 'target_width'))
			toggleWidget(node, findWidgetByName(node, 'target_height'))
			toggleWidget(node, findWidgetByName(node, 'text_g'))
			toggleWidget(node, findWidgetByName(node, 'text_l'))
		} else {
			toggleWidget(node, findWidgetByName(node, 'ascore'), true)
			toggleWidget(node, findWidgetByName(node, 'width'), true)
			toggleWidget(node, findWidgetByName(node, 'height'), true)
			toggleWidget(node, findWidgetByName(node, 'crop_w'), true)
			toggleWidget(node, findWidgetByName(node, 'crop_h'), true)
			toggleWidget(node, findWidgetByName(node, 'target_width'), true)
			toggleWidget(node, findWidgetByName(node, 'target_height'), true)
			toggleWidget(node, findWidgetByName(node, 'text_g'), true)
			toggleWidget(node, findWidgetByName(node, 'text_l'), true)
		}
	}
    // Keep showing the SDXL widget if the node is cloned
    if (widget.name === 'ascore') {
        const ascore_widget = findWidgetByName(node, 'ascore')
        const showing = !(ascore_widget.type == "smZhidden" )
        if (showing)
            toggleWidget(node, findWidgetByName(node, 'with_SDXL'), true)
    }
}

const widgets = ['parser', 'mean_normalization', 'multi_conditioning', 'use_old_emphasis_implementation', 'use_CFGDenoiser', 'with_SDXL']
const getSetWidgets = ['parser', 'ascore', 'with_SDXL']

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
			}
		}
}

function toggleMenuOption(node, widget_name) {
    let widget = findWidgetByName(node, widget_name)
    if (!widget || doesInputWithNameExist(node, widget.name)) return;
    if (!origProps[widget.name]) {
        origProps[widget.name] = { origType: widget.type, origComputeSize: widget.computeSize };
    }
    toggleWidget(node, widget, !(widget.type === origProps[widget.name].origType))
    node.setDirtyCanvas(true);
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

                // Hide SDXL widgets on init
                // const wname = getSetWidgets[2]
                // let wt = findWidgetByName(that, wname)
                // wt.value will always be false and doesInputWithNameExist()
                // doesn't work here since we're inside beforeRegisterNodeDef
                // if (wt && !wt.value)
                // if (wt)
                //     toggleMenuOption(that, wname)

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
