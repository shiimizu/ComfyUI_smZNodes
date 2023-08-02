import { app } from "/scripts/app.js";

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
    if (widget.name === 'with_SDXL') {
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
	} else if (widget.name === 'parser') {
		if (widget.value.includes("comfy")) {
			toggleWidget(node, findWidgetByName(node, 'multi_conditioning'))
			toggleWidget(node, findWidgetByName(node, 'use_old_emphasis_implementation'))
			toggleWidget(node, findWidgetByName(node, 'use_CFGDenoiser'))
		} else {
			toggleWidget(node, findWidgetByName(node, 'multi_conditioning'), true)
			toggleWidget(node, findWidgetByName(node, 'use_old_emphasis_implementation'), true)
			toggleWidget(node, findWidgetByName(node, 'use_CFGDenoiser'), true)
		}
		if (widget.value === "comfy") {
			toggleWidget(node, findWidgetByName(node, 'mean_normalization'))
		} else {
			toggleWidget(node, findWidgetByName(node, 'mean_normalization'), true)
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

// const getSetWidgets = ['parser', 'mean_normalization', 'multi_conditioning', 'use_old_emphasis_implementation', 'use_CFGDenoiser', 'with_SDXL']
const getSetWidgets = ['parser', 'ascore', 'with_SDXL']

function getSetters(node) {
	if (node.widgets)
		for (const w of node.widgets) {
			if (getSetWidgets.includes(w.name)) {
				widgetLogic(node, w);
				let widgetValue = w.value;

				// Define getters and setters for widget values
				Object.defineProperty(w, 'value', {
					get() {
						return widgetValue;
					},
					set(newVal) {
						widgetValue = newVal;
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
	beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeType.title == "CLIP Text Encode++") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                let that = this;
                const r = onNodeCreated ? onNodeCreated.apply(that, arguments) : undefined;

                // Hide SDXL widgets on init
                const wname = getSetWidgets[2]
                let wt = findWidgetByName(that, wname)
                // wt.value will always be false and doesInputWithNameExist()
                // doesn't work here since we're inside beforeRegisterNodeDef
                if (wt && !wt.value)
                    toggleMenuOption(that, wname)

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

                    const customOptions = [
                        {
                            content: "Hide/show ",
                            widget_name: "multi_conditioning"
                        },
                        {
                            content: "Hide/show ",
                            widget_name: "use_old_emphasis_implementation"
                        },
                        {
                            content: "Hide/show ",
                            widget_name: "use_CFGDenoiser"
                        },
                        {
                            content: "Hide/show ",
                            widget_name: "with_SDXL"
                        },
                    ].map(x => create_custom_option(x.content + x.widget_name, x.widget_name))

                    options.unshift(...customOptions);
                }
                return r;
              };
		}
	},

	nodeCreated(node) {
		if (node.getTitle() == "CLIP Text Encode++") {
			getSetters(node)
		}
	}
});
