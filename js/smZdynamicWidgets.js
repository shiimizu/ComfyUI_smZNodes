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
	if (widget.name === 'parser') {
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
}

const getSetWidgets = ['parser', 'mean_normalization', 'multi_conditioning']

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

app.registerExtension({
	name: "comfy.smZ.dynamicWidgets",
	beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeType.title == "CLIP Text Encode++") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                let that = this;
                const r = onNodeCreated ? onNodeCreated.apply(that, arguments) : undefined;
                // console.log(that.getExtraMenuOptions)

                // Save the original options
                const getBaseMenuOptions = that.getExtraMenuOptions;
                // Add new options
                that.getExtraMenuOptions = function(_, options) {
                    // Call the original function for the default menu options
                    getBaseMenuOptions.call(this, _, options);

                    let create_custom_option = (content, widget_name) => ({
                        content: content,
                        callback: () => {
                            let widget = findWidgetByName(this, widget_name)
                            if (!widget || doesInputWithNameExist(this, widget.name)) return;
                            if (!origProps[widget.name]) {
                                origProps[widget.name] = { origType: widget.type, origComputeSize: widget.computeSize };
                            }
                            toggleWidget(this, widget, !(widget.type === origProps[widget.name].origType))
                            this.setDirtyCanvas(true);
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
