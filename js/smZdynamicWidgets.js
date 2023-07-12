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
		} else {
			toggleWidget(node, findWidgetByName(node, 'multi_conditioning'), true)
			toggleWidget(node, findWidgetByName(node, 'use_old_emphasis_implementation'), true)
		}
		if (widget.value === "comfy") {
			toggleWidget(node, findWidgetByName(node, 'mean_normalization'))
			toggleWidget(node, findWidgetByName(node, 'use_CFGDenoiser'))
		} else {
			toggleWidget(node, findWidgetByName(node, 'mean_normalization'), true)
			toggleWidget(node, findWidgetByName(node, 'use_CFGDenoiser'), true)
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
				const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
				this.getExtraMenuOptions = function(_, options) {
					options.unshift(
						{
							content: "Hide/show multi_conditioning",
							callback: () => {
                                let widget = findWidgetByName(this, 'multi_conditioning')
                                if (!widget || doesInputWithNameExist(this, widget.name)) return;
                                if (!origProps[widget.name]) {
                                    origProps[widget.name] = { origType: widget.type, origComputeSize: widget.computeSize };
                                }
                                toggleWidget(this, widget, !(widget.type === origProps[widget.name].origType))
							},
						},
						{
							content: "Hide/show use_old_emphasis_implementation",
							callback: () => {
                                let widget = findWidgetByName(this, 'use_old_emphasis_implementation')
                                if (!widget || doesInputWithNameExist(this, widget.name)) return;
                                if (!origProps[widget.name]) {
                                    origProps[widget.name] = { origType: widget.type, origComputeSize: widget.computeSize };
                                }
                                toggleWidget(this, widget, !(widget.type === origProps[widget.name].origType))
							},
						},
						{
							content: "Hide/show use_CFGDenoiser",
							callback: () => {
                                let widget = findWidgetByName(this, 'use_CFGDenoiser')
                                if (!widget || doesInputWithNameExist(this, widget.name)) return;
                                if (!origProps[widget.name]) {
                                    origProps[widget.name] = { origType: widget.type, origComputeSize: widget.computeSize };
                                }
                                toggleWidget(this, widget, !(widget.type === origProps[widget.name].origType))
							},
						},
					);
				}

				this.onRemoved = function () {
					// When removing this node we need to remove the input from the DOM
					for (let y in this.widgets) {
						if (this.widgets[y].canvas) {
							this.widgets[y].canvas.remove();
						}
					}
				};

				this.onSelected = function () {
					this.selected = true
				}
				this.onDeselected = function () {
					this.selected = false
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
