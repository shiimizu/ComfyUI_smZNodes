from .. import devices
def apply_single_hypernetwork(hypernetwork, context_k, context_v, layer=None):
    hypernetwork_layers = (hypernetwork.layers if hypernetwork is not None else {}).get(context_k.shape[2], None)

    if hypernetwork_layers is None:
        return context_k, context_v

    if layer is not None:
        layer.hyper_k = hypernetwork_layers[0]
        layer.hyper_v = hypernetwork_layers[1]

    context_k = devices.cond_cast_unet(hypernetwork_layers[0](devices.cond_cast_float(context_k)))
    context_v = devices.cond_cast_unet(hypernetwork_layers[1](devices.cond_cast_float(context_v)))
    return context_k, context_v


def apply_hypernetworks(hypernetworks, context, layer=None):
    context_k = context
    context_v = context
    for hypernetwork in hypernetworks:
        context_k, context_v = apply_single_hypernetwork(hypernetwork, context_k, context_v, layer)

    return context_k, context_v