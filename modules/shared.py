import logging

log = logging.getLogger("sd")
options_templates = {}

class Options:
    data = None
    data_labels = options_templates
    typemap = {int: float}

    def __init__(self):
        self.data = {k: v.default for k, v in self.data_labels.items()}

    def __setattr__(self, key, value): # pylint: disable=inconsistent-return-statements
        if self.data is not None:
            if key in self.data or key in self.data_labels:
                # if cmd_opts.freeze:
                #     log.warning(f'Settings are frozen: {key}')
                #     return
                # if cmd_opts.hide_ui_dir_config and key in restricted_opts:
                #     log.warning(f'Settings key is restricted: {key}')
                #     return
                self.data[key] = value
                return
        return super(Options, self).__setattr__(key, value) # pylint: disable=super-with-arguments

    def __getattr__(self, item):
        if self.data is not None:
            if item in self.data:
                return self.data[item]
        if item in self.data_labels:
            return self.data_labels[item].default
        return super(Options, self).__getattribute__(item) # pylint: disable=super-with-arguments


opts = Options()
opts.data['prompt_attention'] = 'A1111 parser'
opts.data['comma_padding_backtrack'] = None
opts.data['prompt_mean_norm'] = True
opts.data["comma_padding_backtrack"] = 20
opts.data["clip_skip"] = None

