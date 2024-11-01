
from modules import shared, ui



def handle_save_settings(state, preset, extensions, show_controls, theme):
    contents = ui.save_settings(state, preset, extensions, show_controls, theme)
    return [
        contents,
        "settings.yaml",
        "./",
        gr.update(visible=True)
    ]


def set_interface_arguments(extensions, bool_active):
    shared.args.extensions = extensions

    bool_list = get_boolean_arguments()

    for k in bool_list:
        setattr(shared.args, k, False)
    for k in bool_active:
        setattr(shared.args, k, True)
        if k == 'api':
            shared.add_extension('openai', last=True)

    shared.need_restart = True


def get_boolean_arguments(active=False):
    exclude = shared.deprecated_args

    cmd_list = vars(shared.args)
    bool_list = sorted([k for k in cmd_list if type(cmd_list[k]) is bool and k not in exclude + ui.list_model_elements()])
    bool_active = [k for k in bool_list if vars(shared.args)[k]]

    if active:
        return bool_active
    else:
        return bool_list
