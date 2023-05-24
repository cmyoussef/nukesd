import os
import subprocess
import threading
from datetime import datetime

import nuke

python_exe = r"D:\stable-diffusion\stable-diffusion-webui\venv\Scripts\python.exe"
script_path = r"D:\stable-diffusion\stable-diffusion-integrator\nukesd\sdexecuters\sdexecutor.py"

model_not_support_controlNet = ["stabilityai/stable-diffusion-2-1"]


class ExecuteThread(threading.Thread):
    def __init__(self, args, callback):
        threading.Thread.__init__(self)
        self.args = args
        self.callback = callback

    def run(self):
        python_exe = self.args.pop('python_exe')
        script_path = self.args.pop('script_path')

        cmd = [python_exe, script_path]

        for key, value in self.args.items():
            cmd.append(f'--{key}')
            cmd.append(str(value))

        subprocess.call(cmd)

        nuke.executeInMainThread(self.callback, args=(self.args['output'],))


def find_node_in_group(node_name, group):
    print("find_node_in_group")
    print(node_name, group.name())
    for node in group.nodes():
        if node.name() == node_name:
            print(node_name, " is found")
            return node
    return None


class SDGizmoInstance:

    @property
    def input_node(self):
        return find_node_in_group("Input1", self.gizmo) or nuke.nodes.Input()

    @property
    def input_nodes(self):
        return [find_node_in_group(f"Input{i+1}", self.gizmo) or nuke.nodes.Input() for i in range(3)]

    @property
    def read_node(self):
        return find_node_in_group("Read1", self.gizmo) or nuke.createNode('Read', inpanel=False)

    @property
    def output_node(self):
        return find_node_in_group("Output1", self.gizmo) or nuke.nodes.Output()

    @staticmethod
    def create_controlNet_menu(menuName, niceName=None):
        niceName = niceName or menuName
        cn = nuke.Enumeration_Knob(menuName, niceName, ["None",
                                                        "lllyasviel/control_v11p_sd15_canny",
                                                        "lllyasviel/control_v11e_sd15_ip2p",
                                                        "lllyasviel/control_v11p_sd15_inpaint",
                                                        "lllyasviel/control_v11p_sd15_mlsd",
                                                        "lllyasviel/control_v11f1p_sd15_depth",
                                                        "lllyasviel/control_v11p_sd15_normalbae",
                                                        "lllyasviel/control_v11p_sd15_seg",
                                                        "lllyasviel/control_v11p_sd15_lineart",
                                                        "lllyasviel/control_v11p_sd15s2_lineart_anime",
                                                        "lllyasviel/control_v11p_sd15_openpose",
                                                        "lllyasviel/control_v11p_sd15_scribble",
                                                        "lllyasviel/control_v11p_sd15_softedge"
                                                        "lllyasviel/control_v11e_sd15_shuffle"
                                                        ])
        return cn

    @property
    def controlNet_menus(self):
        controlNetList = []
        for i in range(3):
            cn = self.gizmo.knob(f'controlNet_menu{i+1}')
            if not cn:
                cn = self.create_controlNet_menu(f'controlNet_menu{i+1}', f'ControlNet Menu{i+1}')
                self.gizmo.addKnob(cn)

            format_mult = self.gizmo.knob(f'cnFormatMult{i+1}')
            if not format_mult:
                format_mult = nuke.Double_Knob(f'cnFormatMult{i+1}', f'CN Format Mult{i+1}')
                format_mult.setValue(1)
                self.gizmo.addKnob(format_mult)
            controlNetList.append((cn, format_mult))
        return controlNetList

    @property
    def model_menu(self):
        model_menu = self.gizmo.knob('model_menu')
        if not model_menu:
            model_menu = nuke.Enumeration_Knob('model_menu', 'Model', ["runwayml/stable-diffusion-v1-5",
                                                                       "stalkeryga/f222",
                                                                       "dreamlike-art/dreamlike-photoreal-2.0",
                                                                       "SG161222/Realistic_Vision_V2.0",
                                                                       "prompthero/openjourney-v4",
                                                                       "stabilityai/stable-diffusion-2-1",
                                                                       "alkzar90/ppaine-landscape",
                                                                       "ckpt/analog-madness-realistic-model",
                                                                       'wdkwdkwdk/chinese_jewelry_fintune'])
            self.gizmo.addKnob(model_menu)
        return model_menu

    @property
    def prompt(self):
        prompt = self.gizmo.knob('prompt')
        if not prompt:
            prompt = nuke.Multiline_Eval_String_Knob('prompt', 'Prompt',
                                                     "ancient egypt pharaohs shiny super clean silver earrings on ")
            self.gizmo.addKnob(prompt)
        return prompt

    @property
    def negative_prompt(self):
        negative_prompt = self.gizmo.knob('negative_prompt')
        if not negative_prompt:
            negative_prompt = nuke.Multiline_Eval_String_Knob('negative_prompt', 'Negative',
                                                              "cartoon ")
            self.gizmo.addKnob(negative_prompt)
        return negative_prompt

    @property
    def format_menu(self):
        format_menu = self.gizmo.knob('format_menu')
        if not format_menu:
            format_menu = nuke.Format_Knob('format_menu', 'Format')
            self.gizmo.addKnob(format_menu)
        return format_menu

    @property
    def format_mult(self):
        format_mult = self.gizmo.knob('formatMult')
        if not format_mult:
            format_mult = nuke.Double_Knob('formatMult', 'Format Mult')
            format_mult.setValue(1)
            self.gizmo.addKnob(format_mult)
        return format_mult

    @property
    def seed(self):
        seed = self.gizmo.knob('seed')
        if not seed:
            seed = nuke.Int_Knob('seed', 'Seed', -1)
            seed.setValue(-1)
            self.gizmo.addKnob(seed)
        return seed

    @property
    def num_inference_steps(self):
        num_inference_steps = self.gizmo.knob('num_inference_steps')
        if not num_inference_steps:
            num_inference_steps = nuke.Int_Knob('num_inference_steps', 'Number of Inference Steps', 20)
            num_inference_steps.setValue(20)
            self.gizmo.addKnob(num_inference_steps)
        return num_inference_steps

    @property
    def directory_knob(self):
        directory_knob = self.gizmo.knob('output_dir')
        if not directory_knob:
            directory_knob = nuke.File_Knob("output_dir", "Output Directory")
            # Set directory_knob value and add other knobs...
            directory_knob.setValue(os.path.join(os.path.expanduser("~"), 'nuke-stable-diffusion'))
            self.gizmo.addKnob(directory_knob)
        return directory_knob

    @property
    def generate_tab_knob(self):
        if not self.gizmo.knob('generate'):
            # generate tab
            generate_tab_knob = nuke.Tab_Knob("generate", 'Generate')
            self.gizmo.addKnob(generate_tab_knob)
        return self.gizmo.knob('generate')

    @property
    def controlNet_tab_knob(self):
        if not self.gizmo.knob('controlNet'):
            # generate tab
            generate_tab_knob = nuke.Tab_Knob("controlNet", 'Control Net')
            self.gizmo.addKnob(generate_tab_knob)
        return self.gizmo.knob('controlNet')

    @property
    def settings_tab_knob(self):
        if not self.gizmo.knob('settings'):
            # generate tab
            settings = nuke.Tab_Knob("settings", 'Settings')
            self.gizmo.addKnob(settings)
        return self.gizmo.knob('settings')

    @property
    def button(self):
        if not self.gizmo.knob('button'):
            button = nuke.PyScript_Knob('button', 'Execute')
            self.gizmo.addKnob(button)
        self.gizmo.knob('button').setCommand('sd_exec_fun()')
        return self.gizmo.knob('button')

    def __init__(self, gizmo=None):
        if gizmo:
            self.gizmo = nuke.toNode(gizmo)
            if self.gizmo is None:
                raise ValueError("No such gizmo named '{}'".format(gizmo))
        else:
            self.gizmo = nuke.createNode('Group', inpanel=False)
            self.gizmo.setName('Nuke_SD')

        self.gizmo.begin()

        # nodes inside gizmo
        _ = self.input_nodes
        self.output_node.setInput(0, self.read_node)

        # generate tab
        _ = self.generate_tab_knob

        _ = self.model_menu

        # Add Prompt
        _ = self.prompt

        # Negative prompts multi-line edit
        _ = self.negative_prompt

        # Format dropdown menu
        _ = self.format_menu

        # formatMult
        _ = self.format_mult

        # seed
        _ = self.seed

        # Number of Inference Steps knob
        _ = self.num_inference_steps

        # Add other knobs...
        _ = self.button

        # control net tab
        _ = self.controlNet_tab_knob
        _ = self.controlNet_menus

        # settings tab
        _ = self.settings_tab_knob

        _ = self.directory_knob

        self.gizmo.end()

    def writeInput(self, outputPath, i=0):
        # Use a Write node to save the input image
        write_node = nuke.nodes.Write()
        write_node.knob('file').setValue(outputPath.replace('\\', '/'))
        write_node.knob('channels').setValue('rgb')
        write_node.setInput(0, self.input_nodes[i])

        # Set the file_type and datatype
        write_node.knob('file_type').setValue('png')
        write_node.knob('datatype').setValue('8 bit')

        # Execute the Write node
        nuke.execute(write_node.name(), nuke.frame(), nuke.frame())

        # Remove the Write node
        nuke.delete(write_node)

    @staticmethod
    def check_and_correct_size(size):
        size = int(size)
        if size % 8 != 0:
            corrected_size = (size // 8) * 8
            nuke.warning(f"Warning: Size {size} is not a multiple of 8. Changing to {corrected_size}.")
            return corrected_size
        return size

    def sd_on_execute(self):
        model = self.model_menu.value()

        output_dir = self.directory_knob.value()

        output_main_dir = os.path.join(output_dir, self.gizmo.name())
        output_dir_path = os.path.join(output_main_dir, 'output', model.rsplit('/', 1)[-1])
        os.makedirs(output_dir_path, exist_ok=True)
        filename = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        output_file_path = os.path.join(output_dir_path, filename)

        # Get the value from the format knob
        format_value = self.format_menu.value()
        width = self.check_and_correct_size(format_value.width() * self.format_mult.value())
        height = self.check_and_correct_size(format_value.height() * self.format_mult.value())
        args = {
            'model': self.model_menu.value(),
            'prompt': self.prompt.value(),
            'negative_prompt': self.negative_prompt.value(),
            'num_inference_steps': int(self.num_inference_steps.value()),
            'width': width,
            'height': height,
            'seed': int(self.seed.value()),
            'output': output_file_path,
            'python_exe': python_exe,
            'script_path': script_path,
        }


        controlNets = []
        # Check if there's an image connected to the input
        for i, (controlNet_menu, controlNet_weight) in enumerate(self.controlNet_menus):
            controlNet = controlNet_menu.value()
            if controlNet != 'None' and model not in model_not_support_controlNet:
                cn_dir_path = os.path.join(output_main_dir, 'controlNet')
                os.makedirs(cn_dir_path, exist_ok=True)
                cn_file_path = os.path.join(cn_dir_path, controlNet.rsplit('/', 1)[-1] + f'{i+1}.png')

                self.writeInput(cn_file_path, i)
                # Add the control_net argument
                controlNets.append([controlNet, cn_file_path, controlNet_weight.value()])

        args['controlNet'] = controlNets
        # Define a callback to be run after the external script finishes
        def callback(output_file):
            # Start editing the Gizmo
            self.gizmo.begin()
            # Set Read node file to the temporary image file
            self.read_node.knob('file').setValue(output_file.replace('\\', '/'))
            self.read_node.knob('reload').execute()  # Reload the Read node
            self.read_node.knob('file').setValue(output_file.replace('\\', '/'))
            self.read_node.knob('reload').execute()  # Reload the Read node
            # read_node.knob('file').setValue(str(output_file.replace('\\', '/')))
            # End editing the Gizmo
            self.gizmo.end()
        print('-' * 100)
        print("ExecuteSD: ", args)
        # Start the thread
        thread = ExecuteThread(args, callback)
        thread.start()

    def callback(self, output_file):
        # Start editing the Gizmo
        self.gizmo.begin()
        # Set Read node file to the temporary image file
        self.read_node.knob('file').setValue(output_file.replace('\\', '/'))
        # Reload the Read node
        self.read_node.knob('reload').execute()
        # End editing the Gizmo
        self.gizmo.end()


# to declare it once
try:
    gizmo_instances
except NameError:
    gizmo_instances = {}


def create_sd_gizmo_instance():
    sdInstance = SDGizmoInstance()
    gizmo_instances[sdInstance.gizmo.name()] = sdInstance


def sd_exec_fun():
    global gizmo_instances
    try:
        gizmo_instances
    except NameError:
        gizmo_instances = {}
    gizmoName = nuke.thisNode().name()
    if gizmoName not in gizmo_instances:
        gizmo_instances[gizmoName] = SDGizmoInstance(gizmoName)

    gizmo_instances[gizmoName].sd_on_execute()


if __name__ == '__main__':
    create_sd_gizmo_instance()
