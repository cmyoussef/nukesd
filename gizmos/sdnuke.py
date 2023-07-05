import inspect
import json
import os
import subprocess
import threading
import time
from datetime import datetime

import nuke

DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))

python_exe = r"D:\stable-diffusion\stable-diffusion-webui\venv\Scripts\python.exe"
script_path = os.path.join(DIR_PATH, 'sdexecuters', 'sdpipeline.py')

config_path = os.path.join(DIR_PATH, 'config', 'model_config.json')
with open(config_path, 'r') as f:
    model_config = json.load(f)

model_not_support_controlNet = model_config['model_not_support_controlNet']
controlNetModels = model_config['controlNetModels']


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


class StableDiffusionBase:
    _instances = {}  # Keep track of instances here
    models = model_config['models']
    unsupported_args = []

    def __new__(cls, gizmo=None):
        cls._instances.setdefault(cls.__name__, {})
        if gizmo and gizmo in cls._instances[cls.__name__]:
            instance = cls._instances[cls.__name__][gizmo]
        else:
            instance = super(StableDiffusionBase, cls).__new__(cls)
            cls._instances[cls.__name__][gizmo] = instance
        return instance

    def __init__(self, gizmo=None):

        if gizmo:
            self.gizmo = nuke.toNode(gizmo)
        else:
            self.gizmo = nuke.createNode('Group', inpanel=False)
            self.gizmo.setName(f'nuke_{self.__class__.__name__}')

        self.gizmo.begin()
        if not self.gizmo.knob("gizmo_class_type"):
            self.gizmo_class_type = nuke.String_Knob("gizmo_class_type", "Gizmo class type")
            self.gizmo_class_type.setValue(self.__class__.__name__)
            self.gizmo.addKnob(self.gizmo_class_type)
            self.gizmo_class_type.setVisible(False)

        self.gizmo.knob('User').setFlag(nuke.INVISIBLE)
        self.controlNet_count = model_config.get('controlNetCount', 3)
        self.gizmo.end()

        self.args = {'python_exe': python_exe, 'unsupported_args': self.unsupported_args}

    def create_controlNet_tab(self):
        if not self.gizmo.knob('controlNet'):
            generate_tab_knob = nuke.Tab_Knob("controlNet", 'Control Net')
            self.gizmo.addKnob(generate_tab_knob)
        return self.gizmo.knob('controlNet')

    def create_settings_tab(self):
        if not self.gizmo.knob('settings'):
            settings = nuke.Tab_Knob("settings", 'Settings')
            self.gizmo.addKnob(settings)
        return self.gizmo.knob('settings')

    def create_settings_knobs(self):
        self.create_settings_tab()

        if not self.gizmo.knob("output_dir"):
            self.directory_knob = nuke.File_Knob("output_dir", "Output Directory")
            self.directory_knob.setValue(os.path.join(os.path.expanduser("~"), 'nuke-stable-diffusion'))
            self.gizmo.addKnob(self.directory_knob)

        # time_out
        if not self.gizmo.knob('time_out'):
            time_out = nuke.Int_Knob('time_out', 'Time out', 30)
            time_out.setValue(300)
            self.gizmo.addKnob(time_out)

    def create_generate_tab(self):
        if not self.gizmo.knob('generate'):
            generate_tab_knob = nuke.Tab_Knob("generate", 'Generate')
            self.gizmo.addKnob(generate_tab_knob)
        return self.gizmo.knob('generate')

    def create_generate_knobs(self):
        self.create_generate_tab()

        if not self.gizmo.knob('button'):
            cn_button = nuke.PyScript_Knob('button', 'Execute')
            self.gizmo.addKnob(cn_button)
        self.gizmo.knob('button').setCommand(f'{self.__class__.__name__}.exec_fun()')

    def reorder_knobs(self):
        btn = self.gizmo.knob('button')
        if btn:
            self.gizmo.removeKnob(btn)
            self.gizmo.addKnob(btn)

    @property
    def output_dir(self):
        return self.gizmo.knob("output_dir").value()

    def knobChanged(self, knob):
        if knob.name() == 'controlNet_menu':
            pass

    def find_node_in_group(self, node_name):
        for node in self.gizmo.nodes():
            if node.name() == node_name:
                return node
        return None

    @property
    def input_node(self):
        return self.get_node("Input1", 'Input')

    @property
    def output_node(self):
        return self.get_node("Output1", 'Output')

    def get_model_args(self, model):
        controlNet_option = self.gizmo.knob('controlNet_menu').value()
        args = {}
        if controlNet_option == "lllyasviel/control_v11p_sd15_canny":
            args['min_threshold'] = self.gizmo.knob('min_threshold').value()
            args['max_threshold'] = self.gizmo.knob('max_threshold').value()
        return args

    def update_args(self):
        return NotImplementedError()

    def get_output_dir(self):
        output_dir = self.gizmo.knob("output_dir").value()
        return os.path.join(output_dir, self.gizmo.name())

    def get_node(self, nodeName, nodeType):
        for node in self.gizmo.nodes():
            if node.name() == nodeName:
                return node
        if nodeType == 'Input':
            node = nuke.nodes.Input()

        elif nodeType == 'Output':
            node = nuke.nodes.Output()
        else:
            node = nuke.createNode(nodeType, inpanel=False)

        node.setName(nodeName)
        return node

    def update_output_callback(self, output_batch):
        self.gizmo.begin()
        if not isinstance(output_batch, list):
            output_batch = [output_batch]

        output_nodes = []
        num = 1
        for b, output_files in enumerate(output_batch):
            for i, output_file in enumerate(output_files):
                fileNode = self.get_node(f"Read{num}", 'Read')
                fileNode.knob('file').setValue(output_file.replace('\\', '/'))
                fileNode.knob('reload').execute()
                output_nodes.append(fileNode)
                num += 1
        self.gizmo.end()
        return output_nodes

    def writeInput(self, outputPath):
        write_node = nuke.nodes.Write()
        write_node.knob('file').setValue(outputPath.replace('\\', '/'))
        write_node.knob('channels').setValue('rgb')
        write_node.setInput(0, self.input_node)

        write_node.knob('file_type').setValue('png')
        write_node.knob('datatype').setValue('8 bit')

        nuke.execute(write_node.name(), nuke.frame(), nuke.frame())
        nuke.delete(write_node)

    def on_execute(self):
        print('-' * 100)
        print(f"Execute {self.__class__.__name__}: {self.args}")
        thread = ExecuteThread(self.args, self.update_output_callback)
        thread.start()

    @classmethod
    def exec_fun(cls):
        inst = cls(nuke.thisNode().name())
        return inst.on_execute()

    @staticmethod
    def update_instance_from_node():
        node = nuke.thisNode()
        # Check if the knob changed is the hidden knob
        if node.knob("gizmo_class_type"):
            # Get the value of the hidden knob
            class_name = node.knob("gizmo_class_type").value()
            try:
                # Dynamically create an instance of the class based on the stored name
                return globals()[class_name](node.name())
            except KeyError:
                print("Class not found:", class_name)

    def update_single_read_node(self, node, file_path):
        node.knob('file').setValue(file_path)
        node.knob('reload').execute()
        self.force_valuate_nodes()

    def check_files(self, files, timeout):
        time.sleep(5)
        start_time = time.time()
        remaining_files = set(files)  # Convert list to set for efficient removal.
        while remaining_files:
            for file_path, node in list(remaining_files):  # Create a copy of the set for iteration.
                if os.path.exists(file_path):
                    nuke.executeInMainThread(self.update_single_read_node, args=(node, file_path,))
                    remaining_files.remove((file_path, node))  # Remove from the set.
            # Check if the timeout has been reached
            if time.time() - start_time > timeout:
                print("Timeout reached")
                break
            time.sleep(2)

    @staticmethod
    def disconnect_inputs(node):
        for i in range(node.inputs()):
            node.setInput(i, None)

    def force_valuate_nodes(self):
        n = self.get_node('Output1', 'Output')

        # for n in self.gizmo.nodes():
        knob = n.knob('label')
        current_label = knob.value()
        knob.setValue(current_label + " ")
        knob.setValue(current_label)

    def show_hide_knobs(self, knob_list, show=False):
        if isinstance(knob_list, str):
            knob_list = [knob_list]
        for knob in knob_list:
            knobC = self.gizmo.knob(knob)
            if knobC:
                knobC.setVisible(show)


class StableDiffusionGenerate(StableDiffusionBase):

    @property
    def input_nodes(self):
        return [self.get_node(f"Input{i + 1}", "Input") for i in range(self.controlNet_count)]

    @property
    def controlNet_inputs(self):
        return [self.get_node(f"CN{i + 1}", "Input") for i in range(self.controlNet_count)]

    def __init__(self, gizmo=None):
        super().__init__(gizmo=gizmo)
        self.args['script_path'] = os.path.join(DIR_PATH, 'sdexecuters', 'sdpipeline.py')

        self.gizmo.begin()

        # <editor-fold desc="output contact sheet">
        contact_sheet = self.get_node('Output_contactSheet', "ContactSheet")
        # Create Switch node
        self.switch_node = self.get_node('Output_switch', "Switch")
        self.output_node.setInput(0, self.switch_node)
        # Connect the contact sheet and individual inputs to the switch node
        self.switch_node.setInput(0, contact_sheet)

        # </editor-fold>

        self.create_inputs()
        self.create_generate_knobs()
        self.create_controlNet_knobs()
        self.create_settings_knobs()
        self.gizmo.end()

        self.show_hide_knobs(self.unsupported_args, False)

    def create_inputs(self):
        return self.controlNet_inputs

    def create_controlNet_knobs(self):
        self.create_controlNet_tab()
        return self.controlNet_menus

    def get_output_dir(self):
        output_dir = self.gizmo.knob("output_dir").value()
        model = self.gizmo.knob('model_menu').value()
        return os.path.join(output_dir, self.gizmo.name(), model.rsplit('/', 1)[-1])

    @staticmethod
    def create_controlNet_menu(menuName, niceName=None):
        niceName = niceName or menuName
        cn = nuke.Enumeration_Knob(menuName, niceName, ["None"] + model_config['controlNetModels'])
        return cn

    @property
    def controlNet_menus(self):
        controlNetList = []
        for i in range(3):
            cn = self.gizmo.knob(f'controlNet_menu{i + 1}')
            if not cn:
                cn = self.create_controlNet_menu(f'controlNet_menu{i + 1}', f'ControlNet Menu{i + 1}')
                self.gizmo.addKnob(cn)

            format_mult = self.gizmo.knob(f'cnFormatMult{i + 1}')
            if not format_mult:
                format_mult = nuke.Double_Knob(f'cnFormatMult{i + 1}', f'CN Format Mult{i + 1}')
                format_mult.setValue(1)
                self.gizmo.addKnob(format_mult)
            controlNetList.append((cn, format_mult))
        return controlNetList

    def create_generate_knobs(self):

        self.create_generate_tab()

        # model_menu
        if not self.gizmo.knob('model_menu'):
            model_menu = nuke.Enumeration_Knob('model_menu', 'Model', self.models)
            self.gizmo.addKnob(model_menu)

        # prompt
        if not self.gizmo.knob('prompt'):
            prompt = nuke.Multiline_Eval_String_Knob('prompt', 'Prompt',
                                                     "ancient egypt pharaohs shiny super clean silver earrings on ")
            self.gizmo.addKnob(prompt)

        # negative_prompt
        if not self.gizmo.knob('negative_prompt'):
            negative_prompt = nuke.Multiline_Eval_String_Knob('negative_prompt', 'Negative',
                                                              "cartoon, 3d, ((disfigured)), ((bad art)), ((deformed)), ((poorly drawn)), ((extra limbs)), ((close up)), ((b&w)), weird colors, blurry")
            self.gizmo.addKnob(negative_prompt)

        # format_menu
        if not self.gizmo.knob('format_menu'):
            format_menu = nuke.Format_Knob('format_menu', 'Format')
            self.gizmo.addKnob(format_menu)
            desired_format = None
            for format in nuke.formats():
                if format.width() == 640 and format.height() == 480:
                    desired_format = format
                    break
            if desired_format is None:
                desired_format = nuke.addFormat("640 480 0 0 640 480 1 SD-format")

            # Create the knob and set the value
            format_menu.setValue(desired_format.name())

        # formatMult
        if not self.gizmo.knob('formatMult'):
            format_mult = nuke.Double_Knob('formatMult', 'Format Mult')
            format_mult.setValue(1)
            self.gizmo.addKnob(format_mult)

        # guidance_scale
        if not self.gizmo.knob('guidance_scale'):
            guidance_scale = nuke.Double_Knob('guidance_scale', 'Guidance scale')
            guidance_scale.setValue(7.5)
            guidance_scale.setRange(0, 10)
            self.gizmo.addKnob(guidance_scale)

        # batch_count
        if not self.gizmo.knob('batch_count'):
            seed = nuke.Int_Knob('batch_count', 'Batch count', 1)
            seed.setValue(1)
            self.gizmo.addKnob(seed)

        # batch_count
        if not self.gizmo.knob('batch_size'):
            seed = nuke.Int_Knob('batch_size', 'Batch size', 1)
            seed.setValue(1)
            self.gizmo.addKnob(seed)

        # seed
        if not self.gizmo.knob('seed'):
            seed = nuke.Int_Knob('seed', 'Seed', -1)
            seed.setValue(-1)
            self.gizmo.addKnob(seed)

        if not self.gizmo.knob('num_inference_steps'):
            num_inference_steps = nuke.Int_Knob('num_inference_steps', 'Number of Inference Steps', 20)
            num_inference_steps.setValue(20)
            self.gizmo.addKnob(num_inference_steps)

        if not self.gizmo.knob('output_switch'):
            # Create a Double_Knob
            output_switch = nuke.Double_Knob("output_switch", "Output switch")
            self.gizmo.addKnob(output_switch)
            # Link the 'which' knob of the switch_node to the 'slider_knob'
            self.switch_node.knob('which').setExpression(output_switch.name())
        super().create_generate_knobs()

    @staticmethod
    def check_and_correct_size(size):
        size = int(size)
        if size % 8 != 0:
            corrected_size = (size // 8) * 8
            nuke.warning(f"Warning: Size {size} is not a multiple of 8. Changing to {corrected_size}.")
            return corrected_size
        return size

    def writeInput(self, outputPath, node):
        # Use a Write node to save the input image
        write_node = nuke.nodes.Write()
        write_node.knob('file').setValue(outputPath.replace('\\', '/'))
        write_node.knob('channels').setValue('rgb')
        write_node.setInput(0, node)

        # Set the file_type and datatype
        write_node.knob('file_type').setValue('png')
        write_node.knob('datatype').setValue('8 bit')

        # Execute the Write node
        nuke.execute(write_node.name(), nuke.frame(), nuke.frame())

        # Remove the Write node
        nuke.delete(write_node)

    def get_controlNet(self):

        controlNets = []
        model = self.gizmo.knob('model_menu').value()
        output_main_dir = self.get_output_dir()
        # Check if there's an image connected to the input
        for i, (controlNet_menu, controlNet_weight) in enumerate(self.controlNet_menus):
            controlNet = controlNet_menu.value()
            if controlNet != 'None' and model not in model_not_support_controlNet:
                cn_dir_path = os.path.join(output_main_dir, 'controlNet')
                os.makedirs(cn_dir_path, exist_ok=True)
                cn_file_path = os.path.join(cn_dir_path, controlNet.rsplit('/', 1)[-1] + f'{i + 1}.png')

                self.writeInput(cn_file_path, self.controlNet_inputs[i])
                # Add the control_net argument
                controlNets.append([controlNet, cn_file_path, controlNet_weight.value()])
        return controlNets

    def update_args(self):
        # get the model
        model = self.gizmo.knob('model_menu').value()

        self.args['model'] = model
        self.args['prompt'] = self.gizmo.knob('prompt').value()
        self.args['negative_prompt'] = self.gizmo.knob('negative_prompt').value()
        self.args['guidance_scale'] = self.gizmo.knob('guidance_scale').value()
        self.args['num_inference_steps'] = int(self.gizmo.knob('num_inference_steps').value())

        format_mult = self.gizmo.knob('formatMult').value()
        format_value = self.gizmo.knob('format_menu').value()
        self.args['width'] = self.check_and_correct_size(format_value.width() * format_mult)
        self.args['height'] = self.check_and_correct_size(format_value.height() * format_mult)
        self.args['seed'] = int(self.gizmo.knob('seed').value())

        self.args['batch_count'] = int(self.gizmo.knob('batch_count').value())
        self.args['num_images_per_prompt'] = int(self.gizmo.knob('batch_size').value())

        # get the output dir and make it
        output_dir_path = os.path.join(self.get_output_dir(), f'{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(output_dir_path, exist_ok=True)

        output_file_list = []
        for bc in range(self.args['batch_count']):
            output_batch_list = []
            for i in range(self.args['num_images_per_prompt']):
                filename = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{bc + 1}_{i + 1}.png'
                filePath = os.path.join(output_dir_path, filename).replace('\\', '/')
                output_batch_list.append(filePath)
            output_file_list.append(output_batch_list)
        self.args['output'] = output_file_list

        return self.args

    def on_execute(self):
        # Get the value from the format knob
        self.update_args()
        controlNets = self.get_controlNet()
        if controlNets:
            self.args['controlNet'] = controlNets

        batch_count = self.args.pop('batch_count')
        output_files = self.args.get('output', [])
        super().on_execute()
        self.update_output_readNodes(output_files)

    def update_output_readNodes(self, output_files):
        self.gizmo.begin()
        i = 0
        concatenated_file_nodes = []
        concatenated_file_path = []
        for fileList in output_files:
            if not isinstance(fileList, list):
                fileList = [fileList]
            for f in fileList:
                fileNode = self.get_node(f"Read{i + 1}", 'Read')
                fileNode.knob('file').setValue('')
                concatenated_file_nodes.append(fileNode)
                concatenated_file_path.append(f)
                i += 1

        self.create_contact_sheet(concatenated_file_nodes)

        self.check_multiple_files(concatenated_file_nodes, concatenated_file_path)
        self.gizmo.end()

    def check_multiple_files(self, node_names, file_paths):
        # Create a thread for each file
        timeout = self.gizmo.knob('time_out').value()
        files = zip(file_paths, node_names)
        thread = threading.Thread(target=self.check_files, args=(files, timeout))
        thread.start()

        return thread

    def create_contact_sheet(self, nodes):

        if len(nodes) == 1:
            self.output_node.setInput(0, nodes[0])
            return

        # Create LayerContactSheet node
        contact_sheet = self.get_node('Output_contactSheet', "ContactSheet")
        contact_sheet['roworder'].setValue('TopBottom')
        # Create Switch node
        switch_node = self.get_node('Output_switch', "Switch")
        self.output_node.setInput(0, switch_node)
        # Disconnect all inputs from the contact sheet and switch node
        self.disconnect_inputs(contact_sheet)
        self.disconnect_inputs(switch_node)

        # Connect nodes to the contact sheet
        for index, node in enumerate(nodes):
            contact_sheet.setInput(index, node)

        # Calculate rows and columns based on the number of images
        num_images = len(nodes)
        columns = int(num_images ** 0.5)  # Square root of num_images rounded down
        if columns ** 2 < num_images:  # Check if the columns need to be increased
            columns += 1
        rows = (num_images // columns)  # Divide num_images by columns
        if num_images % columns != 0:  # Check if there is an extra row needed
            rows += 1

        # Set resolution based on the first input node
        first_input = nodes[0]
        width = first_input.width()
        height = first_input.height()
        contact_sheet.knob('width').setValue(width * columns)
        contact_sheet.knob('height').setValue(height * rows)

        # Set rows, columns, and gap
        contact_sheet.knob('rows').setValue(rows)
        contact_sheet.knob('columns').setValue(columns)
        contact_sheet.knob('gap').setValue(width * 0.02)

        # Connect the contact sheet and individual inputs to the switch node
        switch_node.setInput(0, contact_sheet)
        for index, node in enumerate(nodes):
            switch_node.setInput(index + 1, node)

        self.gizmo.knob('output_switch').setRange(0, len(nodes))
        # Return the contact sheet and switch node
        return contact_sheet, switch_node


class StableDiffusionSAGPipeline(StableDiffusionGenerate):

    def __init__(self, gizmo=None):
        super().__init__(gizmo=gizmo)
        self.args['SAG'] = True


class StableDiffusionImg2Img(StableDiffusionGenerate):

    def __init__(self, gizmo=None):
        self.unsupported_args = ['format_menu', 'width', 'height', 'latents']
        super().__init__(gizmo=gizmo)

    @property
    def input_node(self):
        return self.get_node("init_img", 'Input')

    def create_generate_knobs(self):
        # strength
        self.create_generate_tab()

        if not self.gizmo.knob('strength'):
            strength = nuke.Double_Knob('strength', 'Strength')
            strength.setValue(.7)
            strength.setRange(0, 1)
            self.gizmo.addKnob(strength)

        if not self.gizmo.knob('use_depth'):
            use_depth = nuke.Boolean_Knob('use_depth', 'Use depth')

            self.gizmo.addKnob(use_depth)
        super().create_generate_knobs()

    def create_controlNet_knobs(self):
        return

    def create_inputs(self):
        return self.input_node

    def get_controlNet(self):
        return

    def on_execute(self):
        # Get the value from the format knob
        self.update_args()

        img2img_dir = os.path.join(self.get_output_dir(), 'source_img2img')
        os.makedirs(img2img_dir, exist_ok=True)

        img2img_path = os.path.join(img2img_dir, 'init_img.png')
        strength = self.gizmo.knob('strength').value()
        self.args['img2img'] = (img2img_path, strength)
        self.args['num_images_per_prompt'] = 1
        self.writeInput(img2img_path, self.input_node)

        if self.gizmo.knob('use_depth').value():
            self.args['use_depth'] = True
        batch_count = self.args.pop('batch_count')
        output_files = self.args.get('output', [])
        super().on_execute()
        self.update_output_readNodes(output_files)


class StableDiffusionImageVariationPipeline(StableDiffusionGenerate):

    def __init__(self, gizmo=None):
        self.models = ["lambdalabs/sd-image-variations-diffusers"]
        self.unsupported_args = ['prompt', 'negative_prompt']
        super().__init__(gizmo=gizmo)

    @property
    def input_node(self):
        return self.get_node("init_img", 'Input')

    def create_controlNet_knobs(self):
        return

    def create_inputs(self):
        return self.input_node

    def get_controlNet(self):
        return

    def on_execute(self):
        # Get the value from the format knob
        self.update_args()

        img2img_dir = os.path.join(self.get_output_dir(), 'source_img2img')
        os.makedirs(img2img_dir, exist_ok=True)

        img2img_path = os.path.join(img2img_dir, 'init_img.png')
        self.args['imageVariation'] = (img2img_path)
        self.writeInput(img2img_path, self.input_node)

        batch_count = self.args.pop('batch_count')
        output_files = self.args.get('output', [])
        super().on_execute()
        self.update_output_readNodes(output_files)


class StableDiffusionUpscalePipeline(StableDiffusionGenerate):

    def __init__(self, gizmo=None):
        self.models = ['stabilityai/stable-diffusion-x4-upscaler']
        self.unsupported_args = ['format_menu', 'width', 'height', 'latents']

        super().__init__(gizmo=gizmo)

        # self.show_hide_knobs(knobs_to_hide, False)

    @property
    def input_node(self):
        return self.get_node("init_img", 'Input')

    def create_controlNet_knobs(self):
        return

    def create_inputs(self):
        return self.input_node

    def get_controlNet(self):
        return

    def on_execute(self):
        # Get the value from the format knob
        self.update_args()

        img2img_dir = os.path.join(self.get_output_dir(), 'source_img2img')
        os.makedirs(img2img_dir, exist_ok=True)

        img2img_path = os.path.join(img2img_dir, 'init_img.png')
        self.args['upScale'] = (img2img_path)
        self.writeInput(img2img_path, self.input_node)

        batch_count = self.args.pop('batch_count')
        output_files = self.args.get('output', [])
        super().on_execute()
        self.update_output_readNodes(output_files)


class StableDiffusionInpaintPipeline(StableDiffusionGenerate):

    def __init__(self, gizmo=None):
        self.models = ['runwayml/stable-diffusion-inpainting',
                       "emilianJR/CyberRealistic_V3",
                       'stabilityai/stable-diffusion-2-inpainting']

        super().__init__(gizmo=gizmo)
        # self.create_inputs()
        # self.create_generate_knobs()
        # self.create_settings_knobs()

    @property
    def input_node(self):
        return self.get_node("init_img", 'Input')

    @property
    def mask_node(self):
        return self.get_node("mask_image", 'Input')

    def create_generate_knobs(self):
        # strength
        self.create_generate_tab()

        if not self.gizmo.knob('strength'):
            strength = nuke.Double_Knob('strength', 'Strength')
            strength.setValue(.7)
            strength.setRange(0, 1)
            self.gizmo.addKnob(strength)

        super().create_generate_knobs()

    def create_controlNet_knobs(self):
        return

    def create_inputs(self):
        return self.input_node, self.mask_node

    def get_controlNet(self):
        return

    def on_execute(self):
        # Get the value from the format knob
        self.update_args()

        img2img_dir = os.path.join(self.get_output_dir(), 'source_img2img')
        os.makedirs(img2img_dir, exist_ok=True)

        img2img_path = os.path.join(img2img_dir, 'init_img.png')
        self.writeInput(img2img_path, self.input_node)
        mask_path = os.path.join(img2img_dir, 'mask_img.png')
        self.writeInput(mask_path, self.mask_node)
        self.args['inPaint'] = (img2img_path, mask_path)

        batch_count = self.args.pop('batch_count')
        output_files = self.args.get('output', [])
        super().on_execute()
        self.update_output_readNodes(output_files)


class StableDiffusionRePaintPipelinePipeline(StableDiffusionGenerate):

    def __init__(self, gizmo=None):
        self.models = ['google/ddpm-ema-celebahq-256']
        self.unsupported_args = ['prompt', 'negative_prompt', 'format_menu', 'width', 'height', 'latents',
                                 'num_images_per_prompt', 'guidance_scale']
        super().__init__(gizmo=gizmo)
        # self.create_inputs()
        # self.create_generate_knobs()
        # self.create_settings_knobs()

    @property
    def input_node(self):
        return self.get_node("init_img", 'Input')

    @property
    def mask_node(self):
        return self.get_node("mask_image", 'Input')

    def create_generate_knobs(self):
        # strength
        self.create_generate_tab()

        if not self.gizmo.knob('strength'):
            strength = nuke.Double_Knob('strength', 'Strength')
            strength.setValue(.7)
            strength.setRange(0, 1)
            self.gizmo.addKnob(strength)

        if not self.gizmo.knob('jump_length'):
            jump_length = nuke.Int_Knob('jump_length', 'Jump length')
            jump_length.setValue(10)
            jump_length.setRange(1, 50)
            self.gizmo.addKnob(jump_length)

        if not self.gizmo.knob('jump_n_sample'):
            jump_n_sample = nuke.Int_Knob('jump_n_sample', 'Jump NLength')
            jump_n_sample.setValue(10)
            jump_n_sample.setRange(1, 50)
            self.gizmo.addKnob(jump_n_sample)

        if not self.gizmo.knob('eta'):
            eta = nuke.Double_Knob('eta', 'eta')
            eta.setValue(0.0)
            eta.setRange(0, 1)
            self.gizmo.addKnob(eta)

        super().create_generate_knobs()

    def create_controlNet_knobs(self):
        return

    def create_inputs(self):
        return self.input_node, self.mask_node

    def get_controlNet(self):
        return

    def on_execute(self):
        # Get the value from the format knob
        self.update_args()

        img2img_dir = os.path.join(self.get_output_dir(), 'source_img2img')
        os.makedirs(img2img_dir, exist_ok=True)

        img2img_path = os.path.join(img2img_dir, 'init_img.png')
        self.writeInput(img2img_path, self.input_node)
        mask_path = os.path.join(img2img_dir, 'mask_img.png')
        self.writeInput(mask_path, self.mask_node)

        jump_length = self.gizmo.knob('jump_length').value()
        jump_n_sample = self.gizmo.knob('jump_n_sample').value()
        eta = self.gizmo.knob('eta').value()
        self.args['rePaintPipeline'] = (img2img_path, mask_path, jump_length, jump_n_sample, eta)

        batch_count = self.args.pop('batch_count')
        output_files = self.args.get('output', [])
        super().on_execute()
        self.update_output_readNodes(output_files)


class StableDiffusionPaintByExample(StableDiffusionGenerate):

    def __init__(self, gizmo=None):
        self.models = ["Fantasy-Studio/Paint-by-Example"]
        self.unsupported_args = ['prompt', 'negative_prompt']
        super().__init__(gizmo=gizmo)
        # self.create_inputs()
        # self.create_generate_knobs()
        # self.create_settings_knobs()

    @property
    def input_node(self):
        return self.get_node("init_img", 'Input')

    @property
    def example_node(self):
        return self.get_node("example_image", 'Input')

    @property
    def mask_node(self):
        return self.get_node("mask_image", 'Input')

    def create_generate_knobs(self):
        # strength
        self.create_generate_tab()

        if not self.gizmo.knob('strength'):
            strength = nuke.Double_Knob('strength', 'Strength')
            strength.setValue(.7)
            strength.setRange(0, 1)
            self.gizmo.addKnob(strength)

        super().create_generate_knobs()

    def create_controlNet_knobs(self):
        return

    def create_inputs(self):
        return self.input_node, self.example_node, self.mask_node

    def get_controlNet(self):
        return

    def on_execute(self):
        # Get the value from the format knob
        self.update_args()

        img2img_dir = os.path.join(self.get_output_dir(), 'source_img2img')
        os.makedirs(img2img_dir, exist_ok=True)

        img2img_path = os.path.join(img2img_dir, 'init_img.png')
        self.writeInput(img2img_path, self.input_node)
        mask_path = os.path.join(img2img_dir, 'mask_img.png')
        self.writeInput(mask_path, self.mask_node)
        example_path = os.path.join(img2img_dir, 'example_image.png')
        self.writeInput(example_path, self.example_node)
        self.args['paintByExample'] = (img2img_path, mask_path, example_path)

        batch_count = self.args.pop('batch_count')
        output_files = self.args.get('output', [])
        super().on_execute()
        self.update_output_readNodes(output_files)
