import inspect
import os
import subprocess
import threading
from datetime import datetime

import nuke

controlNetModels = ["lllyasviel/control_v11p_sd15_canny",
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
                    "lllyasviel/control_v11p_sd15_softedge",
                    "lllyasviel/control_v11e_sd15_shuffle"]

python_exe = r"D:\stable-diffusion\stable-diffusion-webui\venv\Scripts\python.exe"
DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
cn_script_path = os.path.join(DIR_PATH, 'sdexecuters', 'cnpipeline.py')


def find_node_in_group(node_name, group):
    for node in group.nodes():
        if node.name() == node_name:
            return node
    return None


class CNGizmo:
    def __init__(self, gizmo=None):
        if gizmo:
            self.gizmo = nuke.toNode(gizmo)
            if self.gizmo is None:
                raise ValueError("No such gizmo named '{}'".format(gizmo))
        else:
            self.gizmo = nuke.createNode('Group', inpanel=False)
            self.gizmo.setName('Nuke_CN')

        self.gizmo.begin()

        if not self.gizmo.knob("output_dir"):
            self.directory_knob = nuke.File_Knob("output_dir", "Output Directory")
            self.directory_knob.setValue(os.path.join(os.path.expanduser("~"), 'nuke-stable-diffusion'))
            self.gizmo.addKnob(self.directory_knob)

        if not self.gizmo.knob('controlNet_menu'):
            self.cn_controlNet_menu = nuke.Enumeration_Knob('controlNet_menu', 'ControlNet Menu', controlNetModels)
            self.gizmo.addKnob(self.cn_controlNet_menu)

        if not self.gizmo.knob('min_threshold'):
            self.min_threshold_knob = nuke.Double_Knob('min_threshold', 'Min Threshold')
            self.min_threshold_knob.setFlag(nuke.STARTLINE)
            self.min_threshold_knob.setVisible(True)
            self.min_threshold_knob.setValue(100)
            self.gizmo.addKnob(self.min_threshold_knob)

        if not self.gizmo.knob('max_threshold'):
            self.max_threshold_knob = nuke.Double_Knob('max_threshold', 'Max Threshold')
            self.max_threshold_knob.setFlag(nuke.STARTLINE)
            self.max_threshold_knob.setVisible(True)
            self.max_threshold_knob.setValue(200)
            self.gizmo.addKnob(self.max_threshold_knob)

        self.gizmo.knob('controlNet_menu').setFlag(nuke.STARTLINE)
        self.gizmo.knob('controlNet_menu').setTooltip('Select ControlNet model')
        self.gizmo.knob('controlNet_menu').setAnimated(False)
        self.gizmo.knob('controlNet_menu').setValue('0')
        self.gizmo.knob('knobChanged').setValue(
            'cn_gizmo_instances["{}"].knobChanged(nuke.thisKnob())'.format(self.gizmo.name()))

        if not self.gizmo.knob('button'):
            self.cn_button = nuke.PyScript_Knob('button', 'Execute')
            self.gizmo.addKnob(self.cn_button)
        self.gizmo.knob('button').setCommand('cn_exec_fun()')

        _ = self.input_node
        self.output_node.setInput(0, self.read_node)

        self.gizmo.end()

    def knobChanged(self, knob):
        if knob.name() == 'controlNet_menu':
            controlNet_option = knob.value()
            if controlNet_option == "lllyasviel/control_v11p_sd15_canny":
                self.gizmo.knob('min_threshold').setVisible(True)
                self.gizmo.knob('max_threshold').setVisible(True)
            else:
                self.gizmo.knob('min_threshold').setVisible(False)
                self.gizmo.knob('max_threshold').setVisible(False)

    @property
    def read_node(self):
        return find_node_in_group("Read1", self.gizmo) or nuke.createNode('Read', inpanel=False)

    @property
    def input_node(self):
        return find_node_in_group("Input1", self.gizmo) or nuke.nodes.Input()

    @property
    def output_node(self):
        return find_node_in_group("Output1", self.gizmo) or nuke.nodes.Output()

    def get_model_args(self, model):
        controlNet_option = self.gizmo.knob('controlNet_menu').value()
        args = {}
        if controlNet_option == "lllyasviel/control_v11p_sd15_canny":
            args['min_threshold'] = self.gizmo.knob('min_threshold').value()
            args['max_threshold'] = self.gizmo.knob('max_threshold').value()
        return args

    def on_execute(self):
        model = self.gizmo.knob('controlNet_menu').value()
        output_dir = self.gizmo.knob("output_dir").value()
        output_dir_path = os.path.join(output_dir, self.gizmo.name(), model.rsplit('/', 1)[-1])
        os.makedirs(output_dir_path, exist_ok=True)
        filename = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        output_file_path = os.path.join(output_dir_path, filename)

        inputPath = os.path.join(output_dir_path, 'temp.png')

        args = {
            'model': model,
            'input': inputPath,
            'output': output_file_path,
            'python_exe': python_exe,
            'script_path': cn_script_path,
            'model_kwargs': self.get_model_args(model)
        }

        print('*' * 100)
        print("ExecuteCN: ", args)

        self.gizmo.begin()
        self.writeInput(inputPath)
        self.gizmo.end()

        def callback(output_file):
            self.gizmo.begin()
            self.read_node.knob('file').setValue(output_file.replace('\\', '/'))
            self.read_node.knob('reload').execute()
            self.read_node.knob('file').setValue(output_file.replace('\\', '/'))
            self.read_node.knob('reload').execute()
            self.gizmo.end()

        thread = ExecuteThread(args, callback)
        thread.start()

    def writeInput(self, outputPath):
        write_node = nuke.nodes.Write()
        write_node.knob('file').setValue(outputPath.replace('\\', '/'))
        write_node.knob('channels').setValue('rgb')
        write_node.setInput(0, self.input_node)

        write_node.knob('file_type').setValue('png')
        write_node.knob('datatype').setValue('8 bit')

        nuke.execute(write_node.name(), nuke.frame(), nuke.frame())
        nuke.delete(write_node)


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


try:
    cn_gizmo_instances
except NameError:
    cn_gizmo_instances = {}


def create_cn_gizmo_instance():
    instance = CNGizmo()
    cn_gizmo_instances[instance.gizmo.name()] = instance


def cn_exec_fun():
    global cn_gizmo_instances
    try:
        cn_gizmo_instances
    except NameError:
        cn_gizmo_instances = {}
    gizmoName = nuke.thisNode().name()

    if gizmoName not in cn_gizmo_instances:
        cn_gizmo_instances[gizmoName] = CNGizmo(gizmoName)

    cn_gizmo_instances[gizmoName].on_execute()


if __name__ == '__main__':
    create_cn_gizmo_instance()
