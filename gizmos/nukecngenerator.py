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
cn_script_path = r"D:\stable-diffusion\stable-diffusion-integrator\nukesd\sdexecuters\cnexecutor.py"


def find_node_in_group(node_name, group):
    print("find_node_in_group")
    print(node_name, group.name())
    for node in group.nodes():
        if node.name() == node_name:
            print(node_name, " is found")
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

        if not self.gizmo.knob('button'):
            self.cn_button = nuke.PyScript_Knob('button', 'Execute')
            self.gizmo.addKnob(self.cn_button)
        self.gizmo.knob('button').setCommand('cn_exec_fun()')

        _ = self.input_node
        self.output_node.setInput(0, self.read_node)

        self.gizmo.end()

    @property
    def read_node(self):
        return find_node_in_group("Read1", self.gizmo) or nuke.createNode('Read', inpanel=False)

    @property
    def input_node(self):
        return find_node_in_group("Input1", self.gizmo) or nuke.nodes.Input()

    @property
    def output_node(self):
        return find_node_in_group("Output1", self.gizmo) or nuke.nodes.Output()

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
    print('gizmoName, cn_gizmo_instances')
    print(gizmoName, cn_gizmo_instances)
    if gizmoName not in cn_gizmo_instances:
        cn_gizmo_instances[gizmoName] = CNGizmo(gizmoName)

    cn_gizmo_instances[gizmoName].on_execute()


if __name__ == '__main__':
    create_cn_gizmo_instance()
