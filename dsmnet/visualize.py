import sys
import os
from pycore.tikzeng import *
import subprocess


def LeNetD3(input_path):
    name = "lenetd3"
    arch = [
        to_head(os.path.expanduser('~/git/PlotNeuralNet')),
        to_cor(),
        to_begin(),

        to_input(input_path, to='(-5.8,0,0)', width=13, height=13*1.4,
                 name="input", caption="Input (84x60)",
                 caption_adjust=(-0.1, -1.93)),

        to_Conv("conv1", 6, 80, 56, offset="(0,0,0)", to="(0,0,0)", height=80,
                depth=56, width=6, caption="Conv1 (5x5)"),
        to_Pool("pool1", offset="(0,0,0)", to="(conv1-east)",
                width=6, height=40, depth=28, opacity=0.5,
                caption="Pool1"),
        # to_connection("input", "conv1"),

        to_Conv("conv2", 16, 36, 24, offset="(3.5,0,0)", to="(pool1-east)", height=36,
                depth=24, width=16, caption='Conv2 (5x5)'),
        to_Pool("pool2", offset="(0,0,0)", to="(conv2-east)",
                width=16, height=18, depth=12, opacity=0.5,
                caption="Pool2"),
        to_connection("pool1", "conv2"),

        to_Conv("conv3", 16, 14, 8, offset="(2.5,0,0)", to="(pool2-east)", height=14,
                depth=8, width=16, caption="Conv3 (5x5)"),
        to_Pool("pool3", offset="(0,0,0)", to="(conv3-east)",
                width=16, height=7, depth=4, opacity=0.5,
                caption="Pool3"),
        to_connection("pool2", "conv3"),

        to_SoftMax("soft1", 120, "(3.,0,0)", "(pool3-east)", caption="FC1",
                   width=1, height=1, depth=120),
        to_connection("pool3", "soft1"),

        to_SoftMax("soft2", 84,"(2.5,0,0)", "(soft1-east)", caption="FC2",
                   width=1, height=1, depth=84),
        to_connection("soft1", "soft2"),

        to_SoftMax("soft3", 2,"(2.5,0,0)", "(soft2-east)", caption="FC3",
                   width=1, height=1, depth=2),
        to_connection("soft2", "soft3"),

        to_end()
        ]

    return arch, name


def draw(root_path, filename, arch):
    outpath = os.path.join(root_path, filename + '.tex')
    to_generate(arch, outpath)
    subprocess.run(['/usr/bin/pdflatex', outpath])
    subprocess.run(['/usr/bin/rm',
                    filename + '.aux',
                    filename + '.log',
                    outpath])
    subprocess.run(['/usr/bin/mv', filename + '.pdf', root_path])


if __name__ == '__main__':
    arch, filename = LeNetD3('../tests/X_0.png')
    root_path = '../tests/figures/'
    draw(root_path, filename, arch)

