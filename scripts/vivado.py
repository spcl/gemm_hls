#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess as sp

def run(*popenargs, input=None, check=False, **kwargs):
    if input is not None:
        if 'stdin' in kwargs:
            raise ValueError('stdin and input arguments may not both be used.')
        kwargs['stdin'] = sp.PIPE

    process = sp.Popen(*popenargs, **kwargs)
    try:
        stdout, stderr = process.communicate(input)
    except:
        process.kill()
        process.wait()
        raise
    retcode = process.poll()
    if check and retcode:
        raise sp.CalledProcessError(
            retcode, process.args, output=stdout, stderr=stderr)
    return retcode, stdout, stderr

def run_tcl(scriptPath, args):
    proc = run(
        [args["vivadoPath"], "-mode", "batch", "-source", scriptPath],
        cwd=args["tmpDir"])
    if proc[0] != 0:
        raise RuntimeError("Script " + os.path.basename(scriptPath) +
                           " failed.")


if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("command", type=str)
    argParser.add_argument("kernelName", type=str)
    argParser.add_argument("vivadoPath", type=str)
    argParser.add_argument("buildDir", type=str)
    argParser.add_argument("tmpDir", type=str)

    args = vars(argParser.parse_args())

    scriptDir = os.path.dirname(os.path.realpath(__file__))
    projectDir = os.path.join(args["buildDir"], args["kernelName"] + "_ex")

    if not os.path.exists(args["tmpDir"]):
        os.makedirs(args["tmpDir"])

    shutil.copyfile(
        os.path.join(args["buildDir"], "config.tcl"),
        os.path.join(args["tmpDir"], "config.tcl"))

    if args["command"] == "setup":

        run_tcl(os.path.join(scriptDir, "create_project.tcl"), args)
        run_tcl(os.path.join(scriptDir, "create_ips.tcl"), args)
        shutil.copyfile(
            os.path.join(args["buildDir"], "top.v"),
            os.path.join(projectDir, "imports", args["kernelName"] + ".v"))
        run_tcl(os.path.join(scriptDir, "import_sources.tcl"), args)

    elif args["command"] == "package":

        run_tcl(os.path.join(scriptDir, "package_kernel.tcl"), args)
        shutil.copyfile(
            os.path.join(projectDir, "sdx_imports",
                         args["kernelName"] + ".xo"),
            os.path.join(args["buildDir"], args["kernelName"] + ".xo"))

    else:

        os.remove(os.path.join(args["tmpDir"], "config.tcl"))
        raise ValueError("Unknown command \"{}\"".format(args["command"]))

    os.remove(os.path.join(args["tmpDir"], "config.tcl"))
