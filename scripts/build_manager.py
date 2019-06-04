#!/usr/bin/env python3
import argparse
from collections import OrderedDict
import datetime
import itertools
import multiprocessing as mp
import os
import re
import shutil
import signal
import subprocess as sp
import sys
import time

PROJECT_CONFIG = {
    "kernel_name":
    "MatrixMultiplicationKernel",
    "kernel_file":
    "MatrixMultiplication_hw",
    "make_synthesis":
    "synthesis",
    "make_compile":
    "compile_hardware",
    "make_link":
    "link_hardware",
    "build_dir":
    "scan",
    "benchmark_dir":
    "benchmark",
    "options":
    OrderedDict([
        ("data_type", {
            "cmake": "MM_DATA_TYPE",
            "type": str,
            "default": None
        }),
        ("map_op", {
            "cmake": "MM_MAP_OP",
            "type": str,
            "default": None
        }),
        ("reduce_op", {
            "cmake": "MM_REDUCE_OP",
            "type": str,
            "default": None
        }),
        ("parallelism_n", {
            "cmake": "MM_PARALLELISM_N",
            "type": int,
            "default": None
        }),
        ("parallelism_m", {
            "cmake": "MM_PARALLELISM_M",
            "type": int,
            "default": None
        }),
        ("memory_width_m", {
            "cmake": "MM_MEMORY_BUS_WIDTH_M",
            "type": int,
            "default": None
        }),
        ("size_n", {
            "cmake": "MM_SIZE_N",
            "type": int,
            "default": None
        }),
        ("size_k", {
            "cmake": "MM_SIZE_K",
            "type": int,
            "default": None
        }),
        ("size_m", {
            "cmake": "MM_SIZE_M",
            "type": int,
            "default": None
        }),
        ("tile_size_n", {
            "cmake": "MM_MEMORY_TILE_SIZE_N",
            "type": int,
            "default": None
        }),
        ("tile_size_m", {
            "cmake": "MM_MEMORY_TILE_SIZE_M",
            "type": int,
            "default": None
        }),
        ("frequency", {
            "cmake": "MM_TARGET_CLOCK",
            "type": int,
            "default": 200
        }),
        ("target", {
            "cmake": "MM_DSA_NAME",
            "type": str,
            "default": "xilinx_vcu1525_xdma_201830_1"
        }),
    ]),
}


class Configuration(object):
    def __init__(self, *args, **kwargs):
        for opt, val in zip(
                list(PROJECT_CONFIG["options"].keys())[:len(args)], args):
            setattr(self, opt, PROJECT_CONFIG["options"][opt]["type"](val))
        for opt, val in kwargs.items():
            if opt not in PROJECT_CONFIG["options"]:
                raise KeyError("\"" + opt + "\" is not a valid option.")
            if opt in self.__dict__:
                raise KeyError("Option \"" + opt +
                               "\" set both as arg and kwarg")
            setattr(self, opt, PROJECT_CONFIG["options"][opt]["type"](val))
        unsetArgs = PROJECT_CONFIG["options"].keys() - self.__dict__.keys()
        for arg in unsetArgs:
            default = PROJECT_CONFIG["options"][arg]["default"]
            if default != None:
                setattr(self, opt, default)
                unsetArgs.remove(arg)
        if len(unsetArgs) > 0:
            raise TypeError("Missing arguments: {}".format(
                ", ".join(unsetArgs)))

    @staticmethod
    def csv_header():
        return ",".join(PROJECT_CONFIG["options"].keys())

    def to_string(self):
        return "_".join([
            str(getattr(self, opt)).replace(":", "-").replace("_", "-")
            for opt in PROJECT_CONFIG["options"]
        ])

    def to_csv(self):
        return ",".join(
            map(str,
                [getattr(self, opt) for opt in PROJECT_CONFIG["options"]]))

    def build_folder(self):
        return "build_" + self.to_string()

    def benchmark_folder(self):
        return "benchmark_" + self.to_string()

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return self.to_string()

    def cmake_command(self, sourceDir, extra=[]):
        return (["cmake", sourceDir] + [
            "-D{}={}".format(val["cmake"], getattr(self, key))
            for key, val in PROJECT_CONFIG["options"].items()
        ] + extra)

    @staticmethod
    def get_conf(s):
        pattern = ""
        for opt in PROJECT_CONFIG["options"].values():
            if opt["type"] == str:
                pattern += "([^_]+)_"
            elif opt["type"] == int:
                pattern += "([0-9]+)_"
            else:
                raise TypeError("Unsupported type \"{}\".".format(
                    str(opt["type"])))
        m = re.search(pattern[:-1], s)  # Removing trailing underscore
        if not m:
            raise ValueError("Not a valid configuration string: " + s)
        return Configuration(*m.groups())


def run_process(command,
                directory,
                pipe=True,
                logPath="log",
                timeout=None):
    if pipe:
        proc = sp.Popen(
            command,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            universal_newlines=True,
            cwd=directory)
        stdout, stderr = proc.communicate()
        with open(os.path.join(directory, logPath + ".out"), "a") as outFile:
            outFile.write(stdout)
        with open(os.path.join(directory, logPath + ".err"), "a") as outFile:
            outFile.write(stderr)
    else:
        proc = sp.Popen(command, universal_newlines=True, cwd=directory)
        try:
            proc.communicate(timeout=timeout)
        except sp.TimeoutExpired as err:
            proc.terminate()
            raise err
    return proc.returncode


class Consumption(object):
    def __init__(self, conf, status, lut, reg, dsp, bram, clock):
        self.conf = conf
        self.status = status
        self.lut = lut
        self.reg = reg
        self.dsp = dsp
        self.bram = bram
        self.clock = clock

    @staticmethod
    def csv_cols():
        return ("status," + Configuration.csv_header() +
                ",clock,dsp,lut,reg,bram")

    def __repr__(self):
        return ",".join(
            map(str, [
                self.status,
                self.conf.to_csv(), self.clock, self.dsp, self.lut, self.reg,
                self.bram
            ]))


def do_build(conf, cmakeOpts):
    cmakeCommand = conf.cmake_command("$1", cmakeOpts)
    confStr = conf.to_string()
    confDir = os.path.join(PROJECT_CONFIG["build_dir"], conf.build_folder())
    try:
        os.makedirs(confDir)
    except:
        pass
    with open(os.path.join(confDir, "configure.sh"), "w") as confFile:
        confFile.write("#!/bin/sh\n{}".format(" ".join(cmakeCommand)))
    run_build(conf, clean=True, hardware=True)


def time_only(t):
    if isinstance(t, datetime.datetime):
        return t.strftime("%H:%M:%S")
    else:
        totalSeconds = int(t.total_seconds())
        hours, rem = divmod(totalSeconds, 3600)
        minutes, seconds = divmod(rem, 60)
        minutes = "0" + str(minutes) if minutes < 10 else str(minutes)
        seconds = "0" + str(seconds) if seconds < 10 else str(seconds)
        return "{}:{}:{}".format(hours, minutes, seconds)


def print_status(conf, status):
    print("[{}] {}: {}".format(
        time_only(datetime.datetime.now()), str(conf.to_string()), status))


def run_build(conf, clean=True, hardware=True):
    confStr = conf.to_string()
    confDir = os.path.join(PROJECT_CONFIG["build_dir"], conf.build_folder())
    print_status(conf, "Configuring...")
    if run_process([
            "sh", "configure.sh",
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    ], confDir) != 0:
        print_status(conf, "Configuration failed.")
        return
        # raise Exception(confStr + ": Configuration failed.")
    print_status(conf, "Finished configuration.")
    if clean:
        print_status(conf, "cleaning folder...")
        if run_process(["make", "clean"], confDir) != 0:
            raise Exception(confStr + ": Clean failed.")
    print_status(conf, "Building software...")
    if run_process(["make"], confDir) != 0:
        raise Exception(confStr + ": Software build failed.")
    print_status(conf, "Finished building software.")
    if hardware:
        timeStart = datetime.datetime.now()
        print_status(conf, "Starting kernel compilation (HLS)...")
        if run_process(["make", PROJECT_CONFIG["make_compile"]], confDir) != 0:
            print_status(
                conf, "FAILED after {}.".format(
                    time_only(datetime.datetime.now() - timeStart)))
        else:
            print_status(
                conf, "Finished compilation stage in {}.".format(
                    time_only(datetime.datetime.now() - timeStart)))
        print_status(conf, "Starting kernel build...")
        if run_process(["make", PROJECT_CONFIG["make_link"]], confDir) != 0:
            try:
                with open(os.path.join(confDir, "log.out")) as logFile:
                    m = re.search("auto frequency scaling failed",
                                  logFile.read())
                    if not m:
                        print_status(
                            conf, "FAILED after {}.".format(
                                time_only(datetime.datetime.now() -
                                          timeStart)))
                    else:
                        print(conf, "TIMING failed after {}.".format(
                            time_only(datetime.datetime.now() - timeStart)))
            except FileNotFoundError:
                print_status(
                    conf, "FAILED after {}.".format(
                        time_only(datetime.datetime.now() - timeStart)))
        else:
            print_status(
                conf, "SUCCESS in {}.".format(
                    time_only(datetime.datetime.now() - timeStart)))


def extract_result_build(conf):
    buildFolder = os.path.join(PROJECT_CONFIG["build_dir"],
                               conf.build_folder())
    xoccFolder = "_x"
    if not os.path.exists(os.path.join(buildFolder, xoccFolder)):
        conf.consumption = Consumption(conf, "no_intermediate", None, None,
                                       None, None, None)
        return
    kernelFolder = os.path.join(buildFolder, xoccFolder, "link", "vivado")
    implFolder = os.path.join(kernelFolder, "prj", "prj.runs", "impl_1")
    if not os.path.exists(implFolder):
        conf.consumption = Consumption(conf, "no_build", None, None, None,
                                       None, None)
        return
    status = check_build_status(conf)
    reportPath = os.path.join(implFolder, "kernel_util_routed.rpt")
    if not os.path.isfile(reportPath):
        conf.consumption = Consumption(conf, status, None, None, None, None,
                                       None)
        return
    report = open(reportPath).read()
    pattern = ("Used Resources\s*\|\s*(\d+)[^\|]+\|\s*\d+[^\|]+\|\s*(\d+)"
               "[^\|]+\|\s*(\d+)[^\|]+\|\s*\d+[^\|]+\|\s*(\d+)[^\|]+\|")
    matches = re.search(pattern, report)
    luts = matches.group(1)
    regs = matches.group(2)
    bram = matches.group(3)
    dsp = matches.group(4)
    try:
        with open(os.path.join(kernelFolder, "vivado_warning.txt"),
                  "r") as clockFile:
            warningText = clockFile.read()
            m = re.search("automatically changed to ([0-9]+) MHz", warningText)
            if m:
                clock = int(m.group(1))
            else:
                clock = conf.frequency
    except FileNotFoundError:
        clock = conf.frequency
    conf.consumption = Consumption(conf, status, luts, regs, dsp, bram, clock)


def check_build_status(conf):
    buildFolder = os.path.join(PROJECT_CONFIG["build_dir"],
                               conf.build_folder())
    kernelFolder = os.path.join(
        buildFolder, "_x", "link", "vivado")
    logPath = os.path.join(kernelFolder, "vivado.log")
    try:
        log = open(logPath, "r").read()
    except:
        print("No build found for {}.".format(conf))
        return "no_build"
    m = re.search("Implementation Feasibility check failed", log)
    if m:
        return "failed_feasibility"
    m = re.search("Detail Placement failed", log)
    if m:
        return "failed_placement"
    m = re.search("Placer could not place all instances", log)
    if m:
        return "failed_placement"
    m = re.search(
        "Routing results verification failed due to partially-conflicted nets",
        log)
    if m:
        return "failed_routing"
    m = re.search("route_design ERROR", log)
    if m:
        return "failed_routing"
    m = re.search("Internal Data Exception", log)
    if m:
        return "crashed"
    m = re.search("Design failed to meet timing - hold violation.", log)
    if m:
        return "failed_hold"
    m = re.search("auto frequency scaling failed", log)
    if m:
        return "failed_timing"
    m = re.search("Unable to write message .+ as it exceeds maximum size", log)
    if m:
        return "failed_report"
    for fileName in os.listdir(buildFolder):
        if len(fileName) >= 7 and fileName.endswith(".xclbin"):
            return "success"
    print("Unfinished build or unknown error for {} [{}].".format(
        conf, logPath))
    return "failed_unknown"


def get_build_result(buildDir):
    confs = []
    for fileName in os.listdir(buildDir):
        try:
            conf = Configuration.get_conf(fileName)
        except ValueError:
            continue
        print("Extracting {}...".format(fileName))
        extract_result_build(conf)
        confs.append(conf)
    with open(
            os.path.join(PROJECT_CONFIG["build_dir"], "status.csv"),
            "w") as resultFile:
        resultFile.write(Consumption.csv_cols() + "\n")
        for conf in confs:
            resultFile.write(str(conf.consumption) + "\n")


def scan_configurations(numProcs, configurations, cmakeOpts):
    try:
        os.makedirs(PROJECT_CONFIG["build_dir"])
    except FileExistsError:
        pass
    pool = mp.Pool(processes=numProcs)
    try:
        pool.starmap(do_build,
                     zip(configurations,
                         len(configurations) * [cmakeOpts]))
    except KeyboardInterrupt:
        pool.terminate()
        print("Builds successfully aborted.")
    else:
        print("All configurations finished running.")


def files_to_copy(conf, target):
    filesToCopy = ["configure.sh", PROJECT_CONFIG["kernel_file"] + ".xclbin"]
    kernelString = PROJECT_CONFIG["kernel_name"]
    xoccFolder = "_x"
    hlsFolder = os.path.join(xoccFolder, PROJECT_CONFIG["kernel_file"],
                             PROJECT_CONFIG["kernel_name"])
    hlsReportFolder = os.path.join(hlsFolder, PROJECT_CONFIG["kernel_name"],
                                   "solution", "syn", "report")
    kernelFolder = os.path.join(xoccFolder, "link", "vivado")
    runsFolder = os.path.join(
        kernelFolder, "prj", "prj.runs",
        "pfm_dynamic_{}_1_0_synth_1".format(PROJECT_CONFIG["kernel_name"]))
    filesToCopy.append("log.out")
    filesToCopy.append("log.err")
    filesToCopy.append(PROJECT_CONFIG["kernel_file"] + ".xclbin.info")
    filesToCopy.append("xocc_{}.log".format(PROJECT_CONFIG["kernel_file"]))
    filesToCopy.append(os.path.join(hlsFolder, "vivado_hls.log"))
    filesToCopy.append(
        os.path.join(hlsReportFolder,
                     "{}_csynth.rpt".format(PROJECT_CONFIG["kernel_name"])))
    filesToCopy.append(os.path.join(kernelFolder, "vivado.log"))
    filesToCopy.append(os.path.join(kernelFolder, "vivado_warning.txt"))
    implFolder = os.path.join(kernelFolder, "prj", "prj.runs", "impl_1")
    filesToCopy.append(os.path.join(implFolder, "runme.log"))
    filesToCopy.append(os.path.join(implFolder, "kernel_util_routed.rpt"))
    filesToCopy.append(
        os.path.join(
            implFolder, "{}_bb_locked_timing_summary_routed.rpt".format(
                target.replace("-", "_"))))
    filesToCopy.append(os.path.join(runsFolder, "runme.log"))
    filesToCopy.append(
        os.path.join(
            runsFolder, "pfm_dynamic_{}_1_0_utilization_synth.rpt".format(
                PROJECT_CONFIG["kernel_name"])))
    return [runsFolder, implFolder, hlsReportFolder], filesToCopy


def package_configurations(target):
    kernelsPackaged = 0
    for fileName in os.listdir(PROJECT_CONFIG["build_dir"]):
        try:
            conf = Configuration.get_conf(fileName)
        except ValueError:
            continue
        if conf.target != target:
            continue
        sourceDir = os.path.join(PROJECT_CONFIG["build_dir"], fileName)
        packageFolder = os.path.join(target, fileName)
        kernelName = None
        kernelPath = None
        for subFile in os.listdir(sourceDir):
            if subFile.endswith(".xclbin"):
                kernelName = subFile[:-7]
                kernelPath = os.path.join(sourceDir, subFile)
                break
        if kernelPath == None or not os.path.exists(kernelPath):
            continue
        print("Packaging {}...".format(fileName))
        folders, filesToCopy = files_to_copy(conf, target)
        for folder in folders:
            try:
                os.makedirs(os.path.join(packageFolder, folder))
            except FileExistsError:
                pass
        filesCopied = 0
        filesMissing = 0
        for path in filesToCopy:
            try:
                shutil.copy(
                    os.path.join(sourceDir, path),
                    os.path.join(packageFolder, path))
                filesCopied += 1
            except FileNotFoundError:
                filesMissing += 1
        if filesCopied == 0:
            raise FileNotFoundError("Files not found!")
        if filesMissing > 0:
            print("WARNING: only {} / {} files copied ({} files missing).".format(
                filesCopied, filesCopied + filesMissing, filesMissing))
        kernelsPackaged += 1
    if kernelsPackaged > 0:
        print(("Successfully packaged " + str(kernelsPackaged) +
               " kernels and configuration files into \"{}\".").format(target))
    else:
        print("No kernels for target \"{}\" found in \"{}\".".format(
            target, PROJECT_CONFIG["build_dir"]))


def unpackage_configuration(conf):
    confStr = conf.to_string()
    fileName = conf.build_folder()
    print("Unpackaging {}...".format(confStr))
    sourceDir = os.path.join(conf.target, fileName)
    targetDir = os.path.join(PROJECT_CONFIG["build_dir"], fileName)
    folders, filesToCopy = files_to_copy(conf, conf.target)
    for folder in folders:
        try:
            os.makedirs(os.path.join(targetDir, folder))
        except FileExistsError:
            pass
    filesCopied = 0
    filesMissing = 0
    for path in filesToCopy:
        try:
            shutil.copy(
                os.path.join(sourceDir, path), os.path.join(targetDir, path))
            filesCopied += 1
        except FileNotFoundError:
            filesMissing += 1
    if filesCopied == 0:
        raise FileNotFoundError("Files not found!")
    if filesMissing > 0:
        print("WARNING: only {} / {} files copied ({} files missing).".format(
            filesCopied, filesCopied + filesMissing, filesMissing))
    with open(os.path.join(targetDir, "configure.sh"), "r") as inFile:
        confStr = inFile.read()
    with open(os.path.join(targetDir, "configure.sh"), "w") as outFile:
        # Remove specific compiler paths
        fixed = re.sub(" -DCMAKE_C(XX)?_COMPILER=[^ ]*", "", confStr)
        outFile.write(fixed)
    run_build(conf, clean=False, hardware=False)


def unpackage_configurations(target):
    unpackagedSomething = False
    confs = []
    for fileName in os.listdir(target):
        conf = Configuration.get_conf(fileName)
        if not conf:
            continue
        confs.append(conf)
        unpackagedSomething = True
    pool = mp.Pool(processes=mp.cpu_count())
    try:
        pool.map(unpackage_configuration, confs)
    except KeyboardInterrupt:
        pool.terminate()
    if unpackagedSomething:
        print("Successfully unpackaged kernels into \"{}\".".format(
            PROJECT_CONFIG["build_dir"]))
    else:
        print("No kernels found in \"{}\".".format(target))


def extract_benchmarks():
    benchmarkFile = "benchmark.csv"
    wroteHeader = False
    with open(benchmarkFile, "w") as outFile:
        for fileName in os.listdir(PROJECT_CONFIG["build_dir"]):
            try:
                conf = Configuration.get_conf(fileName)
            except ValueError:
                continue
            if not wroteHeader:
                outFile.write(conf.csv_header() +
                              ",time,performance,power,power_efficiency\n")
            wroteHeader = True
            confStr = conf.to_string()
            folderName = conf.benchmark_folder()
            kernelFolder = os.path.join(PROJECT_CONFIG["build_dir"], fileName)
            for resultPath in os.listdir(kernelFolder):
                if (not resultPath.endswith(".out")
                        or not resultPath.startswith("benchmark")):
                    continue
                resultPath = os.path.join(kernelFolder, resultPath)
                with open(resultPath, "r") as resultFile:
                    txt = resultFile.read()
                    m_perf = re.search(
                        "([\d\.]+) seconds[^\d]+([\d\.]+) GOp/s", txt)
                    m_power = re.search("([\d\.]+) W[^\d]+([\d\.]+) GOp/J",
                                        txt)
                    outFile.write("{},{},{},{},{}\n".format(
                        conf.to_csv(), m_perf.group(1), m_perf.group(2),
                        m_power.group(1), m_power.group(2)))
                    outFile.flush()


def benchmark(repetitions, timeout):
    for fileName in os.listdir(PROJECT_CONFIG["build_dir"]):
        try:
            conf = Configuration.get_conf(fileName)
        except ValueError:
            continue
        confStr = conf.to_string()
        folderName = conf.benchmark_folder()
        kernelFolder = os.path.join(PROJECT_CONFIG["build_dir"], fileName)
        kernelString = PROJECT_CONFIG["kernel_file"]
        kernelPath = os.path.join(kernelFolder, kernelString + ".xclbin")
        if not os.path.exists(kernelPath):
            continue
        benchmarkFile = "benchmark.csv"
        print("Running {}...".format(confStr))
        if run_process(["make"], kernelFolder, pipe=False) != 0:
            raise Exception(confStr + ": software build failed.")
        repsDone = 0
        timeouts = 0
        with open(benchmarkFile, "a") as outFile:
            outFile.write(conf.csv_header() +
                          ",time,performance,power,power_efficiency\n")
            while repsDone < repetitions:
                time.sleep(0.5)
                profilePath = os.path.join("benchmark_" +
                                           str(datetime.datetime.now()).
                                           replace(" ", "_").replace(":", "-"))
                print("Running iteration {} / {}...".format(
                    repsDone + 1, repetitions))
                cmd = ["./RunHardware.exe"]
                if "DYNAMIC_SIZES=ON" in open(
                        os.path.join(kernelFolder, "configure.sh")).read():
                    cmd += [
                        str(conf.size_n),
                        str(conf.size_k),
                        str(conf.size_m)
                    ]
                cmd += ["hw", "off"]
                try:
                    ret = run_process(
                        cmd,
                        kernelFolder,
                        pipe=True,
                        logPath=profilePath,
                        timeout=timeout)
                except sp.TimeoutExpired as err:
                    timeouts += 1
                    if timeouts > 10:
                        print(
                            "\n" + confStr +
                            ": exceeded maximum number of timeouts. Skipping.")
                        break
                    else:
                        print(confStr + ": timeout occurred. Retrying...")
                        continue
                if ret != 0:
                    raise Exception(confStr + ": kernel execution failed.")
                repsDone += 1
                timeouts = 0

def merge_files(source, destination):

    lines_out = OrderedDict()

    with open(destination, "r") as in_file:
        for line in in_file:
            lines_out[line] = None

    with open(source, "r") as in_file:
        for line in in_file:
            lines_out[line] = None

    with open(destination, "w") as out_file:
        for line in lines_out:
            out_file.write(line)


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("Specify a command to run: build, extract, "
              "package, unpackage, benchmark, merge_files, extract_benchmarks")
        sys.exit(1)

    if sys.argv[1] == "build":

        argParser = argparse.ArgumentParser()
        argParser.add_argument("procs", type=int)
        for key, val in PROJECT_CONFIG["options"].items():
            if "default" in val and val["default"] != None:
                default = val["default"]
                key = "--" + key
            else:
                default = None
            argParser.add_argument(key, default=str(default))
        argParser.add_argument("--cmake_opts", default="")
        args = argParser.parse_args(sys.argv[2:])

        argDict = vars(args)

        procs = int(argDict["procs"])
        if "cmake_opts" in argDict and argDict["cmake_opts"]:
            cmakeOpts = argDict["cmake_opts"].split(" ")
        else:
            cmakeOpts = []
        del argDict["procs"]
        del argDict["cmake_opts"]

        orderedArgs = OrderedDict()
        for key in argDict:
            orderedArgs[key] = [
                PROJECT_CONFIG["options"][key]["type"](val)
                for val in argDict[key].split(",")
            ]

        product = itertools.product(*orderedArgs.values())
        configs = [
            Configuration(**dict(zip(orderedArgs.keys(), x))) for x in product
        ]

        scan_configurations(procs, configs, cmakeOpts)

    elif sys.argv[1] == "extract":

        get_build_result(PROJECT_CONFIG["build_dir"])

    elif sys.argv[1] == "package":

        argParser = argparse.ArgumentParser()
        argParser.add_argument("target", type=str)
        args = vars(argParser.parse_args(sys.argv[2:]))

        package_configurations(args["target"])

    elif sys.argv[1] == "unpackage":

        argParser = argparse.ArgumentParser()
        argParser.add_argument("target", type=str)
        args = vars(argParser.parse_args(sys.argv[2:]))

        unpackage_configurations(args["target"])

    elif sys.argv[1] == "benchmark":

        argParser = argparse.ArgumentParser()
        argParser.add_argument("repetitions", type=int)
        argParser.add_argument("--timeout", type=float, default=30)
        args = vars(argParser.parse_args(sys.argv[2:]))

        benchmark(args["repetitions"], args["timeout"])

    elif sys.argv[1] == "merge_files":

        argParser = argparse.ArgumentParser()
        argParser.add_argument("source")
        argParser.add_argument("destination")
        args = argParser.parse_args(sys.argv[2:])

        merge_files(args.source, args.destination)

    elif sys.argv[1] == "extract_benchmarks":

        extract_benchmarks()

    else:

        raise ValueError("Unknown command \"{}\".".format(sys.argv[1]))
