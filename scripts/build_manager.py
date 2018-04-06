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
  "kernelName": "MatrixMatrix",
  "kernelFile": "MatrixMatrix",
  "makeSynthesis": "synthesis",
  "makeKernel": "build_kernel",
  "executeKernel": ["./RunMatrixMatrix"],
  "buildDir": "scan",
  "benchmarkDir": "benchmark",
  "options": OrderedDict([
    ("dataType", {
        "cmake": "MM_DATA_TYPE",
        "type": str,
        "default": None
    }),
    ("mapOp", {
        "cmake": "MM_MAP_OP",
        "type": str,
        "default": None
    }),
    ("reduceOp", {
        "cmake": "MM_REDUCE_OP",
        "type": str,
        "default": None
    }),
    ("kernelWidth", {
        "cmake": "MM_KERNEL_WIDTH",
        "type": int,
        "default": None
    }),
    ("sizeN", {
        "cmake": "MM_SIZE_N",
        "type": int,
        "default": None
    }),
    ("sizeM", {
        "cmake": "MM_SIZE_M",
        "type": int,
        "default": None
    }),
    ("sizeP", {
        "cmake": "MM_SIZE_P",
        "type": int,
        "default": None
    }),
    ("tileSizeP", {
        "cmake": "MM_TILE_SIZE_P",
        "type": int,
        "default": None
    }),
    ("tileSizeN", {
        "cmake": "MM_TILE_SIZE_N",
        "type": int,
        "default": None
    }),
    ("targetClock", {
        "cmake": "MM_TARGET_CLOCK",
        "type": int,
        "default": 250
    }),
    ("target", {
        "cmake": "MM_DSA_NAME",
        "type": str,
        "default": "xilinx:xil-accel-rd-ku115:4ddr-xpr:4.0"
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
        raise KeyError("Option \"" + opt + "\" set both as arg and kwarg")
      setattr(self, opt, PROJECT_CONFIG["options"][opt]["type"](val))
    unsetArgs = PROJECT_CONFIG["options"].keys() - self.__dict__.keys()
    for arg in unsetArgs:
      default = PROJECT_CONFIG["options"][arg]["default"]
      if default != None:
        setattr(self, opt, default)
        unsetArgs.remove(arg)
    if len(unsetArgs) > 0:
      raise TypeError("Missing arguments: {}".format(", ".join(unsetArgs)))

  @staticmethod
  def csv_header():
    return ",".join(PROJECT_CONFIG["options"].keys())

  def to_string(self):
    return "_".join([str(getattr(self, opt)).replace(":", "-").replace("_", "-")
                     for opt in PROJECT_CONFIG["options"]])

  def to_csv(self):
    return ",".join(
        map(str, [getattr(self, opt) for opt in PROJECT_CONFIG["options"]]))

  def build_folder(self):
    return "build_" + self.to_string()

  def benchmark_folder(self):
    return "benchmark_" + self.to_string()

  def __str__(self):
    return self.to_string()

  def __repr__(self):
    return self.to_string()

  def cmake_command(self, sourceDir, extra=[]):
    return (["cmake", sourceDir] +
            ["-D{}={}".format(val["cmake"], getattr(self, key))
             for key, val in PROJECT_CONFIG["options"].items()] + extra)

  @staticmethod
  def get_conf(s):
    pattern = ""
    for opt in PROJECT_CONFIG["options"].values():
      if opt["type"] == str:
        pattern += "([^_]+)_"
      elif opt["type"] == int:
        pattern += "([0-9]+)_"
      else:
        raise TypeError("Unsupported type \"{}\".".format(str(opt["type"])))
    m = re.search(pattern[:-1], s) # Removing trailing underscore
    if not m:
      raise ValueError("Not a valid configuration string: " + s)
    return Configuration(*m.groups())


def run_process(command, directory, pipe=True, logPath="log", timeout=None):
  if pipe:
    proc = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE,
                    universal_newlines=True, cwd=directory)
    stdout, stderr = proc.communicate()
    with open(os.path.join(directory, logPath + ".out"), "a") as outFile:
      outFile.write(stdout)
    with open(os.path.join(directory, logPath + ".err"), "a") as outFile:
      outFile.write(stderr)
  else:
    proc = sp.Popen(command,
                    universal_newlines=True, cwd=directory)
    try:
      proc.communicate(timeout=timeout)
    except sp.TimeoutExpired as err:
      proc.terminate()
      raise err
  return proc.returncode

class Consumption(object):

  def __init__(self, conf, status, lut, ff, dsp, bram, power, clock):
    self.conf = conf
    self.status = status
    self.lut = lut
    self.ff = ff
    self.dsp = dsp
    self.bram = bram
    self.power = power
    self.clock = clock

  @staticmethod
  def csv_cols():
    return ("status," + Configuration.csv_header() +
            ",clock,dsp,lut,ff,bram,power")

  def __repr__(self):
    return ",".join(map(str, [
        self.status, self.conf.to_csv(), self.clock,
        self.dsp, self.lut, self.ff, self.bram, self.power]))


def do_build(conf, cmakeOpts):
  cmakeCommand = conf.cmake_command("$1", cmakeOpts)
  confStr = conf.to_string()
  confDir = os.path.join(PROJECT_CONFIG["buildDir"], conf.build_folder())
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
  print("[{}] {}: {}".format(time_only(datetime.datetime.now()),
                             str(conf.to_string()), status))


def run_build(conf, clean=True, hardware=True):
  confStr = conf.to_string()
  confDir = os.path.join(PROJECT_CONFIG["buildDir"], conf.build_folder())
  print_status(conf, "Configuring...")
  if run_process(["sh", "configure.sh",
                  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))],
                 confDir) != 0:
    raise Exception(confStr + ": Configuration failed.")
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
    # timeStart = datetime.datetime.now()
    # print_status(conf, "Running HLS...")
    # if run_process(["make", PROJECT_CONFIG["makeSynthesis"]], confDir) != 0:
    #   raise Exception(confStr + ": HLS failed.")
    # print_status(conf, "Finished HLS after {}.".format(
    #     time_only(datetime.datetime.now() - timeStart)))
    timeStart = datetime.datetime.now()
    print_status(conf, "Starting kernel build...")
    if run_process(["make", PROJECT_CONFIG["makeKernel"]], confDir) != 0:
      try:
        with open(os.path.join(confDir, "log.out")) as logFile:
          m = re.search("auto frequency scaling failed", logFile.read())
          if not m:
            print_status(conf, "FAILED after {}.".format(
                time_only(datetime.datetime.now() - timeStart)))
          else:
            print(conf, "TIMING failed after {}.".format(
                time_only(datetime.datetime.now() - timeStart)))
      except FileNotFoundError:
        print_status(conf, "FAILED after {}.".format(
            time_only(datetime.datetime.now() - timeStart)))
    else:
      print_status(conf, "SUCCESS in {}.".format(
          time_only(datetime.datetime.now() - timeStart)))

def extract_result_build(conf):
  buildFolder = os.path.join(PROJECT_CONFIG["buildDir"], conf.build_folder())
  xoccFolder = ("_xocc_" + PROJECT_CONFIG["kernelName"] + "_" +
                PROJECT_CONFIG["kernelFile"] + ".dir")
  if not os.path.exists(os.path.join(buildFolder, xoccFolder)):
    conf.consumption = Consumption(conf, "no_intermediate", None, None, None,
                                   None, None, None)
    return
  kernelFolder = os.path.join(
      buildFolder, xoccFolder,
      "impl", "build", "system", PROJECT_CONFIG["kernelFile"], "bitstream")
  implFolder = os.path.join(
      kernelFolder, PROJECT_CONFIG["kernelFile"] + "_ipi", "ipiimpl",
      "ipiimpl.runs", "impl_1")
  if not os.path.exists(implFolder):
    conf.consumption = Consumption(conf, "no_build", None, None, None, None,
                                   None, None)
    return
  status = check_build_status(conf)
  reportPath = os.path.join(
      implFolder, "xcl_design_wrapper_utilization_placed.rpt")
  if not os.path.isfile(reportPath):
    conf.consumption = Consumption(conf, status, None, None, None, None, None,
                                   None)
    return
  report = open(reportPath).read()
  try:
    luts = int(re.search(
        "CLB LUTs[ \t]*\|[ \t]*([0-9]+)", report).group(1))
    ff = int(re.search(
        "CLB Registers[ \t]*\|[ \t]*([0-9]+)", report).group(1))
  except AttributeError:
    luts = int(re.search(
        "Slice LUTs[ \t]*\|[ \t]*([0-9]+)", report).group(1))
    ff = int(re.search(
        "Slice Registers[ \t]*\|[ \t]*([0-9]+)", report).group(1))
  bram = int(re.search(
      "Block RAM Tile[ \t]*\|[ \t]*([0-9]+)", report).group(1))
  dsp = int(re.search(
      "DSPs[ \t]*\|[ \t]*([0-9]+)", report).group(1))
  reportPath = os.path.join(implFolder, "xcl_design_wrapper_power_routed.rpt")
  if not os.path.isfile(reportPath):
    power = 0
  else:
    report = open(reportPath).read()
    power = float(re.search(
        "Total On-Chip Power \(W\)[ \t]*\|[ \t]*([0-9\.]+)", report).group(1))
  try:
    with open(os.path.join(kernelFolder, PROJECT_CONFIG["kernelFile"] + "_ipi",
                           "vivado_warning.txt"), "r") as clockFile:
      warningText = clockFile.read()
      m = re.search("automatically changed to ([0-9]+) MHz", warningText)
      if m:
        clock = int(m.group(1))
      else:
        clock = conf.targetClock
  except FileNotFoundError:
    clock = conf.targetClock
  conf.consumption = Consumption(
      conf, status, luts, ff, dsp, bram, power, clock)

def check_build_status(conf):
  buildFolder = os.path.join(PROJECT_CONFIG["buildDir"], conf.build_folder())
  kernelFolder = os.path.join(
      buildFolder, ("_xocc_" + PROJECT_CONFIG["kernelFile"] + "_" +
                    PROJECT_CONFIG["kernelFile"] + ".dir"),
      "impl", "build", "system", PROJECT_CONFIG["kernelName"], "bitstream")
  try:
    log = open(
        os.path.join(buildFolder, "log.out"), "r").read()
  except:
    return "no_build"
  try:
    report = open(
        os.path.join(kernelFolder, PROJECT_CONFIG["kernelName"] + "_ipi",
                     "vivado.log")).read()
  except FileNotFoundError:
    return "no_build"
  m = re.search("Implementation Feasibility check failed", log)
  if m:
    return "failed_feasibility"
  m = re.search("Detail Placement failed", log)
  if m:
    return "failed_placement"
  m = re.search("Placer could not place all instances", report)
  if m:
    return "failed_placement"
  m = re.search("Routing results verification failed due to partially-conflicted nets", report)
  if m:
    return "failed_routing"
  m = re.search("route_design ERROR", log)
  if m:
    return "failed_routing"
  m = re.search("Internal Data Exception", log)
  if m:
    return "crashed"
  m = re.search("Design failed to meet timing - hold violation.", report)
  if m:
    return "failed_hold"
  m = re.search("auto frequency scaling failed", report)
  if m:
    return "failed_timing"
  m = re.search("Unable to write message .+ as it exceeds maximum size", report)
  if m:
    return "failed_report"
  for fileName in os.listdir(kernelFolder):
    if len(fileName) >= 7 and fileName.endswith(".xclbin"):
      return "success"
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
  with open(os.path.join(PROJECT_CONFIG["buildDir"],
                         "build_status.csv"), "w") as resultFile:
    resultFile.write(Consumption.csv_cols() + "\n")
    for conf in confs:
      resultFile.write(str(conf.consumption) + "\n")

def scan_configurations(numProcs, configurations, cmakeOpts):
  try:
    os.makedirs(PROJECT_CONFIG["buildDir"])
  except FileExistsError:
    pass
  pool = mp.Pool(processes=numProcs)
  try:
    pool.starmap(do_build, zip(configurations, len(configurations)*[cmakeOpts]))
  except KeyboardInterrupt:
    pool.terminate()
    print("Builds successfully aborted.")
  else:
    print("All configurations finished running.")

def files_to_copy(conf):
  filesToCopy = ["configure.sh", PROJECT_CONFIG["kernelName"] + ".xclbin"]
  kernelString = PROJECT_CONFIG["kernelName"]
  xoccFolder = ("_xocc_" + PROJECT_CONFIG["kernelFile"] + "_" +
                PROJECT_CONFIG["kernelName"] + ".dir")
  hlsFolder = os.path.join(
      xoccFolder, "impl", "kernels", PROJECT_CONFIG["kernelName"])
  kernelFolder = os.path.join(
      xoccFolder, "impl", "build",
      "system", kernelString, "bitstream", kernelString + "_ipi")
  filesToCopy.append(os.path.join(hlsFolder, "vivado_hls.log"))
  filesToCopy.append(os.path.join(kernelFolder, "vivado.log"))
  filesToCopy.append(os.path.join(kernelFolder, "vivado_warning.txt"))
  implFolder = os.path.join(
      kernelFolder, "ipiimpl", "ipiimpl.runs", "impl_1")
  filesToCopy.append(os.path.join(
      implFolder, "xcl_design_wrapper_utilization_placed.rpt"))
  filesToCopy.append(os.path.join(
      implFolder, "xcl_design_wrapper_power_routed.rpt"))
  return implFolder, hlsFolder, filesToCopy

def package_configurations(target):
  kernelsPackaged = 0
  for fileName in os.listdir(PROJECT_CONFIG["buildDir"]):
    try:
      conf = Configuration.get_conf(fileName)
    except ValueError:
      continue
    if conf.target != target:
      continue
    sourceDir = os.path.join(PROJECT_CONFIG["buildDir"], fileName)
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
    implFolder, hlsFolder, filesToCopy = files_to_copy(conf)
    try:
      os.makedirs(os.path.join(packageFolder, implFolder))
    except FileExistsError:
      pass
    try:
      os.makedirs(os.path.join(packageFolder, hlsFolder))
    except FileExistsError:
      pass
    for path in filesToCopy:
      try:
        shutil.copy(os.path.join(sourceDir, path),
                    os.path.join(packageFolder, path))
      except FileNotFoundError as err:
        if path.endswith("vivado_warning.txt"):
          with open(os.path.join(packageFolder, path), "w") as outFile:
            pass
        else:
          raise err
    kernelsPackaged += 1
  if kernelsPackaged > 0:
    print(("Successfully packaged " + str(kernelsPackaged) +
           " kernels and configuration files into \"{}\".").format(target))
  else:
    print("No kernels for target \"{}\" found in \"{}\".".format(
        target, PROJECT_CONFIG["buildDir"]))

def unpackage_configuration(conf):
  confStr = conf.to_string()
  fileName = conf.build_folder()
  print("Unpackaging {}...".format(confStr))
  sourceDir = os.path.join(conf.target, fileName)
  targetDir = os.path.join(PROJECT_CONFIG["buildDir"], fileName)
  implFolder, hlsFolder, filesToCopy = files_to_copy(conf)
  try:
    os.makedirs(os.path.join(targetDir, implFolder))
  except FileExistsError:
    pass
  try:
    os.makedirs(os.path.join(targetDir, hlsFolder))
  except FileExistsError:
    pass
  for path in filesToCopy:
    shutil.copy(os.path.join(sourceDir, path), os.path.join(targetDir, path))
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
  pool = mp.Pool(processes=len(confs))
  try:
    pool.map(unpackage_configuration, confs)
  except KeyboardInterrupt:
    pool.terminate()
  if unpackagedSomething:
    print("Successfully unpackaged kernels into \"{}\".".format(
        PROJECT_CONFIG["buildDir"]))
  else:
    print("No kernels found in \"{}\".".format(target))

def benchmark(repetitions, timeout):
  for fileName in os.listdir(PROJECT_CONFIG["buildDir"]):
    try:
      conf = Configuration.get_conf(fileName)
    except ValueError:
      continue
    confStr = conf.to_string()
    folderName = conf.benchmark_folder()
    kernelFolder = os.path.join(PROJECT_CONFIG["buildDir"], fileName)
    kernelString = PROJECT_CONFIG["kernelName"]
    kernelPath = os.path.join(kernelFolder, kernelString + ".xclbin")
    if not os.path.exists(kernelPath):
      continue
    benchmarkFolder = os.path.join("benchmark", conf.benchmark_folder())
    try:
      os.makedirs(benchmarkFolder)
    except FileExistsError:
      pass
    shutil.copy(os.path.join(kernelFolder, "configure.sh"), benchmarkFolder)
    print("Running {}...".format(confStr))
    if run_process(["make"], kernelFolder, pipe=False) != 0:
      raise Exception(confStr + ": software build failed.")
    repsDone = 0
    timeouts = 0
    while repsDone < repetitions:
      time.sleep(0.5)
      print("Running iteration {} / {}...".format(repsDone + 1, repetitions))
      try:
        ret = run_process("./RunMatrixMatrix.exe".split(),
                          kernelFolder, pipe=False, timeout=timeout)
      except sp.TimeoutExpired as err:
        timeouts += 1
        if timeouts > 10:
          print("\n" + confStr + ": exceeded maximum number of timeouts. Skipping.")
          break
        else:
          print(confStr + ": timeout occurred. Retrying...")
          continue
      if ret != 0:
        raise Exception(confStr + ": kernel execution failed.")
      repsDone += 1
      timeouts = 0
      profilePath = os.path.join(kernelFolder, "sdaccel_profile_summary.csv")
      shutil.copy(profilePath,
                  os.path.join(benchmarkFolder,
                               str(datetime.datetime.now()).replace(" ", "_")
                               + ".csv"))


if __name__ == "__main__":

  if len(sys.argv) < 2:

    print("Specify a command to run: build, extract, package, unpackage")
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
    argParser.add_argument("--cmakeOpts", default="")
    args = argParser.parse_args(sys.argv[2:])

    argDict = vars(args)

    procs = int(argDict["procs"])
    if "cmakeOpts" in argDict and argDict["cmakeOpts"]:
      cmakeOpts = ["-D" + arg for arg in argDict["cmakeOpts"].split(" ")]
    else:
      cmakeOpts = []
    del argDict["procs"]
    del argDict["cmakeOpts"]

    orderedArgs = OrderedDict()
    for key in argDict:
      orderedArgs[key] = [PROJECT_CONFIG["options"][key]["type"](val) for val in
                          argDict[key].split(",")]

    product = itertools.product(*orderedArgs.values())
    configs = [Configuration(**dict(zip(orderedArgs.keys(), x))) for x in product]

    scan_configurations(procs, configs, cmakeOpts)

  elif sys.argv[1] == "extract":

    get_build_result(PROJECT_CONFIG["buildDir"])

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

  else:

    raise ValueError("Unknown command \"{}\".".format(sys.argv[1]))
