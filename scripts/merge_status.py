#!/usr/bin/env python3
import argparse
import re
import sys

if __name__ == "__main__":

  argParser = argparse.ArgumentParser()
  argParser.add_argument("source", type=str)
  argParser.add_argument("target", type=str)
  args = argParser.parse_args()

  argDict = vars(args)

  #success,float,Multiply,Add,8,4096,4096,4096,4096,32,250,xilinx-xil-accel-rd-ku115-4ddr-xpr-4.0,240,1292,278878,391671,485,22.684
  pattern = "([a-zA-Z_]+),([\-a-zA-Z_0-9]+),([a-zA-Z_]+),([a-zA-Z_]+),([0-9]+),([0-9]+),([0-9]+),([0-9]+),([0-9]+),([0-9]+),([0-9]+)"

  with open(argDict["target"], "r") as targetFile:
    content = targetFile.read()

  with open(argDict["source"], "r") as sourceFile:
    with open(argDict["target"], "a") as targetFile:
      for line in sourceFile:
        m = re.search(pattern, line)
        if m:
          if m.group(1) in ["no_build", "failed_unknown"]:
            print("Skipping unfinished build " + str(m.groups()[1:]))
            continue
          statusStr = ",".join(m.groups()[1:])
          if statusStr in content:
            if line not in content:
              print("WARNING: inconsistency for " + str(m.groups()[1:]))
            continue
          print("Appending: " + str(m.groups()[1:]))
          targetFile.write(line)
