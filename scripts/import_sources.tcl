# Configuration variables
source config.tcl
set prjPath $workspace/${kernelName}_ex/${kernelName}_ex

open_project $prjPath.xpr

# Verilog files out of kernel hls project
import_files $hlsDir 
# Floating point IP cores
foreach file [glob -dir $hlsDir *.tcl] {
  source $file
}

close_project
