# Author: Johannes de Fine Licht (johannes.definelicht@inf.ethz.ch)
# Date:   October 2017

# Implemented according to:
# https://www.xilinx.com/support/documentation/sw_manuals/xilinx2017_2/ug1023-sdaccel-user-guide.pdf

# Load configuration variables
source config.tcl

# Internal variables 
set kernelPrjName ${kernelName}_ex
set wizardDir $tmpDir/kernel_wizard
set kernelDir $workspace/$kernelPrjName

# We first create a dummy project for the kernel wizard, which will create the
# actual kernel project
create_project kernel_wizard $wizardDir -part $partName -force

# Instantiate the SDx kernel wizard IP
create_ip -name sdx_kernel_wizard -vendor xilinx.com -library ip -module_name $kernelName 

# We have to first write the command to a string so the variables are expanded,
# otherwise the variable names (rather than their content) are written to the
# generated tcl files, and the example project will fail.
# TODO: generalize this
set cmd "set_property -dict \[list CONFIG.NUM_INPUT_ARGS {0} CONFIG.NUM_M_AXI {3} CONFIG.M00_AXI_NUM_ARGS {1} CONFIG.M00_AXI_ARG00_NAME {a} CONFIG.M01_AXI_NUM_ARGS {1} CONFIG.M01_AXI_ARG00_NAME {b} CONFIG.M02_AXI_NUM_ARGS {1} CONFIG.M02_AXI_ARG00_NAME {c} CONFIG.KERNEL_NAME {$kernelName} CONFIG.KERNEL_VENDOR {$kernelVendor} CONFIG.KERNEL_LIBRARY {$kernelLibrary}] \[get_ips $kernelName]"
# Configure for correct OpenCL input arguments, including memory interfaces.
eval $cmd

set kernelXci $wizardDir/kernel_wizard.srcs/sources_1/ip/$kernelName/$kernelName.xci

# Generate kernel wizard IP core
generate_target {instantiation_template} [get_files $kernelXci]
set_property generate_synth_checkpoint false [get_files $kernelXci]
generate_target all [get_files $kernelXci]

# Reopen project to generate cache
close_project
open_project $wizardDir/kernel_wizard.xpr

# Export files (potentially a lot of things can be removed here)
export_ip_user_files -of_objects [get_files $kernelXci] -no_script -sync -force -quiet
export_simulation -of_objects [get_files $kernelXci] -directory $wizardDir/kernel_wizard.ip_user_files/sim_scripts -ip_user_files_dir $wizardDir/kernel_wizard.ip_user_files -ipstatic_source_dir $wizardDir/kernel_wizard.ip_user_files/ipstatic -lib_map_path [list {modelsim=$wizardDir/kernel_wizard.cache/compile_simlib/modelsim} {questa=$wizardDir/kernel_wizard.cache/compile_simlib/questa} {ies=$wizardDir/kernel_wizard.cache/compile_simlib/ies} {vcs=$wizardDir/kernel_wizard.cache/compile_simlib/vcs} {riviera=$wizardDir/kernel_wizard.cache/compile_simlib/riviera}] -use_ip_compiled_libs -force -quiet

# The IP will generate a script to generate the example project, which we now
# source. This implicitly closes the wizard project and opens the kernel
# project instead
open_example_project -force -dir $workspace [get_ips $kernelName]

# Close and clean up wizard project
close_project
file delete -force $wizardDir
