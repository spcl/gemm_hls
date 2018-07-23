# Configuration variables
source config.tcl
set prjPath $workspace/${kernelName}_ex/${kernelName}_ex

open_project $prjPath.xpr

# 512-bit to 1 operand memory width converter
create_ip -name axis_dwidth_converter -vendor xilinx.com -library ip -module_name memory_to_n_converter
set cmd "set_property -dict \[list CONFIG.S_TDATA_NUM_BYTES {$memoryWidth} CONFIG.M_TDATA_NUM_BYTES {$kernelWidthN}] \[get_ips memory_to_n_converter]"
eval $cmd
generate_target {instantiation_template} [get_files $prjPath.srcs/sources_1/ip/memory_to_n_converter/memory_to_n_converter.xci]

# 512-bit to compute size in M memory width converter
create_ip -name axis_dwidth_converter -vendor xilinx.com -library ip -module_name memory_to_m_converter
set cmd "set_property -dict \[list CONFIG.S_TDATA_NUM_BYTES {$memoryWidth} CONFIG.M_TDATA_NUM_BYTES {$kernelWidthM}] \[get_ips memory_to_m_converter]"
eval $cmd
generate_target {instantiation_template} [get_files $prjPath.srcs/sources_1/ip/memory_to_m_converter/memory_to_m_converter.xci]

# Compute size in M to 512-bit memory width converter
create_ip -name axis_dwidth_converter -vendor xilinx.com -library ip -module_name m_to_memory_converter
set cmd "set_property -dict \[list CONFIG.S_TDATA_NUM_BYTES {$kernelWidthM} CONFIG.M_TDATA_NUM_BYTES {$memoryWidth}] \[get_ips m_to_memory_converter]"
eval $cmd
generate_target {instantiation_template} [get_files $prjPath.srcs/sources_1/ip/m_to_memory_converter/m_to_memory_converter.xci]

close_project
