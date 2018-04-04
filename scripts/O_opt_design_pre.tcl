puts "Running custom TCL-script to fix kernel to the lower SLR"
set_property DONT_PARTITION TRUE [get_cells -hier -filter {REF_NAME==dr_MatrixMatrix_1_0}]

add_cells_to_pblock pblock_lower [get_cells [list xcl_design_i/expanded_region/u_ocl_region/dr_i/MatrixMatrix_1]]
