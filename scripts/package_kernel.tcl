source config.tcl
set prjDir $workspace/${kernelName}_ex

open_project $prjDir/${kernelName}_ex.xpr

source $prjDir/imports/package_kernel.tcl

if ($callPackageFunction) {

  set path_to_project [get_property DIRECTORY [current_project]]
  set ip_name ${kernel_name}_v1_0
  set xo_file ${path_to_project}/sdx_imports/${kernel_name}.xo
  set kernel_xml_file ${path_to_project}/imports/kernel.xml
  set path_to_packaged ${path_to_project}/${ip_name}
  puts "INFO: Running package_project command"

  if {[catch {package_project $path_to_packaged $path_to_project $kernel_vendor $kernel_library} ret]} {
    puts $ret
    puts "ERROR: Package project failed."
    return 1
  } else {
    puts $ret
    puts "INFO: Successfully packaged project into IP: ${path_to_packaged}"
  }

  # Generate xo file
  if {[file exists $xo_file]} {
    file delete -force $xo_file
  }
  puts "INFO: Running package_xo command"
  if {[catch {set xo [package_xo -xo_path $xo_file -kernel_name $kernel_name -ip_directory $path_to_packaged -kernel_xml $kernel_xml_file]} ret]} {
    puts $ret
    puts "ERROR: Package kernel xo failed."
    return 1
  } else {
    puts $ret
    puts "INFO: Successfully generated kernel xo file: $xo"
  }
}

close_project
