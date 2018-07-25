source config.tcl
set prjDir $workspace/${kernelName}_ex

open_project $prjDir/${kernelName}_ex.xpr

source $prjDir/imports/package_kernel.tcl

if ($callPackageFunction) {

  package_project $prjDir/$kernelName $kernelVendor $kernelLibrary $kernelName 
  package_xo -xo_path $prjDir/sdx_imports/$kernelName.xo -kernel_name $kernelName -ip_directory $prjDir/$kernelName -kernel_xml $prjDir/imports/kernel.xml

}

close_project
