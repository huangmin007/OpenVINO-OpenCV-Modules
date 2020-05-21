rem @echo off 
rem 关闭自动输出
:begin

set /p dirinput=请输OpenVINO安装路径:
set /p yn=确定路径:%dirinput%(y/n)?

if /i "%yn%" == "y" (goto yes) else (goto begin)

:yes
set SETX="C:\Windows\System32\setx.exe"

rem 设置 OpenVINO 安装路径 /M
%SETX% "INTEL_OPENVINO_DIR_T" %dirinput%
%SETX% "INTEL_CVSDK_DIR_T" "%%INTEL_OPENVINO_DIR_T%%"

rem 设置 OpenCV 路径
%SETX% "OpenCV_DIR" "%%INTEL_OPENVINO_DIR_T%%\opencv\cmake"
%SETX% PATH "%%INTEL_OPENVINO_DIR_T%%\opencv\bin;%%PATH%%"

pause
