rem @echo off 
rem �ر��Զ����
:begin

set /p dirinput=����OpenVINO��װ·��:
set /p yn=ȷ��·��:%dirinput%(y/n)?

if /i "%yn%" == "y" (goto yes) else (goto begin)

:yes
set SETX="C:\Windows\System32\setx.exe"

rem ���� OpenVINO ��װ·�� /M
%SETX% "INTEL_OPENVINO_DIR_T" %dirinput%
%SETX% "INTEL_CVSDK_DIR_T" "%%INTEL_OPENVINO_DIR_T%%"

rem ���� OpenCV ·��
%SETX% "OpenCV_DIR" "%%INTEL_OPENVINO_DIR_T%%\opencv\cmake"
%SETX% PATH "%%INTEL_OPENVINO_DIR_T%%\opencv\bin;%%PATH%%"

pause
