# Windows批处理脚本 - 手部重建Demo运行示例

@echo off
chcp 65001 >nul
echo ======================================
echo 3D手部重建Demo - 运行示例
echo ======================================

REM 配置参数
set INPUT_VIDEO=./videos/hand_video.mp4
set OUTPUT_VIDEO=./output/hand_recon_result.mp4

REM 相机内参（根据实际情况修改）
set CAMERA_FX=1000.0
set CAMERA_FY=1000.0

REM 处理参数
set MAX_FRAMES=100
set THRESH=0.5
set FPS=30
set DEVICE=cuda

REM 创建输出目录
if not exist "output" mkdir output

echo.
echo 参数配置：
echo   输入: %INPUT_VIDEO%
echo   输出: %OUTPUT_VIDEO%
echo   焦距: fx=%CAMERA_FX%, fy=%CAMERA_FY%
echo   设备: %DEVICE%
echo.

REM 运行demo
python demo.py ^
    --input %INPUT_VIDEO% ^
    --output %OUTPUT_VIDEO% ^
    --camera_fx %CAMERA_FX% ^
    --camera_fy %CAMERA_FY% ^
    --max_frames %MAX_FRAMES% ^
    --thresh %THRESH% ^
    --fps %FPS% ^
    --device %DEVICE%

echo.
echo ======================================
echo 处理完成！
echo 输出视频: %OUTPUT_VIDEO%
echo ======================================
pause
