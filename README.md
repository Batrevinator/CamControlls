# Motion Capture System

This project implements a real-time human motion detection system using OpenCV in C++.

## Features

- **Live Camera Capture**: Captures video feed from the computer's default camera
- **Motion Detection**: Uses background subtraction (MOG2 algorithm) to detect moving objects
- **Real-time Processing**: Processes frames in real-time with morphological operations and filtering
- **Dual Display**: Shows both the original live feed and the motion detection mask
- **Screenshot Capability**: Press 's' to save the current frame as an image
- **Background Model Reset**: Press 'c' to clear and reset the background model

## Building and Running

### Prerequisites
- OpenCV 4.x installed on your system
- CMake 3.5 or higher
- C++ compiler (g++ recommended)

### Build Instructions
```bash
mkdir build
cd build
cmake ..
make
```

### Running the Application
```bash
./MotionCapture
```

### Controls
- **q** or **ESC**: Quit the application
- **s**: Save current frame as PNG image
- **c**: Clear background model (useful if lighting changes)

## Camera Setup Issues

### If you're running in WSL (Windows Subsystem for Linux)

WSL doesn't have direct access to Windows cameras. Here are your options:

#### Option 1: Run from Windows (Recommended)
1. Install OpenCV for Windows
2. Compile and run the application directly in Windows Command Prompt or PowerShell
3. The camera will work normally

#### Option 2: Use WSL with USB passthrough (Advanced)
1. Use WSL2 with USB device passthrough (requires Windows 11 Pro or Enterprise)
2. Connect your camera as a USB device to WSL
3. This is complex and may not work reliably

#### Option 3: Use Virtual Camera Software
1. Install virtual camera software on Windows
2. Stream Windows camera feed to a virtual device
3. Access the virtual device from WSL

### General Camera Troubleshooting

If you get "No camera found" errors:

1. **Check camera connection**: Make sure your camera is plugged in and working
2. **Check permissions**: Try `sudo chmod 666 /dev/video*`
3. **Check device status**: Run `ls -la /dev/video*` to see available devices
4. **Close other applications**: Make sure no other app is using the camera
5. **Try different camera index**: The app automatically tries indices 0-9

### Testing Camera Access

To verify camera setup:
```bash
# Install diagnostic tools
sudo apt install v4l-utils usbutils

# List video devices
v4l2-ctl --list-devices

# List USB devices
lsusb

# Check video device permissions
ls -la /dev/video*
```

## Technical Details

The system uses:
- **BackgroundSubtractorMOG2**: For robust background modeling and foreground detection
- **Morphological Operations**: Opening and closing to reduce noise in the motion mask
- **Gaussian Blur**: To smooth the detection results
- **Thresholding**: To create a binary motion mask

## Future Enhancements

- Human pose estimation and tracking
- Gesture recognition
- Integration with game controllers
- Multi-camera support
- Performance optimizations