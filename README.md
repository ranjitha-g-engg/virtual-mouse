"# virtual-mouse" 
# 🖱️ Virtual Mouse - Hand Gesture Control

A Python-based virtual mouse system that allows you to control your computer mouse using hand gestures captured through your webcam. This project uses computer vision and machine learning to track hand movements and translate them into mouse actions.

> 📸 Demo images and videos coming soon!

## 🌟 Features

### Current Features
- **Hand Tracking**: Real-time hand detection and landmark tracking
- **Mouse Movement**: Control cursor movement with hand position
- **Gesture Recognition**: Machine learning model for gesture classification
- **Click Actions**: Perform mouse clicks using specific hand gestures
- **Data Collection**: Built-in tool for collecting gesture training data

### 🚧 In Development
- Additional gesture types for advanced mouse actions
- Support for multiple gesture-based actions (right-click, drag, scroll, etc.)
- Improved accuracy and responsiveness
- Customizable gesture mappings

## 🛠️ Technologies Used

- **Python 3.8+**
- **OpenCV** - Computer vision and image processing
- **MediaPipe** - Hand tracking and landmark detection
- **PyAutoGUI** - Mouse control automation
- **scikit-learn** - Machine learning for gesture classification
- **NumPy & Pandas** - Data processing and analysis

## 📋 Prerequisites

Before running this project, make sure you have Python 3.8 or higher installed on your system.

## 🚀 Installation

1. Clone this repository:
```bash
git clone https://github.com/ranjitha-g-engg/virtual-mouse.git
cd virtual-mouse
```

2. Install required dependencies:
```bash
pip install opencv-python mediapipe pyautogui numpy pandas scikit-learn
```

## 📁 Project Structure

```
virtual-mouse/
│
├── virtualmouse.py          # Main virtual mouse application
├── virtualmouseedited.py    # Enhanced version with improvements
├── datacollector.py         # Tool for collecting gesture training data
├── 2_train_model.py         # Script to train the gesture recognition model
├── gesture_model.pkl        # Trained machine learning model
├── gestures.csv             # Dataset of collected gestures
└── README.md                # Project documentation
```

## 💻 Usage

### Running the Virtual Mouse

To start the virtual mouse application:

```bash
python virtualmouse.py
```

Or use the enhanced version:

```bash
python virtualmouseedited.py
```

### Collecting Training Data

To collect your own gesture data for training:

```bash
python datacollector.py
```

Follow the on-screen instructions to record different hand gestures.

### Training the Model

To train the gesture recognition model with collected data:

```bash
python 2_train_model.py
```

This will generate or update the `gesture_model.pkl` file.

## 🎮 How to Use

1. **Launch the Application**: Run the main script and ensure your webcam is working
2. **Position Your Hand**: Place your hand in front of the camera within the detection area
3. **Control the Cursor**: Move your hand to control the mouse cursor
4. **Perform Gestures**: Make specific hand gestures to perform clicks and other actions
5. **Exit**: Press 'q' or 'ESC' to quit the application

## 🎯 Current Gesture Support

> **Note**: Additional gestures are currently in development

- 👆 **Index Finger** -Cursor movement
- ✌️ **Peace Sign** - [In Development]
- 🤏 **Pinch** -  click

*More gestures and actions coming soon!*

## 🔧 Configuration

You can adjust various parameters in the script:
- Camera resolution
- Hand detection confidence
- Gesture sensitivity
- Mouse movement smoothing

## 📊 Model Training

The gesture recognition model is trained using:
- Hand landmark coordinates from MediaPipe
- Custom collected gesture datasets
- Machine learning classification algorithms

To improve accuracy:
1. Collect more diverse gesture samples using `datacollector.py`
2. Retrain the model with `2_train_model.py`
3. Test with different lighting conditions and backgrounds

## 🐛 Known Issues

- Performance may vary based on lighting conditions
- Requires clear background for optimal hand detection
- Gesture recognition accuracy improves with more training data

## 🚀 Future Enhancements

- [ ] Multi-hand gesture support
- [ ] Right-click and double-click gestures
- [ ] Drag and drop functionality
- [ ] Scroll gestures
- [ ] Customizable gesture mapping
- [ ] Improved gesture recognition accuracy
- [ ] GUI for configuration settings
- [ ] Support for different camera angles

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📝 License

This project is open source and available for educational purposes.

## 👩‍💻 Author

**Ranjitha G**
- GitHub: [@ranjitha-g-engg](https://github.com/ranjitha-g-engg)

## 🙏 Acknowledgments

- MediaPipe team for the hand tracking solution
- OpenCV community for computer vision tools
- All contributors and testers

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

⭐ If you find this project helpful, please consider giving it a star!

**Status**: 🚧 Active Development - New gestures and features coming soon!