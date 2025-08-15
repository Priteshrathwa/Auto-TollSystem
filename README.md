# AutoTollSystem

A Python-based simulation of an automatic toll collection system, designed to showcase computer vision, OCR, database management, and CLI development skills for a developer portfolio. The system detects and reads vehicle license plates from images (with planned video support), verifies against a SQLite database, deducts toll fees if the vehicle has sufficient balance, logs transactions, and generates reports.

![Demo GIF](docs/demo.gif) *(To be added after implementation)*

## Table of Contents
- [Features](#features)
- [Technologies](#technologies)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

## Features
- **License Plate Recognition**: Detects and reads license plates from images using OpenCV (contour detection) and EasyOCR.
- **Toll Deduction**: Simulates toll fee deduction from a vehicle's balance stored in a SQLite database.
- **Database Management**: Supports adding vehicles, updating balances, and logging transactions.
- **CLI Interface**: User-friendly commands to process images, manage vehicles, and view transaction reports.
- **Logging**: Tracks all operations (e.g., detection failures, toll deductions) in a log file.
- **Extensibility**: Modular design for easy integration of video processing or alternative OCR engines.
- **Testing**: Unit and integration tests with Pytest for robust code.

## Technologies
- **Python**: 3.12+
- **Libraries**:
  - OpenCV: Image processing for plate detection.
  - EasyOCR: Optical character recognition for reading plates.
  - SQLite3: Lightweight database for vehicle and transaction data.
  - Click: Command-line interface.
  - PyYAML: Configuration file parsing.
  - Pytest: Unit and integration testing.
  - Matplotlib: Optional visualizations (e.g., annotated images).
- **Tools**: Black and Flake8 for linting, Git for version control.

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/AutoTollSystem.git
   cd AutoTollSystem
   ```
2. **Set up a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Initialize the database**:
   ```bash
   python src/main.py init-db
   ```

## Usage
The system is controlled via a CLI interface. Below are the main commands:

- **Process an image** to detect a license plate and deduct toll:
  ```bash
  python src/main.py process --image data/sample_images/car1.jpg
  ```
  *Output example*:
  ```
  Detected License Plate: ABC123
  Toll of $10.0 deducted for John Doe. New balance: $90.0
  ```

- **Add a vehicle** to the database:
  ```bash
  python src/main.py add-vehicle --plate ABC123 --owner "John Doe" --balance 100.0
  ```

- **View transaction report**:
  ```bash
  python src/main.py report
  ```
  *Output example*:
  ```
  Transaction Report:
  ID | Plate  | Amount | Timestamp           | Status
  1  | ABC123 | 10.0   | 2025-08-15 21:00:00 | Success
  2  | GHI789 | 10.0   | 2025-08-15 21:01:00 | Failed (Insufficient balance)
  ```

- **Initialize database** with sample data:
  ```bash
  python src/main.py init-db
  ```

*Note*: Place test images in `data/sample_images/`. Sample images are not included in the repo but can be sourced from open datasets (e.g., Kaggle car plate datasets).

## Project Structure
```
AutoTollSystem/
├── src/
│   ├── main.py               # CLI entry point
│   ├── recognizer.py         # License plate detection and OCR
│   ├── database.py           # SQLite database management
│   ├── toll_processor.py     # Toll deduction logic
│   ├── utils.py              # Helpers (logging, image loading)
│   └── config.py             # Configuration loader
├── data/
│   ├── database.db           # SQLite database
│   └── sample_images/        # Test images/videos
├── tests/
│   ├── test_recognizer.py    # Recognition tests
│   ├── test_database.py      # Database tests
│   ├── test_toll_processor.py # Toll processing tests
│   └── test_utils.py         # Helper tests
├── docs/
│   ├── README.md             # This file
│   └── architecture.md       # System design
├── config.yaml               # Configuration (toll amount, log level)
├── requirements.txt          # Dependencies
└── .gitignore                # Git ignore file
```

## Limitations
- **OCR Accuracy**: Dependent on image quality (lighting, angle, resolution). Preprocessing tweaks are included, but real-world scenarios may require advanced models like YOLO.
- **Simulation**: No real payment processing; balances are stored in SQLite.
- **Single-Threaded**: Image processing is sequential. Video support (planned) will need optimization.
- **Offline**: No external API integration for now (e.g., cloud OCR or payment gateways).

## Future Enhancements
- **Video Support**: Process live webcam feeds or video files using OpenCV's VideoCapture.
- **Web Interface**: Add a Flask/Django frontend for browser-based interaction.
- **Advanced OCR**: Replace EasyOCR with YOLO-based detection for better accuracy.
- **Database Scalability**: Migrate to PostgreSQL for larger-scale deployments.
- **Real-Time Alerts**: Email/SMS notifications for insufficient balance (via APIs).
- **Docker**: Containerize the app for easy deployment.

## Contributing
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit changes (`git commit -m "Add YourFeature"`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

Please follow PEP 8 style guidelines and include tests for new features.

## License
MIT License. See [LICENSE](LICENSE) for details.