# AutoToll System

A comprehensive toll collection system with automated license plate recognition using YOLO and OCR technology.

## Features

- **License Plate Recognition**: Automated detection and recognition using YOLO v8 and EasyOCR
- **Web Interface**: Complete web dashboard for managing toll operations
- **Multiple Input Sources**: Support for camera, image, and video file processing
- **Database Integration**: MySQL database for transaction and vehicle management
- **Real-time Processing**: Live camera feed with detection controls
- **Transaction Management**: Complete toll transaction tracking and payment status management

## Project Structure

```
toll_system/
├── app/
│   ├── main.py                     # Main application entry point
│   ├── config.py                   # Configuration settings
│   ├── recognition/
│   │   └── plate_recognition.py    # License plate recognition logic
│   ├── database/
│   │   └── db_connector.py        # Database connection and operations
│   ├── transactions/
│   │   └── transaction_manager.py  # Transaction processing logic
│   └── web/
│       ├── routes.py              # Flask web routes
│       ├── templates/             # HTML templates
│       └── static/                # CSS, JS, and image files
├── requirements.txt               # Python dependencies
└── logs/                         # Application logs
```

## Installation

1. **Clone/Extract the project**
   ```bash
   cd toll_system
   ```

2. **Activate virtual environment** (already created)
   ```bash
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure database**
   - Ensure MySQL is running
   - Database: `tollsystem`
   - Password: `pritesh10`
   - Tables should already be created as per README.txt

5. **Update YOLO model path**
   - Ensure the YOLO model is at: `D:\AUTOTOLL\yolo model\best.pt`
   - Or update the path in `app/config.py`

## Usage

### Running the Application

```bash
python app/main.py
```

The web interface will be available at: http://localhost:5000

### Web Interface Features

1. **Dashboard** (`/`)
   - System statistics and overview
   - Recent transactions
   - Quick action buttons

2. **Camera Detection** (`/camera`)
   - Live camera feed
   - Start/stop detection
   - Real-time license plate recognition
   - Manual frame capture

3. **File Upload** (`/upload`)
   - Upload images or videos
   - Batch processing
   - Detection results display

4. **Transactions** (`/transactions`)
   - View all transactions
   - Update payment status
   - Search and filter options

5. **Vehicle Management** (`/vehicles`)
   - Registered vehicle information
   - Exemption and blacklist management
   - Vehicle details

### API Endpoints

- `GET /api/stats` - System statistics
- `GET /api/transactions` - Transaction list
- `POST /api/update_payment` - Update payment status
- `POST /start_camera` - Start camera feed
- `POST /stop_camera` - Stop camera feed
- `POST /toggle_detection` - Toggle detection on/off
- `POST /upload_file` - Upload and process files

## Configuration

Edit `app/config.py` to customize:

- Database connection settings
- YOLO model path
- OCR confidence thresholds
- Toll amounts and rules
- Application settings

## Database Schema

### Tables

1. **vehicles**
   - `plate` (Primary Key)
   - `owner_name`
   - `vehicle_type`
   - `fastag_id`
   - `exemption`
   - `blacklist`
   - `created_at`

2. **transactions**
   - `id` (Primary Key)
   - `plate`
   - `toll_amount`
   - `payment_method`
   - `payment_status`
   - `location`
   - `cam_id`
   - `captured_at`
   - `raw_ocr_text`
   - `ocr_confidence`
   - `notes`

3. **toll_rules**
   - `id` (Primary Key)
   - `vehicle_type`
   - `base_amount`
   - `created_at`

## Development

### Adding New Features

1. **Recognition improvements**: Modify `app/recognition/plate_recognition.py`
2. **Database operations**: Extend `app/database/db_connector.py`
3. **Business logic**: Update `app/transactions/transaction_manager.py`
4. **Web interface**: Add routes in `app/web/routes.py` and templates
5. **Styling**: Modify `app/web/static/css/style.css`

### Testing

The system supports multiple input methods for testing:

- **Camera**: Real-time detection from webcam
- **Images**: Upload JPG, PNG, BMP files
- **Videos**: Upload MP4, AVI, MOV, MKV files

## Troubleshooting

### Common Issues

1. **Camera not detected**
   - Check camera permissions
   - Ensure no other application is using the camera
   - Try different camera index (0, 1, 2...)

2. **YOLO model not found**
   - Verify the model path in config.py
   - Ensure the .pt file exists

3. **Database connection error**
   - Check MySQL service is running
   - Verify credentials in config.py
   - Ensure database and tables exist

4. **OCR accuracy issues**
   - Adjust OCR_CONFIDENCE_THRESHOLD 
   - Ensure good lighting conditions
   - Use high-resolution images

### Logs

Application logs are stored in the `logs/` directory:
- `app.log` - Application events
- `detection.log` - Detection-specific logs (if enabled)

## License

This project is developed for educational and demonstration purposes.

## Support

For issues and questions, check the logs and ensure all dependencies are properly installed.
