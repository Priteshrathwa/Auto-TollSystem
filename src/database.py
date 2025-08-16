# src/database.py
"""MySQL database management for vehicles and transactions."""
import mysql.connector
import logging
from datetime import datetime

class DatabaseManager:
    """Manage MySQL database for vehicles and transactions."""
    
    def __init__(self, db_config):
        """Initialize database connection with MySQL config."""
        self.db_config = db_config
        self._setup_connection()
    
    def _setup_connection(self):
        """Set up MySQL connection."""
        try:
            self.conn = mysql.connector.connect(**self.db_config)
            self.cursor = self.conn.cursor()
            self.init_db()
        except mysql.connector.Error as e:
            logging.error(f"Database connection error: {e}")
            raise
    
    def init_db(self):
        """Create vehicles and transactions tables and seed sample data."""
        try:
            # Create vehicles table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS vehicles (
                    plate VARCHAR(20) PRIMARY KEY,
                    owner VARCHAR(100) NOT NULL,
                    balance FLOAT NOT NULL
                )
            ''')
            
            # Create transactions table (allow longer status messages)
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS transactions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    plate VARCHAR(20),
                    amount FLOAT,
                    timestamp DATETIME,
                    status VARCHAR(255),
                    FOREIGN KEY (plate) REFERENCES vehicles(plate)
                )
            ''')
            
            # Seed sample data if vehicles table is empty
            self.cursor.execute("SELECT COUNT(*) FROM vehicles")
            if self.cursor.fetchone()[0] == 0:
                sample_vehicles = [
                    ("ABC123", "John Doe", 100.0),
                    ("DEF456", "Jane Smith", 50.0),
                    ("GHI789", "Alice Johnson", 5.0),
                ]
                self.cursor.executemany(
                    "INSERT INTO vehicles (plate, owner, balance) VALUES (%s, %s, %s)",
                    sample_vehicles
                )
                logging.info("Seeded sample vehicle data.")
            
            self.conn.commit()
            logging.info("Database initialized successfully.")
        except mysql.connector.Error as e:
            logging.error(f"Database initialization error: {e}")
            raise
    
    def add_vehicle(self, plate, owner, balance):
        """Add a vehicle to the database."""
        try:
            self.cursor.execute(
                "INSERT INTO vehicles (plate, owner, balance) VALUES (%s, %s, %s)",
                (plate.upper(), owner, balance)
            )
            self.conn.commit()
            logging.info(f"Added vehicle: {plate}, Owner: {owner}, Balance: {balance}")
        except mysql.connector.IntegrityError:
            logging.error(f"Vehicle with plate {plate} already exists.")
            raise ValueError(f"Vehicle with plate {plate} already exists.")
        except mysql.connector.Error as e:
            logging.error(f"Error adding vehicle {plate}: {e}")
            raise
    
    def get_balance(self, plate):
        """Get vehicle balance by plate number."""
        try:
            self.cursor.execute("SELECT balance FROM vehicles WHERE plate = %s", (plate.upper(),))
            result = self.cursor.fetchone()
            if result is None:
                logging.error(f"Vehicle {plate} not found.")
                raise ValueError(f"Vehicle {plate} not found.")
            return result[0]
        except mysql.connector.Error as e:
            logging.error(f"Error retrieving balance for {plate}: {e}")
            raise
    
    def update_balance(self, plate, amount):
        """Update vehicle balance after toll deduction."""
        try:
            current_balance = self.get_balance(plate)
            new_balance = current_balance - amount
            if new_balance < 0:
                logging.warning(f"Insufficient balance for {plate}: {current_balance} < {amount}")
                raise ValueError("Insufficient balance.")
            self.cursor.execute(
                "UPDATE vehicles SET balance = %s WHERE plate = %s",
                (new_balance, plate.upper())
            )
            self.conn.commit()
            logging.info(f"Updated balance for {plate}: New balance = {new_balance}")
            return new_balance
        except mysql.connector.Error as e:
            logging.error(f"Error updating balance for {plate}: {e}")
            raise
    
    def log_transaction(self, plate, amount, status):
        """Log a toll transaction."""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.cursor.execute(
                "INSERT INTO transactions (plate, amount, timestamp, status) VALUES (%s, %s, %s, %s)",
                (plate.upper(), amount, timestamp, status)
            )
            self.conn.commit()
            logging.info(f"Logged transaction: Plate={plate}, Amount={amount}, Status={status}")
        except mysql.connector.Error as e:
            logging.error(f"Error logging transaction for {plate}: {e}")
            raise
    
    def get_report(self):
        """Generate a transaction report."""
        try:
            self.cursor.execute("SELECT id, plate, amount, timestamp, status FROM transactions")
            transactions = self.cursor.fetchall()
            return [
                {"id": t[0], "plate": t[1], "amount": t[2], "timestamp": str(t[3]), "status": t[4]}
                for t in transactions
            ]
        except mysql.connector.Error as e:
            logging.error(f"Error generating report: {e}")
            raise
    
    def __del__(self):
        """Close database connection when object is destroyed."""
        try:
            if hasattr(self, 'conn') and self.conn:
                try:
                    self.conn.close()
                    logging.info("Database connection closed.")
                except Exception as e:
                    logging.debug(f"Error closing database connection: {e}")
        except Exception:
            # Avoid raising during interpreter shutdown
            pass