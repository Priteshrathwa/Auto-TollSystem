# tests/test_database.py
"""Unit tests for DatabaseManager."""
import pytest
import mysql.connector
from src.database import DatabaseManager

@pytest.fixture
def db_manager():
    """Create a temporary test database for testing."""
    db_config = {
        'host': 'localhost',
        'user': 'autotoll_user',
        'password': 'your_password',
        'database': 'autotoll_test'
    }
    
    # Create test database
    conn = mysql.connector.connect(
        host=db_config['host'],
        user=db_config['user'],
        password=db_config['password']
    )
    cursor = conn.cursor()
    cursor.execute("CREATE DATABASE IF NOT EXISTS autotoll_test")
    conn.commit()
    cursor.close()
    conn.close()
    
    db = DatabaseManager(db_config)
    yield db
    
    # Clean up: Drop test database (use same password key and ignore errors)
    try:
        conn = mysql.connector.connect(
            host=db_config['host'],
            user=db_config['user'],
            password=db_config['password']
        )
        cursor = conn.cursor()
        cursor.execute("DROP DATABASE IF EXISTS autotoll_test")
        conn.commit()
        cursor.close()
        conn.close()
    except Exception:
        # Avoid failing teardown if DB already removed or connection fails
        pass

def test_init_db(db_manager):
    """Test database initialization."""
    db_manager.init_db()
    db_manager.cursor.execute("SHOW TABLES")
    tables = [row[0] for row in db_manager.cursor.fetchall()]
    assert 'vehicles' in tables
    assert 'transactions' in tables

def test_add_vehicle(db_manager):
    """Test adding a vehicle."""
    db_manager.add_vehicle("TEST123", "Test User", 100.0)
    db_manager.cursor.execute("SELECT * FROM vehicles WHERE plate = 'TEST123'")
    result = db_manager.cursor.fetchone()
    assert result == ("TEST123", "Test User", 100.0)