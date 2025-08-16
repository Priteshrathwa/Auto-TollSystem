# src/main.py
"""Command-line interface for the AutoTollSystem using Click."""

import click
import logging
from .utils import setup_logging
from .config import load_config
from .database import DatabaseManager
from .toll_processor import TollProcessor
from .utils import PlateNotDetectedError

@click.group()
def cli():
    """AutoTollSystem CLI: Manage vehicles, process tolls, and generate reports."""
    config = load_config()
    setup_logging(config['log_file'], config['log_level'])
    logging.info("CLI started.")

@cli.command()
@click.option('--image', required=True, help='Path to image for license plate recognition')
def process(image):
    """Process an image to recognize license plate and deduct toll."""
    config = load_config()
    setup_logging(config['log_file'], config['log_level'])
    logging.info(f"Processing image: {image}")
    click.echo(f"Processing image: {image}")

    toll_amount = config.get('toll_amount', 0)
    db_config = config.get('db_config')
    ocr_lang = config.get('ocr_lang', 'en')

    processor = TollProcessor(db_config, toll_amount, ocr_lang)
    result = processor.process_vehicle(image)

    if result['status'] == 'success':
        click.echo(f"Detected Plate: {result['plate']}")
        click.echo(f"Vehicle Type: {result.get('vehicle_type', 'unknown')}")
        click.echo(result['message'])
    elif result['status'] == 'failed':
        click.echo(f"Failed to process plate. Plate: {result.get('plate')}. Vehicle Type: {result.get('vehicle_type', 'unknown')}. Reason: {result.get('message')}")
    else:
        click.echo(f"Error processing image: {result.get('message')}")

@cli.command()
@click.option('--plate', required=True, help='License plate number')
@click.option('--owner', required=True, help='Vehicle owner name')
@click.option('--balance', required=True, type=float, help='Initial balance')
def add_vehicle(plate, owner, balance):
    """Add a vehicle to the database."""
    config = load_config()
    db = DatabaseManager(config['db_config'])
    try:
        db.add_vehicle(plate, owner, balance)
        click.echo(f"Added vehicle: {plate}, Owner: {owner}, Balance: {balance}")
    except ValueError as e:
        click.echo(f"Error: {e}")
    except Exception as e:
        click.echo(f"Unexpected error: {e}")

@cli.command()
def report():
    """Generate a transaction report."""
    config = load_config()
    db = DatabaseManager(config['db_config'])
    try:
        transactions = db.get_report()
        if not transactions:
            click.echo("No transactions found.")
        else:
            click.echo("Transaction Report:")
            click.echo("ID | Plate | Amount | Timestamp | Status")
            for t in transactions:
                click.echo(f"{t['id']} | {t['plate']} | {t['amount']} | {t['timestamp']} | {t['status']}")
    except Exception as e:
        click.echo(f"Error generating report: {e}")

@cli.command()
def init_db():
    """Initialize the database with sample data."""
    config = load_config()
    db = DatabaseManager(config['db_config'])
    try:
        db.init_db()
        click.echo("Database initialized with sample data.")
    except Exception as e:
        click.echo(f"Error initializing database: {e}")

if __name__ == '__main__':
    cli()
