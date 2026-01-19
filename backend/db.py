import sqlite3
from datetime import datetime
import os

def init_db():
    """Initialize the SQLite database"""
    # Use absolute path to backend directory
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(backend_dir, 'invoices.db')
    
    print(f"🔍 Database path: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if tables exist first
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='documents'")
    table_exists = cursor.fetchone() is not None
    
    if not table_exists:
        print("⚠️ Creating new tables...")
        # Drop existing tables if they exist (to recreate with correct schema)
        cursor.execute('DROP TABLE IF EXISTS items')
        cursor.execute('DROP TABLE IF EXISTS documents')
        
        # Create documents table with ALL required columns
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vendor_name TEXT,
                vendor_address TEXT,
                vendor_phone TEXT,
                vendor_email TEXT,
                invoice_number TEXT,
                invoice_date TEXT,
                due_date TEXT,
                po_number TEXT,
                subtotal REAL,
                tax REAL,
                total REAL,
                currency TEXT,
                raw_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create items table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                description TEXT,
                quantity INTEGER,
                unit_price REAL,
                amount REAL,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            )
        ''')
        
        conn.commit()
    else:
        print("✅ Tables already exist")
    
    # Always show record count
    cursor.execute("SELECT COUNT(*) FROM documents")
    doc_count = cursor.fetchone()[0]
    print(f"✅ Database has {doc_count} document(s)")
    
    return conn
