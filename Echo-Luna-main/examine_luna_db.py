#!/usr/bin/env python3
"""
Examine Luna Memories Database
=============================

Script to examine the structure and content of the Luna memories database
to understand how to integrate it with the H.I.M. model.
"""

import sqlite3
import json

def examine_luna_database():
    """Examine the Luna memories database structure and content."""
    print("Examining Luna Memories Database...")
    print("=" * 50)
    
    try:
        # Connect to the database
        conn = sqlite3.connect('luna_memories.db')
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"Tables found: {tables}")
        
        # Examine each table
        for table in tables:
            table_name = table[0]
            print(f"\n--- Table: {table_name} ---")
            
            # Get table schema
            cursor.execute(f"PRAGMA table_info({table_name})")
            schema = cursor.fetchall()
            print("Schema:")
            for column in schema:
                print(f"  {column[1]} ({column[2]})")
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"Row count: {count}")
            
            # Get sample data
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
            sample_data = cursor.fetchall()
            print("Sample data:")
            for i, row in enumerate(sample_data):
                print(f"  Row {i+1}: {row}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error examining database: {e}")

if __name__ == "__main__":
    examine_luna_database()

