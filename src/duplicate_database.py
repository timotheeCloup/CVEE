#!/usr/bin/env python3
"""
Before running this script you need to run in another terminal:
kubectl port-forward -n cvee postgres-5cd58c56f-dm8wr 5433:5432
Script to duplicate a local PostgreSQL database to Supabase.
Automatically backs up the local database and restores it to Supabase,
overwriting existing tables if they already exist.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
import psycopg2
from dotenv import load_dotenv


def load_env():
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
    
    local_config = {
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT"),
        "database": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
    }
    
    supabase_config = {
        "host": os.getenv("SB_HOST"),
        "port": os.getenv("SB_PORT"),
        "database": os.getenv("SB_NAME"),
        "user": os.getenv("SB_USER"),
        "password": os.getenv("SB_PASSWORD"),
    }
    
    return local_config, supabase_config


def get_connection(db_config):
    """Establish a connection to the database."""
    try:
        conn = psycopg2.connect(
            host=db_config["host"],
            port=db_config["port"],
            database=db_config["database"],
            user=db_config["user"],
            password=db_config["password"],
        )
        return conn
    except psycopg2.Error as e:
        print(f"Error: Failed to connect to database {db_config['host']}:{db_config['port']}/{db_config['database']}")
        print(f"Details: {e}")
        sys.exit(1)


def create_backup(db_config):
    """Create a backup from the local database using pg_dump."""
    print("Step 1: Creating backup from local database...")
    
    env = os.environ.copy()
    env["PGPASSWORD"] = db_config["password"]
    
    cmd = [
        "pg_dump",
        "-h", db_config["host"],
        "-p", str(db_config["port"]),
        "-U", db_config["user"],
        "-d", db_config["database"],
        "--no-owner",
        "--no-privileges",
    ]
    
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sql", delete=False) as f:
            output_file = f.name
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                check=True,
            )
        
        file_size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"   Backup created successfully ({file_size:.2f} MB)")
        return output_file
    
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to create backup")
        print(f"Details: {e.stderr}")
        sys.exit(1)


def drop_all_tables(db_config):
    """Drop all tables in the Supabase database."""
    print("Step 2: Dropping existing tables from Supabase...")
    
    try:
        conn = get_connection(db_config)
        conn.autocommit = True
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT tablename FROM pg_tables 
            WHERE schemaname = 'public'
        """)
        tables = cursor.fetchall()
        
        if tables:
            print(f"   Found {len(tables)} table(s) to drop")
            
            for (table,) in tables:
                cursor.execute(f"DROP TABLE IF EXISTS {table} CASCADE;")
            
            print("   All existing tables dropped")
        else:
            print("   No existing tables found")
        
        cursor.close()
        conn.close()
    
    except psycopg2.Error as e:
        print(f"Error: Failed to drop tables from Supabase")
        print(f"Details: {e}")
        sys.exit(1)


def restore_backup(db_config, backup_file):
    """Restore the backup to Supabase database."""
    print("Step 3: Restoring backup to Supabase...")
    
    env = os.environ.copy()
    env["PGPASSWORD"] = db_config["password"]
    
    cmd = [
        "psql",
        "-h", db_config["host"],
        "-p", str(db_config["port"]),
        "-U", db_config["user"],
        "-d", db_config["database"],
    ]
    
    try:
        with open(backup_file, "r") as f:
            result = subprocess.run(
                cmd,
                stdin=f,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
            )
        
        if result.returncode == 0:
            print("   Backup restored successfully to Supabase")
            return True
        else:
            print("Error: Failed to restore backup to Supabase")
            print(f"Details: {result.stderr}")
            return False
    
    except Exception as e:
        print(f"Error: Failed to restore backup")
        print(f"Details: {e}")
        return False


def cleanup_backup(backup_file):
    """Delete the temporary backup file."""
    try:
        os.remove(backup_file)
        print("Step 4: Cleanup completed (temporary backup deleted)")
    except Exception as e:
        print(f"Warning: Could not delete temporary backup file: {e}")


def main():
    """Main function - executes the complete duplication process."""
    print("\n" + "=" * 60)
    print("LOCAL POSTGRESQL TO SUPABASE DATABASE DUPLICATION")
    print("=" * 60 + "\n")
    
    local_config, supabase_config = load_env()
    
    print("Source Configuration:")
    print(f"   Local: {local_config['host']}:{local_config['port']}/{local_config['database']}")
    print("\nTarget Configuration:")
    print(f"   Supabase: {supabase_config['host']}:{supabase_config['port']}/{supabase_config['database']}")
    print()
    
    try:
        print("Verifying connections...")
        
        # Test local connection
        conn = get_connection(local_config)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM pg_tables WHERE schemaname='public'")
        table_count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        print(f"   Local database connected ({table_count} tables)")
        
        # Test Supabase connection
        conn = get_connection(supabase_config)
        conn.close()
        print("   Supabase connected")
        
        print()
        
        # Execute duplication process
        backup_file = create_backup(local_config)
        drop_all_tables(supabase_config)
        restore_backup(supabase_config, backup_file)
        cleanup_backup(backup_file)
        
        print("\n" + "=" * 60)
        print("DUPLICATION COMPLETED SUCCESSFULLY")
        print("=" * 60 + "\n")
    
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
