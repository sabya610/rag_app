#!/usr/bin/env python
"""
Database migration script for Slack integration
Creates slack_messages and slack_threads tables with pgvector support
"""

import os
import sys
from app import create_app
from app.models import db, SlackMessage, SlackThread

def init_slack_db():
    """Initialize Slack tables in the database"""
    app = create_app()
    
    with app.app_context():
        print("[*] Creating Slack integration tables...")
        
        # Create tables if they don't exist
        db.create_all()
        
        # Create indexes for performance
        try:
            from sqlalchemy import text
            
            # Check if tables exist
            inspector = db.inspect(db.engine)
            tables = inspector.get_table_names()
            
            if 'slack_messages' in tables:
                print("✓ slack_messages table created")
                
                # Create indexes
                db.session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_slack_msg_channel 
                    ON slack_messages(channel_id);
                """))
                
                db.session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_slack_msg_thread 
                    ON slack_messages(thread_id);
                """))
                
                db.session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_slack_msg_ts 
                    ON slack_messages(timestamp DESC);
                """))
                
                print("✓ slack_messages indexes created")
            
            if 'slack_threads' in tables:
                print("✓ slack_threads table created")
                
                # Create indexes
                db.session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_slack_thread_channel 
                    ON slack_threads(channel_id);
                """))
                
                db.session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_slack_thread_ts 
                    ON slack_threads(timestamp DESC);
                """))
                
                print("✓ slack_threads indexes created")
            
            db.session.commit()
            print("\n[✓] Slack database migration completed successfully!")
            
        except Exception as e:
            print(f"[!] Warning: {e}")
            db.session.rollback()

if __name__ == '__main__':
    init_slack_db()
