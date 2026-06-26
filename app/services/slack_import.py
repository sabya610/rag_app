"""
Slack import service for rag_app
Handles importing Slack messages into the database with vector embeddings
"""

import logging
import time
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from app.models import db, SlackMessage, SlackThread
from app.config import Config
from app.services.slack_client import SlackClient

logger = logging.getLogger(__name__)


class SlackImportService:
    """Service to import Slack messages and create embeddings"""
    
    def __init__(self):
        self.slack_client = SlackClient()
        self.embedder = SentenceTransformer(Config.EMBEDDING_MODEL)
        logger.info("SlackImportService initialized")
    
    def import_channels(self, channel_list=None, days_back=None):
        """
        Import messages from specified channels or all channels
        
        Args:
            channel_list: List of channel IDs to import. If None, imports all channels
            days_back: Number of days back to import. Uses Config.SLACK_IMPORT_DAYS if None
        
        Returns:
            dict with import statistics
        """
        if days_back is None:
            days_back = Config.SLACK_IMPORT_DAYS
        
        if channel_list is None:
            channels = self.slack_client.get_channels()
            channel_list = [ch['id'] for ch in channels]
        
        logger.info(f"Starting import for {len(channel_list)} channels (last {days_back} days)")
        
        stats = {
            'total_messages': 0,
            'total_threads': 0,
            'failed_channels': [],
            'channels_processed': 0
        }
        
        oldest_timestamp = self._get_oldest_timestamp(days_back)
        
        for channel_id in channel_list:
            try:
                channel_stats = self._import_channel(channel_id, oldest_timestamp)
                stats['total_messages'] += channel_stats['messages']
                stats['total_threads'] += channel_stats['threads']
                stats['channels_processed'] += 1
                logger.info(f"✓ Imported {channel_stats['messages']} messages from channel {channel_id}")
            except Exception as e:
                logger.error(f"✗ Failed to import channel {channel_id}: {str(e)}")
                stats['failed_channels'].append(channel_id)
            
            # Rate limiting
            time.sleep(1)
        
        logger.info(f"Import complete: {stats['total_messages']} messages, {stats['total_threads']} threads")
        return stats
    
    def _import_channel(self, channel_id, oldest_timestamp):
        """Import messages from a single channel"""
        channel_info = self.slack_client.get_channel_info(channel_id)
        channel_name = channel_info['name'] if channel_info else channel_id
        
        messages = self.slack_client.get_channel_history(
            channel_id,
            limit=Config.SLACK_IMPORT_LIMIT,
            oldest=oldest_timestamp
        )
        
        stats = {'messages': 0, 'threads': 0}
        processed_threads = set()
        
        for msg in messages:
            # Skip bot messages and messages without text
            if msg.get('subtype') or not msg.get('text'):
                continue
            
            user_id = msg.get('user')
            if not user_id:
                continue
            
            # Get user info
            user_info = self.slack_client.get_user_info(user_id)
            user_name = user_info['real_name'] if user_info else user_id
            
            # Import root message
            message_id = msg['ts']
            thread_id = msg.get('thread_ts')
            
            # Check if this is a threaded message (has thread_ts and it's not the root)
            if thread_id and thread_id != message_id:
                # This is a reply in a thread, import it as SlackMessage
                self._create_slack_message(
                    message_id=message_id,
                    channel_id=channel_id,
                    channel_name=channel_name,
                    user_id=user_id,
                    user_name=user_name,
                    text=msg['text'],
                    thread_id=thread_id,
                    timestamp=datetime.fromtimestamp(float(message_id))
                )
                stats['messages'] += 1
            else:
                # This is a root message
                self._create_slack_message(
                    message_id=message_id,
                    channel_id=channel_id,
                    channel_name=channel_name,
                    user_id=user_id,
                    user_name=user_name,
                    text=msg['text'],
                    thread_id=None,
                    timestamp=datetime.fromtimestamp(float(message_id))
                )
                stats['messages'] += 1
                
                # If this message has replies, import the thread
                if msg.get('reply_count', 0) > 0 and message_id not in processed_threads:
                    thread_stats = self._import_thread(
                        channel_id, channel_name, message_id, msg, user_id, user_name
                    )
                    stats['threads'] += thread_stats['threads']
                    stats['messages'] += thread_stats['messages']
                    processed_threads.add(message_id)
        
        return stats
    
    def _import_thread(self, channel_id, channel_name, thread_ts, root_msg, root_user_id, root_user_name):
        """Import all replies in a thread"""
        try:
            thread_messages = self.slack_client.get_thread_replies(channel_id, thread_ts)
            
            # Create SlackThread record from root message
            root_embedding = self.embedder.encode(root_msg['text'])
            
            existing_thread = SlackThread.query.filter_by(thread_id=thread_ts).first()
            if not existing_thread:
                thread_record = SlackThread(
                    thread_id=thread_ts,
                    channel_id=channel_id,
                    channel_name=channel_name,
                    root_user_id=root_user_id,
                    root_user_name=root_user_name,
                    root_text=root_msg['text'],
                    thread_embedding=root_embedding,
                    reply_count=root_msg.get('reply_count', 0),
                    timestamp=datetime.fromtimestamp(float(thread_ts))
                )
                db.session.add(thread_record)
            
            # Import thread replies
            stats = {'threads': 1, 'messages': 0}
            
            for reply_msg in thread_messages:
                # Skip the root message (already imported)
                if reply_msg['ts'] == thread_ts:
                    continue
                
                # Skip bot messages and messages without text
                if reply_msg.get('subtype') or not reply_msg.get('text'):
                    continue
                
                reply_user_id = reply_msg.get('user')
                if not reply_user_id:
                    continue
                
                user_info = self.slack_client.get_user_info(reply_user_id)
                user_name = user_info['real_name'] if user_info else reply_user_id
                
                self._create_slack_message(
                    message_id=reply_msg['ts'],
                    channel_id=channel_id,
                    channel_name=channel_name,
                    user_id=reply_user_id,
                    user_name=user_name,
                    text=reply_msg['text'],
                    thread_id=thread_ts,
                    timestamp=datetime.fromtimestamp(float(reply_msg['ts']))
                )
                stats['messages'] += 1
            
            db.session.commit()
            return stats
        except Exception as e:
            logger.error(f"Error importing thread {thread_ts}: {str(e)}")
            db.session.rollback()
            return {'threads': 0, 'messages': 0}
    
    def _create_slack_message(self, message_id, channel_id, channel_name, user_id, user_name, text, thread_id, timestamp):
        """Create a SlackMessage record with embedding"""
        try:
            # Check if message already exists
            existing = SlackMessage.query.filter_by(message_id=message_id).first()
            if existing:
                return
            
            # Generate embedding
            embedding = self.embedder.encode(text)
            
            # Create record
            slack_msg = SlackMessage(
                message_id=message_id,
                channel_id=channel_id,
                channel_name=channel_name,
                user_id=user_id,
                user_name=user_name,
                text=text,
                embedding=embedding,
                thread_id=thread_id,
                timestamp=timestamp
            )
            
            db.session.add(slack_msg)
            db.session.commit()
        except Exception as e:
            logger.error(f"Error creating slack message: {str(e)}")
            db.session.rollback()
    
    @staticmethod
    def _get_oldest_timestamp(days_back):
        """Get Unix timestamp for N days ago"""
        now = datetime.utcnow()
        oldest = now - timedelta(days=days_back)
        return str(int(oldest.timestamp()))
