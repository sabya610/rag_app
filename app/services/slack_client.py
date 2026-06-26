"""
Slack API client integration for rag_app
Handles authentication and basic Slack API operations
"""

import logging
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from app.config import Config

logger = logging.getLogger(__name__)


class SlackClient:
    """Wrapper around Slack SDK WebClient"""
    
    def __init__(self):
        self.token = Config.SLACK_BOT_TOKEN
        if not self.token:
            raise ValueError("SLACK_BOT_TOKEN environment variable not set")
        
        self.client = WebClient(token=self.token)
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify Slack connection by testing auth"""
        try:
            response = self.client.auth_test()
            logger.info(f"✓ Slack connected as: {response['user']} in workspace: {response['team']}")
            return True
        except SlackApiError as e:
            logger.error(f"✗ Slack connection failed: {e.response['error']}")
            raise
    
    def get_channels(self, exclude_archived=True, limit=100):
        """Get list of channels the bot is member of"""
        try:
            channels = []
            cursor = None
            
            while True:
                response = self.client.conversations_list(
                    exclude_archived=exclude_archived,
                    limit=limit,
                    cursor=cursor,
                    types="public_channel,private_channel"
                )
                channels.extend(response['channels'])
                
                cursor = response.get('response_metadata', {}).get('next_cursor')
                if not cursor:
                    break
            
            logger.info(f"Found {len(channels)} channels")
            return channels
        except SlackApiError as e:
            logger.error(f"Error fetching channels: {e.response['error']}")
            raise
    
    def get_channel_history(self, channel_id, limit=100, oldest=None):
        """Get message history from a channel"""
        try:
            kwargs = {
                'channel': channel_id,
                'limit': limit,
            }
            if oldest:
                kwargs['oldest'] = oldest
            
            response = self.client.conversations_history(**kwargs)
            messages = response.get('messages', [])
            logger.info(f"Fetched {len(messages)} messages from channel {channel_id}")
            return messages
        except SlackApiError as e:
            logger.error(f"Error fetching channel history: {e.response['error']}")
            raise
    
    def get_thread_replies(self, channel_id, thread_ts, limit=100):
        """Get all replies in a thread"""
        try:
            response = self.client.conversations_replies(
                channel=channel_id,
                ts=thread_ts,
                limit=limit,
                inclusive=True  # Include the root message
            )
            messages = response.get('messages', [])
            logger.info(f"Fetched {len(messages)} messages from thread {thread_ts}")
            return messages
        except SlackApiError as e:
            logger.error(f"Error fetching thread replies: {e.response['error']}")
            raise
    
    def get_user_info(self, user_id):
        """Get user information"""
        try:
            response = self.client.users_info(user=user_id)
            return response['user']
        except SlackApiError as e:
            logger.error(f"Error fetching user info: {e.response['error']}")
            return None
    
    def get_channel_info(self, channel_id):
        """Get channel information"""
        try:
            response = self.client.conversations_info(channel=channel_id)
            return response['channel']
        except SlackApiError as e:
            logger.error(f"Error fetching channel info: {e.response['error']}")
            return None
