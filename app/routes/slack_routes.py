"""
Slack integration routes for rag_app
Endpoints for importing Slack messages and searching/querying them
"""

import logging
from flask import Blueprint, request, jsonify
from sqlalchemy import func, or_
from app.models import db, SlackMessage, SlackThread
from app.services.slack_import import SlackImportService
from app.utils import get_embedder

logger = logging.getLogger(__name__)

slack_routes = Blueprint('slack', __name__, url_prefix='/api/slack')


@slack_routes.route('/import', methods=['POST'])
def import_slack_messages():
    """
    Import Slack messages from channels into database
    POST data: {
        "channel_ids": ["C123", "C456"],  # optional, imports all if not provided
        "days_back": 30  # optional, defaults to Config.SLACK_IMPORT_DAYS
    }
    """
    try:
        data = request.get_json() or {}
        channel_ids = data.get('channel_ids')
        days_back = data.get('days_back')
        
        import_service = SlackImportService()
        stats = import_service.import_channels(channel_ids, days_back)
        
        return jsonify({
            'status': 'success',
            'message': 'Slack messages imported successfully',
            'stats': stats
        }), 200
    except Exception as e:
        logger.error(f"Error importing Slack messages: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@slack_routes.route('/search', methods=['POST'])
def search_slack_messages():
    """
    Search Slack messages using keyword and/or semantic search
    POST data: {
        "query": "help with setup",
        "search_type": "semantic|keyword|hybrid",  # default: hybrid
        "limit": 10,
        "channel_filter": "general"  # optional
    }
    """
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        search_type = data.get('search_type', 'hybrid')
        limit = min(int(data.get('limit', 10)), 50)
        channel_filter = data.get('channel_filter')
        
        if not query:
            return jsonify({'status': 'error', 'message': 'Query is required'}), 400
        
        if search_type == 'keyword':
            results = _keyword_search(query, limit, channel_filter)
        elif search_type == 'semantic':
            results = _semantic_search(query, limit, channel_filter)
        else:  # hybrid
            results = _hybrid_search(query, limit, channel_filter)
        
        return jsonify({
            'status': 'success',
            'query': query,
            'search_type': search_type,
            'result_count': len(results),
            'results': results
        }), 200
    except Exception as e:
        logger.error(f"Error searching Slack messages: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@slack_routes.route('/threads/<thread_id>', methods=['GET'])
def get_thread(thread_id):
    """
    Get a complete thread with all replies
    """
    try:
        thread = SlackThread.query.filter_by(thread_id=thread_id).first()
        if not thread:
            return jsonify({'status': 'error', 'message': 'Thread not found'}), 404
        
        # Get all replies in thread
        replies = SlackMessage.query.filter_by(thread_id=thread_id).order_by(
            SlackMessage.timestamp
        ).all()
        
        return jsonify({
            'status': 'success',
            'thread': thread.to_dict(),
            'replies': [msg.to_dict() for msg in replies],
            'reply_count': len(replies)
        }), 200
    except Exception as e:
        logger.error(f"Error fetching thread: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@slack_routes.route('/channels', methods=['GET'])
def get_channels_stats():
    """
    Get statistics about imported Slack channels
    """
    try:
        channels = db.session.query(
            SlackMessage.channel_id,
            SlackMessage.channel_name,
            func.count(SlackMessage.id).label('message_count')
        ).group_by(SlackMessage.channel_id, SlackMessage.channel_name).all()
        
        stats = [
            {
                'channel_id': ch[0],
                'channel_name': ch[1],
                'message_count': ch[2]
            }
            for ch in channels
        ]
        
        return jsonify({
            'status': 'success',
            'total_channels': len(stats),
            'channels': stats
        }), 200
    except Exception as e:
        logger.error(f"Error fetching channel stats: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@slack_routes.route('/stats', methods=['GET'])
def get_slack_stats():
    """
    Get overall statistics about imported Slack data
    """
    try:
        total_messages = db.session.query(func.count(SlackMessage.id)).scalar()
        total_threads = db.session.query(func.count(SlackThread.id)).scalar()
        total_channels = db.session.query(func.count(func.distinct(SlackMessage.channel_id))).scalar()
        
        return jsonify({
            'status': 'success',
            'total_messages': total_messages,
            'total_threads': total_threads,
            'total_channels': total_channels
        }), 200
    except Exception as e:
        logger.error(f"Error fetching stats: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


def _keyword_search(query, limit, channel_filter=None):
    """Keyword search using SQL ILIKE"""
    q = SlackMessage.query.filter(
        SlackMessage.text.ilike(f'%{query}%')
    )
    
    if channel_filter:
        q = q.filter(SlackMessage.channel_name.ilike(f'%{channel_filter}%'))
    
    results = q.order_by(SlackMessage.timestamp.desc()).limit(limit).all()
    
    return [
        {
            **msg.to_dict(),
            'relevance': 'keyword_match',
            'score': 1.0
        }
        for msg in results
    ]


def _semantic_search(query, limit, channel_filter=None):
    """Semantic search using pgvector similarity"""
    try:
        embedder = get_embedder()
        query_embedding = embedder.encode(query)
        
        # Use pgvector cosine similarity
        from sqlalchemy import func as sa_func, cast, String
        from pgvector.sqlalchemy import Vector
        
        similarity = (1 - SlackMessage.embedding.cosine_distance(query_embedding)).label('similarity')
        
        q = db.session.query(SlackMessage, similarity).order_by(similarity.desc())
        
        if channel_filter:
            q = q.filter(SlackMessage.channel_name.ilike(f'%{channel_filter}%'))
        
        results = q.limit(limit).all()
        
        return [
            {
                **msg[0].to_dict(),
                'relevance': 'semantic_match',
                'score': float(msg[1]) if msg[1] else 0.0
            }
            for msg in results
        ]
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        return []


def _hybrid_search(query, limit, channel_filter=None):
    """Hybrid search combining keyword and semantic search"""
    keyword_results = _keyword_search(query, limit // 2, channel_filter)
    semantic_results = _semantic_search(query, limit // 2, channel_filter)
    
    # Combine and deduplicate by message_id, prefer higher scores
    combined = {}
    for result in keyword_results + semantic_results:
        msg_id = result['message_id']
        if msg_id not in combined or result['score'] > combined[msg_id]['score']:
            combined[msg_id] = result
    
    # Sort by score and return
    return sorted(combined.values(), key=lambda x: x['score'], reverse=True)[:limit]
