import zmq 

import pickle 

from datetime import datetime

from PIL import Image

from io import BytesIO

from os import path 

from time import sleep, time 

import zmq 
import logging 

import asyncio 

from deep_translator import GoogleTranslator

from urllib.parse import urlparse

from typing import Dict, List, Optional, Iterable, Tuple 
from glob import glob 

from openai.types.chat import ChatCompletionChunk

from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    filters, 
    Application,
    ContextTypes,
    CommandHandler,
    ConversationHandler,
    MessageHandler
)
from os import path 

from src.embedding import Embedding
from src.schema.knowledge import Knowledge, KnowledgeItem
from src.schema.wikipedia import WikiResource

import wikipedia

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


class BOTServer:
    def __init__(self, token:str, api_key:str, port:int):
        self.token = token 
        self.apikey = api_key
        self.endpoint = f'tcp://localhost:{port}'
        self.translator = GoogleTranslator(source='auto', target='en')        
        self.user2conversation:Dict[str, List[Dict[str, str]]] = {}
        self.application = Application.builder().token(token).build()
        self.ctx = zmq.Context()
        
    async def start(self, update:Update, context:ContextTypes.DEFAULT_TYPE):
        user = update.message.from_user
        logger.debug(f'{user.first_name} is connected')
        self.user2conversation[user.first_name] = []
        await update.message.reply_text(text=f"Hello {user.first_name}")
        return 0 
    
    async def chatting(self, update:Update, context:ContextTypes.DEFAULT_TYPE):
        user = update.message.from_user
        text = update.message.text
        if text == "create-index":
            await update.message.reply_text(text="cool, lets create the index, please give us a wikipedia url")
            return 1 
        return 0 

    async def handle_url(self, update:Update, context:ContextTypes.DEFAULT_TYPE):
        user = update.message.from_user
        text = update.message.text
        await update.message.reply_text(text="thanks for the url, index will be created soon")
        await update.message.reply_text(text="use /monitor <index_id> to track the progression")
        return 0
    
    async def stop(self, update:Update, context:ContextTypes.DEFAULT_TYPE):
        user = update.message.from_user
        del self.user2conversation[user.first_name]
       
        await update.message.reply_text(
            text=f"""
                Merci de votre visite {user.first_name} et n'hésitez pas à revenir si vous avez d'autres questions ou besoins d'assistance. 
                À bientôt !
                NeuralMap Team <3!
            """
        )
        return ConversationHandler.END
    
    
    def listen(self):
        self.application.run_polling()

    def __enter__(self):
        try:
            handler = ConversationHandler(
                entry_points=[CommandHandler('start', self.start)],
                states={
                    0: [CommandHandler('stop', self.stop), MessageHandler(filters.TEXT, self.chatting)],
                    1: [CommandHandler('stop', self.stop), MessageHandler(filters.TEXT, self.handle_url)]
                },
                fallbacks=[CommandHandler('stop', self.stop)]
            )
            self.application.add_handler(handler)
        except Exception as e:
            logger.error(e)
        return self 

    def __exit__(self, exc, val, traceback):
        logger.debug('GPT ... shutdown')
