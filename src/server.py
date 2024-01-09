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

from typing import Dict, List, Optional, Iterable
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
    
    def _retrieve_capture(self) -> bytes:
        dealer_socket:zmq.Socket = self.ctx.socket(zmq.DEALER)
        dealer_socket.connect(self.endpoint)

        dealer_socket.send_multipart([b'', b''])
        _, binarystream = dealer_socket.recv_multipart()
        dealer_socket.close(linger=0)
        return binarystream

    async def start(self, update:Update, context:ContextTypes.DEFAULT_TYPE):
        user = update.message.from_user
        
        logger.debug(f'{user.first_name} is connected')
        if user.first_name not in self.user2conversation:
            self.user2conversation[user.first_name] = []
        await update.message.reply_text(text=f"Hello {user.first_name}")
        return 0 
    
    
    async def chatting(self, update:Update, context:ContextTypes.DEFAULT_TYPE):
        user = update.message.from_user
        text = update.message.text
        return 0 
    
    async def capture(self, update:Update, context:ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(text="ok, voici une capture de l'état actuel")
        loop = asyncio.get_event_loop()
        binarystream = await loop.run_in_executor(None, self._retrieve_capture)
        buffer = pickle.loads(binarystream)
        photo = BytesIO(buffer)
        caption = datetime.now()
        await update.message.reply_photo(photo=photo, caption=f'{caption}')
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
                    0: [CommandHandler('stop', self.stop), CommandHandler('capture', self.capture), MessageHandler(filters.TEXT|filters.VOICE|filters.PHOTO, self.chatting)],
                },
                fallbacks=[CommandHandler('stop', self.stop)]
            )
            self.application.add_handler(handler)
        except Exception as e:
            logger.error(e)
        return self 

    def __exit__(self, exc, val, traceback):
        logger.debug('GPT ... shutdown')
