import json
from channels.generic.websocket import AsyncWebsocketConsumer

class BotConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.channel_layer.group_add("bot_updates", self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard("bot_updates", self.channel_name)

    async def bot_update(self, event):
        message = event['message']
        await self.send(text_data=json.dumps(message))
