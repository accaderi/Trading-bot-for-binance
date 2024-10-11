from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import logging
from .trading_bot import TradingBot
import pandas as pd
from django.conf import settings

logger = logging.getLogger(__name__)

bot_instance = None

def index(request):
    return render(request, 'index.html')

@csrf_exempt
def start_trading(request):
    global bot_instance
    if request.method == 'POST':
        data = json.loads(request.body)
        symbol = data['symbol']
        interval = data['interval']
        ema_slow = int(data['emaSlow'])
        ema_fast = int(data['emaFast'])
        adx_period = int(data['adxPeriod'])
        adx_threshold = float(data['adxThreshold'])
        chop_period = int(data['chopPeriod'])
        chop_threshold = float(data['chopThreshold'])
        initial_trade_percentage = float(data['initialTradePercentage'])
        
        # Replace with your actual API keys
        api_key = settings.API_KEY
        api_secret = settings.API_SECRET

        if bot_instance:
            bot_instance.stop()
            bot_instance.join()  # Wait for the thread to finish

        bot_instance = TradingBot(symbol, interval, ema_slow, ema_fast, adx_period, adx_threshold, chop_period, chop_threshold, initial_trade_percentage, api_key, api_secret)
        bot_instance.start()
        logger.info("Trading bot started")
        return JsonResponse({'status': 'success', 'message': 'Trading bot started'})
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

@csrf_exempt
def stop_trading(request):
    global bot_instance
    if request.method == 'POST':
        if bot_instance:
            bot_instance.stop()
            bot_instance.join()  # Wait for the thread to finish
            bot_instance = None
            logger.info("Trading bot stopped")
            return JsonResponse({'status': 'success', 'message': 'Trading bot stopped'})
        else:
            return JsonResponse({'status': 'error', 'message': 'No active trading bot'})
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

@csrf_exempt
def bot_status(request):
    global bot_instance
    if bot_instance:
        try:
            df = pd.read_csv(bot_instance.filename)
            if df.empty:
                return JsonResponse({
                    'status': 'running',
                    'message': 'Bot is running, but no trades have been made yet',
                    'last_update': None
                })
            
            last_trade = df.iloc[-1]
            last_update = last_trade['timestamp']
            
            if 'last_update' in request.GET:
                client_last_update = request.GET['last_update']
                if client_last_update == last_update:
                    return JsonResponse({
                        'status': 'no_update',
                        'message': 'No new data available'
                    }, status=204)  # No Content
            
            latest_trades = df.tail(10).to_dict('records')
            return JsonResponse({
                'status': 'success',
                'message': 'Bot is running',
                'trades': latest_trades,
                'last_update': last_update
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': f'Error reading bot data: {str(e)}'
            })
    else:
        return JsonResponse({
            'status': 'inactive',
            'message': 'Bot is not running'
        })