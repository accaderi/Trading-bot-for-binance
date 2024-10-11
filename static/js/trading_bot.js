let socket = null;
let priceChart;
let adxThreshold = 25;  // Default value
let chopThreshold = 50;  // Default value


function displayMessage(message, color = 'black') {
    const outputElement = document.getElementById('output');
    outputElement.innerHTML += `<p style="color: ${color};">${message}</p>`;
    outputElement.scrollTop = outputElement.scrollHeight;
}

function connectWebSocket() {
    const emaSlowLength = parseInt(document.getElementById('emaSlow').value);
    const symbol = document.getElementById('symbol').value;
    const interval = document.getElementById('interval').value;

    socket = new WebSocket(`ws://${window.location.host}/ws/bot_updates/?ema_slow=${emaSlowLength}&symbol=${symbol}&interval=${interval}`);

    socket.onmessage = function(e) {
        const data = JSON.parse(e.data);
        if (data.status === 'success') {
            if (data.adx_threshold) adxThreshold = data.adx_threshold;
            if (data.chop_threshold) chopThreshold = data.chop_threshold;
            if (data.message) {
                displayMessage(data.message);
            }
            if (data.ohlc_data && data.ohlc_data.length > 0) {
                updateDataTable(data.trades);
                updatePriceChart(data.ohlc_data, emaSlowLength);
            }
        } else if (data.status === 'error') {
            console.error('Error:', data.message);
            displayMessage(`Error: ${data.message}`, 'red');
        }
    };

    socket.onclose = function(e) {
        console.error('WebSocket closed unexpectedly');
    };
}

function updateDataTable(trades) {
    const tableBody = document.querySelector('#dataTable tbody');
    
    // Check if trades is an array and has at least one element
    if (!Array.isArray(trades) || trades.length === 0) {
        return; // Exit the function if there's no valid trade data
    }

    // Clear the existing table
    tableBody.innerHTML = '';

    // Insert all trades into the table, starting from the oldest (lowest index)
    for (let i = 0; i < trades.length; i++) {
        const trade = trades[i];
        const newRow = tableBody.insertRow(-1); // Insert at the bottom
        
        // Index (right-aligned)
        const indexCell = newRow.insertCell(0);
        indexCell.textContent = trade.index;
        indexCell.style.textAlign = 'right';
        
        // Timestamp (right-aligned)
        const timestampCell = newRow.insertCell(1);
        timestampCell.textContent = trade.timestamp ? new Date(trade.timestamp).toLocaleString() : 'N/A';
        timestampCell.style.textAlign = 'center';
        
        // Side (center-aligned)
        const sideCell = newRow.insertCell(2);
        sideCell.textContent = trade.side || 'N/A';
        sideCell.style.textAlign = 'center';
        
        // Price (2 decimal places, right-aligned)
        const priceCell = newRow.insertCell(3);
        priceCell.textContent = trade.price !== null && trade.price !== undefined ? 
            parseFloat(trade.price).toFixed(2) : 'N/A';
        priceCell.style.textAlign = 'right';
        
        // Quantity (4 decimal places, right-aligned)
        const quantityCell = newRow.insertCell(4);
        quantityCell.textContent = trade.quantity !== null && trade.quantity !== undefined ? 
            parseFloat(trade.quantity).toFixed(4) : 'N/A';
        quantityCell.style.textAlign = 'right';
        
        // Total Gain/Loss USDT (2 decimal places, right-aligned)
        const totalGainLossCell = newRow.insertCell(5);
        totalGainLossCell.textContent = trade.Total_gain_loss_USDT !== null && trade.Total_gain_loss_USDT !== undefined ? 
            parseFloat(trade.Total_gain_loss_USDT).toFixed(2) : 'N/A';
        totalGainLossCell.style.textAlign = 'right';
        
        // Total Gain/Loss Percent (2 decimal places, right-aligned)
        const totalGainLossPercentCell = newRow.insertCell(6);
        totalGainLossPercentCell.textContent = trade.Total_gain_loss_percent !== null && trade.Total_gain_loss_percent !== undefined ? 
            parseFloat(trade.Total_gain_loss_percent).toFixed(2) : 'N/A';
        totalGainLossPercentCell.style.textAlign = 'right';
        
        // Last Trade Gain/Loss USDT (2 decimal places, right-aligned)
        const lastTradeGainLossCell = newRow.insertCell(7);
        lastTradeGainLossCell.textContent = trade.Last_trade_gain_loss_USDT !== null && trade.Last_trade_gain_loss_USDT !== undefined ? 
            parseFloat(trade.Last_trade_gain_loss_USDT).toFixed(2) : 'N/A';
        lastTradeGainLossCell.style.textAlign = 'right';
        
        // Last Trade Gain/Loss Percent (2 decimal places, right-aligned)
        const lastTradeGainLossPercentCell = newRow.insertCell(8);
        lastTradeGainLossPercentCell.textContent = trade.Last_trade_gain_loss_percent !== null && trade.Last_trade_gain_loss_percent !== undefined ? 
            parseFloat(trade.Last_trade_gain_loss_percent).toFixed(2) : 'N/A';
        lastTradeGainLossPercentCell.style.textAlign = 'right';
    }
}

function initializePriceChart() {
    const trace = {
        x: [],
        close: [],
        high: [],
        low: [],
        open: [],
        type: 'candlestick',
        xaxis: 'x',
        yaxis: 'y'
    };

    const data = [trace];

    const layout = {
        dragmode: 'zoom',
        showlegend: true,
        xaxis: {
            rangeslider: { visible: false },
            title: 'Date'
        },
        yaxis: {
            title: 'Price'
        },
        // Adding secondary y-axis for ADX and Choppiness
        yaxis2: {
            title: 'ADX / Choppiness',
            overlaying: 'y',
            side: 'right',
            showgrid: false,
            zeroline: false
        }
    };

    Plotly.newPlot('priceChart', data, layout);
    priceChart = document.getElementById('priceChart');
}

function updatePriceChart(ohlcData, plotLength) {
    if (!ohlcData || ohlcData.length === 0) {
        console.error('No OHLC data received');
        return;
    }

    // Ensure we only plot the last 'plotLength' data points
    const plotData = ohlcData.slice(-plotLength);

    // Filter out NaN values for ADX and Choppiness
    const validADX = plotData.filter(d => !isNaN(d.ADX));
    const validChoppiness = plotData.filter(d => !isNaN(d.Choppiness));

    // Candlestick trace
    const trace1 = {
        x: plotData.map(d => d.timestamp),
        close: plotData.map(d => d.close),
        high: plotData.map(d => d.high),
        low: plotData.map(d => d.low),
        open: plotData.map(d => d.open),
        type: 'candlestick',
        xaxis: 'x',
        yaxis: 'y',
        name: 'OHLC'
    };

    // EMA Slow trace
    const trace2 = {
        x: plotData.map(d => d.timestamp),
        y: plotData.map(d => d.ema_slow),
        type: 'scatter',
        mode: 'lines',
        line: { color: 'blue' },
        name: 'EMA Slow'
    };

    // EMA Fast trace
    const trace3 = {
        x: plotData.map(d => d.timestamp),
        y: plotData.map(d => d.ema_fast),
        type: 'scatter',
        mode: 'lines',
        line: { color: 'red' },
        name: 'EMA Fast'
    };

    // ADX trace
    const trace4 = {
        x: validADX.map(d => d.timestamp),
        y: validADX.map(d => d.ADX),
        type: 'scatter',
        mode: 'lines',
        line: { color: 'purple' },
        name: 'ADX',
        yaxis: 'y2'
    };

    // Choppiness Index trace
    const trace5 = {
        x: validChoppiness.map(d => d.timestamp),
        y: validChoppiness.map(d => d.Choppiness),
        type: 'scatter',
        mode: 'lines',
        line: { color: 'green' },
        name: 'Choppiness',
        yaxis: 'y3'
    };

    // Add buy/sell markers
    const buyMarkers = {
        x: plotData.filter(d => d.action === 'BUY').map(d => d.timestamp),
        y: plotData.filter(d => d.action === 'BUY').map(d => d.close),
        mode: 'markers',
        type: 'scatter',
        marker: { symbol: 'triangle-up', size: 10, color: 'green' },
        name: 'Buy',
        yaxis: 'y'
    };

    const sellMarkers = {
        x: plotData.filter(d => d.action === 'SELL').map(d => d.timestamp),
        y: plotData.filter(d => d.action === 'SELL').map(d => d.close),
        mode: 'markers',
        type: 'scatter',
        marker: { symbol: 'triangle-down', size: 10, color: 'red' },
        name: 'Sell',
        yaxis: 'y'
    };

    const layout = {
        grid: {rows: 2, columns: 1, pattern: 'independent'},
        height: 800,
    
        // First row: Price plot, no timestamp title and ticks
        // Single x-axis for both plots
        xaxis: {
            title: 'Timestamp',
            domain: [0, 1],
            showticklabels: true,
            showgrid: true,
            zeroline: true,
            rangeslider: {visible: false}
        },
        yaxis: {title: 'Price', domain: [0.55, 1]},
    
        // Second row: ADX and Choppiness, with timestamp title and ticks
        xaxis2: {title: 'Timestamp', domain: [0, 1], anchor: 'y2', rangeslider: {visible: false}, showticklabels: true},
        yaxis2: {title: 'ADX', domain: [0, 0.45], side: 'left'},
        yaxis3: {title: 'Choppiness', domain: [0, 0.45], side: 'right', overlaying: 'y2'},
    
        // Lines for thresholds
        shapes: [
            {type: 'line', y0: adxThreshold, y1: adxThreshold, x0: 0, x1: 1, yref: 'y2', xref: 'paper', 
             line: {color: 'purple', width: 1, dash: 'dash'}, name: 'ADX Threshold'},
            {type: 'line', y0: chopThreshold, y1: chopThreshold, x0: 0, x1: 1, yref: 'y3', xref: 'paper', 
             line: {color: 'green', width: 1, dash: 'dash'}, name: 'Choppiness Threshold'}
        ]
    };
    
    // Add threshold lines as separate traces for the legend
    const adxThresholdLine = {
        x: [plotData[0].timestamp, plotData[plotData.length - 1].timestamp],
        y: [adxThreshold, adxThreshold],
        mode: 'lines',
        line: {color: 'purple', width: 1, dash: 'dash'},
        name: 'ADX Threshold',
        yaxis: 'y2'
    };

    const chopThresholdLine = {
        x: [plotData[0].timestamp, plotData[plotData.length - 1].timestamp],
        y: [chopThreshold, chopThreshold],
        mode: 'lines',
        line: {color: 'green', width: 1, dash: 'dash'},
        name: 'Choppiness Threshold',
        yaxis: 'y3'
    };

    const data = [trace1, trace2, trace3, trace4, trace5, buyMarkers, sellMarkers, adxThresholdLine, chopThresholdLine];
    
    Plotly.newPlot('priceChart', data, layout);
}

const startButton = document.getElementById('startTradingBtn');
const stopButton = document.getElementById('stopTradingBtn');

startButton.addEventListener('click', function() {
    startButton.disabled = true;
    stopButton.disabled = false;

    const params = {
        symbol: document.getElementById('symbol').value,
        interval: document.getElementById('interval').value,
        emaSlow: document.getElementById('emaSlow').value,
        emaFast: document.getElementById('emaFast').value,
        adxPeriod: document.getElementById('adxPeriod').value,
        adxThreshold: document.getElementById('adxThreshold').value,
        chopPeriod: document.getElementById('chopPeriod').value,
        chopThreshold: document.getElementById('chopThreshold').value,
        initialTradePercentage: document.getElementById('initialTradePercentage').value
    };
    
    fetch('/start_trading/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(params),
    })
    .then(response => response.json())
    .then(data => {
        displayMessage(data.message);
        connectWebSocket();
    })
    .catch(error => {
        displayMessage(`Error: ${error}`, 'red');
        startButton.disabled = false;
        stopButton.disabled = true;
    });
});

stopButton.addEventListener('click', function() {
    fetch('/stop_trading/', {
        method: 'POST',
    })
    .then(response => response.json())
    .then(data => {
        displayMessage(data.message);
        if (socket) {
            socket.close();
            socket = null;
        }
        startButton.disabled = false;
        stopButton.disabled = true;
    })
    .catch(error => {
        displayMessage(`Error: ${error}`, 'red');
    });
});

const emaSlowSlider = document.getElementById('emaSlow');
const emaSlowValue = document.getElementById('emaSlowValue');
const emaFastSlider = document.getElementById('emaFast');
const emaFastValue = document.getElementById('emaFastValue');

emaSlowSlider.addEventListener('input', function() {
    emaSlowValue.textContent = this.value;
    if (parseInt(emaSlowValue.textContent) <= parseInt(emaFastValue.textContent)) {
        // Adjust emaFastSlider to be less than emaSlowSlider while staying within its min and max range
        let newFastValue = Math.floor((parseInt(emaSlowValue.textContent) - 10) / 5) * 5;
        emaFastSlider.value = Math.max(emaFastSlider.min, newFastValue); // Ensure it's not below the minimum
        emaFastValue.textContent = emaFastSlider.value;
    }
});


emaFastSlider.addEventListener('input', function() {
    emaFastValue.textContent = this.value;
    if (parseInt(emaFastValue.textContent) >= parseInt(emaSlowValue.textContent)) {
        // Adjust emaSlowSlider to the next higher step while staying within its min and max range
        let newSlowValue = Math.ceil((parseInt(emaFastValue.textContent) + 5) / 10) * 10;
        emaSlowSlider.value = Math.min(emaSlowSlider.max, newSlowValue); // Ensure it's not above the maximum
        emaSlowValue.textContent = emaSlowSlider.value;
    }
});


// Call initializePriceChart when the page loads
document.addEventListener('DOMContentLoaded', initializePriceChart);

// Initialize button states
startButton.disabled = false;
stopButton.disabled = true;