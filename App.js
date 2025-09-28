import React, { useState } from 'react';
import {
  Container,
  TextField,
  Button,
  Card,
  CardContent,
  Typography,
  Grid,
  Box,
  Alert,
  CircularProgress,
  AppBar,
  Toolbar,
  Chip
} from '@mui/material';
import { Search, TrendingUp } from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import axios from 'axios';
import './App.css';

const API_BASE = 'http://localhost:5000/api';

function App() {
  const [symbol, setSymbol] = useState('AAPL');
  const [days, setDays] = useState(30);
  const [predictionData, setPredictionData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handlePredict = async () => {
    if (!symbol) {
      setError('Please enter a stock symbol');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await axios.post(`${API_BASE}/predict`, {
        symbol: symbol.toUpperCase(),
        days: parseInt(days)
      });
      
      setPredictionData(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to fetch predictions');
    } finally {
      setLoading(false);
    }
  };

  const formatChartData = () => {
    if (!predictionData) return [];
    
    const historical = predictionData.historical_data.map(item => ({
      date: item.date,
      price: item.price,
      type: 'Historical'
    }));
    
    const predictions = predictionData.predictions.map(item => ({
      date: item.date,
      price: item.predicted_price,
      type: 'Predicted'
    }));
    
    return [...historical, ...predictions];
  };

  const popularStocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META'];

  return (
    <div className="App">
      <AppBar position="static" sx={{ mb: 4 }}>
        <Toolbar>
          <TrendingUp sx={{ mr: 2 }} />
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Stock Prediction App
          </Typography>
        </Toolbar>
      </AppBar>

      <Container maxWidth="lg">
        <Grid container spacing={4}>
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="h5" gutterBottom>
                  Stock Prediction
                </Typography>
                
                <TextField
                  fullWidth
                  label="Stock Symbol"
                  value={symbol}
                  onChange={(e) => setSymbol(e.target.value)}
                  margin="normal"
                  placeholder="e.g., AAPL"
                />
                
                <TextField
                  fullWidth
                  label="Prediction Days"
                  type="number"
                  value={days}
                  onChange={(e) => setDays(e.target.value)}
                  margin="normal"
                  inputProps={{ min: 1, max: 90 }}
                />
                
                <Button
                  fullWidth
                  variant="contained"
                  onClick={handlePredict}
                  disabled={loading}
                  startIcon={loading ? <CircularProgress size={20} /> : <Search />}
                  sx={{ mt: 2 }}
                >
                  {loading ? 'Predicting...' : 'Predict'}
                </Button>

                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Popular Stocks:
                  </Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                    {popularStocks.map((stock) => (
                      <Chip
                        key={stock}
                        label={stock}
                        onClick={() => setSymbol(stock)}
                        variant={symbol === stock ? 'filled' : 'outlined'}
                        color="primary"
                        size="small"
                      />
                    ))}
                  </Box>
                </Box>

                {error && (
                  <Alert severity="error" sx={{ mt: 2 }}>
                    {error}
                  </Alert>
                )}
              </CardContent>
            </Card>

            {predictionData && (
              <Card sx={{ mt: 2 }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Prediction Summary
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Symbol: <strong>{predictionData.symbol}</strong>
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Current Price: <strong>${predictionData.current_price}</strong>
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Model Accuracy: <strong>{predictionData.accuracy_score}%</strong>
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Prediction Period: <strong>{predictionData.predictions.length} days</strong>
                  </Typography>
                  
                  {predictionData.predictions.length > 0 && (
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="subtitle2">
                        Next Prediction: ${predictionData.predictions[0].predicted_price}
                      </Typography>
                    </Box>
                  )}
                </CardContent>
              </Card>
            )}
          </Grid>

          <Grid item xs={12} md={8}>
            <Card>
              <CardContent>
                <Typography variant="h5" gutterBottom>
                  Price Chart
                </Typography>
                
                {predictionData ? (
                  <ResponsiveContainer width="100%" height={400}>
                    <LineChart data={formatChartData()}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line 
                        type="monotone" 
                        dataKey="price" 
                        stroke="#8884d8" 
                        name="Price"
                        strokeWidth={2}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <Box 
                    sx={{ 
                      height: 400, 
                      display: 'flex', 
                      alignItems: 'center', 
                      justifyContent: 'center',
                      bgcolor: 'grey.50'
                    }}
                  >
                    <Typography variant="h6" color="textSecondary">
                      Enter a stock symbol to see predictions
                    </Typography>
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Container>
    </div>
  );
}

export default App;