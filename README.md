# AI trading bot

This is a Trading Bot mainly intendet to trade Crypto Currencies. It analysis the Sentiment of news headlines and Tweets to predict the development of the price for the asset you intend to trade.

## Setup
This project uses a few apis to collect data.
- the **twitter api** is used to collect tweets about financial news
- the **coinbase api** is used to trade bitcoin and get price information

Insert your coinbase and twitter api key in the ./env/lib/python3.7/site-packages/_set_envs.pth file

### Server
This project is ment to be hosted on a server in order to have a reliable 24/7 service for trading. The easiest way to use this on a server is to deploy this project using docker.