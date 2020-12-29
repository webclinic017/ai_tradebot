FROM python:3.8

ENV COINBASE_API_KEY='<Your Coinbase Api Key>'
ENV COINBASE_API_SECRET='<Your Coinbase Api Secret>'
ENV TWITTER_API_KEY='<Your Twitter Api Key>'
ENV TWITTER_API_SECRET='<Your Twitter Api Secret>'

WORKDIR /ai_tradebot

COPY . .

RUN pip install -r requirements.txt

CMD [ "python", "main.py" ]