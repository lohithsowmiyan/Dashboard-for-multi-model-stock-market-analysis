from cProfile import label
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from GoogleNews import GoogleNews
from newspaper import Article
from newspaper import Config
from wordcloud import WordCloud, STOPWORDS
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize,pos_tag
import yfinance as yf
from pandas_datareader.data import DataReader
import yahoo_fin.stock_info as stock_details
from datetime import datetime,date
import requests
from streamlit_option_menu import option_menu
from statsmodels.tsa.seasonal import seasonal_decompose
import babel

st.markdown(f'<div align="center"><h1>EIC DASHBOARD</h1></div>',unsafe_allow_html=True)

def remove_stopwords(text):
    li = []
    for word in text:
        if word not in en_stopwords:
            li.append(word)
    return li
def remove_punct(text):
    
    tokenizer = RegexpTokenizer(r"\w+")
    lst=tokenizer.tokenize(' '.join(text))
    return lst

def lemmatization(text):
    
    result=[]
    wordnet = WordNetLemmatizer()
    for token,tag in pos_tag(text):
        pos=tag[0].lower()
        
        if pos not in ['a', 'r', 'n', 'v']:
            pos='n'
            
        result.append(wordnet.lemmatize(token,pos))
    
    return result

eic = option_menu(
    menu_title=None,
    options=['Economy','Industry','Company'],
    default_index = 0,
    orientation = "horizontal",
)
if eic == "Economy":
    country = pd.read_csv('gdp.csv',nrows=61)
   
    country = country.set_index(keys='Date')
    country_name = st.selectbox("Select the country",country.columns)
    #country[country_name] = country[country_name].apply(make_decimal)
    #st.write(country)
    st.table(country[country_name])
    
    
#st.markdown('<h1>Hello all </h1>',unsafe_allow_html=True)
if eic == "Industry":
        st.title("UBC data")
        m_gubc = pd.read_csv('/Users/lohithsowmiyan/machine learning/Stock market sentiment analysis/nic_data.csv',delimiter='\t')
        st.write(m_gubc)
        m_gubc['Date'] = pd.to_datetime(m_gubc['Date'],format = '%b-%y')
        m_gubc.columns =m_gubc.columns.str.replace(" ","_")
        m_gubc.loc[m_gubc['Primary_goods']=='#','Primary_goods'] = 4
        m_gubc.loc[m_gubc['Capital_goods']=='#','Capital_goods'] = 3
        m_gubc.loc[m_gubc['Intermediate_goods']=='#','Intermediate_goods'] = 3
        m_gubc.loc[m_gubc['Infrastructure/_construction_goods']=='#','Infrastructure/_construction_goods'] = 3
        m_gubc.loc[m_gubc['Consumer_durables']=='#','Consumer_durables'] = 4
        m_gubc.loc[m_gubc['Consumer_non-durables']=='#','Consumer_non-durables'] = 4
        new_date =  m_gubc['Date']
        m_gubc = m_gubc.drop(columns=['Date']).apply(pd.to_numeric)
        m_gubc['Date'] = new_date
        
        ubc_column = st.selectbox("Choose the industry",("Primary_goods","Capital_goods","Intermediate_goods","Infrastructure/_construction_goods","Consumer_durables","Consumer_non-durables"))
        figure1,axes1 = plt.subplots()
        st.write(ubc_column)
        sns.lineplot(x=m_gubc['Date'],y=m_gubc[ubc_column],color='green')
        
        st.pyplot(figure1)
        res2 = seasonal_decompose(m_gubc[ubc_column], model='additive',period=12,extrapolate_trend='freq')
        figure4 = res2.plot()
        st.markdown('<h3>Seasonal Decomposition</h3>',unsafe_allow_html=True)
        
        st.pyplot(figure4)

        st.title("NIC data")

        #nic analysis
        m_iip = pd.read_csv('/Users/lohithsowmiyan/machine learning/Stock market sentiment analysis/ubc_monthly.csv')
        st.dataframe(m_iip)
        #m_iip.head()
        m_iip['Date'] = pd.to_datetime(m_iip['Date'],format = '%b-%y')
        description = list(m_iip[:0][1:])
        columns_indices = [x for x in range(0, 28)]
        new_names = ['Date','Food','Beverages','Tobacco','Textiles','Wearing Apparel','Leather','Wood and Cork','Paper','Recorded Media','Petroleum','Chemicals','Pharmaceutical','Rubber','Non-Metallic','Basic metals','Fabricated Metal','Computer','Electrical','Machinery','Motor Vehicles','Other Transport','Furniture','Other Manufacturing','Mining','Manufacturing','Electricity','General']
        old_names = m_iip.columns[columns_indices]
        m_iip.rename(columns=dict(zip(old_names, new_names)), inplace=True)
        m_iip.columns = m_iip.columns.str.replace(' ','_')

        nic_column = st.selectbox("Choose sector",m_iip.drop('Date',axis=1).columns)
        figure2,axes2 = plt.subplots()
        st.write(nic_column)
        sns.lineplot(x=m_iip['Date'],y=m_iip[nic_column],color='green')
        
        st.pyplot(figure2)

        #Seasonal Decomposition
        st.markdown('<h3>Seasonal Decomposition</h3>',unsafe_allow_html=True)
        
        
        dm_iip = m_iip.copy()
        dm_iip['Year'] = pd.DatetimeIndex(dm_iip['Date']).year
        dm_iip['Month'] = pd.DatetimeIndex(dm_iip['Date']).month
        dm_iip['Week_of_year'] = pd.DatetimeIndex(dm_iip['Date']).weekofyear
        dm_iip['Quarter'] = pd.DatetimeIndex(dm_iip['Date']).quarter
        res = seasonal_decompose(dm_iip[nic_column], model='additive',period=12,extrapolate_trend='freq')
        figure3 = res.plot()
        
        st.pyplot(figure3)




if eic == "Company":
        st.title('Company Analysis')
        now = dt.date.today()
        now = now.strftime('%m-%d-%Y')
        yesterday = dt.date.today() - dt.timedelta(days = 30)
        yesterday = yesterday.strftime('%m-%d-%Y')
        print("Date Yesterday",yesterday,"Today",now)

        user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_5) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/85 Version/11.1.1 Safari/605.1.15'
        config = Config()
        config.browser_user_agent = user_agent

        company_name = st.text_input("Enter the company name")
        #As long as the company name is valid, not empty...

        if company_name != '':
            print(f'Searching for and analyzing {company_name}, Please be patient, it might take a while...')

            #Extract News with Google News
            googlenews = GoogleNews(start=yesterday,end=now)
            
            googlenews.search(company_name)
            result = googlenews.result()
            #store the results
            df = pd.DataFrame(result)
            #st.write(df)
        try:
            list =[] #creating an empty list 
            for i in df.index:
                dict = {} #creating an empty dictionary to append an article in every single iteration
                article = Article(df['link'][i],config=config) #providing the link
                try:
                        article.download() #downloading the article 
                        article.parse() #parsing the article
                        article.nlp() #performing natural language processing (nlp)
                except:
                        pass 
                #storing results in our empty dictionary
                dict['Date']=df['date'][i] 
                dict['Media']=df['media'][i]
                dict['Title']=article.title
                dict['Article']=article.text
                dict['Summary']=article.summary
                dict['Key_words']=article.keywords
                list.append(dict)
            check_empty = not any(list)
            # print(check_empty)
            if check_empty == False:
                news_df=pd.DataFrame(list) #creating dataframe
                #st.write(news_df)

        except Exception as e:
            #exception handling
            print("exception occurred:" + str(e))
            print('Looks like, there is some error in retrieving the data, Please try again or try with a different ticker.' )

        #preprocessing
        if company_name != "":
            data = news_df['Summary']
        #tokenizing
            data = data.apply(lambda text : word_tokenize(text))

        #removing empty rows
            for row,i in zip(data,data.index):
                if row == []:
                    data.drop(i,inplace=True)

        #stopwords removal
            en_stopwords = stopwords.words('english')
            data = data.apply(remove_stopwords)
        #removing speacial characters
            data=data.apply(remove_punct)
        #lemmatization
            data =data.apply(lemmatization)

            #st.write(data)
        #sentiment analyzer

        si = SentimentIntensityAnalyzer()

        def classify_sentiments(text):
            string = " ".join(text)
            return si.polarity_scores(string)

        if company_name!="":
            scores = data.apply(classify_sentiments)
            sentiments = pd.DataFrame({'text':data,'scores':scores})
        #sentiments['label'] = scores.apply(lambda label: np.argmax(label))
            #st.write(sentiments.head())

        #plotting sentiments

        def percentage(part,whole):
            return 100 * float(part)/float(whole)
        
        if company_name!="":
            st.write('Customer Sentiments')
            #Assigning Initial Values
            positive = 0
            negative = 0
            neutral = 0
            #Creating empty lists
            news_list = []
            neutral_list = []
            negative_list = []
            positive_list = []

            #Iterating over the tweets in the dataframe
            for score in sentiments['scores']:
                
                if score['neg'] > score['pos']:
                    negative += 1 #increasing the count by 1
                elif score['pos'] > score['neg']:
                    positive += 1 #increasing the count by 1
                elif score['pos'] == score['neg']:
                    neutral += 1 #increasing the count by 1 

            positive = percentage(positive,len(sentiments['text'])) #percentage is the function defined above
            negative = percentage(negative,len(sentiments['text']))
            neutral = percentage(neutral,len(sentiments['text']))

            #Converting lists to pandas dataframe

            #using len(length) function for counting
            print("Positive Sentiment:", '%.2f' % positive, end='\n')
            print("Neutral Sentiment:", '%.2f' % neutral, end='\n')
            print("Negative Sentiment:", '%.2f' % negative, end='\n')

            #Creating PieCart
            labels = ['Positive ['+str(round(positive))+'%]' , 'Neutral ['+str(round(neutral))+'%]','Negative ['+str(round(negative))+'%]']
            sizes = [positive, neutral, negative]
            colors = ['yellowgreen', 'blue','red']
            fig, ax = plt.subplots()
            ax.pie(sizes,colors=colors, startangle=90)

            ax.legend(labels)
            #ax.title("Sentiment Analysis Result for stock= "+company_name+"" )
            ax.axis('equal')
            st.pyplot(fig)

            if positive > negative and positive > neutral:
                st.write("The customer sentiments are on a positive note so you can expect the stock performance to be good")
            elif negative>positive and negative>neutral:
                st.write("The customer sentiments are quite bad, So it must not be safe in investing the stock")
        
        #get stock data
        stock_name_list = []
        def getStock(search_term,period=0):
            results = []
            query = requests.get(f'https://yfapi.net/v6/finance/autocomplete?region=IN&lang=en&query={search_term}', 
            headers={
                'accept': 'application/json',
                'X-API-KEY': 'NRd7nboojG6C4BoXUYojV5T7SBLJcrXO2U9u83la'
            })
            response = query.json()
            for i in response['ResultSet']['Result']:
                final = i['symbol']
                results.append(final)
                stock_name_list.append(final)
                
            
            stock_list = [results[0]]

            end = datetime.now()
            start = datetime(end.year-period,1,end.day)

            for stock in stock_list:
                globals()[stock] = yf.download(stock ,start ,end)
            #print(globals()[results[0]])

        if company_name!="":
            st.title("Stock Performance")
            period = st.slider("How many years?",0,10,0)
            #st.write(period)
            getStock(company_name,period)
            

        #stock performance
            comp = globals()[stock_name_list[0]]

            ma1 = st.slider("1st MA",5,20,5)
            ma2 = st.slider("2nd MA",20,50,20)
            ma3 = st.slider("3rd MA",50,200,60)

            ma = [ma1,ma2,ma3]
        
            st.markdown(f'<h3>Moving averages for {company_name}</h3>',unsafe_allow_html=True)
            count =1
            for i in ma:
                column_name = "MA for {ma}".format(ma=count)
                comp[column_name] = comp['Close'].rolling(i).mean()
                count+=1

            #plotting stock market
            fig2,ax2 = plt.subplots(figsize=(30,15))
            sns.lineplot(x = comp.index,y=comp['Adj Close'],axes=ax2,color='green')
            sns.lineplot(x = comp.index,y=comp['MA for 1'],axes=ax2,color='blue',label=f'ma for{ma[0]} days')
            sns.lineplot(x =comp.index,y=comp['MA for 2'],axes=ax2,color='yellow',label=f'ma for{ma[1]} 20 days')
            sns.lineplot(x = comp.index,y=comp['MA for 3'],axes=ax2,color='red',label=f'ma for {ma[2]} days')
            st.pyplot(fig2)

            #volume of stock
            st.markdown(f'<h3>Volume for {company_name}</h3>',unsafe_allow_html=True)
            fig3,ax3 = plt.subplots(figsize=(30,15))
            sns.lineplot(x =comp.index,y=comp['Volume'],axes=ax3,color='blue',label='volume')
            st.pyplot(fig3)

            #increase decrease of stock price
            st.markdown(f'<h3>Percentage Change for {company_name}</h3>',unsafe_allow_html=True)
            comp['Daily return'] = comp['Adj Close'].pct_change()
            fig4,ax4 = plt.subplots(figsize=(30,15))
            sns.barplot(x=comp.index,y=comp['Daily return'],axes=ax4,color='black',label='Daily changes')
            st.pyplot(fig4)


            #other attributes:
            attributes = st.selectbox("Select an attribute",("sector",
            "fullTimeEmployees",
            "longBusinessSummary",
            "city",
            "phone",
            "state",
            "country",
            "companyOfficers",
            "website",
            "maxAge",
            "address1",
            "industry",
            "ebitdaMargins",
            "profitMargins",
            "grossMargins",
            "operatingCashflow",
            "revenueGrowth",
            "operatingMargins",
            "ebitda",
            "targetLowPrice",
            "recommendationKey",
            "grossProfits",
            "freeCashflow",
            "targetMedianPrice",
            "currentPrice",
            "earningsGrowth",
            "currentRatio",
            "returnOnAssets",
            "numberOfAnalystOpinions",
            "targetMeanPrice",
            "debtToEquity",
            "returnOnEquity",
            "targetHighPrice",
            "totalCash",
            "totalDebt",
            "totalRevenue",
            "totalCashPerShare",
            "financialCurrency",
            "revenuePerShare",
            "quickRatio",
            "recommendationMean",
            "exchange",
            "shortName",
            "longName",
            "exchangeTimezoneName",
            "exchangeTimezoneShortName",
            "isEsgPopulated",
            "gmtOffSetMilliseconds",
            "quoteType",
            "symbol",
            "messageBoardId",
            "market",
            "annualHoldingsTurnover",
            "enterpriseToRevenue",
            "beta3Year",
            "enterpriseToEbitda",
            "52WeekChange",
            "morningStarRiskRating",
            "forwardEps",
            "revenueQuarterlyGrowth",
            "sharesOutstanding",
            "fundInceptionDate",
            "annualReportExpenseRatio",
            "totalAssets",
            "bookValue",
            "sharesShort",
            "sharesPercentSharesOut",
            "fundFamily",
            "lastFiscalYearEnd",
            "heldPercentInstitutions",
            "netIncomeToCommon",
            "trailingEps",
            "lastDividendValue",
            "SandP52WeekChange",
            "priceToBook",
            "heldPercentInsiders",
            "nextFiscalYearEnd",
            "yield",
            "mostRecentQuarter",
            "shortRatio",
            "sharesShortPreviousMonthDate",
            "floatShares",
            "beta",
            "enterpriseValue",
            "priceHint",
            "threeYearAverageReturn",
            "lastSplitDate",
            "lastSplitFactor",
            "legalType",
            "lastDividendDate",
            "morningStarOverallRating",
            "earningsQuarterlyGrowth",
            "priceToSalesTrailing12Months",
            "dateShortInterest",
            "pegRatio",
            "ytdReturn",
            "forwardPE",
            "lastCapGain",
            "shortPercentOfFloat",
            "sharesShortPriorMonth",
            "impliedSharesOutstanding",
            "category",
            "fiveYearAverageReturn",
            "previousClose",
            "regularMarketOpen",
            "twoHundredDayAverage",
            "trailingAnnualDividendYield",
            "payoutRatio",
            "volume24Hr",
            "regularMarketDayHigh",
            "navPrice",
            "averageDailyVolume10Day",
            "regularMarketPreviousClose",
            "fiftyDayAverage",
            "trailingAnnualDividendRate",
            "open",
            "toCurrency",
            "averageVolume10days",
            "expireDate",
            "algorithm",
            "dividendRate",
            "exDividendDate",
            "circulatingSupply",
            "startDate",
            "regularMarketDayLow",
            "currency",
            "trailingPE",
            "regularMarketVolume",
            "lastMarket",
            "maxSupply",
            "openInterest",
            "marketCap",
            "volumeAllCurrencies",
            "strikePrice",
            "averageVolume",
            "dayLow",
            "ask",
            "askSize",
            "volume",
            "fiftyTwoWeekHigh",
            "fromCurrency",
            "fiveYearAvgDividendYield",
            "fiftyTwoWeekLow",
            "bid",
            "tradeable",
            "dividendYield",
            "bidSize",
            "dayHigh",
            "regularMarketPrice",
            "preMarketPrice",
            "logo_url",
            "trailingPegRatio"))
            company = yf.Ticker(stock_name_list[0])
            st.write(company.info[attributes])

            #st.write(stock_details.get_stats_valuation(stock_name_list[0]))
            income_statement = stock_details.get_income_statement(stock_name_list[0])
            income_statement = income_statement.transpose()
            #st.write(income_statement)

            #plotting income details
            fig1, ax1 = plt.subplots()
            income_col = st.selectbox("Select an attribute",income_statement.columns)
            sns.lineplot(income_statement.index,income_statement[income_col],axes=ax1,label=income_col)
            #ax1.set_xlim(1,10)
            st.write(fig1)

