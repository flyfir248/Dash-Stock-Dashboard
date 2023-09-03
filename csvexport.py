import os
import yfinance as yf
from nsetools import Nse
from datetime import datetime

# Define a dictionary that maps company names to stock symbols
company_symbol_mapping = {
    "Reliance Industries Ltd.": "RELIANCE.BO",
    "Tata Consultancy Services Ltd.": "TCS.BO",
    "HDFC Bank Ltd.": "HDFCBANK.BO",
    "ICICI Bank Ltd.": "ICICIBANK.BO",
    "Infosys Ltd.": "INFY.BO",
    "Hindustan Unilever Ltd.": "HINDUNILVR.BO",
    "ITC Ltd.": "ITC.BO",
    "Bharti Airtel Ltd.": "BHARTIARTL.BO",
    "State Bank Of India": "SBIN.BO",
    "Bajaj Finance Ltd.": "BAJFINANCE.BO",
    "Larsen & Toubro Ltd.": "LT.BO",
    "Kotak Mahindra Bank Ltd.": "KOTAKBANK.BO",
    "HCL Technologies Ltd.": "HCLTECH.BO",
    "Asian Paints Ltd.": "ASIANPAINT.BO",
    "Maruti Suzuki India Ltd.": "MARUTI.BO",
    "Axis Bank Ltd.": "AXISBANK.BO",
    "Adani Enterprises Ltd.": "ADANIENT.BO",
    "Titan Company Ltd.": "TITAN.BO",
    "Sun Pharmaceutical Industries Ltd.": "SUNPHARMA.BO",
    "Avenue Supermarts Ltd.": "DMART.BO",
    "Bajaj Finserv Ltd.": "BAJAJFINSV.BO",
    "Ultratech Cement Ltd.": "ULTRACEMCO.BO",
    "Oil & Natural Gas Corporation Ltd.": "ONGC.BO",
    "NTPC Ltd.": "NTPC.BO",
    "Wipro Ltd.": "WIPRO.BO",
    "Nestle India Ltd.": "NESTLEIND.BO",
    "Tata Motors Ltd.": "TATAMOTORS.BO",
    "Mahindra & Mahindra Ltd.": "M&M.BO",
    "JSW Steel Ltd.": "JSWSTEEL.BO",
    "Power Grid Corporation Of India Ltd.": "POWERGRID.BO",
    "Adani Ports and Special Economic Zone Ltd.": "ADANIPORTS.BO",
    "LTIMindtree Ltd.": "MINDTREE.BO",
    "Tata Steel Ltd.": "TATASTEEL.BO",
    "Coal India Ltd.": "COALINDIA.BO",
    "Siemens Ltd.": "SIEMENS.BO",
    "HDFC Life Insurance Company Ltd.": "HDFCLIFE.BO",
    "SBI Life Insurance Company Ltd.": "SBILIFE.BO",
    "Bajaj Auto Ltd.": "BAJAJ-AUTO.BO",
    "Adani Power Ltd.": "ADANIPOWER.BO",
    "Pidilite Industries Ltd.": "PIDILITIND.BO",
    "Indian Oil Corporation Ltd.": "IOC.BO",
    "DLF Ltd.": "DLF.BO",
    "Tech Mahindra Ltd.": "TECHM.BO",
    "Grasim Industries Ltd.": "GRASIM.BO",
    "Varun Beverages Ltd.": "VBL.BO",
    "IndusInd Bank Ltd.": "INDUSINDBK.BO",
    "Britannia Industries Ltd.": "BRITANNIA.BO",
    "Hindalco Industries Ltd.": "HINDALCO.BO",
    "Godrej Consumer Products Ltd.": "GODREJCP.BO",
    "Bharat Electronics Ltd.": "BEL.BO",
    "Cipla Ltd.": "CIPLA.BO",
    "Bank Of Baroda": "BANKBARODA.BO",
    "Dabur India Ltd.": "DABUR.BO",
    "Divis Laboratories Ltd.": "DIVISLAB.BO",
    "Interglobe Aviation Ltd.": "INDIGO.BO",
    "Eicher Motors Ltd.": "EICHERMOT.BO",
    "Dr. Reddys Laboratories Ltd.": "DRREDDY.BO",
    "Cholamandalam Investment and Finance Company Ltd.": "CHOLAFIN.BO",
    "Vedanta Ltd.": "VEDL.BO",
    "Shree Cement Ltd.": "SHREECEM.BO",
    "Havells India Ltd.": "HAVELLS.BO",
    "Ambuja Cements Ltd.": "AMBUJACEM.BO",
    "Zomato Ltd.": "ZOMATO.BO",
    "Bajaj Holdings & Investment Ltd.": "BAJAJHLDNG.BO",
    "Tata Power Company Ltd.": "TATAPOWER.BO",
    "ICICI Prudential Life Insurance Company Ltd.": "ICICIPRULI.BO",
    "GAIL (India) Ltd.": "GAIL.BO",
    "SBI Cards And Payment Services Ltd.": "SBICARD.BO",
    "Tata Consumer Products Ltd.": "TATACONSUM.BO",
    "Bharat Petroleum Corporation Ltd.": "BPCL.BO",
    "United Spirits Ltd.": "MCDOWELL-N.BO",
    "Marico Ltd.": "MARICO.BO",
    "Trent Ltd.": "TRENT.BO",
    "Shriram Finance Ltd.": "SRF.BO",
    "Apollo Hospitals Enterprise Ltd.": "APOLLOHOSP.BO",
    "ICICI Lombard General Insurance Company Ltd.": "ICICIGI.BO",
    "The Indian Hotels Company Ltd.": "INDHOTEL.BO",
    "Hero MotoCorp Ltd.": "HEROMOTOCO.BO",
    "Tube Investments of India Ltd.": "TUBEINVEST.BO",
    "Info Edge (India) Ltd.": "NAUKRI.BO",
    "Max Healthcare Institute Ltd.": "MAXHEALTH.BO",
    "Indian Railway Catering And Tourism Corporation Ltd.": "IRCTC.BO",
    "PI Industries Ltd.": "PIIND.BO",
    "Ashok Leyland Ltd.": "ASHOKLEY.BO",
    "Colgate-Palmolive (India) Ltd.": "COLPAL.BO",
    "Bharat Forge Ltd.": "BHARATFORG.BO",
    "Lupin Ltd.": "LUPIN.BO",
    "AU Small Finance Bank Ltd.": "AUBANK.BO",
    "Mphasis Ltd.": "MPHASIS.BO",
    "Tata Elxsi Ltd.": "TATAELXSI.BO",
    "UPL Ltd.": "UPL.BO",
    "Page Industries Ltd.": "PAGEIND.BO",
    "Bandhan Bank Ltd.": "BANDHANBNK.BO",
    "ACC Ltd.": "ACC.BO",
    "The Federal Bank Ltd.": "FEDERALBNK.BO",
    "Jubilant FoodWorks Ltd.": "JUBLFOOD.BO",
    "Voltas Ltd.": "VOLTAS.BO",
    "Zee Entertainment Enterprises Ltd.": "ZEEL.BO",
    "Crompton Greaves Consumer Electricals Ltd.": "CROMPTON.BO",
}


def download_historical_data(stock_symbol, output_directory, start_date, end_date):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Convert end_date to the current system date
    if end_date == "system_date":
        end_date = datetime.today().strftime('%Y-%m-%d')

    # Fetch historical stock data using yfinance
    data = yf.download(stock_symbol, start=start_date, end=end_date)

    # Create a CSV file name based on the stock symbol
    csv_filename = os.path.join(output_directory, f"{stock_symbol}.csv")

    # Save the data to a CSV file with date as the index
    data.to_csv(csv_filename)

    print(f"Saved historical data for {stock_symbol} to {csv_filename}")

if __name__ == "__main__":
    # Directory to store CSV files
    output_directory = "historical_stock_data"

    # Date range for historical data
    start_date = "2000-01-01"
    end_date = "system_date"  # Set end_date to "system_date" for the current system date

    # Get user input for the company name
    user_input = input("Enter a company name: ")
    stock_symbol = company_symbol_mapping.get(user_input)

    if stock_symbol:
        # Download historical data for the specified company
        download_historical_data(stock_symbol, output_directory, start_date, end_date)
    else:
        print("Company not found in the mapping.")