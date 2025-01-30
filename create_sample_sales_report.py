import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_sample_sales_report(years=3):
    # Aloitetaan kolme vuotta sitten
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)
    
    dates = []
    sales_data = []
    weekdays = []
    customers = []
    paydays = []
    events = []
    
    current_date = start_date
    while current_date <= end_date:
        # Perusmyynti arkipäiville (1000-1400€)
        base_sales = np.random.normal(1200, 200)
        
        # Viikonlopun korotus (pe-la +40%, su +20%)
        weekday = current_date.weekday()
        if weekday in [4, 5]:  # pe-la
            base_sales *= 1.4
        elif weekday == 6:  # su
            base_sales *= 1.2
            
        # Palkkapäivien vaikutus (15. ja viimeinen päivä)
        is_payday = 1 if current_date.day in [15, -1] else 0
        if is_payday:
            base_sales *= 1.3
            
        # Kausivaihtelu (kesällä enemmän myyntiä)
        day_of_year = current_date.timetuple().tm_yday
        summer_effect = 1 + 0.3 * np.sin(2 * np.pi * (day_of_year - 172) / 365)  # Huippu heinäkuussa
        base_sales *= summer_effect
        
        # Lisätään satunnaiset tapahtumat (10% päivistä)
        has_event = 1 if np.random.random() < 0.1 else 0
        if has_event:
            base_sales *= 1.25
        
        # Arvioidaan asiakasmäärä
        avg_customers_per_euro = np.random.normal(0.05, 0.01)  # keskimäärin 1 asiakas / 20€
        customer_count = int(base_sales * avg_customers_per_euro)
        
        # Tallennetaan päivän tiedot
        dates.append(current_date)
        sales_data.append(round(base_sales, 2))
        weekdays.append(weekday + 1)  # 1-7, ma-su
        customers.append(customer_count)
        paydays.append(is_payday)
        events.append(has_event)
        
        current_date += timedelta(days=1)
    
    # Luodaan DataFrame
    df = pd.DataFrame({
        'Päivämäärä': dates,
        'Kokonaismyynti': sales_data,
        'Viikonpäivä': weekdays,
        'Asiakkaita': customers,
        'Palkkapäivä': paydays,
        'Tapahtuma': events
    })
    
    # Tallennetaan CSV-tiedostoon
    df.to_csv('myyntiraportti.csv', sep=';', decimal=',', index=False, 
              date_format='%d.%m.%Y')
    
    print(f"Luotu myyntiraportti {len(df)} päivälle välille "
          f"{start_date.strftime('%d.%m.%Y')} - {end_date.strftime('%d.%m.%Y')}")

if __name__ == "__main__":
    create_sample_sales_report(years=3) 