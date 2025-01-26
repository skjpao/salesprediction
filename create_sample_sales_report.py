import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Luodaan realistinen myyntiraportti
def create_sample_sales_report():
    # Aloitetaan vuoden alusta
    start_date = datetime(2023, 1, 1)
    dates = []
    sales_data = []
    
    # Luodaan 365 päivän data
    for i in range(365):
        current_date = start_date + timedelta(days=i)
        
        # Perusmyynti arkipäiville
        base_sales = np.random.normal(1200, 200)
        
        # Viikonlopun korotus (pe-la +40%, su +20%)
        if current_date.weekday() in [4, 5]:
            base_sales *= 1.4
        elif current_date.weekday() == 6:
            base_sales *= 1.2
            
        # Palkkapäivien vaikutus (15. ja viimeinen päivä)
        is_payday = 1 if current_date.day in [15, -1] else 0
        if is_payday:
            base_sales *= 1.3
            
        # Kausivaihtelu (kesällä enemmän myyntiä)
        summer_effect = 1 + 0.2 * np.sin(2 * np.pi * (i - 172) / 365)  # Huippu heinäkuussa
        base_sales *= summer_effect
        
        # Lisätään satunnaiset tapahtumat (10% päivistä)
        has_event = 1 if np.random.random() < 0.1 else 0
        if has_event:
            base_sales *= 1.25
            
        # Pyöristetään myynti kahden desimaalin tarkkuuteen
        total_sales = round(base_sales, 2)
        
        # Jaetaan myynti eri kategorioihin
        ruoka = total_sales * np.random.normal(0.6, 0.05)  # n. 60% ruokaa
        juomat = total_sales * np.random.normal(0.3, 0.05)  # n. 30% juomia
        muut = total_sales - ruoka - juomat  # loput muuta myyntiä
        
        # Lisätään päivän tiedot
        sales_data.append({
            'Päivämäärä': current_date.strftime('%d.%m.%Y'),
            'Viikonpäivä': ['Ma', 'Ti', 'Ke', 'To', 'Pe', 'La', 'Su'][current_date.weekday()],
            'Kokonaismyynti': round(total_sales, 2),
            'Ruokamyynti': round(ruoka, 2),
            'Juomamyynti': round(juomat, 2),
            'Muu myynti': round(muut, 2),
            'Asiakkaita': int(total_sales / np.random.normal(25, 5)),  # keskiostos 20-30€
            'Palkkapäivä': is_payday,
            'Tapahtuma': has_event
        })
    
    # Luodaan DataFrame ja tallennetaan CSV-tiedostoksi
    df = pd.DataFrame(sales_data)
    df.to_csv('myyntiraportti.csv', index=False, decimal=',', sep=';')
    print("Myyntiraportti luotu tiedostoon 'myyntiraportti.csv'")

if __name__ == "__main__":
    create_sample_sales_report() 