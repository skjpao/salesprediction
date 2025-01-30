import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def load_sales_data(filename='myyntiraportti.csv'):
    try:
        # Luetaan CSV-tiedosto
        df = pd.read_csv(filename, sep=';', decimal=',', parse_dates=['Päivämäärä'], 
                        date_parser=lambda x: datetime.strptime(x, '%d.%m.%Y'))
        
        print(f"\nLuettu {len(df)} päivän myyntitiedot.")
        print(f"Aikaväli: {df['Päivämäärä'].min().strftime('%d.%m.%Y')} - {df['Päivämäärä'].max().strftime('%d.%m.%Y')}")
        print(f"\nKeskimääräinen päivämyynti: {df['Kokonaismyynti'].mean():.2f}€")
        print(f"Paras päivä: {df.loc[df['Kokonaismyynti'].idxmax(), 'Päivämäärä'].strftime('%d.%m.%Y')}: {df['Kokonaismyynti'].max():.2f}€")
        print(f"Heikoin päivä: {df.loc[df['Kokonaismyynti'].idxmin(), 'Päivämäärä'].strftime('%d.%m.%Y')}: {df['Kokonaismyynti'].min():.2f}€")
        
        # Muodostetaan feature matrix
        X = np.column_stack([
            df['Viikonpäivä'],      # Viikonpäivä (1-7)
            df['Asiakkaita'],       # Asiakasmäärä
            df['Palkkapäivä'],      # Palkkapäivä (0/1)
            df['Tapahtuma']         # Tapahtuma (0/1)
        ])
        
        y = df['Kokonaismyynti'].values
        
        return X, y, df
        
    except FileNotFoundError:
        print(f"Virhe: Tiedostoa '{filename}' ei löydy.")
        print("Varmista että olet luonut myyntiraportin ajamalla ensin create_sample_sales_report.py")
        exit(1)
    except Exception as e:
        print(f"Virhe tiedoston lukemisessa: {e}")
        exit(1)

def predict_sales(model, df, weekday, is_payday, has_event):
    """Ennustaa päivän myynnin ja asiakasmäärän"""
    if not (1 <= weekday <= 7):
        raise ValueError("Viikonpäivän tulee olla välillä 1-7")
    
    # Arvioidaan asiakasmäärä historiallisen datan perusteella
    avg_customers = df[df['Päivämäärä'].dt.weekday == weekday]['Asiakkaita'].mean()
    
    # Säädetään asiakasmäärää palkkapäivän ja tapahtumien mukaan
    if is_payday:
        avg_customers *= 1.3
    if has_event:
        avg_customers *= 1.25
    
    estimated_customers = int(avg_customers)
    
    # Tehdään myyntiennuste
    input_data = tf.convert_to_tensor([[weekday, estimated_customers, is_payday, has_event]], 
                                    dtype=tf.float32)
    prediction = model.predict(input_data)
    predicted_sales = max(0, prediction[0][0])
    
    return predicted_sales, estimated_customers

def create_prediction_for_period(model, df, start_date, end_date):
    """Luo ennusteet annetulle aikavälille"""
    dates = []
    predictions = []
    current_date = start_date
    
    while current_date <= end_date:
        # Määritellään päivän ominaisuudet
        weekday = current_date.weekday() + 1  # 1-7
        is_payday = 1 if current_date.day in [15, -1] else 0
        
        # Oletetaan tapahtumia olevan viikonloppuisin todennäköisemmin
        has_event = 1 if weekday in [5, 6, 7] and np.random.random() < 0.3 else 0
        
        # Haetaan historiallinen keskimääräinen asiakasmäärä tälle viikonpäivälle
        avg_customers = df[df['Viikonpäivä'] == weekday]['Asiakkaita'].mean()
        
        # Säädetään asiakasmäärää palkkapäivän ja tapahtumien mukaan
        if is_payday:
            avg_customers *= 1.3
        if has_event:
            avg_customers *= 1.25
        
        estimated_customers = int(avg_customers)
        
        # Tehdään ennuste
        input_data = tf.convert_to_tensor([[weekday, estimated_customers, is_payday, has_event]], 
                                        dtype=tf.float32)
        prediction = model.predict(input_data, verbose=0)[0][0]
        
        dates.append(current_date)
        predictions.append(prediction)
        
        current_date += timedelta(days=1)
    
    return dates, predictions

def visualize_period_prediction(df, start_date, end_date, model):
    """Visualisoi ennusteet ja toteutuneet myynnit aikavälillä"""
    # Haetaan toteutuneet myynnit aikaväliltä
    mask = (df['Päivämäärä'] >= start_date) & (df['Päivämäärä'] <= end_date)
    actual_data = df[mask]
    
    # Luodaan ennusteet samalle aikavälille
    dates, predictions = create_prediction_for_period(model, df, start_date, end_date)
    
    # Piirretään kuvaaja
    plt.figure(figsize=(15, 7))
    
    # Toteutuneet myynnit
    if not actual_data.empty:
        plt.plot(actual_data['Päivämäärä'], actual_data['Kokonaismyynti'], 
                label='Toteutunut myynti', alpha=0.6, color='blue')
    
    # Ennusteet
    plt.plot(dates, predictions, label='Ennuste', alpha=0.6, color='red', linestyle='--')
    
    plt.title('Myyntiennuste ja toteutuneet myynnit')
    plt.xlabel('Päivämäärä')
    plt.ylabel('Myynti (€)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Asetetaan y-akselin minimi nollaan
    y_max = max(max(predictions), actual_data['Kokonaismyynti'].max() if not actual_data.empty else max(predictions))
    margin = y_max * 0.1
    plt.ylim(0, y_max + margin)
    
    # Muokataan x-akselin päivämäärien näyttämistä
    ax = plt.gca()
    
    # Määritellään sopiva määrä x-akselin merkintöjä
    days = (end_date - start_date).days
    if days <= 14:
        # Alle 2 viikon jaksolla näytetään kaikki päivät
        interval = 1
    elif days <= 31:
        # Kuukauden jaksolla näytetään joka toinen päivä
        interval = 2
    else:
        # Pidemmällä jaksolla näytetään noin viikon välein
        interval = max(days // 20, 7)
    
    # Luodaan x-akselin merkinnät
    dates_list = [start_date + timedelta(days=x) for x in range(0, days + 1)]
    ticks = dates_list[::interval]
    
    # Luodaan merkintöjen tekstit (päivämäärä + viikonpäivä)
    viikonpaivat = ['Maanantai', 'Tiistai', 'Keskiviikko', 'Torstai', 'Perjantai', 'Lauantai', 'Sunnuntai']
    labels = [f"{d.strftime('%d.%m.')}\n{viikonpaivat[d.weekday()]}" for d in ticks]
    
    plt.xticks(ticks, labels, rotation=45, ha='right')
    
    # Säädetään marginaaleja niin että kaikki tekstit näkyvät
    plt.tight_layout()
    
    # Tallennetaan kuva
    plt.savefig('myyntiennuste_aikavali.png', dpi=300, bbox_inches='tight')
    plt.close()

def get_date_input(prompt):
    """Pyytää käyttäjältä päivämäärän"""
    while True:
        try:
            date_str = input(prompt)
            return datetime.strptime(date_str, '%d.%m.%Y')
        except ValueError:
            print("Virheellinen päivämäärä. Käytä muotoa pp.mm.vvvv")

def main():
    # Ladataan myyntidata
    X, y, df = load_sales_data()
    
    # Muunnetaan numpy arrayt TensorFlow tensoreiksi
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)
    
    # Määritellään malli
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    # Kompiloidaan malli
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )
    
    print("\nKoulutetaan mallia...")
    history = model.fit(
        X_tensor, 
        y_tensor,
        epochs=1000,
        verbose=1,
        batch_size=32
    )
    
    print("Mallin koulutus valmis!")
    
    while True:
        print("\nValitse toiminto:")
        print("1 = Tee yksittäisen päivän ennuste")
        print("2 = Tee aikavälin ennuste")
        print("3 = Lopeta")
        
        valinta = input("\nValintasi: ")
        
        if valinta == "1":
            print("\nTee myyntiennuste:")
            viikonpaivat = ['Maanantai', 'Tiistai', 'Keskiviikko', 'Torstai', 'Perjantai', 'Lauantai', 'Sunnuntai']
            
            for i, paiva in enumerate(viikonpaivat):
                print(f"{i+1} = {paiva}")
            
            try:
                weekday = int(input("\nValitse viikonpäivä (1-7): "))
                if not (1 <= weekday <= 7):
                    raise ValueError("Viikonpäivän tulee olla välillä 1-7")
                weekday -= 1  # Muunnetaan indeksiksi (0-6)
                
                is_payday = 1 if input("Onko palkkapäivä? (k/e): ").lower() == 'k' else 0
                has_event = 1 if input("Onko lähistöllä tapahtuma? (k/e): ").lower() == 'k' else 0
                
                predicted_sales, estimated_customers = predict_sales(model, df, weekday+1, is_payday, has_event)
                
                print(f"\nEnnuste päivälle:")
                print(f"Päivä: {viikonpaivat[weekday]}")
                print(f"Ennustettu asiakasmäärä: {estimated_customers}")
                print(f"Palkkapäivä: {'Kyllä' if is_payday else 'Ei'}")
                print(f"Tapahtuma: {'Kyllä' if has_event else 'Ei'}")
                print(f"Ennustettu myynti: {predicted_sales:.2f}€")
                print(f"Ennustettu keskiostos: {predicted_sales/estimated_customers:.2f}€/asiakas")
                
                # Näytetään vertailu historialliseen dataan
                hist_avg = df[df['Viikonpäivä'] == weekday+1]['Kokonaismyynti'].mean()
                print(f"\nVertailu:")
                
                # Muodostetaan päätteen teksti viikonpäivän mukaan
                paate = "na" if weekday == 2 else "sin"  # Keskiviikko = na, muut = sin
                print(f"Historiallinen keskimyynti {viikonpaivat[weekday]}{paate}: {hist_avg:.2f}€")
                print(f"Ennusteen ero keskiarvoon: {((predicted_sales/hist_avg) - 1)*100:.1f}%")
                
            except (ValueError, IndexError) as e:
                print(f"Virhe: {e}")
                continue
            
        elif valinta == "2":
            try:
                print("\nAnna aikaväli ennusteelle:")
                start_date = get_date_input("Alkupäivä (pp.mm.vvvv): ")
                end_date = get_date_input("Loppupäivä (pp.mm.vvvv): ")
                
                if end_date < start_date:
                    raise ValueError("Loppupäivä ei voi olla ennen alkupäivää")
                
                if (end_date - start_date).days > 365:
                    raise ValueError("Aikaväli voi olla korkeintaan vuoden")
                
                visualize_period_prediction(df, start_date, end_date, model)
                print(f"\nEnnuste luotu välille {start_date.strftime('%d.%m.%Y')} - {end_date.strftime('%d.%m.%Y')}")
                print("Kuvaaja tallennettu tiedostoon 'myyntiennuste_aikavali.png'")
                
            except ValueError as e:
                print(f"Virhe: {e}")
                continue
                
        elif valinta == "3":
            break
        else:
            print("Virheellinen valinta")

if __name__ == "__main__":
    main()