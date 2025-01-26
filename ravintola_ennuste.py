import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Luodaan keinotekoista dataa demonstraatiota varten
np.random.seed(42)

# Generoidaan 100 päivän data
n_samples = 100

# Tekijät jotka vaikuttavat myyntiin:
# - Viikonpäivä (1-7, 1=maanantai)
# - Lämpötila (Celsius)
# - Onko palkkapäivä (0 tai 1)
# - Onko tapahtuma lähistöllä (0 tai 1)

weekdays = np.random.randint(1, 8, n_samples)
temperatures = np.random.uniform(10, 30, n_samples)
paydays = np.random.randint(0, 2, n_samples)
events = np.random.randint(0, 2, n_samples)

# Luodaan myynti perustuen tekijöihin ja lisätään hieman satunnaisuutta
base_sales = 1000
weekday_effect = weekdays * 50  # Viikonloppua kohti nouseva myynti
temp_effect = (temperatures - 20) * 30  # Optimaalinen lämpötila 20°C
payday_effect = paydays * 300
event_effect = events * 400

sales = (base_sales + weekday_effect + temp_effect + payday_effect + 
        event_effect + np.random.normal(0, 100, n_samples))

# Muodostetaan feature matrix
X = np.column_stack([weekdays, temperatures, paydays, events])
y = sales

# Muunnetaan numpy arrayt TensorFlow tensoreiksi
X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)

# Luodaan lineaarinen malli
class RestaurantModel(tf.Module):
    def __init__(self):
        self.weights = tf.Variable(tf.random.normal([4, 1]))
        self.bias = tf.Variable(tf.zeros([1]))
        
    def __call__(self, x):
        return tf.matmul(x, self.weights) + self.bias

# Määritellään loss-funktio
def loss_fn(model, x, y):
    pred = model(x)
    return tf.reduce_mean(tf.square(pred - tf.reshape(y, [-1, 1])))

# Luodaan malli ja optimoija
model = RestaurantModel()
optimizer = tf.optimizers.Adam(learning_rate=0.1)

# Koulutetaan malli
n_epochs = 200
for epoch in range(n_epochs):
    with tf.GradientTape() as tape:
        current_loss = loss_fn(model, X_tensor, y_tensor)
    
    gradients = tape.gradient(current_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {current_loss.numpy():.2f}")

# Tehdään ennuste uudelle päivälle
def predict_sales(model, df, weekday, is_payday, has_event):
    """Ennustaa päivän myynnin ja asiakasmäärän"""
    if not (0 <= weekday <= 6):
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
    prediction = model(input_data)
    predicted_sales = max(0, prediction.numpy()[0][0])
    
    return predicted_sales, estimated_customers

# Lisätään mallin suorituskyvyn mittarit
def calculate_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R² score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

# Lisätään käyttöliittymä ennusteiden tekemiseen
def get_user_input():
    print("\nSyötä tiedot ennustetta varten:")
    
    # Viikonpäivän syöttö
    print("Viikonpäivä (1=ma, 2=ti, 3=ke, 4=to, 5=pe, 6=la, 7=su):")
    while True:
        try:
            weekday = int(input("Anna viikonpäivä (1-7): "))
            if 1 <= weekday <= 7:
                break
            print("Virheellinen viikonpäivä. Käytä numeroita 1-7.")
        except ValueError:
            print("Virheellinen syöte. Käytä numeroita 1-7.")
    
    # Lämpötilan syöttö
    while True:
        try:
            temperature = float(input("Anna lämpötila (°C): "))
            if 0 <= temperature <= 40:
                break
            print("Virheellinen lämpötila. Käytä arvoja väliltä 0-40.")
        except ValueError:
            print("Virheellinen syöte. Käytä numeroita.")
    
    # Palkkapäivän syöttö
    while True:
        payday = input("Onko palkkapäivä? (k/e): ").lower()
        if payday in ['k', 'e']:
            is_payday = 1 if payday == 'k' else 0
            break
        print("Virheellinen syöte. Käytä k tai e.")
    
    # Tapahtuman syöttö
    while True:
        event = input("Onko lähistöllä tapahtuma? (k/e): ").lower()
        if event in ['k', 'e']:
            has_event = 1 if event == 'k' else 0
            break
        print("Virheellinen syöte. Käytä k tai e.")
    
    return weekday, temperature, is_payday, has_event

def load_or_generate_data():
    print("\nHaluatko käyttää omaa myyntidataa vai testidataa?")
    while True:
        valinta = input("Valitse (1=oma data, 2=testidata): ")
        if valinta == "1":
            return load_user_data()
        elif valinta == "2":
            return generate_test_data()
        print("Virheellinen valinta. Valitse 1 tai 2.")

def load_user_data():
    print("\nSyötä historiadataa vähintään 30 päivältä:")
    data = []
    
    while True:
        try:
            paivien_maara = int(input("Kuinka monen päivän tiedot syötät? (min. 30): "))
            if paivien_maara >= 30:
                break
            print("Tarvitaan vähintään 30 päivän tiedot.")
        except ValueError:
            print("Virheellinen syöte. Käytä numeroita.")
    
    for i in range(paivien_maara):
        print(f"\nPäivä {i+1}/{paivien_maara}:")
        try:
            # Päivämäärä
            while True:
                pvm = input("Anna päivämäärä (pp.kk.vvvv): ")
                try:
                    from datetime import datetime
                    dt = datetime.strptime(pvm, "%d.%m.%Y")
                    weekday = dt.weekday()  # 0-6, ma-su
                    break
                except ValueError:
                    print("Virheellinen päivämäärä. Käytä muotoa pp.kk.vvvv")
            
            # Myynti
            myynti = float(input("Anna päivän kokonaismyynti (€): "))
            if myynti < 0:
                raise ValueError("Myynnin tulee olla positiivinen luku")
            
            # Lämpötila
            temp = float(input("Anna päivän keskilämpötila (°C): "))
            if not -30 <= temp <= 40:
                raise ValueError("Lämpötilan tulee olla välillä -30...40°C")
            
            # Palkkapäivä
            while True:
                payday = input("Oliko palkkapäivä? (k/e): ").lower()
                if payday in ['k', 'e']:
                    is_payday = 1 if payday == 'k' else 0
                    break
                print("Virheellinen syöte. Käytä k tai e.")
            
            # Tapahtumat
            while True:
                event = input("Oliko lähistöllä tapahtumia? (k/e): ").lower()
                if event in ['k', 'e']:
                    has_event = 1 if event == 'k' else 0
                    break
                print("Virheellinen syöte. Käytä k tai e.")
            
            data.append([weekday, temp, is_payday, has_event, myynti])
            
        except ValueError as e:
            print(f"Virhe: {e}")
            print("Aloitetaan päivän tiedot alusta.")
            i -= 1  # Palataan takaisin samaan päivään
    
    data = np.array(data)
    X = data[:, :-1]  # Kaikki paitsi viimeinen sarake
    y = data[:, -1]   # Viimeinen sarake (myynti)
    
    return X, y

def generate_test_data():
    # Alkuperäinen testidatan generointi
    n_samples = 100
    weekdays = np.random.randint(1, 8, n_samples)
    temperatures = np.random.uniform(10, 30, n_samples)
    paydays = np.random.randint(0, 2, n_samples)
    events = np.random.randint(0, 2, n_samples)
    
    base_sales = 1000
    weekday_effect = weekdays * 50
    temp_effect = (temperatures - 20) * 30
    payday_effect = paydays * 300
    event_effect = events * 400
    
    sales = (base_sales + weekday_effect + temp_effect + payday_effect + 
            event_effect + np.random.normal(0, 100, n_samples))
    
    X = np.column_stack([weekdays, temperatures, paydays, events])
    return X, sales

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
            df['Päivämäärä'].dt.weekday,  # Viikonpäivä (1-7)
            df['Asiakkaita'],             # Asiakasmäärä
            df['Palkkapäivä'],            # Palkkapäivä (0/1)
            df['Tapahtuma']               # Tapahtuma (0/1)
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

def main():
    # Ladataan myyntidata
    X, y, df = load_sales_data()
    
    # Muunnetaan numpy arrayt TensorFlow tensoreiksi
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)
    
    # Koulutetaan malli
    model = RestaurantModel()
    optimizer = tf.optimizers.Adam(learning_rate=0.01)
    
    print("\nKoulutetaan mallia...")
    n_epochs = 1000
    for epoch in range(n_epochs):
        with tf.GradientTape() as tape:
            current_loss = loss_fn(model, X_tensor, y_tensor)
        
        gradients = tape.gradient(current_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {current_loss.numpy():.2f}")
            
    print("Mallin koulutus valmis!")
    
    # Analysoidaan tulokset
    predictions = model(X_tensor).numpy().flatten()
    
    # Visualisoidaan viikonpäivien vaikutus
    plt.figure(figsize=(12, 6))
    df['Viikonpäivä'] = pd.Categorical(df['Viikonpäivä'], 
                                      categories=['Ma', 'Ti', 'Ke', 'To', 'Pe', 'La', 'Su'])
    df.boxplot(column='Kokonaismyynti', by='Viikonpäivä', figsize=(10, 6))
    plt.title('Myynti viikonpäivittäin')
    plt.suptitle('')  # Poista automaattinen otsikko
    plt.ylabel('Myynti (€)')
    plt.savefig('myynti_viikonpaivittain.png')
    plt.close()
    
    # Tehdään ennusteita
    while True:
        print("\nTee myyntiennuste:")
        viikonpaivat = ['Maanantai', 'Tiistai', 'Keskiviikko', 'Torstai', 'Perjantai', 'Lauantai', 'Sunnuntai']
        
        for i, paiva in enumerate(viikonpaivat):
            print(f"{i} = {paiva}")
        
        try:
            weekday = int(input("\nValitse viikonpäivä (1-7): "))
            is_payday = 1 if input("Onko palkkapäivä? (k/e): ").lower() == 'k' else 0
            has_event = 1 if input("Onko lähistöllä tapahtuma? (k/e): ").lower() == 'k' else 0
            
            predicted_sales, estimated_customers = predict_sales(model, df, weekday, is_payday, has_event)
            
            print(f"\nEnnuste päivälle:")
            print(f"Päivä: {viikonpaivat[weekday]}")
            print(f"Ennustettu asiakasmäärä: {estimated_customers}")
            print(f"Palkkapäivä: {'Kyllä' if is_payday else 'Ei'}")
            print(f"Tapahtuma: {'Kyllä' if has_event else 'Ei'}")
            print(f"Ennustettu myynti: {predicted_sales:.2f}€")
            print(f"Ennustettu keskiostos: {predicted_sales/estimated_customers:.2f}€/asiakas")
            
            # Näytetään vertailu historialliseen dataan
            hist_avg = df[df['Päivämäärä'].dt.weekday == weekday]['Kokonaismyynti'].mean()
            print(f"\nVertailu:")
            print(f"Historiallinen keskimyynti {viikonpaivat[weekday]}isin: {hist_avg:.2f}€")
            print(f"Ennusteen ero keskiarvoon: {((predicted_sales/hist_avg) - 1)*100:.1f}%")
            
        except (ValueError, IndexError) as e:
            print(f"Virhe: {e}")
            continue
        
        if input("\nHaluatko tehdä uuden ennusteen? (k/e): ").lower() != 'k':
            break

if __name__ == "__main__":
    main() 