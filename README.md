# Ravintolan Myyntiennuste

Sovellus ravintolan myynnin ennustamiseen koneoppimisen avulla. Ohjelma käyttää historiallista myyntidataa oppiakseen myyntiin vaikuttavat tekijät ja tekee ennusteita tuleville päiville.

## Ominaisuudet

- Yksittäisen päivän myyntiennuste
- Aikavälin myyntiennuste visualisointeineen
- Huomioi seuraavat tekijät:
  - Viikonpäivä
  - Asiakasmäärä
  - Palkkapäivät
  - Tapahtumat lähistöllä
- Vertailu historiallisiin keskiarvoihin
- Visuaalinen esitys ennusteista

## Käyttöönotto

1. Varmista, että sinulla on Python 3.8 tai uudempi asennettuna
2. Asenna tarvittavat kirjastot: 

pip install tensorflow numpy pandas matplotlib


## Käyttö

1. Luo ensin testidataa (tai käytä omaa myyntidataa):

python create_sample_sales_report.py

2. Käynnistä ennustus:

python ravintola_ennuste.py


3. Valitse haluamasi toiminto:
   - 1 = Yksittäisen päivän ennuste
   - 2 = Aikavälin ennuste
   - 3 = Lopeta

## Tiedostot

- `ravintola_ennuste.py` - Pääohjelma, sisältää ennustemallin ja käyttöliittymän
- `create_sample_sales_report.py` - Luo testidataa CSV-muodossa
- `myyntiraportti.csv` - Myyntidata (luodaan automaattisesti tai korvaa omalla)
- `myyntiennuste_aikavali.png` - Aikavälin ennusteen visualisointi (luodaan automaattisesti)

## Datan muoto

Ohjelma odottaa CSV-tiedoston sisältävän seuraavat sarakkeet:
- Päivämäärä (pp.mm.vvvv)
- Kokonaismyynti (€)
- Viikonpäivä (1-7, ma-su)
- Asiakkaita (lukumäärä)
- Palkkapäivä (0/1)
- Tapahtuma (0/1)

## Rajoitukset

- Aikavälin ennuste voi olla korkeintaan vuoden mittainen
- Vaatii vähintään 30 päivän historiadatan toimiakseen luotettavasti
- Ennusteet perustuvat historialliseen dataan, eivätkä huomioi erikoistapahtumia