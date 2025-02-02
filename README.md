## Ravintolan Myyntiennuste

Tämä sovellus ennustaa ravintolan tulevaa myyntiä hyödyntäen historiallista myyntidataa ja koneoppimismalleja. Ennusteet auttavat ravintolan johtoa tekemään tietoon perustuvia päätöksiä esimerkiksi henkilöstön ja varastojen hallinnassa.

## Koneoppimismallin kouluttaminen TensorFlow'lla

Sovellus käyttää TensorFlow'ta myynnin ennustamiseen. Alla on yhteenveto mallin koulutusprosessista:

# Kirjastot

- tensorflow
- numpy
- pandas
- matplotlib

# Datan lataaminen

Historiallinen myyntidata ladataan CSV-tiedostosta myyntiraportti.csv. Data sisältää seuraavat sarakkeet:

- Päivämäärä: Myyntipäivä
- Kokonaismyynti: Päivän kokonaismyynti euroina
- Viikonpäivä: Päivän järjestysnumero viikossa (1=maanantai, 7=sunnuntai)
- Asiakkaita: Päivän asiakasmäärä

# Mallin koulutus

Mallia koulutetaan 1000 epochilla.
Koulutettu malli pyrkii ennustamaan tulevan päivän myynnin syötettyjen piirteiden perusteella.

## Testidatan luominen

Sovellus sisältää skriptin create_sample_sales_report.py, joka generoi testidataa seuraavasti:

- Aikaväli: Dataa luodaan halutulle ajanjaksolle (oletuksena yksi vuosi).
- Päivittäinen myynti: Perusmyynti arkipäiville (maanantai–torstai) arvotaan normaalijakaumasta (keskiarvo 1200€, keskihajonta 200€). Viikonloppuisin myyntiä korotetaan: perjantaina ja lauantaina +40%, sunnuntaina +20%.
- Kausivaihtelu: Myyntiin lisätään kausivaihtelua siten, että kesäkuukausina (huippu heinäkuussa) myynti on korkeampaa.
- Asiakasmäärä: Asiakasmäärä arvioidaan suhteessa myyntiin siten, että keskimäärin yksi asiakas tuo 20€ myyntiä.
- Datan tallennus: Luotu data tallennetaan CSV-tiedostoon myyntiraportti.csv.

Tämä testidata mahdollistaa mallin kehittämisen ja testaamisen ilman oikeaa historiallista myyntidataa.

## Käyttöönotto

1. Varmista, että sinulla on Python 3.8 tai uudempi asennettuna
2. Asenna tarvittavat kirjastot terminaalissa: 

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