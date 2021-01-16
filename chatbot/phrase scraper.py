import requests
from bs4 import BeautifulSoup
import smtplib
import numpy
import string

url = 'https://bestlifeonline.com/dirty-jokes/'
headers = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36'


def scrape_text():
    page = requests.get(url, headers)
    soup = BeautifulSoup(page.content, 'html.parser')
    text = soup.get_text()
    with open("phrases.txt", "w", encoding="utf-8") as phrases:
        phrases.write(str(text))


def clean_data():
    data = open("phrases.txt", "r", encoding="utf-8").readlines()
    i = 0
    for Line in "phrases.txt":
        data[i] = Line.split(",")
        i += 1
    while ("\n" in data):
        data.remove("\n")
    del data[0:11]
    data = [Line.lower() for Line in data]
    data = [''.join(c for c in s if c not in string.punctuation) for s in data]
    data = [Line.strip() for Line in data]
    print(data)
    digits = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
              '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'}
    for Line in data:
        for x in Line:
            if x.isdigit():
                data=[x.replace(x, digits[x]) in data]
    print(data)
    return data
def array():
    data = clean_data()
    data = numpy.array(data)
    print(data)



def main():
    scrape_text()
    clean_data()
    array()

main()
