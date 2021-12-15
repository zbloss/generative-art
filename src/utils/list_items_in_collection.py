from selenium.webdriver import Firefox, Chrome
from selenium.webdriver.chrome.options import Options
from selenium import webdriver

from glob import glob
import os
import time
from tqdm import tqdm
import time
import requests

url = "https://api.opensea.io/api/v1/assets"
permalinks = []
for offset in tqdm(range(0, 1100, 50)):
    time.sleep(2)
    querystring = {"order_direction":"desc","offset":str(offset),"limit":"50","collection":"monsters-evolution"}
    response = requests.request("GET", url, params=querystring)
    assert int(response.status_code) == 200, f'Returned Non-200 HTTP Response Code'
    assets = response.json()['assets']
    for asset in assets:
        permalinks.append(asset['permalink'])

permalinks = list(set(permalinks))

options = Options()
options.add_argument("--disable-infobars")
options.add_argument("--enable-file-cookies")
options.add_experimental_option('excludeSwitches', ['enable-logging'])
options.add_extension('~/Downloads/coinbase_wallet.crx')
browser = Chrome(options=options)
browser.get('https://opensea.io/')


x = input("Please log in to your ETH wallet, then click [ENTER] to continue")

for link in tqdm(permalinks):
    try:
        browser.get(link)
        time.sleep(3)
        buttons = browser.find_elements_by_xpath("//*[contains(text(), 'Sell')]")
        buttons[0].click()
        time.sleep(3)
        
        browser.find_element_by_class_name('Input--input').send_keys('0.1')
        buttons = browser.find_elements_by_xpath("//*[contains(text(), 'Post your listing')]")
        buttons[0].click()
        time.sleep(8)
        print(f'Listed: {link}')
    except:
        pass