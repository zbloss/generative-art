from selenium.webdriver import Firefox, Chrome
from selenium.webdriver.chrome.options import Options
from selenium import webdriver

from glob import glob
import os
import time
from tqdm.notebook import tqdm


options = Options()
options.add_argument("--disable-infobars")
options.add_argument("--enable-file-cookies")
options.add_experimental_option('excludeSwitches', ['enable-logging'])
options.add_extension('C:/Users/altoz/Downloads/coinbase_wallet.crx')
browser = Chrome(options=options)
browser.get('https://opensea.io/')


x = input("Please log in to your ETH wallet, then click [ENTER] to continue")
while True:
    try:
        browser.get('https://opensea.io/assets/monsters-evolution?collectionSlug=monsters-evolution&search[sortAscending]=false&search[sortBy]=LISTING_DATE')
        
        for _ in range(50):
            browser.execute_script("window.scrollTo(0,document.body.scrollHeight)")
            time.sleep(1)

        browser.find_element_by_class_name('AssetMedia--img').click()
        time.sleep(4)

        buttons = browser.find_elements_by_xpath("//*[contains(text(), 'Lower price')]")
        time.sleep(4)

        buttons = browser.find_elements_by_xpath("//*[contains(text(), 'Sell')]")
        buttons[0].click()
        time.sleep(4)
        
        browser.find_element_by_class_name('Input--input').send_keys('0.1')
        buttons = browser.find_elements_by_xpath("//*[contains(text(), 'Post your listing')]")
        buttons[0].click()
        time.sleep(8)
        
    except:
        pass